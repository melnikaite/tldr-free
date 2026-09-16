"""Periodic retention sweep — delete jobs older than ``config.storage.retention_days``.

Started from ``main.lifespan`` as a long-running coroutine. One sweep on
startup, then every ``_INTERVAL_SECONDS`` forever — the loop never exits,
even when retention is disabled (``retention_days == 0``), because
``retention_days`` is now editable at runtime via ``PATCH /config``'s
``storage.retention_days`` (see ``api/config.py``). Reading the config once
before the loop and exiting outright when disabled were both bugs against
that: a value changed from the options page couldn't take effect without a
daemon restart, and a value that was ``0`` at startup could never be turned
back on at all (the coroutine was already gone). So the config is re-read
EVERY cycle, and a ``0`` (or negative, shouldn't happen but defensively
treated the same) value just skips that cycle's sweep instead of exiting.
Log lines fire only when the effective value actually changes between
cycles, not on every quiet 6-hourly pass.

Shutdown: cancellation from the lifespan. The current sweep is cheap
(single DELETE per old row) so we don't try to interrupt mid-sweep.

Video frames (``workers/frames.py``) piggyback on this same sweep rather
than getting their own loop. ``repo.delete_job`` and
``repo.delete_jobs_older_than`` already call ``frames.delete_job_frames``
for every row THEY remove, so the age-based half of the job below needs no
extra code — deleting an aged-out job here already takes its frames with
it. What's missing, and what ``_sweep_orphaned_frame_dirs`` below adds, is
the case where a job ROW is gone (deleted through some other path, or a
crash between the row delete and the frame unlink) but its frame directory
was left behind with nothing left to ever revisit it — the age sweep never
sees it again because there's no row to compare a cutoff against.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

from src.config import get_config
from src.storage import repo
from src.workers import frames

log = logging.getLogger(__name__)

_INTERVAL_SECONDS = 6 * 60 * 60     # 6 hours


def _sweep_orphaned_frame_dirs() -> None:
    """Remove frame directories whose job row no longer exists in the DB.

    Cheap (frame directories are expected to be rare — see workers.md) and
    exactly the kind of thing a periodic retention pass is for. Runs every
    cycle regardless of ``storage.retention_days`` — this is a DB/disk
    consistency check, not an age-based policy, so disabling age-based
    retention must not disable it.

    Must never raise out of here and abort the rest of the sweep — same
    "degrade and log" contract as the age-based sweep above. A transient
    failure (e.g. disk I/O hiccup listing the frame root) just gets retried
    next cycle.
    """
    try:
        job_ids = frames.all_frame_job_ids()
    except Exception:
        log.exception("retention sweep: failed to list frame directories; will retry")
        return

    removed = 0
    for job_id in job_ids:
        try:
            if repo.get_job(job_id) is not None:
                continue
            if frames.delete_job_frames(job_id):
                removed += 1
        except Exception:
            log.exception("retention sweep: failed to clean up orphaned frames for job %s", job_id)

    if removed:
        log.info("retention sweep: removed %d orphaned frame director(y/ies)", removed)
    else:
        log.debug("retention sweep: no orphaned frame directories (checked %d)", len(job_ids))


async def retention_worker() -> None:
    """Loop forever. Each cycle re-reads ``storage.retention_days`` fresh
    (rather than once before the loop) and either sweeps or skips —
    disabled is never a reason to return, only to sit out that cycle. A
    change made via ``PATCH /config`` takes effect on the next cycle, i.e.
    within ``_INTERVAL_SECONDS`` — that latency is an accepted tradeoff, not
    a bug (see module docstring)."""
    last_logged_days: int | None = None
    while True:
        try:
            days = get_config().storage.retention_days
            if days != last_logged_days:
                if days > 0:
                    log.info("retention worker: enabled (retention_days=%d)", days)
                else:
                    log.info("retention worker: disabled (retention_days=%d)", days)
                last_logged_days = days

            if days > 0:
                cutoff = datetime.utcnow() - timedelta(days=days)
                n = repo.delete_jobs_older_than(cutoff)
                if n:
                    log.info("retention sweep deleted %d job(s) older than %s", n, cutoff)
                else:
                    log.debug("retention sweep deleted 0 jobs (cutoff=%s)", cutoff)
            else:
                log.debug("retention sweep skipped (disabled)")

            _sweep_orphaned_frame_dirs()
        except asyncio.CancelledError:
            log.info("retention worker cancelled")
            raise
        except Exception:
            log.exception("retention sweep failed; will retry on next interval")
        await asyncio.sleep(_INTERVAL_SECONDS)


__all__ = ["retention_worker"]
