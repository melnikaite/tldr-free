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
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

from src.config import get_config
from src.storage import repo

log = logging.getLogger(__name__)

_INTERVAL_SECONDS = 6 * 60 * 60     # 6 hours


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
        except asyncio.CancelledError:
            log.info("retention worker cancelled")
            raise
        except Exception:
            log.exception("retention sweep failed; will retry on next interval")
        await asyncio.sleep(_INTERVAL_SECONDS)


__all__ = ["retention_worker"]
