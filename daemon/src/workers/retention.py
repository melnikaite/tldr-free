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

Orphaned audio (``_sweep_orphaned_audio`` below) is the same idea applied to
``<data_dir>/audio``, for three confirmed leak sources none of which a
DB-driven delete path can ever reach (there's no row to key off of):

  1. A job row is gone (explicit delete, or a crash between the download and
     ``repo.set_audio``) while ``audio_path`` was still NULL — the file is
     now unreferenced forever.
  2. yt-dlp's pre-conversion intermediate (``workers/youtube.py``'s
     ``_download_audio_sync`` downloads e.g. ``<id>.m4a`` then runs
     ``FFmpegExtractAudio`` to produce ``<id>.opus``; the daemon only ever
     learns about the final file, so the intermediate is never referenced).
  3. Stale ``tldr-chunks-*`` / ``tldr-recut-*`` temp directories that
     ``workers/transcribe.py`` creates INSIDE the audio directory
     (``tempfile.mkdtemp(dir=audio_path.parent)``) and only cleans up on the
     happy path — a crash or kill mid-transcription leaves them behind.

Like the frame sweep, this runs every cycle regardless of
``storage.retention_days`` (a disk/DB consistency check, not an age-based
policy) and must never raise out of the sweep.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

from src.config import get_config
from src.storage import repo
from src.workers import frames

log = logging.getLogger(__name__)

_INTERVAL_SECONDS = 6 * 60 * 60     # 6 hours

# How long an audio-directory entry must sit unreferenced before the sweep
# will remove it. This is load-bearing, not a nicety: between the start of
# a yt-dlp download and the ``repo.set_audio(...)`` call in
# ``runner.py`` (~line 341) a perfectly live file has zero DB references —
# it's mid-download, the row simply hasn't been told about it yet. yt-dlp's
# own ``.part`` files and pre-conversion intermediates (see module
# docstring, leak source 2) never get a DB reference at all, ever, even on
# success. 24 hours is far beyond any plausible download-plus-transcription
# for a single job, and a clean multiple of ``_INTERVAL_SECONDS`` (6h), so
# an entry that just missed being "referenced" on one cycle is essentially
# guaranteed to be judged again at least three times before it's old enough
# to remove.
_AUDIO_ORPHAN_GRACE_SECONDS = 24 * 60 * 60     # 24 hours

# Prefixes ``workers/transcribe.py`` passes to ``tempfile.mkdtemp(dir=...)``
# for its chunking/recut scratch directories (see its module docstring,
# leak source 3 above). Duplicated here as plain string constants rather
# than imported — importing ``workers.transcribe`` from this module would
# pull in a much heavier dependency chain for two string literals, and
# ``transcribe.py`` is a higher layer than this file has any other reason to
# touch (same reasoning as ``_sweep_orphaned_frame_dirs`` importing
# ``workers.frames`` lazily-in-spirit rather than folding frame logic into
# ``storage``: keep the layering one-directional).
_CHUNK_TMP_DIR_PREFIX = "tldr-chunks-"
_RECUT_TMP_DIR_PREFIX = "tldr-recut-"


def _audio_dir() -> Path:
    """``<data_dir>/audio`` — mirrors ``workers.runner._audio_dir()``
    exactly (same ``config.storage.data_dir`` join), but computed locally
    instead of imported: ``workers.runner`` imports ``workers.transcribe``,
    and this module has no other reason to load that dependency chain just
    to reuse a one-line path join. Does NOT create the directory (unlike
    ``runner``'s version) — a missing audio directory is a legitimate no-op
    for this sweep, not something to conjure into existence."""
    return Path(get_config().storage.data_dir) / "audio"


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


def _newest_mtime_under(path: Path) -> float:
    """Newest mtime found anywhere inside ``path`` (recursively), or
    ``path``'s own mtime if it's empty (or on any stat error while walking
    it — conservative: treat as "just touched" rather than risk judging an
    unreadable directory as stale).

    Used to decide whether a ``tldr-chunks-*`` / ``tldr-recut-*`` scratch
    directory is safe to remove: a chunk file written a minute ago inside a
    directory created two hours ago means the transcription that owns it is
    still live, even though the directory itself looks old. The directory's
    own mtime is only consulted as the empty-directory fallback below — once
    there's real content, ONLY that content's mtimes decide, because
    creating the directory itself can predate its newest (or oldest) file by
    hours and must not mask either."""
    newest = path.stat().st_mtime  # fallback: what an empty directory is judged by
    found_any = False
    try:
        for entry in path.rglob("*"):
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                return time.time()  # unreadable entry — conservative: treat as just touched
            if not found_any or mtime > newest:
                newest = mtime
            found_any = True
    except OSError:
        return time.time()
    return newest


def _sweep_orphaned_audio() -> None:
    """Remove ``<data_dir>/audio`` entries that no job row references AND
    that have sat that way for longer than ``_AUDIO_ORPHAN_GRACE_SECONDS``.

    Deletion rule (BOTH must hold):
      - not referenced by any ``Job.audio_path`` in the DB, AND
      - old enough (see the grace-period constant's comment for why this
        matters — a fresh unreferenced file is very likely just mid-flight,
        not a leak).

    Only looks at the audio directory's own entries, plus one level of
    special-casing for ``tldr-chunks-*`` / ``tldr-recut-*`` scratch
    directories (judged by newest content mtime, see
    ``_newest_mtime_under``) — no other recursion, no following of anything
    else that happens to live under ``<data_dir>/audio``.

    Runs every cycle regardless of ``storage.retention_days`` — same
    disk/DB consistency-check status as ``_sweep_orphaned_frame_dirs``, not
    an age-based policy. Tolerates a missing audio directory (nothing has
    ever downloaded audio yet, or it hasn't been created this run) as a
    plain no-op.

    Must never raise out of here and abort the rest of the sweep — same
    "degrade and log" contract as the other sweeps in this module.
    """
    audio_dir = _audio_dir()
    if not audio_dir.is_dir():
        log.debug("retention sweep: audio directory does not exist yet, skipping")
        return

    try:
        entries = list(audio_dir.iterdir())
    except Exception:
        log.exception("retention sweep: failed to list audio directory; will retry")
        return

    try:
        referenced = repo.all_referenced_audio_paths()
    except Exception:
        log.exception("retention sweep: failed to read referenced audio paths; will retry")
        return

    # "Is this entry referenced?" used to be a single exact-string test
    # against ``referenced`` (built from ``runner._audio_dir()`` on one side
    # and ``repo.set_audio``'s ``str(audio_path)`` on the other). That's
    # fragile in a way whose blast radius is silent data loss: ``runner.py``
    # deliberately KEEPS a downloaded audio file (and its DB row) when a job
    # fails after download so a retry can skip yt-dlp — such a file can sit
    # on disk for weeks, way past the grace period above, protected ONLY by
    # this comparison. A trailing slash in ``storage.data_dir``, a
    # symlinked data dir (macOS ``/var`` vs ``/private/var``, ``/tmp``
    # likewise), a differently-expanded ``~``, or any future normalization
    # change on either side of the comparison, and the string stops
    # matching while the file is still very much in use. So three
    # independent ways to match, ANY of which is enough to call an entry
    # referenced — a false "unreferenced" costs a retry its cached
    # download, a false "referenced" costs nothing but one extra grace
    # period, so the check errs toward keeping:
    #   - the exact string, as before;
    #   - the resolved path, tolerating either side failing to resolve;
    #   - the plain basename, safe here specifically because this sweep
    #     only ever looks at one flat directory — a basename collision
    #     between a reference and an entry here is either the same file or
    #     a name clash we should refuse to delete anyway.
    # All three sets are built ONCE before the loop, not per entry.
    referenced_resolved: set[Path] = set()
    for ref in referenced:
        with contextlib.suppress(OSError):  # unresolvable reference just doesn't contribute
            referenced_resolved.add(Path(ref).resolve())
    referenced_basenames = {Path(ref).name for ref in referenced}

    now = time.time()
    removed = 0
    for entry in entries:
        try:
            name = entry.name
            if name.startswith(_CHUNK_TMP_DIR_PREFIX) or name.startswith(_RECUT_TMP_DIR_PREFIX):
                if not entry.is_dir():
                    continue
                newest = _newest_mtime_under(entry)
                if (now - newest) > _AUDIO_ORPHAN_GRACE_SECONDS:
                    shutil.rmtree(entry)
                    removed += 1
                continue

            if not entry.is_file():
                continue
            if str(entry) in referenced:
                continue
            if name in referenced_basenames:
                continue
            try:
                if entry.resolve() in referenced_resolved:
                    continue
            except OSError:
                pass  # entry itself unresolvable — fall through to the mtime check below
            mtime = entry.stat().st_mtime
            if (now - mtime) > _AUDIO_ORPHAN_GRACE_SECONDS:
                entry.unlink()
                removed += 1
        except Exception:
            log.exception("retention sweep: failed to clean up audio entry %s", entry)

    if removed:
        log.info("retention sweep: removed %d orphaned audio file(s)/dir(s)", removed)
    else:
        log.debug("retention sweep: no orphaned audio entries (checked %d)", len(entries))


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
            _sweep_orphaned_audio()
        except asyncio.CancelledError:
            log.info("retention worker cancelled")
            raise
        except Exception:
            log.exception("retention sweep failed; will retry on next interval")
        await asyncio.sleep(_INTERVAL_SECONDS)


__all__ = ["retention_worker"]
