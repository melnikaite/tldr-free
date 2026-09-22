"""Tests for src.workers.retention — periodic job-retention sweep.

The worker loops forever, so we break the loop by making the monkeypatched
``asyncio.sleep`` raise a sentinel exception after the first (or Nth)
iteration. The actual DB delete (``repo.delete_jobs_older_than``) is stubbed
so we can assert on call args / cutoff math without standing up a database.

Covers:
  - retention_days re-read from config EVERY cycle, not once before the loop
    — a change (e.g. via PATCH /config) takes effect on the next cycle
    without a daemon restart.
  - disabled (retention_days <= 0) skips the sweep but keeps looping —
    the worker must NEVER return, since a value that's 0 today could be
    turned back on tomorrow via the options page, and nothing re-spawns
    this coroutine after it exits.
  - cutoff is computed as now - retention_days
  - sweep runs and logs deletions vs no deletions
  - asyncio.CancelledError propagates (clean shutdown)
  - generic exceptions are swallowed and the loop continues to the next sleep
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.workers import retention


class _StopLoop(Exception):
    """Sentinel raised from the patched sleep to terminate the infinite loop."""


def _patch_config(monkeypatch: pytest.MonkeyPatch, days: int) -> None:
    cfg = retention.get_config()
    monkeypatch.setattr(cfg.storage, "retention_days", days)


def _stop_after(n: int):
    """Return an ``asyncio.sleep`` stand-in that raises ``_StopLoop`` on the
    Nth call (1-indexed) so a test can let the loop run a fixed number of
    cycles before terminating it."""
    calls = 0

    async def fake_sleep(_seconds: float) -> None:
        nonlocal calls
        calls += 1
        if calls >= n:
            raise _StopLoop

    return fake_sleep


# ---------------------------------------------------------------------------
# disabled retention — must skip the sweep but NEVER return (see module
# docstring: a value edited via PATCH /config must be able to re-enable
# retention without a daemon restart, which requires the coroutine to still
# be running).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_skips_sweep_but_keeps_looping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_config(monkeypatch, 0)
    called = False

    def fake_delete(_cutoff: datetime) -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(retention.repo, "delete_jobs_older_than", fake_delete)
    monkeypatch.setattr(retention.asyncio, "sleep", _stop_after(1))

    # Must NOT return on its own — the sentinel from the patched sleep is
    # what actually terminates the loop.
    with pytest.raises(_StopLoop):
        await retention.retention_worker()
    assert called is False


@pytest.mark.asyncio
async def test_disabled_when_retention_days_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_config(monkeypatch, -5)

    def fake_delete(_cutoff: datetime) -> int:  # pragma: no cover - must not run
        raise AssertionError("should not sweep when disabled")

    monkeypatch.setattr(retention.repo, "delete_jobs_older_than", fake_delete)
    monkeypatch.setattr(retention.asyncio, "sleep", _stop_after(1))

    with pytest.raises(_StopLoop):
        await retention.retention_worker()


@pytest.mark.asyncio
async def test_config_reread_each_cycle_can_re_enable_without_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug this fixes: reading config once before the loop meant a
    retention_days that was 0 at startup could never be turned back on. The
    fix re-reads get_config() every cycle — simulate a PATCH /config landing
    between two cycles and confirm the very next cycle picks it up."""
    cfg = retention.get_config()
    monkeypatch.setattr(cfg.storage, "retention_days", 0)

    cutoffs: list[datetime] = []

    def fake_delete(cutoff: datetime) -> int:
        cutoffs.append(cutoff)
        return 0

    cycle = 0

    async def fake_sleep(_seconds: float) -> None:
        nonlocal cycle
        cycle += 1
        if cycle == 1:
            # Simulate an admin flipping retention back on via the options
            # page in between the first and second cycle.
            monkeypatch.setattr(cfg.storage, "retention_days", 14)
        else:
            raise _StopLoop

    monkeypatch.setattr(retention.repo, "delete_jobs_older_than", fake_delete)
    monkeypatch.setattr(retention.asyncio, "sleep", fake_sleep)

    with pytest.raises(_StopLoop):
        await retention.retention_worker()

    # First cycle (disabled) swept nothing; second cycle (re-enabled)
    # actually called delete_jobs_older_than.
    assert len(cutoffs) == 1


# ---------------------------------------------------------------------------
# active retention — single sweep then break out of the loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_runs_with_correct_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_config(monkeypatch, 7)
    cutoffs: list[datetime] = []

    def fake_delete(cutoff: datetime) -> int:
        cutoffs.append(cutoff)
        return 3

    monkeypatch.setattr(retention.repo, "delete_jobs_older_than", fake_delete)
    monkeypatch.setattr(retention.asyncio, "sleep", _stop_after(1))

    before = datetime.now(UTC)
    with pytest.raises(_StopLoop):
        await retention.retention_worker()
    after = datetime.now(UTC)

    assert len(cutoffs) == 1
    cutoff = cutoffs[0]
    # cutoff == now - 7 days, computed inside the loop.
    assert before - timedelta(days=7) - timedelta(seconds=2) <= cutoff
    assert cutoff <= after - timedelta(days=7) + timedelta(seconds=2)


@pytest.mark.asyncio
async def test_sweep_with_zero_deletions_still_loops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_config(monkeypatch, 30)
    calls = 0

    def fake_delete(_cutoff: datetime) -> int:
        nonlocal calls
        calls += 1
        return 0

    monkeypatch.setattr(retention.repo, "delete_jobs_older_than", fake_delete)
    monkeypatch.setattr(retention.asyncio, "sleep", _stop_after(1))

    with pytest.raises(_StopLoop):
        await retention.retention_worker()
    assert calls == 1


@pytest.mark.asyncio
async def test_sleep_interval_passed_to_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_config(monkeypatch, 10)
    slept: list[float] = []

    monkeypatch.setattr(retention.repo, "delete_jobs_older_than", lambda _c: 0)

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        raise _StopLoop

    monkeypatch.setattr(retention.asyncio, "sleep", fake_sleep)

    with pytest.raises(_StopLoop):
        await retention.retention_worker()
    assert slept == [retention._INTERVAL_SECONDS]


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancelled_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_config(monkeypatch, 7)

    def fake_delete(_cutoff: datetime) -> int:
        raise asyncio.CancelledError

    monkeypatch.setattr(retention.repo, "delete_jobs_older_than", fake_delete)

    # CancelledError must NOT be swallowed — it signals clean shutdown.
    with pytest.raises(asyncio.CancelledError):
        await retention.retention_worker()


@pytest.mark.asyncio
async def test_generic_exception_swallowed_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_config(monkeypatch, 7)
    calls = 0

    def fake_delete(_cutoff: datetime) -> int:
        nonlocal calls
        calls += 1
        raise RuntimeError("db is down")

    monkeypatch.setattr(retention.repo, "delete_jobs_older_than", fake_delete)
    monkeypatch.setattr(retention.asyncio, "sleep", _stop_after(1))

    with pytest.raises(_StopLoop):
        await retention.retention_worker()

    # Swept once, raised, was caught, then proceeded to sleep.
    assert calls == 1


@pytest.mark.asyncio
async def test_loop_survives_error_then_succeeds_next_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First sweep raises (swallowed), second sweep succeeds, then we stop."""
    _patch_config(monkeypatch, 7)
    outcomes = iter([RuntimeError("transient"), 2])
    deletes: list[int] = []

    def fake_delete(_cutoff: datetime) -> int:
        result = next(outcomes)
        if isinstance(result, Exception):
            raise result
        deletes.append(result)
        return result

    monkeypatch.setattr(retention.repo, "delete_jobs_older_than", fake_delete)
    monkeypatch.setattr(retention.asyncio, "sleep", _stop_after(2))

    with pytest.raises(_StopLoop):
        await retention.retention_worker()

    # Second iteration succeeded with 2 deletions.
    assert deletes == [2]


# ---------------------------------------------------------------------------
# orphaned frame directories — a job row deleted through a path that missed
# the frame-cleanup hook (or a crash between the two) leaves a frame
# directory nothing else will ever revisit. See workers/frames.py's
# all_frame_job_ids() and this module's _sweep_orphaned_frame_dirs().
# ---------------------------------------------------------------------------


def test_orphan_sweep_removes_only_dirs_with_no_job_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        retention.frames, "all_frame_job_ids", lambda: ["job-with-row", "job-orphaned"]
    )

    def fake_get_job(job_id: str) -> object | None:
        return object() if job_id == "job-with-row" else None

    monkeypatch.setattr(retention.repo, "get_job", fake_get_job)

    deleted: list[str] = []
    monkeypatch.setattr(
        retention.frames, "delete_job_frames", lambda job_id: deleted.append(job_id) or True
    )

    retention._sweep_orphaned_frame_dirs()

    assert deleted == ["job-orphaned"]


def test_orphan_sweep_no_op_when_nothing_orphaned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retention.frames, "all_frame_job_ids", lambda: ["job-a", "job-b"])
    monkeypatch.setattr(retention.repo, "get_job", lambda _job_id: object())

    def fail_delete(_job_id: str) -> bool:  # pragma: no cover - must not run
        raise AssertionError("should not delete frames for a job that still has a row")

    monkeypatch.setattr(retention.frames, "delete_job_frames", fail_delete)

    retention._sweep_orphaned_frame_dirs()  # must not raise


def test_orphan_sweep_swallows_listing_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_list() -> list[str]:
        raise OSError("disk hiccup")

    monkeypatch.setattr(retention.frames, "all_frame_job_ids", fail_list)

    retention._sweep_orphaned_frame_dirs()  # must not raise, must not abort the caller


def test_orphan_sweep_swallows_per_job_delete_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retention.frames, "all_frame_job_ids", lambda: ["job-a", "job-b"])
    monkeypatch.setattr(retention.repo, "get_job", lambda _job_id: None)

    calls: list[str] = []

    def flaky_delete(job_id: str) -> bool:
        calls.append(job_id)
        if job_id == "job-a":
            raise OSError("permission denied")
        return True

    monkeypatch.setattr(retention.frames, "delete_job_frames", flaky_delete)

    retention._sweep_orphaned_frame_dirs()  # must not raise

    # job-a's failure didn't stop job-b from being attempted.
    assert calls == ["job-a", "job-b"]


@pytest.mark.asyncio
async def test_retention_worker_runs_orphan_sweep_every_cycle_even_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orphan sweep is a DB/disk consistency check, not an age-based
    policy — disabling storage.retention_days must not disable it."""
    _patch_config(monkeypatch, 0)
    monkeypatch.setattr(retention.repo, "delete_jobs_older_than", lambda _c: 0)

    sweep_calls = 0

    def fake_sweep() -> None:
        nonlocal sweep_calls
        sweep_calls += 1

    monkeypatch.setattr(retention, "_sweep_orphaned_frame_dirs", fake_sweep)
    monkeypatch.setattr(retention.asyncio, "sleep", _stop_after(1))

    with pytest.raises(_StopLoop):
        await retention.retention_worker()

    assert sweep_calls == 1


@pytest.mark.asyncio
async def test_retention_worker_calls_orphan_sweep_each_active_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_config(monkeypatch, 7)
    monkeypatch.setattr(retention.repo, "delete_jobs_older_than", lambda _c: 0)

    sweep_calls = 0

    def fake_sweep() -> None:
        nonlocal sweep_calls
        sweep_calls += 1

    monkeypatch.setattr(retention, "_sweep_orphaned_frame_dirs", fake_sweep)
    monkeypatch.setattr(retention.asyncio, "sleep", _stop_after(2))

    with pytest.raises(_StopLoop):
        await retention.retention_worker()

    assert sweep_calls == 2


# ---------------------------------------------------------------------------
# orphaned audio — files/directories under <data_dir>/audio that no job row
# references. See retention._sweep_orphaned_audio()'s docstring (and the
# module docstring) for the three confirmed leak sources and the
# grace-period reasoning. Uses tmp_path throughout — never the real data
# dir — via monkeypatching retention._audio_dir().
# ---------------------------------------------------------------------------

_OLD_SECONDS = retention._AUDIO_ORPHAN_GRACE_SECONDS + 60  # comfortably past the grace period
_FRESH_SECONDS = 60  # well inside the grace period


def _age(path: Path, seconds_ago: float) -> None:
    """Backdate a file/dir's mtime (and atime) by ``seconds_ago`` seconds,
    so tests can simulate "old" vs "fresh" without real wall-clock waits."""
    ts = time.time() - seconds_ago
    os.utime(path, (ts, ts))


def test_audio_sweep_keeps_referenced_file_even_when_old(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retention, "_audio_dir", lambda: tmp_path)
    referenced = tmp_path / "referenced.opus"
    referenced.write_bytes(b"data")
    _age(referenced, _OLD_SECONDS)
    monkeypatch.setattr(retention.repo, "all_referenced_audio_paths", lambda: {str(referenced)})

    retention._sweep_orphaned_audio()

    assert referenced.exists()


def test_audio_sweep_keeps_stale_file_referenced_via_redundant_path_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The DB reference and the on-disk entry can both point at the same
    file without being string-identical (e.g. a redundant ``./`` segment
    from some other normalization). The resolved-path fallback must still
    recognize them as the same file."""
    monkeypatch.setattr(retention, "_audio_dir", lambda: tmp_path)
    referenced = tmp_path / "referenced.opus"
    referenced.write_bytes(b"data")
    _age(referenced, _OLD_SECONDS)
    equivalent_ref = f"{tmp_path}/./referenced.opus"
    assert equivalent_ref != str(referenced)
    monkeypatch.setattr(retention.repo, "all_referenced_audio_paths", lambda: {equivalent_ref})

    retention._sweep_orphaned_audio()

    assert referenced.exists()


def test_audio_sweep_keeps_stale_file_referenced_via_symlinked_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symlinked data dir (e.g. macOS /var vs /private/var) means the DB
    reference and the audio-dir entry can resolve to the same file while
    looking like different strings on either side of the symlink. The
    resolved-path fallback must catch this."""
    real_dir = tmp_path / "real_audio"
    real_dir.mkdir()
    link_dir = tmp_path / "link_audio"
    try:
        os.symlink(real_dir, link_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available in this environment")

    monkeypatch.setattr(retention, "_audio_dir", lambda: real_dir)
    referenced = real_dir / "referenced.opus"
    referenced.write_bytes(b"data")
    _age(referenced, _OLD_SECONDS)
    # DB stores the path as seen through the symlinked parent.
    referenced_via_link = str(link_dir / "referenced.opus")
    assert referenced_via_link != str(referenced)
    monkeypatch.setattr(retention.repo, "all_referenced_audio_paths", lambda: {referenced_via_link})

    retention._sweep_orphaned_audio()

    assert referenced.exists()


def test_audio_sweep_keeps_stale_file_referenced_by_basename_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reference whose full path differs entirely but whose basename
    matches an entry in the (single, flat) audio directory is treated as
    the same file — see the basename-matching rationale in
    ``_sweep_orphaned_audio``."""
    monkeypatch.setattr(retention, "_audio_dir", lambda: tmp_path)
    entry = tmp_path / "shared-name.opus"
    entry.write_bytes(b"data")
    _age(entry, _OLD_SECONDS)
    unrelated_reference = str(Path("/some/other/data/dir/audio/shared-name.opus"))
    monkeypatch.setattr(retention.repo, "all_referenced_audio_paths", lambda: {unrelated_reference})

    retention._sweep_orphaned_audio()

    assert entry.exists()


def test_audio_sweep_removes_stale_file_with_merely_similar_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A referenced file with a different name must not shield an
    unreferenced file that merely looks similar — basename matching is
    exact, not fuzzy."""
    monkeypatch.setattr(retention, "_audio_dir", lambda: tmp_path)
    orphan = tmp_path / "orphan.m4a"
    orphan.write_bytes(b"data")
    _age(orphan, _OLD_SECONDS)
    referenced_elsewhere = str(tmp_path / "orphan-but-not-quite.m4a")
    monkeypatch.setattr(retention.repo, "all_referenced_audio_paths", lambda: {referenced_elsewhere})

    retention._sweep_orphaned_audio()

    assert not orphan.exists()


def test_audio_sweep_keeps_unreferenced_fresh_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retention, "_audio_dir", lambda: tmp_path)
    fresh = tmp_path / "fresh.m4a"
    fresh.write_bytes(b"data")
    _age(fresh, _FRESH_SECONDS)
    monkeypatch.setattr(retention.repo, "all_referenced_audio_paths", lambda: set())

    retention._sweep_orphaned_audio()

    assert fresh.exists()


def test_audio_sweep_removes_unreferenced_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retention, "_audio_dir", lambda: tmp_path)
    orphan = tmp_path / "orphan.m4a"
    orphan.write_bytes(b"data")
    _age(orphan, _OLD_SECONDS)
    monkeypatch.setattr(retention.repo, "all_referenced_audio_paths", lambda: set())

    retention._sweep_orphaned_audio()

    assert not orphan.exists()


def test_audio_sweep_keeps_chunk_dir_with_one_fresh_file_inside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tldr-chunks-* dir created two hours ago but with a chunk written a
    minute ago is a live transcription, not a leak — judged by the newest
    mtime found INSIDE it, not the directory's own timestamp."""
    monkeypatch.setattr(retention, "_audio_dir", lambda: tmp_path)
    chunk_dir = tmp_path / "tldr-chunks-abc123"
    chunk_dir.mkdir()
    _age(chunk_dir, _OLD_SECONDS)
    stale_chunk = chunk_dir / "chunk0.wav"
    stale_chunk.write_bytes(b"data")
    _age(stale_chunk, _OLD_SECONDS)
    live_chunk = chunk_dir / "chunk1.wav"
    live_chunk.write_bytes(b"data")
    _age(live_chunk, _FRESH_SECONDS)
    monkeypatch.setattr(retention.repo, "all_referenced_audio_paths", lambda: set())

    retention._sweep_orphaned_audio()

    assert chunk_dir.is_dir()
    assert live_chunk.exists()
    assert stale_chunk.exists()


def test_audio_sweep_removes_fully_stale_chunk_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retention, "_audio_dir", lambda: tmp_path)
    chunk_dir = tmp_path / "tldr-recut-def456"
    chunk_dir.mkdir()
    _age(chunk_dir, _OLD_SECONDS)
    old_chunk = chunk_dir / "recut.wav"
    old_chunk.write_bytes(b"data")
    _age(old_chunk, _OLD_SECONDS)
    monkeypatch.setattr(retention.repo, "all_referenced_audio_paths", lambda: set())

    retention._sweep_orphaned_audio()

    assert not chunk_dir.exists()


def test_audio_sweep_missing_audio_dir_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "does-not-exist"
    monkeypatch.setattr(retention, "_audio_dir", lambda: missing)

    def fail_referenced() -> set[str]:  # pragma: no cover - must not run
        raise AssertionError("should not query the DB when the audio dir is missing")

    monkeypatch.setattr(retention.repo, "all_referenced_audio_paths", fail_referenced)

    retention._sweep_orphaned_audio()  # must not raise


def test_audio_sweep_swallows_oserror_during_removal_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retention, "_audio_dir", lambda: tmp_path)
    broken = tmp_path / "broken.m4a"
    broken.write_bytes(b"data")
    _age(broken, _OLD_SECONDS)
    other_orphan = tmp_path / "other.m4a"
    other_orphan.write_bytes(b"data")
    _age(other_orphan, _OLD_SECONDS)
    monkeypatch.setattr(retention.repo, "all_referenced_audio_paths", lambda: set())

    real_unlink = Path.unlink

    def flaky_unlink(self: Path, *args: object, **kwargs: object) -> None:
        if self == broken:
            raise OSError("permission denied")
        real_unlink(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    retention._sweep_orphaned_audio()  # must not raise

    # broken's removal failed (logged, not raised) — the file is still
    # there — but the other orphan was still cleaned up afterwards.
    assert broken.exists()
    assert not other_orphan.exists()


@pytest.mark.asyncio
async def test_retention_worker_runs_audio_sweep_every_cycle_even_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same DB/disk-consistency-check status as the frame-dir sweep —
    disabling storage.retention_days must not disable it."""
    _patch_config(monkeypatch, 0)
    monkeypatch.setattr(retention.repo, "delete_jobs_older_than", lambda _c: 0)

    sweep_calls = 0

    def fake_sweep() -> None:
        nonlocal sweep_calls
        sweep_calls += 1

    monkeypatch.setattr(retention, "_sweep_orphaned_audio", fake_sweep)
    monkeypatch.setattr(retention.asyncio, "sleep", _stop_after(1))

    with pytest.raises(_StopLoop):
        await retention.retention_worker()

    assert sweep_calls == 1
