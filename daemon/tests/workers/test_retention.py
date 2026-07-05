"""Tests for src.workers.retention — periodic job-retention sweep.

The worker loops forever, so we break the loop by making the monkeypatched
``asyncio.sleep`` raise a sentinel exception after the first iteration. The
actual DB delete (``repo.delete_jobs_older_than``) is stubbed so we can assert
on call args / cutoff math without standing up a database.

Covers:
  - retention disabled when retention_days <= 0 (==0 and negative)
  - cutoff is computed as now - retention_days
  - sweep runs and logs deletions vs no deletions
  - asyncio.CancelledError propagates (clean shutdown)
  - generic exceptions are swallowed and the loop continues to the next sleep
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from src.workers import retention


class _StopLoop(Exception):
    """Sentinel raised from the patched sleep to terminate the infinite loop."""


def _patch_config(monkeypatch: pytest.MonkeyPatch, days: int) -> None:
    cfg = retention.get_config()
    monkeypatch.setattr(cfg.storage, "retention_days", days)


# ---------------------------------------------------------------------------
# disabled retention
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_when_retention_days_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_config(monkeypatch, 0)
    called = False

    def fake_delete(_cutoff: datetime) -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(retention.repo, "delete_jobs_older_than", fake_delete)

    # Returns immediately without ever sweeping or sleeping.
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

    await retention.retention_worker()


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

    async def fake_sleep(_seconds: float) -> None:
        raise _StopLoop

    monkeypatch.setattr(retention.repo, "delete_jobs_older_than", fake_delete)
    monkeypatch.setattr(retention.asyncio, "sleep", fake_sleep)

    before = datetime.utcnow()
    with pytest.raises(_StopLoop):
        await retention.retention_worker()
    after = datetime.utcnow()

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

    async def fake_sleep(_seconds: float) -> None:
        raise _StopLoop

    monkeypatch.setattr(retention.repo, "delete_jobs_older_than", fake_delete)
    monkeypatch.setattr(retention.asyncio, "sleep", fake_sleep)

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

    async def fake_sleep(_seconds: float) -> None:
        # The loop must reach sleep even though the sweep raised — proving the
        # generic exception was caught. Break out on the first sleep.
        raise _StopLoop

    monkeypatch.setattr(retention.repo, "delete_jobs_older_than", fake_delete)
    monkeypatch.setattr(retention.asyncio, "sleep", fake_sleep)

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

    sleep_calls = 0

    async def fake_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 2:
            raise _StopLoop

    monkeypatch.setattr(retention.repo, "delete_jobs_older_than", fake_delete)
    monkeypatch.setattr(retention.asyncio, "sleep", fake_sleep)

    with pytest.raises(_StopLoop):
        await retention.retention_worker()

    # Second iteration succeeded with 2 deletions.
    assert deletes == [2]
    assert sleep_calls == 2
