"""Tests for workers.queue — singleton, snapshot, re-enqueue on startup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.workers import queue as queue_mod
from src.workers.queue import WhisperQueue, WhisperTask, get_queue, re_enqueue_pending, reset_queue


@pytest.fixture(autouse=True)
def _reset_queue_singleton() -> Any:
    reset_queue()
    yield
    reset_queue()


def test_get_queue_returns_singleton() -> None:
    a = get_queue()
    b = get_queue()
    assert a is b


@pytest.mark.asyncio
async def test_put_get_basic_round_trip() -> None:
    q = WhisperQueue()
    task = WhisperTask(job_id="j1", url="https://youtu.be/x", cookies=[])
    await q.put(task)
    got = await q.get()
    assert got is task
    q.task_done()


def test_snapshot_initial_zero() -> None:
    q = WhisperQueue()
    assert q.snapshot() == (0, 0)


@pytest.mark.asyncio
async def test_snapshot_reflects_size_and_running_state() -> None:
    q = WhisperQueue()
    await q.put(WhisperTask(job_id="j1", url="https://youtu.be/x"))
    await q.put(WhisperTask(job_id="j2", url="https://youtu.be/y"))
    assert q.snapshot() == (2, 0)
    q.mark_running(True)
    assert q.snapshot()[1] == 1
    q.mark_running(False)
    assert q.snapshot()[1] == 0


# ---------------------------------------------------------------------------
# re-enqueue from repo.find_pending_for_restart
# ---------------------------------------------------------------------------


@dataclass
class _FakeJob:
    id: str
    url: str
    kind: str


class _FakeRepo:
    def __init__(self, rows: list[_FakeJob]) -> None:
        self._rows = rows
        self.failed_calls: list[tuple[str, str]] = []

    def find_pending_for_restart(self) -> list[_FakeJob]:
        return self._rows

    def mark_failed(self, job_id: str, *, error: str) -> None:
        self.failed_calls.append((job_id, error))


@pytest.mark.asyncio
async def test_re_enqueue_youtube_jobs() -> None:
    rows = [
        _FakeJob(id="a", url="https://www.youtube.com/watch?v=aaaaaaaaaaa", kind="youtube"),
        _FakeJob(id="b", url="https://www.youtube.com/watch?v=bbbbbbbbbbb", kind="youtube"),
    ]
    repo = _FakeRepo(rows)
    q = WhisperQueue()
    n = await re_enqueue_pending(q, repo)
    assert n == 2
    assert q.snapshot() == (2, 0)
    # Drain to verify ordering & contents.
    first = await q.get()
    assert first.job_id == "a"
    second = await q.get()
    assert second.job_id == "b"


@pytest.mark.asyncio
async def test_re_enqueue_marks_stale_page_jobs_failed() -> None:
    rows = [
        _FakeJob(id="p1", url="https://example.com/", kind="page"),
        _FakeJob(id="y1", url="https://www.youtube.com/watch?v=ccccccccccc", kind="youtube"),
    ]
    repo = _FakeRepo(rows)
    q = WhisperQueue()
    n = await re_enqueue_pending(q, repo)
    # Only the youtube job is re-enqueued. The page job is marked failed.
    assert n == 1
    assert q.snapshot() == (1, 0)
    assert len(repo.failed_calls) == 1
    assert repo.failed_calls[0][0] == "p1"


@pytest.mark.asyncio
async def test_re_enqueue_skips_rows_without_id() -> None:
    rows = [
        _FakeJob(id="", url="https://x", kind="youtube"),
        _FakeJob(id="ok", url="https://www.youtube.com/watch?v=ddddddddddd", kind="youtube"),
    ]
    repo = _FakeRepo(rows)
    q = WhisperQueue()
    n = await re_enqueue_pending(q, repo)
    assert n == 1


@pytest.mark.asyncio
async def test_module_level_helpers_align_with_singleton() -> None:
    # The queue module exposes get_queue() — ensure it actually persists state.
    q = get_queue()
    await q.put(WhisperTask(job_id="z", url="https://youtu.be/z"))
    # Re-fetch the singleton — it must be the same Q with the same item.
    again = queue_mod.get_queue()
    assert again is q
    assert again.snapshot()[0] == 1
