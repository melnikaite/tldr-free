"""Persistent asyncio queue for deferred Whisper jobs.

Behavior:
- Single ``asyncio.Queue`` holding ``WhisperTask`` items.
- The runner coroutine in ``runner.py`` consumes the queue serially.
- A module-level singleton (``get_queue()``) lets ``api/jobs.py`` and
  ``api/health.py`` see the same queue without dependency injection.
- On daemon startup ``re_enqueue_pending`` scans
  ``repo.find_pending_for_restart()`` and pushes back any rows left in
  ``queued`` / ``running`` from a previous run.
- ``snapshot()`` returns ``(queue_size, running_count)`` for ``/health``.

The queue itself is not durable — durability comes from the SQLite Job
rows. We rebuild the in-memory queue on startup from those rows.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from src.api.schemas import Cookie

log = logging.getLogger(__name__)


@dataclass
class WhisperTask:
    """One item in the deferred queue.

    The job row already exists (status=queued); the worker only reads the
    URL and cookies from this struct (no DB lookup needed in the hot path).
    """

    job_id: str
    url: str
    cookies: list[Cookie] = field(default_factory=list)


class WhisperQueue:
    """An ``asyncio.Queue`` with a small status surface area for ``/health``.

    Pause/resume lives at the global ``workers.control`` level, not here —
    the same gate also throttles synchronous pipeline tasks (page summary,
    YouTube fast path), so flipping pause covers all background ML work.
    """

    def __init__(self) -> None:
        self._q: asyncio.Queue[WhisperTask] = asyncio.Queue()
        self._running: int = 0  # 0 or 1 for v1 (single-worker)

    async def put(self, task: WhisperTask) -> None:
        await self._q.put(task)

    async def get(self) -> WhisperTask:
        return await self._q.get()

    def task_done(self) -> None:
        self._q.task_done()

    def snapshot(self) -> tuple[int, int]:
        """Return ``(queue_size, running_count)`` for /health."""
        return (self._q.qsize(), self._running)

    def mark_running(self, on: bool) -> None:
        self._running = 1 if on else 0


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_queue: WhisperQueue | None = None


def get_queue() -> WhisperQueue:
    """Lazy-init module-level singleton.

    Note: ``asyncio.Queue`` instances bind to the running event loop on first
    use. This singleton is created the first time it's accessed — typically
    inside the lifespan, which means the FastAPI loop is already running by
    then.
    """
    global _queue
    if _queue is None:
        _queue = WhisperQueue()
    return _queue


def reset_queue() -> None:
    """For tests: drop the singleton so the next ``get_queue()`` rebuilds it."""
    global _queue
    _queue = None


# ---------------------------------------------------------------------------
# Re-enqueue on startup
# ---------------------------------------------------------------------------


async def re_enqueue_pending(queue: WhisperQueue, repo_module: object) -> int:
    """Scan ``repo.find_pending_for_restart()`` and push tasks back.

    ``repo_module`` is the storage repo module (``src.storage.repo``). We pass
    it as a parameter so tests can inject a fake without touching globals.
    Returns the number of tasks re-enqueued.
    """
    find = repo_module.find_pending_for_restart  # type: ignore[attr-defined]
    rows = find()
    n = 0
    for row in rows:
        # Only re-enqueue YouTube jobs — pages are sync-only and shouldn't be
        # in queued/running on startup, but if they are, we mark them failed
        # so the user sees something rather than letting them hang silently.
        kind = getattr(row, "kind", None)
        job_id = getattr(row, "id", None)
        url = getattr(row, "url", "")
        if not job_id:
            continue
        if kind == "youtube":
            await queue.put(WhisperTask(job_id=job_id, url=url, cookies=[]))
            n += 1
        else:
            # A page job left in queued/running can't be resumed (no audio,
            # the request body is gone). Mark failed so the row reflects reality.
            mark_failed = getattr(repo_module, "mark_failed", None)
            if mark_failed is not None:
                try:
                    mark_failed(job_id, error="daemon restarted; page job not resumable")
                except Exception:
                    log.exception("failed to mark stale page job %s as failed", job_id)
    if n:
        log.info("queue: re-enqueued %d pending youtube job(s) on startup", n)
    return n


__all__ = [
    "WhisperQueue",
    "WhisperTask",
    "get_queue",
    "re_enqueue_pending",
    "reset_queue",
]
