"""Persistent asyncio queue for deferred Whisper jobs.

Behavior:
- Single ``asyncio.Queue`` holding ``WhisperTask`` items.
- A small pool of worker coroutines in ``runner.py`` (sized by
  ``whisper.max_concurrent_jobs``) consumes the queue concurrently — no
  longer a single serial consumer. This is what lets a short job's chunks
  interleave with a long job's instead of queueing behind all of it. Actual
  Whisper HTTP calls are separately gated (global + per-job semaphores in
  ``transcribe.py``) so the worker-pool size and the network concurrency are
  two independent knobs — see ``.claude/workers.md``.
- A module-level singleton (``get_queue()``) lets ``api/jobs.py`` and
  ``api/health.py`` see the same queue without dependency injection.
- On daemon startup ``re_enqueue_pending`` scans
  ``repo.find_pending_for_restart()`` and pushes back any rows left in
  ``queued`` / ``running`` from a previous run.
- ``snapshot()`` returns ``(queue_size, running_count)`` for ``/health``.
- ``position(job_id)`` returns a job's 1-based position in the FIFO while
  it's still waiting to be picked up by a pool worker, ``None`` once
  dequeued (or if it was never in the queue) — surfaced via
  ``JobDetails.whisper_queue_position``.

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
    # Extension-supplied page text, carried transiently for kind=media jobs
    # only (mirrors how ``media_url`` itself lives only inside the in-flight
    # task — see ``re_enqueue_pending``'s comment below). Used by
    # ``runner._process_one`` as a fallback summary source when the media
    # clip is too short to contain speech or Whisper returns an empty
    # transcript. Lost on daemon restart, same as ``media_url`` — a restart
    # already marks in-flight media jobs failed, so this adds no new gap.
    page_text: str | None = None


class WhisperQueue:
    """An ``asyncio.Queue`` with a small status surface area for ``/health``.

    Pause/resume lives at the global ``workers.control`` level, not here —
    the same gate also throttles synchronous pipeline tasks (page summary,
    YouTube fast path), so flipping pause covers all background ML work.
    """

    def __init__(self) -> None:
        self._q: asyncio.Queue[WhisperTask] = asyncio.Queue()
        self._running: int = 0  # real counter now — up to whisper.max_concurrent_jobs
        # FIFO of job_ids still sitting in the queue, mirroring self._q's own
        # order. Maintained alongside put()/get() rather than derived from
        # self._q (asyncio.Queue exposes no way to peek its contents) so
        # position() can answer "where in line is this job" without draining
        # anything.
        self._pending_ids: list[str] = []

    async def put(self, task: WhisperTask) -> None:
        await self._q.put(task)
        self._pending_ids.append(task.job_id)

    async def get(self) -> WhisperTask:
        task = await self._q.get()
        # Remove by VALUE, not by index 0: with multiple worker coroutines
        # all awaiting get() concurrently, the coroutine that wakes up isn't
        # guaranteed to be the one "at the front" of our own list by the time
        # it runs. asyncio.Queue.get() guarantees whoever wakes up got
        # exactly the item that was put() for them, in FIFO order, so
        # removing that exact job_id is always correct and race-free — no
        # other job with the same id can be enqueued concurrently (job ids
        # are unique).
        self._pending_ids.remove(task.job_id)
        return task

    def task_done(self) -> None:
        self._q.task_done()

    def snapshot(self) -> tuple[int, int]:
        """Return ``(queue_size, running_count)`` for /health."""
        return (self._q.qsize(), self._running)

    def mark_running(self, on: bool) -> None:
        """Increment/decrement the running counter. No longer a 0/1 flag —
        with a worker pool (``whisper.max_concurrent_jobs``), multiple tasks
        can be running at once. Clamped at 0 as a defensive floor."""
        self._running = max(0, self._running + (1 if on else -1))

    def position(self, job_id: str) -> int | None:
        """1-based position of ``job_id`` in the FIFO while it's still
        waiting, or ``None`` once a worker has dequeued it (or if it was
        never queued). "Waiting" means "still sitting behind the worker-pool
        limit" — the instant ``get()`` removes it, this returns ``None``,
        even if the worker hasn't reached Whisper yet."""
        try:
            return self._pending_ids.index(job_id) + 1
        except ValueError:
            return None


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
            # Page / media / pdf jobs left in queued/running can't be
            # resumed by the queue:
            #   - page: page_text from the request body is gone
            #   - media: media_url lives only inside the in-flight WhisperTask
            #   - pdf:  pdf_bytes (file://) aren't persisted; http URLs
            #     survive but restart goes through the user clicking
            #     summarize again, not the queue
            # Mark failed so the row reflects reality.
            mark_failed = getattr(repo_module, "mark_failed", None)
            if mark_failed is not None:
                reasons = {
                    "media": (
                        "daemon restarted mid-stream; media url not persisted, "
                        "please re-summarize from the extension"
                    ),
                    "pdf": (
                        "daemon restarted mid-stream; PDF not resumable, "
                        "please re-summarize from the extension"
                    ),
                }
                kind_key = kind if isinstance(kind, str) else ""
                reason = reasons.get(kind_key, "daemon restarted; job not resumable")
                try:
                    mark_failed(job_id, error=reason)
                except Exception:
                    log.exception("failed to mark stale %s job %s as failed", kind, job_id)
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
