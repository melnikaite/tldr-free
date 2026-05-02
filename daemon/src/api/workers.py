"""Worker control endpoints — global pause/resume.

A single in-memory flag (``workers.control``) is honoured by both the
Whisper queue worker and the pipeline coroutine that runs every
``POST /jobs``. While paused:

- The Whisper worker stops picking up new tasks (in-flight task finishes).
- New pipelines park before the LLM call (already-streaming summaries
  finish; QA is unaffected since the user is actively waiting).

This covers all background ML work, regardless of which backend (mlx,
Ollama, LM Studio, ...) the daemon is configured against.

``WorkerControl.pause()`` / ``resume()`` publish ``workers_event`` to the
global broker themselves, so the response just snapshots the new state.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from src.workers.control import get_control
from src.workers.queue import get_queue

router = APIRouter(prefix="/workers", tags=["workers"])


class WorkersState(BaseModel):
    paused: bool
    queue_size: int
    running: int


def _snapshot() -> WorkersState:
    size, running = get_queue().snapshot()
    return WorkersState(paused=get_control().paused, queue_size=size, running=running)


# GET status is sync `def` — runs in threadpool, no emit involved.
# pause/resume stay `async def` because control.pause()/resume() publish
# into the event broker, and asyncio.Queue.put_nowait is NOT safe from a
# threadpool thread (it wakes pending getters via loop.call_soon, which
# must run in the loop's own thread).


@router.get("", response_model=WorkersState)
def workers_status() -> WorkersState:
    return _snapshot()


@router.post("/pause", response_model=WorkersState)
async def workers_pause() -> WorkersState:
    get_control().pause()
    return _snapshot()


@router.post("/resume", response_model=WorkersState)
async def workers_resume() -> WorkersState:
    get_control().resume()
    return _snapshot()
