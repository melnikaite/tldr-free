"""POST /ai/stream — unified streaming endpoint for ALL AI responses.

Two modes (selected by whether `question` is set on the request body):

Summary mode  (no `question`):
    Subscribe to the job's extraction + summarization lifecycle.
    Live: forward events from the broker (stage, delta, done).
    Cached: if the job is already done, replay summary_md as one delta + done.
    Failed: emit error.

QA mode  (`question` set):
    Trigger a fresh QA call. Persist the user message, stream the answer
    deltas, persist the assistant message, emit done with message_id.

Both modes emit the same event shapes (api/schemas AIStreamEvent variants),
so the client side can use one parser.

Stream framing: text/event-stream, frames separated by ``\\n\\n``, payload
is ``data: <json>\\n\\n``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from src.api.schemas import AIStreamRequest, JobStatus
from src.config import get_config
from src.llm import qa as llm_qa
from src.storage import repo
from src.workers.broker import (
    delta_event,
    done_event,
    error_event,
    get_broker,
    stage_event,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["ai"])


def _sse(event: dict[str, Any]) -> str:
    """Encode one broker-shaped event as a single SSE frame."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post(
    "/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": (
                "SSE stream of AIStageEvent / AIDeltaEvent / AIDoneEvent (or AIErrorEvent). "
                "Without `question` → summary mode (subscribe to job's summarization, "
                "or replay cached). With `question` → QA mode (trigger + stream + persist)."
            ),
        }
    },
)
async def stream(req: AIStreamRequest) -> StreamingResponse:
    job = repo.get_job(req.job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"job {req.job_id} not found")

    gen = _qa_stream(job, req.question.strip()) if req.question else _summary_stream(job)
    return StreamingResponse(gen, media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Summary mode
# ---------------------------------------------------------------------------


async def _summary_stream(job: Any) -> AsyncIterator[str]:
    """Subscribe to a job's lifecycle. Replay cached state when the job is done.

    Behavior matrix:
    - status=done  → emit one delta with summary_md, then done. No broker.
    - status=failed → emit error.
    - status=queued|running → subscribe to broker; if there are no live
      events incoming because no producer is currently publishing, we
      still emit the current stage so the user sees something immediately.
      Then forward events as they arrive until done|error.
    """
    job_id = job.id
    broker = get_broker()

    if job.status == JobStatus.DONE.value and job.summary_md:
        yield _sse(delta_event(job.summary_md))
        yield _sse(done_event(job.summary_md))
        return

    if job.status == JobStatus.FAILED.value:
        yield _sse(error_event(job.error or "job failed"))
        return

    # Subscribe FIRST so we don't miss events that fire while we're emitting
    # the initial snapshot.
    queue = broker.subscribe(job_id)

    try:
        # Snapshot: if there's an existing progress_stage on the row, surface it
        # so the client immediately sees "extracting" / "transcribing" etc.
        if job.progress_stage:
            yield _sse(stage_event(job.progress_stage))

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except TimeoutError:
                # Heartbeat-style: re-poll the row in case the producer
                # finished without publishing (defensive — shouldn't happen,
                # but keeps the stream from hanging if it does).
                refreshed = repo.get_job(job_id)
                if refreshed is None:
                    yield _sse(error_event("job disappeared"))
                    return
                if refreshed.status == JobStatus.DONE.value and refreshed.summary_md:
                    yield _sse(delta_event(refreshed.summary_md))
                    yield _sse(done_event(refreshed.summary_md))
                    return
                if refreshed.status == JobStatus.FAILED.value:
                    yield _sse(error_event(refreshed.error or "job failed"))
                    return
                # Otherwise loop and wait again.
                continue

            yield _sse(event)
            if event.get("type") in ("done", "error"):
                return
    finally:
        broker.unsubscribe(job_id, queue)


# ---------------------------------------------------------------------------
# QA mode
# ---------------------------------------------------------------------------


async def _qa_stream(job: Any, question: str) -> AsyncIterator[str]:
    """Trigger a QA call, stream the answer, persist user + assistant messages."""
    job_id = job.id
    cfg = get_config()

    # Persist the user's question first — even if the LLM crashes mid-stream,
    # the question stays in history.
    try:
        repo.add_message(job_id, role="user", content=question)
    except Exception as exc:
        log.exception("failed to persist user message for job %s", job_id)
        yield _sse(error_event(f"could not save message: {exc}"))
        return

    # Need raw_text or summary_md to ground the answer.
    if not (job.raw_text or job.summary_md):
        yield _sse(error_event("this job has no extracted content yet"))
        return

    yield _sse(stage_event("thinking"))

    parts: list[str] = []
    try:
        async for delta in llm_qa.stream_answer(
            job=job,
            question=question,
            output_language=cfg.output.language_name,
        ):
            parts.append(delta)
            yield _sse(delta_event(delta))
    except Exception as exc:
        log.exception("QA stream failed for job %s", job_id)
        yield _sse(error_event(f"qa failed: {exc}"))
        return

    answer = "".join(parts).strip()
    if not answer:
        yield _sse(error_event("LLM returned empty answer"))
        return

    try:
        assistant = repo.add_message(job_id, role="assistant", content=answer)
        message_id = assistant.id
    except Exception:
        log.exception("failed to persist assistant message for job %s", job_id)
        # Still emit done so the client sees the answer; just no message_id.
        yield _sse(done_event(answer, message_id=None))
        return

    yield _sse(done_event(answer, message_id=message_id))
