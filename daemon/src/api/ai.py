"""POST /ai/qa — streaming Q&A endpoint.

Triggers a fresh QA call against the job's extracted content, streams
answer tokens, persists the user + assistant messages in SQLite, and
emits a final `done` event with the assistant message_id.

Stream framing: text/event-stream, each frame is ``data: <json>\\n\\n``.
The stream ends with either `done` or `error`.
"""

from __future__ import annotations

import json
import logging
import re
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
    stage_event,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["ai"])

# A small local model finishes the answer, then degenerates into a repetition
# loop — a run of <br> tags, a stray closing tag, or bare [MM:SS] markers — and
# keeps emitting until max_tokens. The real answer is already complete by then,
# so once the output ENDS with 6+ consecutive such filler tokens we stop reading
# the stream (cuts the live spam and the wasted generation). clean_answer()
# trims whatever partial filler slipped through before the cutoff.
_DEGEN_TAIL_RE = re.compile(
    r"(?:\s*(?:<[^>]+>|\[\d{1,2}:\d{2}(?::\d{2})?\])\s*){6,}\Z"
)


def _sse(event: dict[str, Any]) -> str:
    """Encode one broker-shaped event as a single SSE frame."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post(
    "/qa",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": (
                "SSE stream of AIStageEvent / AIDeltaEvent / AIDoneEvent (or AIErrorEvent). "
                "Triggers a new QA call, streams the answer, persists both messages."
            ),
        }
    },
)
async def qa(req: AIStreamRequest) -> StreamingResponse:
    job = repo.get_job(req.job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"job {req.job_id} not found")
    if job.status != JobStatus.DONE.value:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"job {req.job_id} is not done yet (status={job.status!r}); "
            "wait for the summary before asking questions",
        )
    return StreamingResponse(_qa_stream(job, req.question.strip()), media_type="text/event-stream")


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
        async for item in llm_qa.stream_answer(
            job=job,
            question=question,
            output_language=cfg.output.language_name,
        ):
            if isinstance(item, str):
                parts.append(item)
                yield _sse(delta_event(item))
                # Stop as soon as the output degenerates into a filler loop —
                # don't keep streaming junk to the client or burning tokens.
                if _DEGEN_TAIL_RE.search("".join(parts)[-400:]):
                    break
            else:
                # Stage event from tool use (e.g. {"type": "stage", "stage": "searching"}).
                yield _sse(item)
    except Exception as exc:
        log.exception("QA stream failed for job %s", job_id)
        yield _sse(error_event(f"qa failed: {exc}"))
        return

    # Final tidy: strip any partial filler that slipped through before the
    # degeneration cutoff above (bare [MM:SS] dumps, runs of <br>, stray tags)
    # so the stored — and reloaded — message is clean.
    answer = llm_qa.clean_answer("".join(parts).strip())
    if not answer:
        yield _sse(error_event("LLM returned empty answer"))
        return

    try:
        assistant = repo.add_message(job_id, role="assistant", content=answer)
        message_id = assistant.id
    except Exception:
        log.exception("failed to persist assistant message for job %s", job_id)
        yield _sse(done_event(answer, message_id=None))
        return

    yield _sse(done_event(answer, message_id=message_id))
