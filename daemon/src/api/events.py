"""GET /events?types=... — single global SSE stream for the extension UIs.

Replaces per-page polling. The Library and Side panel each open one
long-lived connection here and react to events filtered by type / job_id.

Subscribers narrow the firehose with the ``types`` query param —
comma-separated list of event types they care about. Server-side filter
keeps LLM token deltas (high-volume) off connections that don't need them
(notably Library, which only renders status badges).

Event types:
- ``job``     — list change (created / updated / deleted), with full JobSummary
- ``workers`` — pause + queue snapshot
- ``stage``   — pipeline phase transition (job_id present)
- ``delta``   — LLM token chunk (job_id present, batched ~10 Hz)
- ``done``    — summary finished (job_id present)
- ``error``   — pipeline failed (job_id present)

A 30-second timeout pulls the loop out of ``queue.get()`` and emits a
keep-alive comment so intermediaries (and our own readyState watchers)
know the stream is healthy.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from src.workers.broker import get_event_broker

log = logging.getLogger(__name__)

router = APIRouter(prefix="/events", tags=["events"])


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.get("", response_class=StreamingResponse)
async def events(
    types: str | None = Query(
        default=None,
        description=(
            "Comma-separated list of event types to receive. "
            "Default = everything. Pass e.g. 'job,workers,done,error' from "
            "Library to skip the high-volume 'delta' / 'stage' chatter."
        ),
    ),
) -> StreamingResponse:
    allowed = (
        {t.strip() for t in types.split(",") if t.strip()} if types else None
    )
    broker = get_event_broker()
    queue = broker.subscribe()

    async def gen() -> AsyncIterator[str]:
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if allowed is not None and event.get("type") not in allowed:
                    continue
                yield _sse(event)
        finally:
            broker.unsubscribe(queue)

    return StreamingResponse(gen(), media_type="text/event-stream")
