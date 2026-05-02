"""Event brokers for SSE fan-out.

Two brokers live here:

- ``JobEventBroker`` — keyed by ``job_id``. Per-job summary streaming.
  Subscribed by ``POST /ai/stream`` (summary mode) so a client following
  one specific job sees only its events.

- ``EventBroker`` — single global channel. Used by ``GET /events`` so the
  Library and Side panel can react to ANY job change, worker state flip,
  or per-job stage/delta in real time without polling. Subscribers filter
  by event type (and ``job_id`` field where present).

Why two? The job-stream is a stable per-job lifecycle replay (subscribe,
get all events for that job until done). The global stream is a firehose
for UIs that need awareness across all jobs.

Producers publish to BOTH:
  * the per-job stream so /ai/stream subscribers see live progress
  * the global stream so Library/Side panel react instantly

Subscribers are stored in dicts keyed by ``id(queue)`` so unsubscribe is
O(1) — important once a few panels are open and disconnect/reconnect on
network blips.

Helper constructors (stage_event, delta_event, done_event, error_event,
job_event, workers_event) keep payload shapes aligned with api/schemas.py.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)

_QUEUE_MAX = 256        # bounded so a slow subscriber can't grow memory unbounded


class JobEventBroker:
    """In-memory pub/sub keyed by job_id."""

    def __init__(self) -> None:
        # job_id → {id(queue): queue} so unsubscribe is O(1).
        self._subs: dict[str, dict[int, asyncio.Queue[dict[str, Any]]]] = {}

    def subscribe(self, job_id: str) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._subs.setdefault(job_id, {})[id(q)] = q
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue[dict[str, Any]]) -> None:
        subs = self._subs.get(job_id)
        if not subs:
            return
        subs.pop(id(q), None)
        if not subs:
            self._subs.pop(job_id, None)

    def publish(self, job_id: str, event: dict[str, Any]) -> None:
        """Fan out an event to every active subscriber for ``job_id``.

        Drops the event for any subscriber whose queue is full (back-pressure
        protection). Slow subscribers see gaps rather than blocking the producer.

        Also mirrors the event into the global ``EventBroker`` with ``job_id``
        attached, so /events subscribers (Library, Side panel) react in
        real-time without per-job subscribe calls.
        """
        # Iterate a snapshot so a concurrent unsubscribe doesn't mutate
        # the dict mid-loop.
        for q in list(self._subs.get(job_id, {}).values()):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("broker: dropping event for slow subscriber on job %s", job_id)
        _event_broker.publish({**event, "job_id": job_id})

    def reset(self) -> None:
        """Drop all subscriptions. Test helper."""
        self._subs.clear()


class EventBroker:
    """Single global pub/sub for app-wide events (jobs, workers).

    Every subscriber gets every published event. Subscribers filter by
    event ``type`` and (where present) ``job_id``.
    """

    def __init__(self) -> None:
        # id(queue) → queue so unsubscribe is O(1).
        self._subs: dict[int, asyncio.Queue[dict[str, Any]]] = {}

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._subs[id(q)] = q
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subs.pop(id(q), None)

    def publish(self, event: dict[str, Any]) -> None:
        """Fan out to every active subscriber. Drop on a full queue rather
        than blocking the producer."""
        for q in list(self._subs.values()):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("event broker: dropping event for slow subscriber")

    def reset(self) -> None:
        self._subs.clear()


_broker = JobEventBroker()
_event_broker = EventBroker()


def get_broker() -> JobEventBroker:
    return _broker


def get_event_broker() -> EventBroker:
    return _event_broker


def reset_broker() -> None:
    """Test helper — wipe all subscriptions."""
    _broker.reset()
    _event_broker.reset()


# ---------------------------------------------------------------------------
# Event constructors — keep payload shapes aligned with api/schemas.py
# ---------------------------------------------------------------------------


def stage_event(stage: str, detail: str | None = None) -> dict[str, Any]:
    return {"type": "stage", "stage": stage, "detail": detail}


def delta_event(delta: str) -> dict[str, Any]:
    return {"type": "delta", "delta": delta}


def done_event(content: str, message_id: int | None = None) -> dict[str, Any]:
    return {"type": "done", "content": content, "message_id": message_id}


def error_event(error: str) -> dict[str, Any]:
    return {"type": "error", "error": error}


# ---------------------------------------------------------------------------
# Global-stream event constructors (the /events endpoint payloads)
# ---------------------------------------------------------------------------


def job_event(action: str, job: dict[str, Any]) -> dict[str, Any]:
    """Job-list change.

    ``action`` ∈ {"created", "updated", "deleted"}. ``job`` is a JSON-ready
    JobSummary (id, status, kind, title, …) — Library renders rows from this
    directly without a follow-up GET.
    """
    return {"type": "job", "action": action, "job": job}


def workers_event(state: dict[str, Any]) -> dict[str, Any]:
    """Workers control / queue snapshot — paused flag + queue counters."""
    return {"type": "workers", "state": state}


def job_stage_event(job_id: str, stage: str, detail: str | None = None) -> dict[str, Any]:
    """Wrap stage_event with job_id for the global stream."""
    return {"type": "stage", "job_id": job_id, "stage": stage, "detail": detail}


def job_delta_event(job_id: str, delta: str) -> dict[str, Any]:
    return {"type": "delta", "job_id": job_id, "delta": delta}


def job_done_event(job_id: str, content: str) -> dict[str, Any]:
    return {"type": "done", "job_id": job_id, "content": content}


def job_error_event(job_id: str, error: str) -> dict[str, Any]:
    return {"type": "error", "job_id": job_id, "error": error}
