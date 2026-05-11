"""Event brokers for SSE fan-out.

Two brokers live here:

- ``JobEventBroker`` — per-job event publisher. Maintains the stream replay
  buffer (so reconnecting clients can resume mid-generation) and mirrors every
  published event into the global ``EventBroker`` with ``job_id`` attached.
  Producers (pipeline, runner) call ``publish(job_id, event)``; no one
  subscribes per-job — all consumers use the global stream.

- ``EventBroker`` — single global channel. Used by ``GET /events`` so the
  Library and Side panel can react to ANY job change, worker state flip,
  or per-job stage/delta in real time without polling. Subscribers filter
  by event type (and ``job_id`` field where present).

Why two? The job broker is a thin routing layer that adds ``job_id`` and
maintains the replay buffer; the global broker is the actual fan-out.

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

# ---------------------------------------------------------------------------
# Stream replay buffer
# ---------------------------------------------------------------------------
# Accumulates the full delta text for every running job so a client that
# (re-)connects mid-generation — or opens the browser fresh — can fetch the
# buffered text via GET /jobs/{id} and resume from the right position.
# Keyed by job_id; cleared when the job finishes (done / error).

_stream_buffers: dict[str, str] = {}


def get_stream_buffer(job_id: str) -> str:
    """Return accumulated delta text for a still-running job (empty string if none)."""
    return _stream_buffers.get(job_id, "")


def _clear_stream_buffer(job_id: str) -> None:
    _stream_buffers.pop(job_id, None)


class JobEventBroker:
    """Per-job event publisher with stream replay buffer.

    Maintains ``_stream_buffers`` so ``GET /jobs/{id}`` can return
    ``partial_summary`` for in-progress jobs. Every published event is also
    mirrored to the global ``EventBroker`` with ``job_id`` attached, making
    it visible to ``GET /events`` subscribers (Library, Side panel).
    """

    def publish(self, job_id: str, event: dict[str, Any]) -> None:
        """Publish ``event`` for ``job_id``.

        - Appends delta text to the stream replay buffer.
        - Clears the buffer on terminal events (``done`` / ``error``).
        - Mirrors to the global broker with ``job_id`` attached.
        """
        event_type = event.get("type")
        if event_type == "delta":
            _stream_buffers[job_id] = _stream_buffers.get(job_id, "") + event.get("delta", "")
        elif event_type in ("done", "error"):
            _clear_stream_buffer(job_id)

        _event_broker.publish({**event, "job_id": job_id})

    def reset(self) -> None:
        """Test helper — clear replay buffers."""
        _stream_buffers.clear()


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
    """Test helper — wipe all subscriptions and replay buffers."""
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
