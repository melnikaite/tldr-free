"""Tests for the global EventBroker — fan-out, back-pressure, unsubscribe.

Focused on guarantees the rest of the daemon (and the extension) relies on:

- Every subscriber gets every published event.
- A slow subscriber doesn't block the producer (events drop on full queue).
- ``unsubscribe`` removes the queue without raising.
- The per-job broker mirrors events into the global broker with ``job_id``
  attached, so /events surfaces them to filter-aware clients.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.workers.broker import (
    delta_event,
    get_broker,
    get_event_broker,
    job_event,
    reset_broker,
)


@pytest.fixture(autouse=True)
def _reset() -> Any:
    reset_broker()
    yield
    reset_broker()


# ---------------------------------------------------------------------------
# EventBroker fan-out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_broker_fans_out_to_all_subscribers() -> None:
    broker = get_event_broker()
    q1 = broker.subscribe()
    q2 = broker.subscribe()

    payload = job_event("created", {"id": "abc"})
    broker.publish(payload)

    e1 = await asyncio.wait_for(q1.get(), timeout=1.0)
    e2 = await asyncio.wait_for(q2.get(), timeout=1.0)
    assert e1 == payload
    assert e2 == payload


@pytest.mark.asyncio
async def test_event_broker_unsubscribe_stops_delivery() -> None:
    broker = get_event_broker()
    q1 = broker.subscribe()
    q2 = broker.subscribe()

    broker.unsubscribe(q1)

    broker.publish(job_event("updated", {"id": "x"}))

    e2 = await asyncio.wait_for(q2.get(), timeout=1.0)
    assert e2["job"]["id"] == "x"
    # q1 must not receive it.
    assert q1.empty()


@pytest.mark.asyncio
async def test_event_broker_unsubscribe_is_idempotent() -> None:
    """A double-unsubscribe (e.g. SSE generator finally + endpoint cleanup
    racing) must not raise."""
    broker = get_event_broker()
    q = broker.subscribe()
    broker.unsubscribe(q)
    broker.unsubscribe(q)


def test_event_broker_drops_event_for_slow_subscriber() -> None:
    """When a subscriber's queue is full the event is dropped instead of
    blocking the producer — keeps fast clients (and the LLM stream) flowing."""
    broker = get_event_broker()
    q = broker.subscribe()
    # Fill the queue right up to the cap.
    for i in range(q.maxsize):
        q.put_nowait({"type": "delta", "delta": str(i)})

    # One more publish — must not raise, must not block.
    broker.publish({"type": "delta", "delta": "overflow"})

    # The dropped event never made it into the queue.
    assert q.qsize() == q.maxsize


# ---------------------------------------------------------------------------
# JobEventBroker mirrors into EventBroker with job_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_job_broker_mirrors_into_global_broker_with_job_id() -> None:
    job_broker = get_broker()
    event_broker = get_event_broker()
    global_q = event_broker.subscribe()

    job_broker.publish("job-1", delta_event("hello"))

    e = await asyncio.wait_for(global_q.get(), timeout=1.0)
    assert e["type"] == "delta"
    assert e["delta"] == "hello"
    assert e["job_id"] == "job-1"


@pytest.mark.asyncio
async def test_job_broker_publish_to_missing_job_id_still_mirrors() -> None:
    """A delta for a job nobody subscribed to should still hit /events
    subscribers — that's how the Library learns about background jobs."""
    job_broker = get_broker()
    event_broker = get_event_broker()
    global_q = event_broker.subscribe()

    job_broker.publish("nobody-listening", delta_event("x"))

    e = await asyncio.wait_for(global_q.get(), timeout=1.0)
    assert e["job_id"] == "nobody-listening"
