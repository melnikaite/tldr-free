"""``WorkerControl.pause()`` / ``resume()`` toggle the global flag and publish
``workers_event`` to the global event broker so the UI reflects the new state
without polling.

The pause-broken-in-prod regression was that the publish was happening from a
threadpool thread (``asyncio.Queue.put_nowait`` is loop-only). Tests below
exercise pause/resume from the loop thread to lock in the working path.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.workers import broker as broker_mod
from src.workers.broker import get_event_broker
from src.workers.control import get_control, reset_control
from src.workers.queue import reset_queue


@pytest.fixture(autouse=True)
def _reset() -> Any:
    reset_control()
    reset_queue()
    broker_mod.reset_broker()
    yield
    reset_control()
    reset_queue()
    broker_mod.reset_broker()


@pytest.mark.asyncio
async def test_pause_flips_flag_and_emits_workers_event() -> None:
    queue = get_event_broker().subscribe()

    get_control().pause()

    assert get_control().paused is True
    event = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert event["type"] == "workers"
    assert event["state"]["paused"] is True
    assert "queue_size" in event["state"]
    assert "running" in event["state"]


@pytest.mark.asyncio
async def test_resume_flips_flag_and_emits_workers_event() -> None:
    get_control().pause()  # one event we'll skip
    queue = get_event_broker().subscribe()

    get_control().resume()

    assert get_control().paused is False
    event = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert event["type"] == "workers"
    assert event["state"]["paused"] is False


@pytest.mark.asyncio
async def test_pause_called_twice_still_publishes_each_time() -> None:
    """Idempotent flag flip but publish stays on every call so a re-render
    on the UI side never silently misses state."""
    queue = get_event_broker().subscribe()

    get_control().pause()
    get_control().pause()

    e1 = await asyncio.wait_for(queue.get(), timeout=1.0)
    e2 = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert e1["state"]["paused"] is True
    assert e2["state"]["paused"] is True


@pytest.mark.asyncio
async def test_wait_if_paused_blocks_until_resume() -> None:
    """The pipeline relies on this exact contract — park here while paused."""
    get_control().pause()
    waiter = asyncio.create_task(get_control().wait_if_paused())

    # Give the waiter a chance to spin once.
    await asyncio.sleep(0.05)
    assert not waiter.done()

    get_control().resume()
    await asyncio.wait_for(waiter, timeout=2.0)
