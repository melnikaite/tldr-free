"""``llm.client._acquire_llm_slot`` is the bottleneck pause has to defend.

Real bug: pipeline-level ``if control.paused`` checks happen BEFORE a job
queues for the LLM semaphore. With several jobs in flight the second /
third / fourth all sneak past the check, then sit on ``_llm_lock`` waiting
for their turn. By the time pause flips, they've already passed every
gate the pipeline knows about and run their LLM call regardless.

The fix is here at the lock layer: every acquire (with respect_pause=True)
re-checks paused AFTER grabbing the semaphore, releases and waits if pause
landed while the caller was queued, then retries.

Soft-pause contract: in-flight streams complete normally. We do NOT abort
mid-token — the lock-level re-check only catches the next caller in line.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.llm import client as llm_client
from src.workers import broker as broker_mod
from src.workers import control as control_mod
from src.workers import queue as queue_mod
from src.workers.control import get_control


@pytest.fixture(autouse=True)
def _reset() -> Any:
    # Drop the cached semaphore so each test starts with a fresh one bound
    # to its own event loop.
    llm_client._llm_lock.cache_clear()
    control_mod.reset_control()
    queue_mod.reset_queue()
    broker_mod.reset_broker()
    yield
    llm_client._llm_lock.cache_clear()
    control_mod.reset_control()
    queue_mod.reset_queue()
    broker_mod.reset_broker()


@pytest.mark.asyncio
async def test_acquire_blocks_when_already_paused() -> None:
    get_control().pause()

    waiter = asyncio.create_task(llm_client._acquire_llm_slot(respect_pause=True))
    await asyncio.sleep(0.05)
    assert not waiter.done(), "acquire must wait while paused"

    get_control().resume()
    await asyncio.wait_for(waiter, timeout=2.0)
    llm_client._llm_lock().release()


@pytest.mark.asyncio
async def test_pause_during_queued_wait_releases_and_re_blocks() -> None:
    """Two callers queued; first holds the slot, user pauses; the second
    must NOT proceed when the first releases — pause has flipped between
    its initial check and acquire."""
    # Caller 1 grabs the slot and just holds it.
    await llm_client._acquire_llm_slot(respect_pause=True)

    # Caller 2 starts queued behind caller 1.
    caller2 = asyncio.create_task(llm_client._acquire_llm_slot(respect_pause=True))
    await asyncio.sleep(0.05)
    assert not caller2.done()

    # User pauses while caller 2 is still waiting on the semaphore.
    get_control().pause()
    # Caller 1 finishes and releases.
    llm_client._llm_lock().release()

    # Caller 2 must still be blocked — it grabbed the lock briefly, saw
    # paused=True, released and went back to waiting on the pause flag.
    await asyncio.sleep(0.1)
    assert not caller2.done(), "caller 2 must re-block on paused"

    # Resume → caller 2 finally proceeds.
    get_control().resume()
    await asyncio.wait_for(caller2, timeout=2.0)
    llm_client._llm_lock().release()


@pytest.mark.asyncio
async def test_respect_pause_false_bypasses_paused_flag() -> None:
    """Q&A path: user is actively waiting; pause must not freeze the answer."""
    get_control().pause()

    # Should NOT block.
    await asyncio.wait_for(
        llm_client._acquire_llm_slot(respect_pause=False), timeout=1.0,
    )
    llm_client._llm_lock().release()


# ---------------------------------------------------------------------------
# stream_complete: mid-stream cancel on pause flip
# ---------------------------------------------------------------------------


class _FakeChunk:
    def __init__(self, text: str) -> None:
        self.choices = [type("Choice", (), {"delta": type("Delta", (), {"content": text})()})()]


class _FakeStream:
    """Async-iterable fake of an OpenAI streaming response."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self._i = 0

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> _FakeChunk:
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        text = self._chunks[self._i]
        self._i += 1
        await asyncio.sleep(0)  # let the caller's loop schedule
        return _FakeChunk(text)


class _FakeChat:
    def __init__(self, stream: _FakeStream) -> None:
        self._stream = stream
        self.completions = self

    async def create(self, **_: Any) -> _FakeStream:
        return self._stream


class _FakeClient:
    def __init__(self, stream: _FakeStream) -> None:
        self.chat = _FakeChat(stream)


@pytest.mark.asyncio
async def test_stream_complete_runs_to_end_even_when_pause_flips_mid_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Soft-pause contract: an in-flight stream completes normally. The
    lock-level pause check only gates the NEXT acquire, not the running one."""
    fake_stream = _FakeStream(["one ", "two ", "three ", "four "])
    fake_client = _FakeClient(fake_stream)

    monkeypatch.setattr(llm_client, "_client", lambda: fake_client)

    received: list[str] = []
    async for delta in llm_client.stream_complete("prompt", respect_pause=True):
        received.append(delta)
        if len(received) == 2:
            # Simulate the user pressing Pause mid-stream.
            get_control().pause()

    # Stream completes — we don't abort mid-token.
    assert received == ["one ", "two ", "three ", "four "]
    assert get_control().paused is True
