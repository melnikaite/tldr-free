"""OpenAI-compatible LLM client. Default backend: mlx-server.

    async def complete(prompt, *, max_tokens, temperature, respect_pause=True) -> str
        Non-streaming. Used by the summary map phase.

    async def stream_complete(prompt, *, max_tokens, temperature, respect_pause=True) -> AsyncIterator[str]
        Streaming. Yields token strings as they arrive — used by the
        single-pass / reduce summary path and by QA.

    async def complete_with_messages(messages, *, tools, max_tokens, ...) -> Any
        Non-streaming, accepts a messages list (for tool-calling flows).

    async def stream_with_messages(messages, *, max_tokens, ...) -> AsyncIterator[str]
        Streaming from a messages list. ``complete`` and ``stream_complete``
        are thin wrappers around these two primitives.

Built lazily from config.llm.{base_url, api_key, model}.

Concurrency + pause: one global semaphore (``_LLM_LOCK``) serialises every
call — ``complete``, ``stream_complete``, parallel map chunks, QA — so the
local mlx-server is never asked to run two Qwen completions at the same
time. On a single Apple Silicon box this avoids thrashing the GPU /
Neural Engine and keeps the fan tolerable. The Whisper handler runs in a
separate mlx-server slot and its own single-worker queue, so transcription
and summarisation can still overlap when both happen to be in flight.

Lock acquisition is **pause-aware**: callers with ``respect_pause=True``
(the default — every summary path) wait for ``WorkerControl.paused`` to
clear BEFORE grabbing the semaphore, and re-check after grabbing in case
pause flipped while they were queued. This is the only place that defends
against the "5 pipelines all queued behind the LLM lock all sneak past
pause" race — pipeline-level paused checks happen before the queue, not
after, so they don't help once a job is already waiting on the semaphore.

QA passes ``respect_pause=False`` because the user is actively waiting
on the answer; pausing the workers should not freeze a chat reply.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.config import get_config


@lru_cache(maxsize=1)
def _client() -> AsyncOpenAI:
    config = get_config()
    return AsyncOpenAI(
        base_url=config.llm.base_url,
        api_key=config.llm.api_key,
    )


def _model() -> str:
    return get_config().llm.model


def _extra_body() -> dict[str, str] | None:
    """Return extra request-body fields needed by the configured backend.

    ``reasoning_effort`` disables chain-of-thought on models like Gemma 4 in
    LM Studio. Without it the model exhausts ``max_tokens`` on thinking tokens
    and emits no ``delta.content``. Set ``llm.reasoning_effort: "none"`` in
    config/tldr.yaml to activate. Backends that don't know the field ignore it
    (mlx-openai-server, Ollama, llama-server) or return a 400 — in which case
    remove the key from the config for that backend.
    """
    effort = get_config().llm.reasoning_effort
    if effort is not None:
        return {"reasoning_effort": effort}
    return None


@lru_cache(maxsize=1)
def _llm_lock() -> asyncio.Semaphore:
    """Lazy-init semaphore so it binds to the running event loop."""
    n = max(1, get_config().llm.max_concurrent_calls)
    return asyncio.Semaphore(n)


def _is_paused() -> bool:
    """Best-effort read of ``WorkerControl.paused``. Late-imported so the LLM
    layer doesn't take a compile-time dependency on workers/."""
    try:
        from src.workers.control import get_control
    except Exception:
        return False
    try:
        return get_control().paused
    except Exception:
        return False


async def _wait_paused() -> None:
    try:
        from src.workers.control import get_control
    except Exception:
        return
    try:
        await get_control().wait_if_paused()
    except Exception:
        return


async def _acquire_llm_slot(respect_pause: bool) -> None:
    """Wait for the semaphore AND (when respect_pause) the pause flag.

    Critical: the pause re-check happens AFTER acquire so a flip that
    landed while we were queued still holds us off. If we were paused at
    that point we release the slot and try again — otherwise a paused
    daemon would still drain a backlog of waiters one-by-one.
    """
    while True:
        if respect_pause:
            await _wait_paused()
        await _llm_lock().acquire()
        if not respect_pause or not _is_paused():
            return
        _llm_lock().release()
        # Loop and wait for the pause to clear before retrying acquire.


# ---------------------------------------------------------------------------
# Core primitives — messages API
# ---------------------------------------------------------------------------


async def complete_with_messages(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int = 1500,
    temperature: float = 0.3,
    respect_pause: bool = False,
) -> Any:
    """Non-streaming chat completion from a messages list. Returns the raw
    OpenAI ``ChatCompletion`` object so the caller can inspect tool_calls."""
    await _acquire_llm_slot(respect_pause)
    try:
        kwargs: dict[str, Any] = dict(
            model=_model(),
            messages=cast(list[ChatCompletionMessageParam], messages),
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=_extra_body(),
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return await _client().chat.completions.create(**kwargs)
    finally:
        _llm_lock().release()


async def stream_with_messages(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 1500,
    temperature: float = 0.3,
    respect_pause: bool = False,
) -> AsyncIterator[str]:
    """Streaming chat completion from a messages list. Yields delta strings.

    Per-chunk timeout: if the backend stops sending tokens for
    ``config.llm.stream_chunk_timeout_seconds`` (default 60 s) we raise
    ``TimeoutError`` instead of waiting forever.
    """
    await _acquire_llm_slot(respect_pause)
    try:
        stream = await _client().chat.completions.create(
            model=_model(),
            messages=cast(list[ChatCompletionMessageParam], messages),
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            extra_body=_extra_body(),
        )
        chunk_timeout = get_config().llm.stream_chunk_timeout_seconds
        stream_iter = stream.__aiter__()
        while True:
            try:
                chunk = await asyncio.wait_for(stream_iter.__anext__(), timeout=chunk_timeout)
            except StopAsyncIteration:
                return
            except TimeoutError as e:
                raise TimeoutError(
                    f"llm stream stalled: no chunk for {chunk_timeout:.0f}s",
                ) from e
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    finally:
        _llm_lock().release()


# ---------------------------------------------------------------------------
# Convenience wrappers — single-prompt API (backward-compatible)
# ---------------------------------------------------------------------------


async def complete(
    prompt: str,
    *,
    max_tokens: int = 1500,
    temperature: float = 0.3,
    respect_pause: bool = True,
) -> str:
    """Non-streaming chat completion. Returns the full assistant response."""
    response = await complete_with_messages(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        respect_pause=respect_pause,
    )
    return response.choices[0].message.content or ""


async def stream_complete(
    prompt: str,
    *,
    max_tokens: int = 1500,
    temperature: float = 0.3,
    respect_pause: bool = True,
) -> AsyncIterator[str]:
    """Streaming chat completion. Yields delta.content strings as they arrive.

    Pause is enforced at acquire time (``_acquire_llm_slot``): an in-flight
    stream completes normally — we don't abort it mid-token. The next LLM
    call (next chunk in map-reduce, or the next pipeline's summary) blocks
    on the pause flag.
    """
    async for delta in stream_with_messages(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        respect_pause=respect_pause,
    ):
        yield delta
