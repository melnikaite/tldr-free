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
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

from openai import AsyncOpenAI, BadRequestError
from openai.types.chat import ChatCompletionMessageParam

from src.config import get_config


@lru_cache(maxsize=1)
def _client() -> AsyncOpenAI:
    config = get_config()
    return AsyncOpenAI(
        base_url=config.llm.base_url,
        api_key=config.llm.effective_api_key,
    )


def _model() -> str:
    return get_config().llm.model


def reset_caches() -> None:
    """Drop the cached ``AsyncOpenAI`` client and dialect guess after a
    config change (``PATCH /config``), so the next call rebuilds them from
    the fresh ``base_url``/``api_key``/``model``/dialect-related fields.

    Deliberately does NOT touch ``_llm_lock()`` — its ``asyncio.Semaphore``
    is bound to the running event loop and can't be resized in place; a
    changed ``max_concurrent_calls`` needs a process restart instead (see
    ``restart_required`` in the ``/config`` API responses).
    """
    _client.cache_clear()
    _dialect.cache_clear()


# ---------------------------------------------------------------------------
# Backend "dialect" auto-detection
#
# Local backends (mlx-server, LocalAI, Ollama, llama.cpp) and OpenAI's
# o-series/gpt-5 models disagree on three request-body dimensions:
#   - the token-limit kwarg name (`max_tokens` vs `max_completion_tokens`)
#   - whether a non-default `temperature` is accepted at all
#   - whether `reasoning_effort` is a known field
# Sending the wrong shape gets an HTTP 400 back. Rather than hardcode a
# backend allowlist, we start optimistic (today's local-first defaults) and
# flip the relevant flag the first time we see a 400 that blames it,
# retrying the same logical call once per flip. The result is cached for the
# rest of the process lifetime (not persisted to disk) so subsequent calls
# don't pay the round-trip again.
# ---------------------------------------------------------------------------


@dataclass
class _Dialect:
    token_param: str = "max_tokens"
    send_temperature: bool = True
    send_reasoning_effort: bool = True


def _new_dialect() -> _Dialect:
    """Build a starting dialect from config-pinned overrides (if any),
    defaulting to the optimistic local-first assumptions otherwise. Exposed
    (uncached) for callers that need a throwaway dialect scoped to a single
    call — e.g. ``POST /config/test`` probing a candidate backend that may
    not be the currently-configured/cached one."""
    cfg = get_config().llm
    dialect = _Dialect()
    if cfg.token_param != "auto":
        dialect.token_param = cfg.token_param
    if cfg.send_temperature is not None:
        dialect.send_temperature = cfg.send_temperature
    return dialect


@lru_cache(maxsize=1)
def _dialect() -> _Dialect:
    return _new_dialect()


def _detect_400_dimension(exc: BadRequestError) -> str | None:
    """Classify a 400 as one of the three known dialect mismatches, or None
    if it's unrelated (in which case the caller should just propagate it)."""
    haystack = f"{exc.param or ''} {exc}".lower()
    if "max_tokens" in haystack:
        return "max_tokens"
    if "temperature" in haystack:
        return "temperature"
    if "reasoning_effort" in haystack:
        return "reasoning_effort"
    return None


def _extra_body(dialect: _Dialect) -> dict[str, str] | None:
    """Return extra request-body fields needed by the configured backend.

    ``reasoning_effort`` disables chain-of-thought on models like Gemma 4 in
    LM Studio. Without it the model exhausts the token budget on thinking
    tokens and emits no ``delta.content``. Set ``llm.reasoning_effort: "none"``
    in config/tldr.yaml to activate. Suppressed once the dialect has learned
    (via a 400) that the backend rejects the field.
    """
    effort = get_config().llm.reasoning_effort
    if effort is not None and dialect.send_reasoning_effort:
        return {"reasoning_effort": effort}
    return None


def _token_kwargs(dialect: _Dialect, requested_max_tokens: int) -> dict[str, int]:
    """Build the single token-limit kwarg, applying the configured ceiling
    and (for `max_completion_tokens`) reasoning headroom."""
    cfg = get_config().llm
    limit = requested_max_tokens
    if cfg.max_output_tokens is not None:
        limit = min(limit, cfg.max_output_tokens)
    if dialect.token_param == "max_completion_tokens":
        limit += cfg.reasoning_headroom_tokens
    return {dialect.token_param: limit}


def _build_kwargs(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    dialect: _Dialect,
    stream: bool,
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=cast(list[ChatCompletionMessageParam], messages),
        extra_body=_extra_body(dialect),
    )
    kwargs.update(_token_kwargs(dialect, max_tokens))
    if dialect.send_temperature:
        kwargs["temperature"] = temperature
    if stream:
        kwargs["stream"] = True
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice if tool_choice is not None else "auto"
    return kwargs


# One attempt to LEARN each of the three known dialect dimensions
# (token_param, send_temperature, send_reasoning_effort) + one final attempt
# made with the fully-adapted dialect = 3 + 1 = 4. A backend that rejects
# all three at once but reports them one violation per response (the real
# case that motivated this: a cloud gpt-5/o-series config that still had
# `reasoning_effort: "none"` left over from a LocalAI config) would
# otherwise spend every attempt learning and never get a real shot with the
# corrected dialect. If a 4th adaptable dimension is ever added, bump this
# to 5 (dimensions + 1) — the "+1" is what guarantees a real attempt after
# the last adaptation, not just another learning round.
_MAX_DIALECT_ATTEMPTS = 4


async def call_with_dialect_adaptation(
    *,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    stream: bool,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    client: AsyncOpenAI | None = None,
    model: str | None = None,
    dialect: _Dialect | None = None,
) -> Any:
    """``chat.completions.create(...)``, auto-adapting the dialect on a 400.

    By default runs against the cached prod client/model/dialect (the
    process-lifetime-cached backend this daemon is configured to talk to).
    Pass an explicit ``client``/``model``/``dialect`` to run the identical
    adaptation logic against a DIFFERENT backend — e.g. ``POST /config/test``
    probing a candidate (possibly not-yet-saved) config — without touching
    the cached prod state. This keeps dialect-mismatch handling in exactly
    one place instead of two diverging copies.

    Must be called from INSIDE an already-acquired ``_llm_lock()`` slot when
    operating on the cached prod client (the default) — it does not touch
    the semaphore itself, so it's safe to retry internally without any risk
    of double-acquire/double-release. Callers supplying their own throwaway
    ``client`` (like the config-test probe) don't go through the semaphore
    at all, which is fine — it exists to protect the shared prod backend,
    not a one-off connectivity check.

    For the streaming path this returns the (unconsumed) stream object: the
    400 always happens at ``create(..., stream=True)`` time, before any
    chunk is read, so retrying here never risks re-emitting partial output.
    """
    resolved_client = client if client is not None else _client()
    resolved_model = model if model is not None else _model()
    resolved_dialect = dialect if dialect is not None else _dialect()
    cfg = get_config().llm
    last_exc: BadRequestError | None = None
    for _ in range(_MAX_DIALECT_ATTEMPTS):
        kwargs = _build_kwargs(
            model=resolved_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            dialect=resolved_dialect,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
        )
        try:
            return await resolved_client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            dimension = _detect_400_dimension(e)
            if dimension == "max_tokens" and cfg.token_param == "auto":
                resolved_dialect.token_param = "max_completion_tokens"
            elif dimension == "temperature" and cfg.send_temperature is None:
                resolved_dialect.send_temperature = False
            elif dimension == "reasoning_effort":
                resolved_dialect.send_reasoning_effort = False
            else:
                # Not one of the three known dimensions, or the relevant
                # escape hatch is pinned by config — nothing to adapt.
                raise
            last_exc = e
    if last_exc is None:
        # Unreachable in practice: every non-returning loop iteration above
        # either raises immediately (unknown/pinned dimension) or sets
        # last_exc before looping again, so exhausting the loop without
        # returning guarantees last_exc is set. Kept as an explicit check
        # rather than `assert` because `python -O` strips asserts, which
        # would turn this into a confusing `raise None`.
        raise RuntimeError(
            "call_with_dialect_adaptation exhausted attempts without capturing an exception"
        )
    raise last_exc


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
    tool_choice: str | dict[str, Any] | None = None,
    max_tokens: int = 1500,
    temperature: float = 0.3,
    respect_pause: bool = False,
) -> Any:
    """Non-streaming chat completion from a messages list. Returns the raw
    OpenAI ``ChatCompletion`` object so the caller can inspect tool_calls.

    ``tool_choice`` mirrors the OpenAI API field:
    - None / not set → "auto" when tools are provided (default)
    - "required" → the model MUST call one of the tools
    - {"type": "function", "function": {"name": "…"}} → force a specific tool
    """
    await _acquire_llm_slot(respect_pause)
    try:
        return await call_with_dialect_adaptation(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            tools=tools,
            tool_choice=tool_choice,
        )
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
        stream = await call_with_dialect_adaptation(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
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
