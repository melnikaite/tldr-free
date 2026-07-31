"""Backend "dialect" auto-adaptation in ``llm.client``.

Cloud OpenAI o-series/gpt-5 models reject ``max_tokens``, non-default
``temperature``, and ``reasoning_effort`` with an HTTP 400. ``_dialect()``
starts optimistic (today's local-first defaults) and flips the relevant flag
the first time a 400 blames it, retrying the same logical call once. The
adapted state is process-cached (``lru_cache``) so a later, independent call
doesn't pay the 400 round-trip again.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from openai import BadRequestError

from src import config as config_mod
from src.llm import client as llm_client


@pytest.fixture(autouse=True)
def _reset_caches() -> Any:
    llm_client._llm_lock.cache_clear()
    llm_client._dialect.cache_clear()
    yield
    llm_client._llm_lock.cache_clear()
    llm_client._dialect.cache_clear()


def _bad_request(message: str, param: str | None) -> BadRequestError:
    request = httpx.Request("POST", "http://example.test/v1/chat/completions")
    response = httpx.Response(400, request=request)
    # Mirrors what the real SDK passes as `body` (already unwrapped from the
    # top-level "error" key — see openai._client.OpenAI._make_status_error).
    return BadRequestError(message, response=response, body={"message": message, "param": param})


class _FakeCompletion:
    def __init__(self, text: str) -> None:
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": text})()})()]


class _RecordingChat:
    """Fake ``client.chat.completions`` — records kwargs, replays canned
    responses/exceptions in order."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.completions = self

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _RecordingClient:
    def __init__(self, responses: list[Any]) -> None:
        self.chat = _RecordingChat(responses)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.chat.calls


# ---------------------------------------------------------------------------
# max_tokens -> max_completion_tokens
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_tokens_adapts_to_max_completion_tokens_with_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config_mod.get_config()
    headroom = cfg.llm.reasoning_headroom_tokens

    err = _bad_request(
        "Unsupported parameter: 'max_tokens' is not supported with this model. "
        "Use 'max_completion_tokens' instead.",
        param="max_tokens",
    )
    client = _RecordingClient([err, _FakeCompletion("ok")])
    monkeypatch.setattr(llm_client, "_client", lambda: client)

    result = await llm_client.complete_with_messages(
        [{"role": "user", "content": "hi"}], max_tokens=1000
    )
    assert result.choices[0].message.content == "ok"

    assert len(client.calls) == 2
    first, second = client.calls
    assert first["max_tokens"] == 1000
    assert "max_completion_tokens" not in first
    assert "max_tokens" not in second
    assert second["max_completion_tokens"] == 1000 + headroom

    # Cached: a fresh logical call goes straight to max_completion_tokens —
    # no second 400 needed.
    client2 = _RecordingClient([_FakeCompletion("ok2")])
    monkeypatch.setattr(llm_client, "_client", lambda: client2)
    await llm_client.complete_with_messages([{"role": "user", "content": "hi2"}], max_tokens=500)
    assert len(client2.calls) == 1
    assert client2.calls[0]["max_completion_tokens"] == 500 + headroom


@pytest.mark.asyncio
async def test_max_tokens_adaptation_works_on_streaming_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 400 happens at create(..., stream=True) before any chunk is read —
    retrying must not have consumed a partial stream."""
    err = _bad_request("'max_tokens' is not supported", param="max_tokens")

    class _FakeStream:
        def __aiter__(self) -> Any:
            return self

        async def __anext__(self) -> Any:
            raise StopAsyncIteration

    client = _RecordingClient([err, _FakeStream()])
    monkeypatch.setattr(llm_client, "_client", lambda: client)

    deltas = [d async for d in llm_client.stream_with_messages(
        [{"role": "user", "content": "hi"}], max_tokens=200
    )]
    assert deltas == []
    assert len(client.calls) == 2
    assert client.calls[0]["stream"] is True
    assert "max_completion_tokens" in client.calls[1]


# ---------------------------------------------------------------------------
# temperature
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_temperature_adapts_to_not_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    err = _bad_request(
        "Unsupported value: 'temperature' does not support 0.3 with this model.",
        param="temperature",
    )
    client = _RecordingClient([err, _FakeCompletion("ok")])
    monkeypatch.setattr(llm_client, "_client", lambda: client)

    await llm_client.complete_with_messages(
        [{"role": "user", "content": "hi"}], max_tokens=100, temperature=0.3
    )
    first, second = client.calls
    assert first["temperature"] == 0.3
    assert "temperature" not in second

    # Cached for a later independent call.
    client2 = _RecordingClient([_FakeCompletion("ok2")])
    monkeypatch.setattr(llm_client, "_client", lambda: client2)
    await llm_client.complete_with_messages([{"role": "user", "content": "hi2"}], max_tokens=100)
    assert "temperature" not in client2.calls[0]


# ---------------------------------------------------------------------------
# reasoning_effort
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reasoning_effort_adapts_to_not_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = config_mod.get_config()
    monkeypatch.setattr(cfg.llm, "reasoning_effort", "none")

    err = _bad_request(
        "Unsupported parameter: 'reasoning_effort' is not supported with this model.",
        param="reasoning_effort",
    )
    client = _RecordingClient([err, _FakeCompletion("ok")])
    monkeypatch.setattr(llm_client, "_client", lambda: client)

    await llm_client.complete_with_messages([{"role": "user", "content": "hi"}], max_tokens=100)
    first, second = client.calls
    assert first["extra_body"] == {"reasoning_effort": "none"}
    assert second["extra_body"] is None

    # Cached for a later independent call.
    client2 = _RecordingClient([_FakeCompletion("ok2")])
    monkeypatch.setattr(llm_client, "_client", lambda: client2)
    await llm_client.complete_with_messages([{"role": "user", "content": "hi2"}], max_tokens=100)
    assert client2.calls[0]["extra_body"] is None


# ---------------------------------------------------------------------------
# Escape hatches pin the dialect and skip adaptation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pinned_token_param_does_not_adapt_and_propagates_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config_mod.get_config()
    monkeypatch.setattr(cfg.llm, "token_param", "max_tokens")

    err = _bad_request("'max_tokens' is not supported", param="max_tokens")
    client = _RecordingClient([err])
    monkeypatch.setattr(llm_client, "_client", lambda: client)

    with pytest.raises(BadRequestError):
        await llm_client.complete_with_messages(
            [{"role": "user", "content": "hi"}], max_tokens=100
        )
    assert len(client.calls) == 1  # no retry — nothing to adapt


# ---------------------------------------------------------------------------
# Unrelated 400s propagate immediately, un-adapted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unrelated_400_propagates_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    err = _bad_request("Invalid 'messages[0].role': must be one of ...", param="messages")
    client = _RecordingClient([err])
    monkeypatch.setattr(llm_client, "_client", lambda: client)

    with pytest.raises(BadRequestError):
        await llm_client.complete_with_messages(
            [{"role": "user", "content": "hi"}], max_tokens=100
        )
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_recurring_400_gives_up_after_attempt_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 400 that keeps recurring for an undetected reason must not loop
    forever — the attempt cap (4: three dialect dimensions to learn + one
    final real attempt) bounds it and the exception propagates once
    exhausted, instead of retrying indefinitely."""
    err1 = _bad_request("'max_tokens' is not supported", param="max_tokens")
    err2 = _bad_request("'max_tokens' is not supported", param="max_tokens")
    err3 = _bad_request("'max_tokens' is not supported", param="max_tokens")
    err4 = _bad_request("'max_tokens' is not supported", param="max_tokens")
    client = _RecordingClient([err1, err2, err3, err4])
    monkeypatch.setattr(llm_client, "_client", lambda: client)

    with pytest.raises(BadRequestError):
        await llm_client.complete_with_messages(
            [{"role": "user", "content": "hi"}], max_tokens=100
        )
    # Exactly 4 attempts (the cap) — never a 5th, even though the mock would
    # happily keep 400ing forever.
    assert len(client.calls) == 4


# ---------------------------------------------------------------------------
# All three dimensions rejected one at a time — must still succeed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_three_dimensions_reject_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real-world case that motivated raising the attempt cap: a cloud
    gpt-5/o-series config still has ``reasoning_effort: "none"`` left over
    from a LocalAI setup, so the backend rejects all three dialect
    dimensions — but (as providers typically do) reports only one offending
    parameter per response. Learning all three must still leave room for one
    real attempt with the fully-adapted dialect, instead of exhausting the
    cap on learning alone."""
    cfg = config_mod.get_config()
    monkeypatch.setattr(cfg.llm, "reasoning_effort", "none")

    err_tokens = _bad_request("'max_tokens' is not supported", param="max_tokens")
    err_temp = _bad_request(
        "Unsupported value: 'temperature' does not support 0.3 with this model.",
        param="temperature",
    )
    err_reasoning = _bad_request(
        "Unsupported parameter: 'reasoning_effort' is not supported with this model.",
        param="reasoning_effort",
    )
    client = _RecordingClient([err_tokens, err_temp, err_reasoning, _FakeCompletion("ok")])
    monkeypatch.setattr(llm_client, "_client", lambda: client)

    result = await llm_client.complete_with_messages(
        [{"role": "user", "content": "hi"}], max_tokens=100, temperature=0.3
    )
    assert result.choices[0].message.content == "ok"
    assert len(client.calls) == 4

    dialect = llm_client._dialect()
    assert dialect.token_param == "max_completion_tokens"
    assert dialect.send_temperature is False
    assert dialect.send_reasoning_effort is False
