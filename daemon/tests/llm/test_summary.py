"""Tests for llm.summary.stream_summarize — single-pass vs map-reduce branching.

Mocks `llm.client.stream_complete` (and `complete` for the map phase) so no
real LLM call is made.
"""

from __future__ import annotations

import pytest

from src.llm import client as llm_client
from src.llm import summary as summary_mod


async def _async_iter(items: list[str]):
    for it in items:
        yield it


async def _collect(text: str, *, title: str | None, output_language: str) -> str:
    parts: list[str] = []
    async for delta in summary_mod.stream_summarize(
        text, title=title, output_language=output_language
    ):
        parts.append(delta)
    return "".join(parts).strip()


@pytest.mark.asyncio
async def test_short_text_uses_single_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single-pass path goes through stream_complete only."""
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_stream(prompt: str, **kwargs: object):
        calls.append((prompt, dict(kwargs)))
        return _async_iter(["## Сводка\n\n", "Тестовый ", "ответ."])

    monkeypatch.setattr(llm_client, "stream_complete", fake_stream)

    text = "Короткий исходник для проверки." * 5
    result = await _collect(text, title="Test", output_language="English")

    assert result == "## Сводка\n\nТестовый ответ."
    assert len(calls) == 1, "Single-pass should make exactly one stream_complete call"
    prompt = calls[0][0]
    assert "English" in prompt
    assert "Test" in prompt
    assert text in prompt


@pytest.mark.asyncio
async def test_long_text_uses_map_reduce(monkeypatch: pytest.MonkeyPatch) -> None:
    """Map: each chunk → complete() (non-streaming). Reduce → stream_complete()."""

    from src import config as config_mod

    cfg = config_mod.get_config()
    monkeypatch.setattr(cfg.llm, "single_pass_token_limit", 200)

    map_calls: list[str] = []
    stream_calls: list[str] = []

    async def fake_complete(prompt: str, **kwargs: object) -> str:
        map_calls.append(prompt)
        return f"Partial summary #{len(map_calls)}"

    def fake_stream(prompt: str, **kwargs: object):
        stream_calls.append(prompt)
        return _async_iter(["## Final ", "summary"])

    monkeypatch.setattr(llm_client, "complete", fake_complete)
    monkeypatch.setattr(llm_client, "stream_complete", fake_stream)

    paragraph = (
        "Это тестовый параграф с достаточным количеством текста для того, "
        "чтобы при разбиении на чанки получилось несколько частей. "
    )
    text = "\n\n".join(f"Параграф {i}. {paragraph * 30}" for i in range(40))

    result = await _collect(text, title="Long doc", output_language="English")
    assert result == "## Final summary"

    # Map phase: at least 2 chunks → at least 2 complete() calls.
    assert len(map_calls) >= 2
    # Reduce phase: exactly one stream_complete() call carrying the joined partials.
    assert len(stream_calls) == 1
    assert "Chunk summaries:" in stream_calls[0]
    for p in map_calls:
        assert "English" in p
        assert "Long doc" in p
        assert "Chunk summaries:" not in p


@pytest.mark.asyncio
async def test_empty_input_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_stream(prompt: str, **kwargs: object):
        raise AssertionError("stream_complete should not be called for empty input")

    monkeypatch.setattr(llm_client, "stream_complete", fake_stream)

    assert await _collect("", title=None, output_language="English") == ""
    assert await _collect("   ", title=None, output_language="English") == ""
