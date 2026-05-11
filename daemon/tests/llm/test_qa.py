"""Tests for llm.qa.stream_answer.

Mocks ``llm_client.complete_with_messages`` (and ``stream_with_messages``
for the tool-call path) to avoid any real LLM calls. Verifies output_language
/ title / context threading, the web_search tool call path, and fallback.
"""

from __future__ import annotations

import types
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from src.llm import client as llm_client
from src.llm import qa as qa_mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeJob:
    title: str | None
    raw_text: str | None
    summary_md: str | None


def _completion(content: str) -> Any:
    """Minimal ChatCompletion mock with no tool_calls."""
    msg = types.SimpleNamespace(content=content, tool_calls=None)
    choice = types.SimpleNamespace(message=msg)
    return types.SimpleNamespace(choices=[choice])


def _completion_with_tool(tool_name: str, arguments: str) -> Any:
    """Minimal ChatCompletion mock that asks for one tool call."""
    func = types.SimpleNamespace(name=tool_name, arguments=arguments)
    tc = types.SimpleNamespace(id="call_test123", type="function", function=func)
    msg = types.SimpleNamespace(content=None, tool_calls=[tc])
    choice = types.SimpleNamespace(message=msg)
    return types.SimpleNamespace(choices=[choice])


# ---------------------------------------------------------------------------
# Direct answer (no tool call)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_answer_yields_content(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_messages: list[list[dict]] = []

    async def fake_complete(messages: list[dict], **kwargs: object) -> Any:
        captured_messages.append(messages)
        return _completion("Hello, world.")

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)

    job = _FakeJob(
        title="Some video",
        raw_text="[00:30] First moment.\n\n[01:00] Second moment.",
        summary_md="## Summary\nA summary.",
    )

    out = [item async for item in qa_mod.stream_answer(
        job=job, question="What happens at 00:30?", output_language="English"
    )]

    assert "".join(out) == "Hello, world."
    assert len(captured_messages) == 1
    prompt = captured_messages[0][0]["content"]
    assert "English" in prompt
    assert "What happens at 00:30?" in prompt
    assert "Some video" in prompt
    # raw_text fits, so it should be in the context (not the summary).
    assert "First moment." in prompt


@pytest.mark.asyncio
async def test_falls_back_to_summary_when_raw_too_long(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When raw_text doesn't fit in context, summary_md is used instead."""
    from src import config as config_mod

    cfg = config_mod.get_config()
    # Force a tiny budget so raw_text never fits (budget = 1 token).
    monkeypatch.setattr(cfg.llm, "context_length", 4001)

    captured_messages: list[list[dict]] = []

    async def fake_complete(messages: list[dict], **kwargs: object) -> Any:
        captured_messages.append(messages)
        return _completion("")

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)

    big_raw = "Очень длинный сырой текст. " * 1000
    job = _FakeJob(title="Long video", raw_text=big_raw, summary_md="## Краткая выжимка\n- Пункт")

    async for _ in qa_mod.stream_answer(job=job, question="?", output_language="English"):
        pass

    prompt = captured_messages[0][0]["content"]
    assert "Краткая выжимка" in prompt
    assert "Очень длинный сырой текст." not in prompt


@pytest.mark.asyncio
async def test_handles_missing_title(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_messages: list[list[dict]] = []

    async def fake_complete(messages: list[dict], **kwargs: object) -> Any:
        captured_messages.append(messages)
        return _completion("ok")

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)

    job = _FakeJob(title=None, raw_text="some text", summary_md=None)
    out = [item async for item in qa_mod.stream_answer(job=job, question="q", output_language="English")]
    assert out == ["ok"]
    assert "{title}" not in captured_messages[0][0]["content"]


# ---------------------------------------------------------------------------
# Tool call path (web_search)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_web_search_tool_called_and_results_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM requests web_search → DDG runs → results injected → final answer streamed."""
    searched: list[str] = []

    async def fake_ddg(query: str, **kwargs: object) -> list[dict[str, Any]]:
        searched.append(query)
        return [{"title": "T", "href": "https://ex.com", "body": f"info about {query}"}]

    # qa.py imports `search` as `_search` (module reference) → patch the module attr.
    monkeypatch.setattr(qa_mod._search, "ddg_search", fake_ddg)

    # First (non-streaming) call → tool call.
    async def fake_complete(messages: list[dict], **kwargs: object) -> Any:
        return _completion_with_tool("web_search", '{"query": "hantavirus germany"}')

    streamed_messages: list[list[dict]] = []

    async def fake_stream(messages: list[dict], **kwargs: object) -> AsyncIterator[str]:
        streamed_messages.append(messages)
        yield "final answer"

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)

    job = _FakeJob(title="T", raw_text="text", summary_md=None)
    items = [item async for item in qa_mod.stream_answer(
        job=job, question="how many in germany", output_language="Russian"
    )]

    stage_events = [i for i in items if isinstance(i, dict)]
    deltas = [i for i in items if isinstance(i, str)]

    # Searching stage event emitted.
    assert len(stage_events) == 1
    assert stage_events[0]["stage"] == "searching"
    assert stage_events[0]["detail"] == "hantavirus germany"

    # DDG was called.
    assert searched == ["hantavirus germany"]

    # Final answer streamed.
    assert "".join(deltas) == "final answer"

    # The streaming call received all messages including the tool result.
    final_msgs = streamed_messages[0]
    roles = [m["role"] for m in final_msgs]
    assert roles == ["user", "assistant", "tool"]
    assert "info about hantavirus germany" in final_msgs[-1]["content"]


# ---------------------------------------------------------------------------
# Fallback when backend rejects tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_falls_back_to_stream_complete_on_tool_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If complete_with_messages raises, stream_complete is used as fallback."""

    async def boom(messages: list[dict], **kwargs: object) -> Any:
        raise RuntimeError("backend does not support tools")

    captured: list[str] = []

    async def fake_stream_complete(prompt: str, **kwargs: object) -> AsyncIterator[str]:
        captured.append(prompt)
        yield "fallback answer"

    monkeypatch.setattr(llm_client, "complete_with_messages", boom)
    monkeypatch.setattr(llm_client, "stream_complete", fake_stream_complete)

    job = _FakeJob(title="T", raw_text="some text", summary_md=None)
    out = [item async for item in qa_mod.stream_answer(
        job=job, question="?", output_language="English"
    )]

    assert "".join(out) == "fallback answer"
    assert len(captured) == 1
    assert "?" in captured[0]
