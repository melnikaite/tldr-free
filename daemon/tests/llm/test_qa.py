"""Tests for llm.qa.stream_answer.

Mocks ``llm_client.complete_with_messages`` (the forced-tool routing call)
and ``llm_client.stream_with_messages`` (the grounded-answer stream) to avoid
any real LLM calls. Verifies output_language / title / context threading, the
language-agnostic two-tool routing (web_search vs answer_from_material), and
the plain-stream fallback when the backend lacks tool support.

Routing note: the QA flow uses tool_choice="required" with two tools. The
model ALWAYS returns a tool call — either web_search or answer_from_material.
There is no "model returns content directly" happy path anymore (that only
happens as a defensive fallback if a backend honours "required" loosely).
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


def _completion_with_tool(tool_name: str, arguments: str) -> Any:
    """ChatCompletion mock that asks for one tool call."""
    func = types.SimpleNamespace(name=tool_name, arguments=arguments)
    tc = types.SimpleNamespace(id="call_test123", type="function", function=func)
    msg = types.SimpleNamespace(content=None, tool_calls=[tc])
    choice = types.SimpleNamespace(message=msg)
    return types.SimpleNamespace(choices=[choice])


def _completion_no_tool(content: str) -> Any:
    """ChatCompletion mock with no tool call (loose-'required' backend)."""
    msg = types.SimpleNamespace(content=content, tool_calls=None)
    choice = types.SimpleNamespace(message=msg)
    return types.SimpleNamespace(choices=[choice])


# ---------------------------------------------------------------------------
# answer_from_material path (model routes to the material, no search)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answer_from_material_streams_grounded_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model picks answer_from_material → no DDG → final answer streamed."""
    captured_messages: list[list[dict]] = []

    async def fake_complete(messages: list[dict], **kwargs: object) -> Any:
        captured_messages.append(messages)
        # Verify the router was forced to choose a tool.
        assert kwargs.get("tool_choice") == "required"
        assert len(kwargs.get("tools") or []) == 2
        return _completion_with_tool("answer_from_material", "{}")

    streamed_messages: list[list[dict]] = []

    async def fake_stream(messages: list[dict], **kwargs: object) -> AsyncIterator[str]:
        streamed_messages.append(messages)
        yield "Hello, world."

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)

    job = _FakeJob(
        title="Some video",
        raw_text="[00:30] First moment.\n\n[01:00] Second moment.",
        summary_md="## Summary\nA summary.",
    )

    out = [item async for item in qa_mod.stream_answer(
        job=job, question="What happens at 00:30?", output_language="English"
    )]

    assert "".join(s for s in out if isinstance(s, str)) == "Hello, world."
    # Router prompt threaded language / question / title / raw_text context.
    prompt = captured_messages[0][0]["content"]
    assert "English" in prompt
    assert "What happens at 00:30?" in prompt
    assert "Some video" in prompt
    assert "First moment." in prompt
    # No searching stage event on the material path.
    assert not [i for i in out if isinstance(i, dict)]
    # The stream got user + assistant(tool_call) + tool(ack) messages.
    roles = [m["role"] for m in streamed_messages[0]]
    assert roles == ["user", "assistant", "tool"]


@pytest.mark.asyncio
async def test_falls_back_to_summary_when_raw_too_long(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When raw_text doesn't fit in context, summary_md is used instead."""
    from src import config as config_mod

    cfg = config_mod.get_config()
    monkeypatch.setattr(cfg.llm, "context_length", 4001)

    captured_messages: list[list[dict]] = []

    async def fake_complete(messages: list[dict], **kwargs: object) -> Any:
        captured_messages.append(messages)
        return _completion_with_tool("answer_from_material", "{}")

    async def fake_stream(messages: list[dict], **kwargs: object) -> AsyncIterator[str]:
        yield ""

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)

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
        return _completion_with_tool("answer_from_material", "{}")

    async def fake_stream(messages: list[dict], **kwargs: object) -> AsyncIterator[str]:
        yield "ok"

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)

    job = _FakeJob(title=None, raw_text="some text", summary_md=None)
    out = [item async for item in qa_mod.stream_answer(job=job, question="q", output_language="English")]
    assert [s for s in out if isinstance(s, str)] == ["ok"]
    assert "{title}" not in captured_messages[0][0]["content"]


# ---------------------------------------------------------------------------
# web_search path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_web_search_tool_called_and_results_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model picks web_search → page-fetching DDG runs → results injected → answer streamed."""
    searched: list[str] = []

    async def fake_ddg_with_content(query: str, **kwargs: object) -> list[dict[str, Any]]:
        searched.append(query)
        return [
            {
                "title": "T",
                "href": "https://ex.com",
                "body": "snippet",
                "content": f"full cleaned page about {query}",
            }
        ]

    # qa.py calls _search.ddg_search_with_content for the enriched path.
    monkeypatch.setattr(qa_mod._search, "ddg_search_with_content", fake_ddg_with_content)

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

    assert len(stage_events) == 1
    assert stage_events[0]["stage"] == "searching"
    assert stage_events[0]["detail"] == "hantavirus germany"
    assert searched == ["hantavirus germany"]
    assert "".join(deltas) == "final answer"

    # The streaming call received user + assistant(tool_call) + tool(result),
    # and the tool result carries the CLEANED PAGE CONTENT (not just the snippet).
    final_msgs = streamed_messages[0]
    roles = [m["role"] for m in final_msgs]
    assert roles == ["user", "assistant", "tool"]
    assert "full cleaned page about hantavirus germany" in final_msgs[-1]["content"]


# ---------------------------------------------------------------------------
# Fallback when backend rejects tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_falls_back_to_stream_complete_on_tool_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If complete_with_messages raises (no tool/required support), stream_complete is used."""

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

    assert "".join(s for s in out if isinstance(s, str)) == "fallback answer"
    assert len(captured) == 1
    assert "?" in captured[0]


@pytest.mark.asyncio
async def test_loose_required_backend_yields_content_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive: a backend that honours 'required' loosely (returns content,
    no tool call) should still yield that content rather than hang."""

    async def fake_complete(messages: list[dict], **kwargs: object) -> Any:
        return _completion_no_tool("direct answer despite required")

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)

    job = _FakeJob(title="T", raw_text="text", summary_md=None)
    out = [item async for item in qa_mod.stream_answer(
        job=job, question="q", output_language="English"
    )]
    assert "".join(s for s in out if isinstance(s, str)) == "direct answer despite required"
