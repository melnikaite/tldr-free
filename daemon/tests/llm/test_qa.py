"""Tests for llm.qa.stream_answer — plan → search → synthesis flow.

Mocks ``llm_client.complete_with_messages`` (the forced ``plan`` tool call)
and ``llm_client.stream_with_messages`` (the grounded-answer stream) to avoid
any real LLM calls. Verifies context/title/language threading, the layered
material→knowledge→web flow, and that the model can NOT dodge a search:
anything other than a clean ``material_sufficient=true`` (uncertainty, parse
failure, a backend tool error) falls through to a web search.
"""

from __future__ import annotations

import types
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from src.llm import client as llm_client
from src.llm import qa as qa_mod


@dataclass
class _FakeJob:
    title: str | None
    raw_text: str | None
    summary_md: str | None


def _plan_completion(material_sufficient: Any, search_query: str) -> Any:
    """ChatCompletion mock carrying a `plan` tool call."""
    import json

    args = json.dumps(
        {"material_sufficient": material_sufficient, "search_query": search_query}
    )
    func = types.SimpleNamespace(name="plan", arguments=args)
    tc = types.SimpleNamespace(id="call_plan", type="function", function=func)
    msg = types.SimpleNamespace(content=None, tool_calls=[tc])
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _no_tool_completion() -> Any:
    """ChatCompletion mock with no tool call (malformed plan)."""
    msg = types.SimpleNamespace(content="oops", tool_calls=None)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


# ---------------------------------------------------------------------------
# material_sufficient=true → no web search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sufficient_material_skips_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_kwargs: list[dict] = []

    async def fake_complete(messages: list[dict], **kwargs: object) -> Any:
        plan_kwargs.append(dict(kwargs))
        return _plan_completion(True, "irrelevant query")

    def boom_search(*a: object, **k: object) -> Any:  # must NOT be called
        raise AssertionError("search should not run when material is sufficient")

    streamed: list[list[dict]] = []

    async def fake_stream(messages: list[dict], **kwargs: object) -> AsyncIterator[str]:
        streamed.append(messages)
        yield "Hello, world."

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    monkeypatch.setattr(qa_mod._search, "ddg_search_with_content", boom_search)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)

    job = _FakeJob(
        title="Some video",
        raw_text="[00:30] First moment.\n\n[01:00] Second moment.",
        summary_md="## Summary",
    )
    out = [
        item
        async for item in qa_mod.stream_answer(
            job=job, question="What happens at 00:30?", output_language="English", from_audio=True
        )
    ]

    assert "".join(s for s in out if isinstance(s, str)) == "Hello, world."
    # No searching stage event.
    assert not [i for i in out if isinstance(i, dict)]
    # Plan call was forced to the plan tool.
    assert plan_kwargs[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "plan"},
    }
    # Synthesis prompt threaded language / question / title / context, and
    # recorded that no search ran.
    prompt = streamed[0][0]["content"]
    assert "English" in prompt
    assert "What happens at 00:30?" in prompt
    assert "Some video" in prompt
    assert "First moment." in prompt
    assert "no web search was run" in prompt


@pytest.mark.asyncio
async def test_falls_back_to_summary_when_raw_too_long(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import config as config_mod

    cfg = config_mod.get_config()
    monkeypatch.setattr(cfg.llm, "context_length", 4001)

    captured: list[list[dict]] = []

    async def fake_complete(messages: list[dict], **kwargs: object) -> Any:
        captured.append(messages)
        return _plan_completion(True, "q")

    async def fake_stream(messages: list[dict], **kwargs: object) -> AsyncIterator[str]:
        yield ""

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)

    big_raw = "Очень длинный сырой текст. " * 1000
    job = _FakeJob(
        title="Long video", raw_text=big_raw, summary_md="## Краткая выжимка\n- Пункт"
    )
    async for _ in qa_mod.stream_answer(
        job=job, question="?", output_language="English", from_audio=True
    ):
        pass

    # Plan call uses summary, not the oversized raw_text.
    plan_prompt = captured[0][0]["content"]
    assert "Краткая выжимка" in plan_prompt
    assert "Очень длинный сырой текст." not in plan_prompt


@pytest.mark.asyncio
async def test_handles_missing_title(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_complete(messages: list[dict], **kwargs: object) -> Any:
        return _plan_completion(True, "q")

    streamed: list[list[dict]] = []

    async def fake_stream(messages: list[dict], **kwargs: object) -> AsyncIterator[str]:
        streamed.append(messages)
        yield "ok"

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)

    job = _FakeJob(title=None, raw_text="some text", summary_md=None)
    out = [
        item
        async for item in qa_mod.stream_answer(
            job=job, question="q", output_language="English", from_audio=True
        )
    ]
    assert [s for s in out if isinstance(s, str)] == ["ok"]
    assert "{title}" not in streamed[0][0]["content"]


# ---------------------------------------------------------------------------
# material insufficient → web search runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insufficient_material_triggers_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searched: list[str] = []

    async def fake_ddg(query: str, **kwargs: object) -> list[dict[str, Any]]:
        searched.append(query)
        return [
            {
                "title": "T",
                "href": "https://ex.com",
                "body": "snippet",
                "content": f"full cleaned page about {query}",
            }
        ]

    monkeypatch.setattr(qa_mod._search, "ddg_search_with_content", fake_ddg)

    async def fake_complete(messages: list[dict], **kwargs: object) -> Any:
        return _plan_completion(False, "MSI 4060 Ti VRAM")

    streamed: list[list[dict]] = []

    async def fake_stream(messages: list[dict], **kwargs: object) -> AsyncIterator[str]:
        streamed.append(messages)
        yield "final answer"

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)

    job = _FakeJob(title="T", raw_text="text", summary_md=None)
    items = [
        item
        async for item in qa_mod.stream_answer(
            job=job, question="how much VRAM", output_language="Russian", from_audio=True
        )
    ]

    stages = [i for i in items if isinstance(i, dict)]
    assert len(stages) == 1
    assert stages[0]["stage"] == "searching"
    assert stages[0]["detail"] == "MSI 4060 Ti VRAM"
    assert searched == ["MSI 4060 Ti VRAM"]
    assert "".join(s for s in items if isinstance(s, str)) == "final answer"
    # The synthesis prompt embeds the cleaned page content.
    assert "full cleaned page about MSI 4060 Ti VRAM" in streamed[0][0]["content"]


@pytest.mark.asyncio
async def test_plan_tool_error_defaults_to_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend can't do tools → plan raises → we still search (no dodging)."""
    searched: list[str] = []

    async def boom(messages: list[dict], **kwargs: object) -> Any:
        raise RuntimeError("backend does not support tools")

    async def fake_ddg(query: str, **kwargs: object) -> list[dict[str, Any]]:
        searched.append(query)
        return [{"title": "T", "href": "h", "body": "b", "content": "c"}]

    async def fake_stream(messages: list[dict], **kwargs: object) -> AsyncIterator[str]:
        yield "answered anyway"

    monkeypatch.setattr(llm_client, "complete_with_messages", boom)
    monkeypatch.setattr(qa_mod._search, "ddg_search_with_content", fake_ddg)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)

    job = _FakeJob(title="T", raw_text="text", summary_md=None)
    items = [
        item
        async for item in qa_mod.stream_answer(
            job=job, question="the question", output_language="English", from_audio=True
        )
    ]
    # Falls back to the raw question as the query and searches.
    assert searched == ["the question"]
    assert "".join(s for s in items if isinstance(s, str)) == "answered anyway"


def test_parse_plan_strict_boolean() -> None:
    """Only an explicit True counts as sufficient — truthy non-bools mean search."""
    assert qa_mod._parse_plan(_plan_completion(True, "q")) == (True, "q")
    # Truthy-but-not-True (model emitted 1 / "true") → NOT sufficient.
    assert qa_mod._parse_plan(_plan_completion(1, "q")) == (False, "q")
    assert qa_mod._parse_plan(_plan_completion("true", "q")) == (False, "q")
    assert qa_mod._parse_plan(_plan_completion(False, "q")) == (False, "q")


def test_parse_plan_missing_query() -> None:
    """A null/empty query parses to '' so the caller falls back to the question."""
    assert qa_mod._parse_plan(_plan_completion(False, "")) == (False, "")
    assert qa_mod._parse_plan(_no_tool_completion()) == (False, "")


@pytest.mark.asyncio
async def test_search_failure_still_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the web search itself raises, we answer without web results rather
    than failing the whole turn."""

    async def fake_complete(messages: list[dict], **kwargs: object) -> Any:
        return _plan_completion(False, "some query")

    async def boom_ddg(query: str, **kwargs: object) -> list[dict[str, Any]]:
        raise RuntimeError("DDG down")

    streamed: list[list[dict]] = []

    async def fake_stream(messages: list[dict], **kwargs: object) -> AsyncIterator[str]:
        streamed.append(messages)
        yield "answer without web"

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    monkeypatch.setattr(qa_mod._search, "ddg_search_with_content", boom_ddg)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)

    job = _FakeJob(title="T", raw_text="text", summary_md=None)
    items = [
        item
        async for item in qa_mod.stream_answer(
            job=job, question="q", output_language="English", from_audio=True
        )
    ]
    # Searching stage still emitted; answer streamed despite the failure.
    assert any(isinstance(i, dict) for i in items)
    assert "".join(s for s in items if isinstance(s, str)) == "answer without web"
    # The synthesis prompt records that no usable web results were available.
    assert "no web search was run" in streamed[0][0]["content"]


@pytest.mark.asyncio
async def test_malformed_plan_defaults_to_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan returns no tool call → treated as insufficient → search runs."""
    searched: list[str] = []

    async def fake_complete(messages: list[dict], **kwargs: object) -> Any:
        return _no_tool_completion()

    async def fake_ddg(query: str, **kwargs: object) -> list[dict[str, Any]]:
        searched.append(query)
        return [{"title": "T", "href": "h", "body": "b", "content": "c"}]

    async def fake_stream(messages: list[dict], **kwargs: object) -> AsyncIterator[str]:
        yield "ok"

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    monkeypatch.setattr(qa_mod._search, "ddg_search_with_content", fake_ddg)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)

    job = _FakeJob(title="T", raw_text="text", summary_md=None)
    out = [
        item
        async for item in qa_mod.stream_answer(
            job=job, question="q", output_language="English", from_audio=True
        )
    ]
    # No query parsed → falls back to the raw question.
    assert searched == ["q"]
    assert "".join(s for s in out if isinstance(s, str)) == "ok"


# ---------------------------------------------------------------------------
# timestamp_rules threading — document sources must never see a timecode
# instruction/example; transcript sources keep the inline-timecode rule.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_from_audio_true_uses_transcript_timestamp_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_complete(messages: list[dict], **kwargs: object) -> Any:
        return _plan_completion(True, "q")

    streamed: list[list[dict]] = []

    async def fake_stream(messages: list[dict], **kwargs: object) -> AsyncIterator[str]:
        streamed.append(messages)
        yield "ok"

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)

    job = _FakeJob(title="T", raw_text="[00:30] hi", summary_md=None)
    async for _ in qa_mod.stream_answer(
        job=job, question="q", output_language="English", from_audio=True
    ):
        pass

    prompt = streamed[0][0]["content"]
    assert qa_mod._TIMESTAMP_RULES_TRANSCRIPT in prompt
    assert qa_mod._TIMESTAMP_RULES_DOCUMENT not in prompt


@pytest.mark.asyncio
async def test_from_audio_false_uses_document_timestamp_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_complete(messages: list[dict], **kwargs: object) -> Any:
        return _plan_completion(True, "q")

    streamed: list[list[dict]] = []

    async def fake_stream(messages: list[dict], **kwargs: object) -> AsyncIterator[str]:
        streamed.append(messages)
        yield "ok"

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)

    job = _FakeJob(title="T", raw_text="some article text", summary_md=None)
    async for _ in qa_mod.stream_answer(
        job=job, question="q", output_language="English", from_audio=False
    ):
        pass

    prompt = streamed[0][0]["content"]
    assert qa_mod._TIMESTAMP_RULES_DOCUMENT in prompt
    assert qa_mod._TIMESTAMP_RULES_TRANSCRIPT not in prompt
    # No timecode example must leak into a document prompt at all.
    assert "[MM:SS]" in prompt or "[HH:MM:SS]" in prompt  # rule text itself
    assert "01:30" not in prompt and "02:15" not in prompt
