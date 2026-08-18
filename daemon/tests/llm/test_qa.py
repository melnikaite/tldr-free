"""Tests for llm.qa.stream_answer — plan → look → search → synthesis flow.

Mocks ``llm_client.complete_with_messages`` (the forced ``plan`` tool call,
and the vision call the LOOK step makes over frames) and
``llm_client.stream_with_messages`` (the grounded-answer stream) to avoid
any real LLM calls. Verifies context/title/language threading, the layered
material→frames→knowledge→web flow, and that the model can NOT dodge a
search: anything other than a clean ``material_sufficient=true``
(uncertainty, parse failure, a backend tool error) falls through to a web
search.

The LOOK-step tests below (``TestLookStep``-adjacent functions near the
bottom of this file) mock ``workers.deixis.find_deixis_candidates`` and
``workers.frames.fetch_frames`` — never touching the real regex/network
implementations — and assert on flow: candidates offered to the plan tool,
only the model-chosen indices fetched, the category→resolution mapping,
EXTERNAL never fetching (even if named), every degradation path, and that
a job without a timestamped transcript takes the byte-identical old path.
"""

from __future__ import annotations

import json
import types
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.llm import client as llm_client
from src.llm import qa as qa_mod
from src.workers import frames as frames_mod
from src.workers.deixis import DeixisCandidate, DeixisCategory
from src.workers.errors import FrameExtractionError


@dataclass
class _FakeJob:
    title: str | None
    raw_text: str | None
    summary_md: str | None
    # LOOK-step fields — defaulted so every pre-existing call site (none of
    # which passes these) is unaffected and gets the old page/PDF-shaped
    # behaviour (no deixis candidates -> LOOK step never runs).
    transcript_source: str | None = None
    raw_segments_json: str | None = None
    transcript_language: str | None = None
    url: str | None = None
    id: str | None = None


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
async def test_web_search_disabled_skips_search_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``qa.web_search=False`` must stop SEARCH from running at all — no DDG
    call, no ``searching`` stage — even though PLAN says the material is
    insufficient (the exact condition that triggers a search when the
    setting is on, see ``test_insufficient_material_triggers_search``). The
    answer must still stream, and the synthesis prompt must carry the
    anti-fabrication rule instead of silently going quiet."""
    from src import config as config_mod

    cfg = config_mod.get_config()
    monkeypatch.setattr(cfg.qa, "web_search", False)

    called: list[str] = []

    async def boom_ddg(query: str, **kwargs: object) -> list[dict[str, Any]]:
        called.append(query)
        raise AssertionError("ddg_search_with_content must not be called when web_search=False")

    monkeypatch.setattr(qa_mod._search, "ddg_search_with_content", boom_ddg)

    async def fake_complete(messages: list[dict], **kwargs: object) -> Any:
        return _plan_completion(False, "some query")

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
            job=job, question="how much VRAM", output_language="English", from_audio=True
        )
    ]

    assert called == []
    stages = [i for i in items if isinstance(i, dict)]
    assert stages == []  # no "searching" stage emitted
    assert "".join(s for s in items if isinstance(s, str)) == "final answer"
    prompt = streamed[0][0]["content"]
    assert "no web search was run" in prompt
    assert "Web search is disabled for this answer" in prompt


@pytest.mark.asyncio
async def test_web_search_enabled_by_default_matches_prior_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh config (no explicit override) defaults ``qa.web_search`` to
    True, so insufficient material still triggers a real search — same
    behaviour as before this setting existed."""
    from src import config as config_mod

    cfg = config_mod.get_config()
    assert cfg.qa.web_search is True

    searched: list[str] = []

    async def fake_ddg(query: str, **kwargs: object) -> list[dict[str, Any]]:
        searched.append(query)
        return [{"title": "T", "href": "h", "body": "b", "content": "c"}]

    monkeypatch.setattr(qa_mod._search, "ddg_search_with_content", fake_ddg)

    async def fake_complete(messages: list[dict], **kwargs: object) -> Any:
        return _plan_completion(False, "q")

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
            job=job, question="q", output_language="English", from_audio=True
        )
    ]

    stages = [i for i in items if isinstance(i, dict)]
    assert len(stages) == 1
    assert stages[0]["stage"] == "searching"
    assert searched == ["q"]
    prompt = streamed[0][0]["content"]
    assert "Web search is disabled for this answer" not in prompt


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
    assert qa_mod._parse_plan(_plan_completion(True, "q")) == (True, "q", [])
    # Truthy-but-not-True (model emitted 1 / "true") → NOT sufficient.
    assert qa_mod._parse_plan(_plan_completion(1, "q")) == (False, "q", [])
    assert qa_mod._parse_plan(_plan_completion("true", "q")) == (False, "q", [])
    assert qa_mod._parse_plan(_plan_completion(False, "q")) == (False, "q", [])


def test_parse_plan_missing_query() -> None:
    """A null/empty query parses to '' so the caller falls back to the question."""
    assert qa_mod._parse_plan(_plan_completion(False, "")) == (False, "", [])
    assert qa_mod._parse_plan(_no_tool_completion()) == (False, "", [])


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


# ---------------------------------------------------------------------------
# LOOK step — plan-chosen video moments feed the vision model
# ---------------------------------------------------------------------------


def _plan_completion_with_indices(
    material_sufficient: Any, search_query: str, look_at_indices: list[int]
) -> Any:
    """ChatCompletion mock carrying a `plan` tool call that also names
    `look_at_indices` — the shape the plan tool returns once candidates
    were offered."""
    args = json.dumps(
        {
            "material_sufficient": material_sufficient,
            "search_query": search_query,
            "look_at_indices": look_at_indices,
        }
    )
    func = types.SimpleNamespace(name="plan", arguments=args)
    tc = types.SimpleNamespace(id="call_plan", type="function", function=func)
    msg = types.SimpleNamespace(content=None, tool_calls=[tc])
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _is_plan_call(kwargs: dict[str, Any]) -> bool:
    """Both the PLAN call and the LOOK step's vision call now pass `tools=`
    (forced tool-calling), so `kwargs.get("tools")` alone no longer tells
    them apart — check the forced tool's name instead."""
    tools = kwargs.get("tools") or []
    return bool(tools) and tools[0]["function"]["name"] == "plan"


def _vision_completion(text: str) -> Any:
    """ChatCompletion mock with no tool call (malformed vision response) —
    exercises `_parse_vision_result`'s degrade-to-irrelevant path."""
    msg = types.SimpleNamespace(content=text, tool_calls=None)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _vision_tool_completion(
    finding: str, relevant: Any, best_frame_index: Any
) -> Any:
    """ChatCompletion mock carrying a `report_frame_findings` tool call —
    the shape the LOOK step's vision call returns once frames were sent
    (see `qa_mod._VISION_TOOL` / `qa_mod._parse_vision_result`)."""
    args = json.dumps(
        {
            "finding": finding,
            "relevant": relevant,
            "best_frame_index": best_frame_index,
        }
    )
    func = types.SimpleNamespace(name="report_frame_findings", arguments=args)
    tc = types.SimpleNamespace(id="call_vision", type="function", function=func)
    msg = types.SimpleNamespace(content=None, tool_calls=[tc])
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


_CAND_OBJECT = DeixisCandidate(
    timestamp=12.0, phrase="this cream", category=DeixisCategory.OBJECT, confidence=0.65
)
_CAND_ACTION = DeixisCandidate(
    timestamp=40.0, phrase="watch this", category=DeixisCategory.ACTION, confidence=0.85
)
_CAND_EXTERNAL = DeixisCandidate(
    timestamp=90.0,
    phrase="link in the description",
    category=DeixisCategory.EXTERNAL,
    confidence=0.9,
)


def _audio_job(**overrides: Any) -> _FakeJob:
    """A job that qualifies for the LOOK step: audio transcript source +
    non-empty raw_segments_json. The actual segment content doesn't matter
    since `find_deixis_candidates` is monkeypatched in these tests — only
    its *presence* (so `_deixis_candidates_for_job` doesn't bail early)."""
    defaults: dict[str, Any] = dict(
        title="A demo video",
        raw_text="[00:12] Look at this cream.",
        summary_md=None,
        transcript_source="whisper",
        raw_segments_json=json.dumps([{"start": 0.0, "text": "hi"}]),
        transcript_language="en",
        url="https://example.com/video",
        id="job123",
    )
    defaults.update(overrides)
    return _FakeJob(**defaults)


@pytest.mark.asyncio
async def test_candidates_offered_to_plan_and_only_chosen_fetched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qa_mod._deixis, "find_deixis_candidates", lambda *a, **k: [_CAND_OBJECT, _CAND_ACTION]
    )
    monkeypatch.setattr(qa_mod, "_frame_to_data_uri", lambda p: f"data:fake:{p}")

    fetch_calls: list[dict[str, Any]] = []

    async def fake_fetch_frames(**kwargs: Any) -> list[Path]:
        fetch_calls.append(kwargs)
        return [Path("/tmp/frame_01.jpg")]

    monkeypatch.setattr(qa_mod._frames, "fetch_frames", fake_fetch_frames)

    plan_tools_seen: list[list[dict[str, Any]]] = []
    plan_prompts_seen: list[str] = []

    async def fake_complete(messages: list[dict], **kwargs: Any) -> Any:
        if _is_plan_call(kwargs):
            plan_tools_seen.append(kwargs["tools"])
            plan_prompts_seen.append(messages[0]["content"])
            # Model only picks index 1 (the OBJECT candidate) though two
            # were offered — index 2 (ACTION) must NOT be fetched.
            return _plan_completion_with_indices(False, "q", [1])
        return _vision_tool_completion(
            "A red tub labeled 'ACME Cream 200ml'.", True, 1
        )

    streamed: list[list[dict]] = []

    async def fake_stream(messages: list[dict], **kwargs: Any) -> AsyncIterator[str]:
        streamed.append(messages)
        yield "final answer"

    async def fake_ddg(query: str, **kwargs: object) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)
    monkeypatch.setattr(qa_mod._search, "ddg_search_with_content", fake_ddg)

    job = _audio_job()
    items = [
        item
        async for item in qa_mod.stream_answer(
            job=job, question="what cream is that", output_language="English", from_audio=True
        )
    ]

    # The plan tool offered `look_at_indices` and the plan prompt listed
    # both candidates by index/timecode/phrase.
    assert "look_at_indices" in plan_tools_seen[0][0]["function"]["parameters"]["properties"]
    plan_prompt = plan_prompts_seen[0]
    assert "this cream" in plan_prompt
    assert "watch this" in plan_prompt
    assert "00:12" in plan_prompt and "00:40" in plan_prompt

    # Only ONE fetch happened, for the OBJECT candidate (index 1), at
    # readable resolution (720p) — never for the un-chosen ACTION candidate.
    assert len(fetch_calls) == 1
    assert fetch_calls[0]["timestamp_seconds"] == _CAND_OBJECT.timestamp
    assert fetch_calls[0]["max_height_px"] == frames_mod.SECTION_MAX_HEIGHT_READABLE_PX
    assert fetch_calls[0]["job_id"] == "job123"
    assert fetch_calls[0]["url"] == "https://example.com/video"

    # A "looking" stage event fired with the timecode + phrase in the detail.
    looking_stages = [
        i for i in items if isinstance(i, dict) and i.get("stage") == "looking"
    ]
    assert len(looking_stages) == 1
    assert looking_stages[0]["detail"] == "00:12 — this cream"

    # The synthesis prompt carries the visual finding, attributed to its
    # timecode.
    synthesis_prompt = streamed[0][0]["content"]
    assert "[00:12]" in synthesis_prompt
    assert "ACME Cream 200ml" in synthesis_prompt


@pytest.mark.asyncio
async def test_action_category_uses_lower_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qa_mod._deixis, "find_deixis_candidates", lambda *a, **k: [_CAND_ACTION]
    )
    monkeypatch.setattr(qa_mod, "_frame_to_data_uri", lambda p: f"data:fake:{p}")

    fetch_calls: list[dict[str, Any]] = []

    async def fake_fetch_frames(**kwargs: Any) -> list[Path]:
        fetch_calls.append(kwargs)
        return [Path("/tmp/frame_01.jpg")]

    monkeypatch.setattr(qa_mod._frames, "fetch_frames", fake_fetch_frames)

    async def fake_complete(messages: list[dict], **kwargs: Any) -> Any:
        if _is_plan_call(kwargs):
            return _plan_completion_with_indices(True, "q", [1])
        return _vision_tool_completion("A hand folds the fabric in half.", True, 1)

    async def fake_stream(messages: list[dict], **kwargs: Any) -> AsyncIterator[str]:
        yield "ok"

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)

    job = _audio_job()
    async for _ in qa_mod.stream_answer(
        job=job, question="how do you fold it", output_language="English", from_audio=True
    ):
        pass

    assert len(fetch_calls) == 1
    assert fetch_calls[0]["max_height_px"] == frames_mod.SECTION_MAX_HEIGHT_PX


@pytest.mark.asyncio
async def test_external_candidate_never_fetches_even_if_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plan tool/prompt both tell the model never to pick an EXTERNAL
    index, but this test simulates a model that misbehaves anyway — the
    daemon-side guard in stream_answer's LOOK loop must still refuse."""
    monkeypatch.setattr(
        qa_mod._deixis, "find_deixis_candidates", lambda *a, **k: [_CAND_EXTERNAL]
    )

    async def boom_fetch(**kwargs: Any) -> list[Path]:
        raise AssertionError("fetch_frames must never be called for an EXTERNAL candidate")

    monkeypatch.setattr(qa_mod._frames, "fetch_frames", boom_fetch)

    async def fake_complete(messages: list[dict], **kwargs: Any) -> Any:
        if kwargs.get("tools"):
            # Model names index 1 despite it being EXTERNAL.
            return _plan_completion_with_indices(False, "q", [1])
        raise AssertionError("no vision call should happen either")

    async def fake_stream(messages: list[dict], **kwargs: Any) -> AsyncIterator[str]:
        yield "answer"

    async def fake_ddg(query: str, **kwargs: object) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)
    monkeypatch.setattr(qa_mod._search, "ddg_search_with_content", fake_ddg)

    job = _audio_job()
    items = [
        item
        async for item in qa_mod.stream_answer(
            job=job, question="where do I buy it", output_language="English", from_audio=True
        )
    ]
    assert "".join(s for s in items if isinstance(s, str)) == "answer"
    assert not [i for i in items if isinstance(i, dict) and i.get("stage") == "looking"]


@pytest.mark.asyncio
async def test_frame_extraction_error_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qa_mod._deixis, "find_deixis_candidates", lambda *a, **k: [_CAND_OBJECT]
    )

    async def boom_fetch(**kwargs: Any) -> list[Path]:
        raise FrameExtractionError("section download failed")

    monkeypatch.setattr(qa_mod._frames, "fetch_frames", boom_fetch)

    async def fake_complete(messages: list[dict], **kwargs: Any) -> Any:
        return _plan_completion_with_indices(True, "q", [1])

    streamed: list[list[dict]] = []

    async def fake_stream(messages: list[dict], **kwargs: Any) -> AsyncIterator[str]:
        streamed.append(messages)
        yield "still answers"

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)

    job = _audio_job()
    items = [
        item
        async for item in qa_mod.stream_answer(
            job=job, question="q", output_language="English", from_audio=True
        )
    ]
    assert "".join(s for s in items if isinstance(s, str)) == "still answers"
    assert "(no frames were examined)" in streamed[0][0]["content"]


@pytest.mark.asyncio
async def test_empty_frame_list_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fetch_frames returning [] (per-job budget spent) is not an error —
    just no contribution from this moment."""
    monkeypatch.setattr(
        qa_mod._deixis, "find_deixis_candidates", lambda *a, **k: [_CAND_OBJECT]
    )

    async def empty_fetch(**kwargs: Any) -> list[Path]:
        return []

    monkeypatch.setattr(qa_mod._frames, "fetch_frames", empty_fetch)

    async def fake_complete(messages: list[dict], **kwargs: Any) -> Any:
        return _plan_completion_with_indices(True, "q", [1])

    streamed: list[list[dict]] = []

    async def fake_stream(messages: list[dict], **kwargs: Any) -> AsyncIterator[str]:
        streamed.append(messages)
        yield "still answers"

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)

    job = _audio_job()
    async for _ in qa_mod.stream_answer(
        job=job, question="q", output_language="English", from_audio=True
    ):
        pass
    assert "(no frames were examined)" in streamed[0][0]["content"]


@pytest.mark.asyncio
async def test_vision_call_error_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qa_mod._deixis, "find_deixis_candidates", lambda *a, **k: [_CAND_OBJECT]
    )
    monkeypatch.setattr(qa_mod, "_frame_to_data_uri", lambda p: f"data:fake:{p}")

    async def fake_fetch_frames(**kwargs: Any) -> list[Path]:
        return [Path("/tmp/frame_01.jpg")]

    monkeypatch.setattr(qa_mod._frames, "fetch_frames", fake_fetch_frames)

    async def fake_complete(messages: list[dict], **kwargs: Any) -> Any:
        if _is_plan_call(kwargs):
            return _plan_completion_with_indices(True, "q", [1])
        raise RuntimeError("vision backend errored")

    streamed: list[list[dict]] = []

    async def fake_stream(messages: list[dict], **kwargs: Any) -> AsyncIterator[str]:
        streamed.append(messages)
        yield "still answers"

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)

    job = _audio_job()
    items = [
        item
        async for item in qa_mod.stream_answer(
            job=job, question="q", output_language="English", from_audio=True
        )
    ]
    assert "".join(s for s in items if isinstance(s, str)) == "still answers"
    assert "(no frames were examined)" in streamed[0][0]["content"]


@pytest.mark.asyncio
async def test_no_candidates_keeps_plan_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job with no deixis candidates (the real path for a PDF/page job,
    or an audio job whose transcript has no deixis moments) must see the
    exact pre-LOOK-step plan tool object and prompt -- no VIDEO MOMENTS
    section, no look_at_indices property."""

    def boom_deixis(*a: object, **k: object) -> Any:
        raise AssertionError("find_deixis_candidates must not run for this job")

    monkeypatch.setattr(qa_mod._deixis, "find_deixis_candidates", boom_deixis)

    def boom_fetch(**kwargs: Any) -> Any:
        raise AssertionError("fetch_frames must not run with no candidates")

    monkeypatch.setattr(qa_mod._frames, "fetch_frames", boom_fetch)

    seen_tools: list[list[dict[str, Any]]] = []
    seen_prompts: list[str] = []

    async def fake_complete(messages: list[dict], **kwargs: Any) -> Any:
        seen_tools.append(kwargs["tools"])
        seen_prompts.append(messages[0]["content"])
        return _plan_completion_with_indices(True, "q", [])

    async def fake_stream(messages: list[dict], **kwargs: Any) -> AsyncIterator[str]:
        yield "ok"

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)

    # A PDF-shaped job: no transcript_source, no raw_segments_json at all.
    job = _FakeJob(title="A PDF", raw_text="document text", summary_md=None)
    items = [
        item
        async for item in qa_mod.stream_answer(
            job=job, question="q", output_language="English", from_audio=False
        )
    ]

    # Tool offered is the exact module-level constant -- identity, not just
    # equality -- proving nothing was rebuilt/mutated for this job.
    assert seen_tools[0][0] is qa_mod._PLAN_TOOL
    assert "look_at_indices" not in seen_tools[0][0]["function"]["parameters"]["properties"]
    assert "VIDEO MOMENTS" not in seen_prompts[0]
    assert not [i for i in items if isinstance(i, dict) and i.get("stage") == "looking"]


@pytest.mark.asyncio
async def test_audio_job_without_segments_also_takes_unchanged_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """transcript_source alone isn't enough -- raw_segments_json must also be
    present, otherwise there is nothing for find_deixis_candidates to look
    at and the job must take the same unchanged path as a page/PDF job."""

    def boom_deixis(*a: object, **k: object) -> Any:
        raise AssertionError("find_deixis_candidates must not run without segments")

    monkeypatch.setattr(qa_mod._deixis, "find_deixis_candidates", boom_deixis)

    async def fake_complete(messages: list[dict], **kwargs: Any) -> Any:
        return _plan_completion_with_indices(True, "q", [])

    async def fake_stream(messages: list[dict], **kwargs: Any) -> AsyncIterator[str]:
        yield "ok"

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)

    job = _FakeJob(
        title="T",
        raw_text="[00:05] hi",
        summary_md=None,
        transcript_source="whisper",
        raw_segments_json=None,
    )
    async for _ in qa_mod.stream_answer(
        job=job, question="q", output_language="English", from_audio=True
    ):
        pass


# ---------------------------------------------------------------------------
# _parse_vision_result — the LOOK step's structured vision output (see
# VisionResult / _VISION_TOOL). Same defensive shape as _parse_plan:
# anything malformed degrades to relevant=False/no frame rather than raising.
# ---------------------------------------------------------------------------


def test_parse_vision_result_valid_relevant() -> None:
    resp = _vision_tool_completion("A red tub on the desk.", True, 2)
    result = qa_mod._parse_vision_result(resp, num_frames=3)
    assert result == qa_mod.VisionResult("A red tub on the desk.", True, 2)


def test_parse_vision_result_relevant_false_keeps_finding() -> None:
    """relevant=false still carries the finding text (it goes into VISUAL
    FINDINGS either way) but the frame index doesn't matter downstream —
    stream_answer only reads best_frame_index when relevant is true."""
    resp = _vision_tool_completion(
        "The frames show no information relevant to the question.", False, 1
    )
    result = qa_mod._parse_vision_result(resp, num_frames=3)
    assert result.relevant is False
    assert result.finding == "The frames show no information relevant to the question."


def test_parse_vision_result_malformed_tool_call_degrades() -> None:
    """No tool call at all (backend ignored tool_choice, or errored) ->
    degrade to relevant=False, no frame, empty finding — never raise."""
    resp = _vision_completion("some free-text the model wrote instead")
    result = qa_mod._parse_vision_result(resp, num_frames=3)
    assert result == qa_mod.VisionResult("", False, None)


def test_parse_vision_result_bad_json_degrades() -> None:
    func = types.SimpleNamespace(name="report_frame_findings", arguments="{not json")
    tc = types.SimpleNamespace(id="call_vision", type="function", function=func)
    msg = types.SimpleNamespace(content=None, tool_calls=[tc])
    resp = types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])
    result = qa_mod._parse_vision_result(resp, num_frames=3)
    assert result == qa_mod.VisionResult("", False, None)


def test_parse_vision_result_out_of_range_frame_index_drops_index_only() -> None:
    """A frame index outside [1, num_frames] is dropped (best_frame_index=None)
    but the rest of the parse (finding, relevant) is still honoured —
    mirrors _parse_look_at_indices's "drop rather than fail everything"."""
    resp = _vision_tool_completion("A demonstrated action.", True, 99)
    result = qa_mod._parse_vision_result(resp, num_frames=3)
    assert result.best_frame_index is None
    assert result.relevant is True
    assert result.finding == "A demonstrated action."


def test_parse_vision_result_non_numeric_frame_index_drops_index_only() -> None:
    resp = _vision_tool_completion("Something visible.", True, "two")
    result = qa_mod._parse_vision_result(resp, num_frames=3)
    assert result.best_frame_index is None


def test_parse_vision_result_truthy_non_bool_relevant_is_not_relevant() -> None:
    """Same strict-boolean bar _parse_plan holds material_sufficient to."""
    resp = _vision_tool_completion("Something visible.", 1, 1)
    result = qa_mod._parse_vision_result(resp, num_frames=1)
    assert result.relevant is False


# ---------------------------------------------------------------------------
# stream_answer LOOK loop — "frames" event only for relevant moments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_relevant_moment_emits_frames_event(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        qa_mod._deixis, "find_deixis_candidates", lambda *a, **k: [_CAND_OBJECT]
    )
    monkeypatch.setattr(qa_mod, "_frame_to_data_uri", lambda p: f"data:fake:{p}")

    async def fake_fetch_frames(**kwargs: Any) -> list[Path]:
        return [Path("/tmp/t12/frame_01.jpg"), Path("/tmp/t12/frame_02.jpg")]

    monkeypatch.setattr(qa_mod._frames, "fetch_frames", fake_fetch_frames)

    async def fake_complete(messages: list[dict], **kwargs: Any) -> Any:
        if _is_plan_call(kwargs):
            return _plan_completion_with_indices(True, "q", [1])
        # Model picks frame 2 as the most informative.
        return _vision_tool_completion("A red tub labeled 'ACME Cream'.", True, 2)

    async def fake_stream(messages: list[dict], **kwargs: Any) -> AsyncIterator[str]:
        yield "answer"

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)

    job = _audio_job()
    items = [
        item
        async for item in qa_mod.stream_answer(
            job=job, question="what cream is that", output_language="English", from_audio=True
        )
    ]

    frames_events = [i for i in items if isinstance(i, dict) and i.get("type") == "frames"]
    assert len(frames_events) == 1
    refs = frames_events[0]["items"]
    assert len(refs) == 1
    ref = refs[0]
    assert ref["seconds"] == _CAND_OBJECT.timestamp
    assert ref["timecode"] == "00:12"
    assert ref["phrase"] == _CAND_OBJECT.phrase
    # Points at frame index 2 (frame_02.jpg), the one the model picked, under
    # the job's own frame directory, in the t<second>/frame_NN.jpg shape
    # frames.resolve_frame_path expects.
    assert ref["frame_url"] == f"/jobs/{job.id}/frames/t12/frame_02.jpg"


@pytest.mark.asyncio
async def test_irrelevant_moment_skips_frames_event_but_keeps_finding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The vision model looked and found nothing relevant: the finding text
    still reaches VISUAL FINDINGS (useful context: "we checked"), but no
    'frames' event is emitted — no thumbnail for nothing shown."""
    monkeypatch.setattr(
        qa_mod._deixis, "find_deixis_candidates", lambda *a, **k: [_CAND_OBJECT]
    )
    monkeypatch.setattr(qa_mod, "_frame_to_data_uri", lambda p: f"data:fake:{p}")

    async def fake_fetch_frames(**kwargs: Any) -> list[Path]:
        return [Path("/tmp/t12/frame_01.jpg")]

    monkeypatch.setattr(qa_mod._frames, "fetch_frames", fake_fetch_frames)

    async def fake_complete(messages: list[dict], **kwargs: Any) -> Any:
        if _is_plan_call(kwargs):
            return _plan_completion_with_indices(True, "q", [1])
        return _vision_tool_completion(
            "The frames show no information relevant to the question.", False, 1
        )

    streamed: list[list[dict]] = []

    async def fake_stream(messages: list[dict], **kwargs: Any) -> AsyncIterator[str]:
        streamed.append(messages)
        yield "answer"

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    monkeypatch.setattr(llm_client, "stream_with_messages", fake_stream)

    job = _audio_job()
    items = [
        item
        async for item in qa_mod.stream_answer(
            job=job, question="what cream is that", output_language="English", from_audio=True
        )
    ]

    assert not [i for i in items if isinstance(i, dict) and i.get("type") == "frames"]
    synthesis_prompt = streamed[0][0]["content"]
    assert "no information relevant to the question" in synthesis_prompt
