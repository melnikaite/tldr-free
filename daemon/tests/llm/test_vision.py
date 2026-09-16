"""Tests for llm.vision's stage-2 seam: `fetch_moment_frames` (the download
half of `inspect_moment`, pulled out for the summary-time conveyor) and
`analyze_summary_frames` (the summary-anchored vision call, sibling to the
QA-anchored `_ask_vision_about_frames`).

Mocks `workers.frames.fetch_frames` and `llm.client.complete_with_messages`
— never touching real yt-dlp/ffmpeg or a real LLM. `inspect_moment`'s own
QA behaviour is covered by tests/llm/test_qa.py; these tests are scoped to
the two new pieces and the fact that they raise (rather than degrade) on
failure, per the module's docstring.
"""

from __future__ import annotations

import json
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.llm import client as llm_client
from src.llm import vision as vision_mod
from src.workers import frames as frames_mod
from src.workers.deixis import DeixisCandidate, DeixisCategory
from src.workers.errors import FrameExtractionError


@dataclass
class _FakeJob:
    id: str | None
    url: str | None
    kind: str = "youtube"
    media_url: str | None = None


def _vision_tool_completion(finding: str, relevant: Any, best_frame_index: Any) -> Any:
    args = json.dumps(
        {"finding": finding, "relevant": relevant, "best_frame_index": best_frame_index}
    )
    func = types.SimpleNamespace(name="report_frame_findings", arguments=args)
    tc = types.SimpleNamespace(id="call_vision", type="function", function=func)
    msg = types.SimpleNamespace(content=None, tool_calls=[tc])
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


# ---------------------------------------------------------------------------
# fetch_moment_frames
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_moment_frames_picks_height_by_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    async def fake_fetch_frames(**kwargs: Any) -> list[Path]:
        captured.append(kwargs)
        return [Path("/tmp/frame_01.jpg")]

    monkeypatch.setattr(frames_mod, "fetch_frames", fake_fetch_frames)
    job = _FakeJob(id="job1", url="https://example.com/video")

    obj_cand = DeixisCandidate(
        timestamp=12.0, phrase="this cream", category=DeixisCategory.OBJECT, confidence=0.7
    )
    await vision_mod.fetch_moment_frames(job=job, candidate=obj_cand)
    assert captured[-1]["max_height_px"] == frames_mod.SECTION_MAX_HEIGHT_READABLE_PX
    assert captured[-1]["cookies"] is None
    assert captured[-1]["job_id"] == "job1"
    assert captured[-1]["url"] == "https://example.com/video"

    action_cand = DeixisCandidate(
        timestamp=40.0, phrase="watch this", category=DeixisCategory.ACTION, confidence=0.85
    )
    await vision_mod.fetch_moment_frames(job=job, candidate=action_cand)
    assert captured[-1]["max_height_px"] == frames_mod.SECTION_MAX_HEIGHT_PX


@pytest.mark.asyncio
async def test_fetch_moment_frames_forwards_cookies_when_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The summary-time caller (workers.pipeline) has ingestion-time cookies
    in scope and passes them through — this is the fix for frame analysis
    silently no-op'ing on any video YouTube gates behind a signed-in
    session. See llm.vision.fetch_moment_frames's docstring."""
    captured: list[dict[str, Any]] = []

    async def fake_fetch_frames(**kwargs: Any) -> list[Path]:
        captured.append(kwargs)
        return [Path("/tmp/frame_01.jpg")]

    monkeypatch.setattr(frames_mod, "fetch_frames", fake_fetch_frames)
    job = _FakeJob(id="job1", url="https://example.com/video")
    cand = DeixisCandidate(
        timestamp=12.0, phrase="this cream", category=DeixisCategory.OBJECT, confidence=0.7
    )
    cookies = [{"name": "SID", "value": "abc123"}]

    await vision_mod.fetch_moment_frames(job=job, candidate=cand, cookies=cookies)

    assert captured[-1]["cookies"] == cookies


@pytest.mark.asyncio
async def test_fetch_moment_frames_defaults_to_no_cookies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The QA path (llm.qa's LOOK step, via inspect_moment) calls this
    without a cookies argument at all — behaviour must stay byte-for-byte
    unchanged: no cookies forwarded."""
    captured: list[dict[str, Any]] = []

    async def fake_fetch_frames(**kwargs: Any) -> list[Path]:
        captured.append(kwargs)
        return [Path("/tmp/frame_01.jpg")]

    monkeypatch.setattr(frames_mod, "fetch_frames", fake_fetch_frames)
    job = _FakeJob(id="job1", url="https://example.com/video")
    cand = DeixisCandidate(
        timestamp=12.0, phrase="this cream", category=DeixisCategory.OBJECT, confidence=0.7
    )

    await vision_mod.fetch_moment_frames(job=job, candidate=cand)

    assert captured[-1]["cookies"] is None


@pytest.mark.asyncio
async def test_fetch_moment_frames_raises_when_job_missing_id_or_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(**kwargs: Any) -> list[Path]:
        raise AssertionError("fetch_frames must not be called without id/url")

    monkeypatch.setattr(frames_mod, "fetch_frames", boom)
    cand = DeixisCandidate(
        timestamp=12.0, phrase="this cream", category=DeixisCategory.OBJECT, confidence=0.7
    )
    with pytest.raises(FrameExtractionError):
        await vision_mod.fetch_moment_frames(job=_FakeJob(id=None, url=None), candidate=cand)


@pytest.mark.asyncio
async def test_fetch_moment_frames_propagates_download_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing(**kwargs: Any) -> list[Path]:
        raise FrameExtractionError("section download failed")

    monkeypatch.setattr(frames_mod, "fetch_frames", failing)
    job = _FakeJob(id="job1", url="https://example.com/video")
    cand = DeixisCandidate(
        timestamp=12.0, phrase="this cream", category=DeixisCategory.OBJECT, confidence=0.7
    )
    with pytest.raises(FrameExtractionError):
        await vision_mod.fetch_moment_frames(job=job, candidate=cand)


@pytest.mark.asyncio
async def test_fetch_moment_frames_uses_media_url_for_media_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A kind=media job's video lives at media_url, not the page url — see
    workers.frames.resolve_frame_source_url, which fetch_moment_frames goes
    through instead of reading job.url directly."""
    captured: list[dict[str, Any]] = []

    async def fake_fetch_frames(**kwargs: Any) -> list[Path]:
        captured.append(kwargs)
        return [Path("/tmp/frame_01.jpg")]

    monkeypatch.setattr(frames_mod, "fetch_frames", fake_fetch_frames)
    job = _FakeJob(
        id="job1",
        url="https://example.com/article-with-embedded-video",
        kind="media",
        media_url="https://cdn.example.com/signed/video.mp4?exp=123",
    )
    cand = DeixisCandidate(
        timestamp=12.0, phrase="this cream", category=DeixisCategory.OBJECT, confidence=0.7
    )

    await vision_mod.fetch_moment_frames(job=job, candidate=cand)

    assert captured[-1]["url"] == "https://cdn.example.com/signed/video.mp4?exp=123"


@pytest.mark.asyncio
async def test_fetch_moment_frames_uses_page_url_for_non_media_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-media job (youtube/page/pdf) has no url/media_url split — the
    job's own url is always the fetch target, even if media_url happened to
    be set (it shouldn't be, for a non-media kind, but the rule keys off
    kind, not "is media_url truthy")."""
    captured: list[dict[str, Any]] = []

    async def fake_fetch_frames(**kwargs: Any) -> list[Path]:
        captured.append(kwargs)
        return [Path("/tmp/frame_01.jpg")]

    monkeypatch.setattr(frames_mod, "fetch_frames", fake_fetch_frames)
    job = _FakeJob(
        id="job1",
        url="https://youtube.com/watch?v=abc",
        kind="youtube",
        media_url="https://cdn.example.com/should-be-ignored.mp4",
    )
    cand = DeixisCandidate(
        timestamp=12.0, phrase="this cream", category=DeixisCategory.OBJECT, confidence=0.7
    )

    await vision_mod.fetch_moment_frames(job=job, candidate=cand)

    assert captured[-1]["url"] == "https://youtube.com/watch?v=abc"


@pytest.mark.asyncio
async def test_fetch_moment_frames_media_job_without_stored_media_url_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A media job created before migration v12 (or whose discovery genuinely
    produced nothing) has media_url=None — resolve_frame_source_url must
    return None rather than falling back to job.url, and fetch_moment_frames
    must raise (not silently fetch the page) so the caller's existing
    degrade-to-nothing path (workers.pipeline._fetch_moment_frames_safe /
    llm.qa's inspect_moment) is what decides what happens next."""

    async def boom(**kwargs: Any) -> list[Path]:
        raise AssertionError("fetch_frames must not be called with the page url")

    monkeypatch.setattr(frames_mod, "fetch_frames", boom)
    job = _FakeJob(
        id="job1",
        url="https://example.com/article-with-embedded-video",
        kind="media",
        media_url=None,
    )
    cand = DeixisCandidate(
        timestamp=12.0, phrase="this cream", category=DeixisCategory.OBJECT, confidence=0.7
    )

    with pytest.raises(FrameExtractionError):
        await vision_mod.fetch_moment_frames(job=job, candidate=cand)


# ---------------------------------------------------------------------------
# analyze_summary_frames
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_summary_frames_uses_summary_prompt_and_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vision_mod, "_frame_to_data_uri", lambda p: f"data:fake:{p}")

    captured: dict[str, Any] = {}

    async def fake_complete(messages: list[dict], **kwargs: Any) -> Any:
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return _vision_tool_completion(
            "A red tub labeled 'ACME Cream 200ml'. Worth including.", True, 1
        )

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)

    cand = DeixisCandidate(
        timestamp=12.0, phrase="this cream", category=DeixisCategory.OBJECT, confidence=0.7
    )
    result = await vision_mod.analyze_summary_frames(
        [Path("/tmp/frame_01.jpg")],
        candidate=cand,
        output_language="English",
        job_id="job1",
    )

    assert result.relevant is True
    assert result.best_frame_index == 1
    assert "ACME Cream" in result.finding

    # Forced tool call, same function name QA uses, but summary-worded.
    tools = captured["kwargs"]["tools"]
    assert tools[0]["function"]["name"] == "report_frame_findings"
    assert captured["kwargs"]["tool_choice"] == {
        "type": "function",
        "function": {"name": "report_frame_findings"},
    }

    prompt_text = captured["messages"][0]["content"][0]["text"]
    assert "this cream" in prompt_text
    assert "English" in prompt_text
    # No question anchor — unlike qa_frames.txt, there's no user question to
    # weigh relevance against, only a "does this add anything" judgment.
    assert "The question:" not in prompt_text
    assert "worth" in prompt_text.lower()


@pytest.mark.asyncio
async def test_analyze_summary_frames_raises_on_llm_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vision_mod, "_frame_to_data_uri", lambda p: f"data:fake:{p}")

    async def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("backend errored")

    monkeypatch.setattr(llm_client, "complete_with_messages", boom)
    cand = DeixisCandidate(
        timestamp=12.0, phrase="this cream", category=DeixisCategory.OBJECT, confidence=0.7
    )
    with pytest.raises(RuntimeError):
        await vision_mod.analyze_summary_frames(
            [Path("/tmp/frame_01.jpg")], candidate=cand, output_language="English"
        )


@pytest.mark.asyncio
async def test_analyze_summary_frames_irrelevant_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vision_mod, "_frame_to_data_uri", lambda p: f"data:fake:{p}")

    async def fake_complete(messages: list[dict], **kwargs: Any) -> Any:
        return _vision_tool_completion(
            "Just the speaker talking to camera; nothing shown.", False, 1
        )

    monkeypatch.setattr(llm_client, "complete_with_messages", fake_complete)
    cand = DeixisCandidate(
        timestamp=12.0, phrase="so anyway", category=DeixisCategory.OBJECT, confidence=0.6
    )
    result = await vision_mod.analyze_summary_frames(
        [Path("/tmp/frame_01.jpg")], candidate=cand, output_language="English"
    )
    assert result.relevant is False
