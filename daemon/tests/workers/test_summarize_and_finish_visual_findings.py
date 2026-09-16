"""Tests for workers.pipeline._summarize_and_finish's visual-findings wiring.

Summary-time on-screen findings (workers.pipeline._run_frame_analysis) used
to be threaded into llm.summary.stream_summarize as a separate
`visual_findings` block; they are now woven directly into the transcript
text handed to the summarizer, via workers.timecodes.inject_visual_findings
(see that function's docstring, and llm/summary.py's module docstring, for
why). These tests verify the pipeline wiring around that change:

- the LLM-facing input (`summary_input`) carries the woven-in annotation
- the job's persisted `raw_text` (the `text` argument, unchanged since
  before this feature existed) is byte-for-byte untouched by the injection
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from src.api.schemas import TranscriptSource
from src.config import get_config
from src.storage import repo
from src.storage.db import dispose_engine, init_engine
from src.storage.migrations import run_migrations
from src.workers import broker as broker_mod
from src.workers import control as control_mod
from src.workers import pipeline as pipeline_mod


@pytest.fixture(autouse=True)
def _reset_singletons() -> Any:
    control_mod.reset_control()
    broker_mod.reset_broker()
    yield
    control_mod.reset_control()
    broker_mod.reset_broker()


@pytest.fixture
def isolated_db(tmp_path: Path) -> Any:
    db_path = tmp_path / "summarize-and-finish.db"
    engine = init_engine(db_path)
    run_migrations(engine)
    try:
        yield engine
    finally:
        dispose_engine()


@pytest.mark.asyncio
async def test_raw_text_untouched_while_summary_input_carries_annotation(
    isolated_db: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = repo.create_job(url="https://example.com/video", kind="youtube")

    original_text = "[00:05] First sentence.\n[00:12] Second sentence.\n"

    async def fake_frame_analysis(
        job_id: str, broker: Any, cookies: Any
    ) -> list[dict[str, Any]]:
        return [
            {
                "seconds": 12.0,
                "timecode": "00:12",
                "phrase": "watch this",
                "category": "object",
                "finding": "A red tub labeled 'ACME Cream 200ml'.",
                "frame_url": None,
            }
        ]

    monkeypatch.setattr(pipeline_mod, "_run_frame_analysis", fake_frame_analysis)

    received_inputs: list[str] = []

    async def fake_stream_summarize(text: str, **kwargs: Any) -> AsyncIterator[str]:
        received_inputs.append(text)
        yield "## Overview\n\nSummary."

    monkeypatch.setattr(pipeline_mod.llm_summary, "stream_summarize", fake_stream_summarize)

    await pipeline_mod._summarize_and_finish(
        job.id,
        text=original_text,
        title="Title",
        transcript_source=TranscriptSource.YOUTUBE_API,
        video_id="abc123",
        cfg=get_config(),
        cookies=[],
    )

    # The LLM-facing input carries the woven-in annotation...
    assert len(received_inputs) == 1
    assert "ACME Cream 200ml" in received_inputs[0]
    assert "⟦PICTURE [" in received_inputs[0]

    # ...but the persisted raw_text is byte-for-byte the original.
    stored = repo.get_job(job.id)
    assert stored is not None
    assert stored.raw_text == original_text
    assert "⟦PICTURE [" not in stored.raw_text


@pytest.mark.asyncio
async def test_no_findings_leaves_summary_input_unchanged(
    isolated_db: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = repo.create_job(url="https://example.com/video", kind="youtube")
    original_text = "[00:05] Nothing visual to report here.\n"

    async def fake_frame_analysis(
        job_id: str, broker: Any, cookies: Any
    ) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(pipeline_mod, "_run_frame_analysis", fake_frame_analysis)

    received_inputs: list[str] = []

    async def fake_stream_summarize(text: str, **kwargs: Any) -> AsyncIterator[str]:
        received_inputs.append(text)
        yield "## Overview\n\nSummary."

    monkeypatch.setattr(pipeline_mod.llm_summary, "stream_summarize", fake_stream_summarize)

    await pipeline_mod._summarize_and_finish(
        job.id,
        text=original_text,
        title="Title",
        transcript_source=TranscriptSource.YOUTUBE_API,
        video_id="abc123",
        cfg=get_config(),
        cookies=[],
    )

    assert received_inputs == [original_text]
    stored = repo.get_job(job.id)
    assert stored is not None
    assert stored.raw_text == original_text
