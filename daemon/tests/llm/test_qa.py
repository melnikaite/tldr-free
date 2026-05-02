"""Tests for llm.qa.stream_answer.

Mocks `llm.client.stream_complete` to yield fixed deltas. Verifies that
`output_language` is threaded into the prompt.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from src.llm import client as llm_client
from src.llm import qa as qa_mod


@dataclass
class _FakeJob:
    title: str | None
    raw_text: str | None
    summary_md: str | None


@pytest.mark.asyncio
async def test_stream_answer_yields_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_prompts: list[str] = []

    async def fake_stream(prompt: str, **kwargs: object) -> AsyncIterator[str]:
        captured_prompts.append(prompt)
        for token in ["Hello", ", ", "world", "."]:
            yield token

    # qa.py calls llm_client.stream_complete, where llm_client is `src.llm.client`
    # — patching the canonical module is enough.
    monkeypatch.setattr(llm_client, "stream_complete", fake_stream)

    job = _FakeJob(
        title="Some video",
        raw_text="[00:30] First moment.\n\n[01:00] Second moment.",
        summary_md="## Summary\nA summary.",
    )

    out = []
    async for delta in qa_mod.stream_answer(
        job=job, question="What happens at 00:30?", output_language="English"
    ):
        out.append(delta)

    assert "".join(out) == "Hello, world."
    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
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
    # Force a tiny budget so raw_text never fits.
    monkeypatch.setattr(cfg.llm, "context_length", 4001)
    # Now budget = 4001 - 4000 = 1 token. raw_text below has many tokens.

    captured_prompts: list[str] = []

    async def fake_stream(prompt: str, **kwargs: object) -> AsyncIterator[str]:
        captured_prompts.append(prompt)
        if False:  # pragma: no cover - generator never yields, that's fine
            yield ""
        return

    # qa.py calls llm_client.stream_complete, where llm_client is `src.llm.client`
    # — patching the canonical module is enough.
    monkeypatch.setattr(llm_client, "stream_complete", fake_stream)

    big_raw = "Очень длинный сырой текст. " * 1000
    job = _FakeJob(
        title="Long video",
        raw_text=big_raw,
        summary_md="## Краткая выжимка\n- Пункт",
    )

    async for _ in qa_mod.stream_answer(
        job=job, question="?", output_language="English"
    ):
        pass

    prompt = captured_prompts[0]
    assert "Краткая выжимка" in prompt  # summary used
    # raw_text body should NOT be present (we expect the summary to replace it).
    assert "Очень длинный сырой текст." not in prompt


@pytest.mark.asyncio
async def test_handles_missing_title(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_prompts: list[str] = []

    async def fake_stream(prompt: str, **kwargs: object) -> AsyncIterator[str]:
        captured_prompts.append(prompt)
        for t in ["ok"]:
            yield t

    # qa.py calls llm_client.stream_complete, where llm_client is `src.llm.client`
    # — patching the canonical module is enough.
    monkeypatch.setattr(llm_client, "stream_complete", fake_stream)

    job = _FakeJob(title=None, raw_text="some text", summary_md=None)
    out = [d async for d in qa_mod.stream_answer(
        job=job, question="q", output_language="English"
    )]
    assert out == ["ok"]
    # The {title} placeholder was substituted (no leftover braces).
    assert "{title}" not in captured_prompts[0]
