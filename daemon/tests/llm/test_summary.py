"""Tests for llm.summary.stream_summarize — single-pass vs map-reduce branching.

Mocks `llm.client.stream_complete` (and `complete` for the map phase) so no
real LLM call is made.
"""

from __future__ import annotations

import pytest

from src.llm import client as llm_client
from src.llm import summary as summary_mod
from src.llm.tokens import count_tokens


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
    # Must be > _CHUNK_TARGET_TOKENS (4000) so chunk_target (derived from this
    # threshold, see stream_summarize) stays at its usual 4000 and the ~20
    # small partials this text produces don't themselves need reduce-folding
    # — this test is about the map/reduce split, not folding (see
    # test_hierarchical_reduce_folds_large_partials for that).
    monkeypatch.setattr(cfg.llm, "single_pass_token_limit", 10_000)

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


@pytest.mark.asyncio
async def test_oversized_lone_chunk_never_single_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real scenario, not a mocked splitter bug: with a small configured
    `single_pass_token_limit`, the map-phase chunk budget is derived from it
    (see `chunk_target` in stream_summarize) so a marked-transcript-shaped
    text that split_for_summary would otherwise hand back as one oversized
    chunk gets split into several sub-threshold chunks instead — never
    single-passed just because `len(chunks) == 1` at the default budget."""
    from src import config as config_mod

    cfg = config_mod.get_config()
    threshold = 300
    monkeypatch.setattr(cfg.llm, "single_pass_token_limit", threshold)

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

    # A marked-transcript-shaped single "paragraph" (no blank lines) whose
    # total size sits comfortably above `threshold` (300) but well under
    # split_for_summary's own 4000-token chunk target — so split_for_summary
    # legitimately returns exactly one chunk equal to the whole text.
    lines = [
        f"[{i:02d}:00] Sentence number {i} in a transcript with no blank lines."
        for i in range(120)
    ]
    text = "\n".join(lines)
    assert count_tokens(text) > threshold
    assert count_tokens(text) < 4000

    await _collect(text, title="Video", output_language="English")

    # chunk_target = threshold - 1 already at the first split_for_summary
    # call, so the oversized "one chunk" case is split up front — more than
    # one map call, never a single-pass.
    assert len(map_calls) >= 2, "oversized lone chunk must be split, not single-passed"
    assert len(stream_calls) == 1
    # The final reduce prompt as a whole can exceed `threshold` (it also
    # carries the reduce instructions/title), but it must not carry the
    # ORIGINAL unsplit chunk's raw text.
    assert text not in stream_calls[0]


@pytest.mark.asyncio
async def test_hierarchical_reduce_folds_large_partials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Many large partials whose naive join would blow the reduce budget
    must be folded hierarchically (via _fold_partials) before the final
    reduce — regression for the reduce-phase half of the incident (map
    chunks correctly bounded, but 17 partials joined unchecked)."""
    from src import config as config_mod

    cfg = config_mod.get_config()
    # Sized so that 2 partials fit the budget but 3 don't — pack_lines groups
    # them 2-at-a-time, forcing real folding (not the degenerate case where
    # every partial is already too big to combine with any neighbour at all).
    reduce_budget = 4000
    monkeypatch.setattr(cfg.llm, "single_pass_token_limit", reduce_budget)

    # Bypass the map phase entirely — feed _fold_partials directly with
    # large partials to isolate the reduce-folding logic under test.
    intermediate_calls: list[list[str]] = []

    async def fake_complete(prompt: str, **kwargs: object) -> str:
        # Each intermediate reduce call collapses its group into a short
        # fixed-size folded summary.
        intermediate_calls.append([prompt])
        return f"Folded summary #{len(intermediate_calls)}"

    monkeypatch.setattr(llm_client, "complete", fake_complete)

    big_partial = "Ключевой момент. " * 200  # ~1600 tokens — large but individually under budget
    partials = [big_partial for _ in range(8)]
    assert count_tokens(big_partial) < reduce_budget
    assert count_tokens("\n\n---\n\n".join(partials)) > reduce_budget

    folded = await summary_mod._fold_partials(
        partials,
        title="Video",
        output_language="English",
        source_note="note",
        budget=reduce_budget,
    )

    # Folding must have actually happened: fewer folded partials than
    # originals, at least one intermediate (non-streaming) reduce call made.
    assert len(folded) < len(partials)
    assert len(intermediate_calls) >= 1
    # And the final join must now fit the budget.
    assert count_tokens("\n\n---\n\n".join(folded)) <= reduce_budget
