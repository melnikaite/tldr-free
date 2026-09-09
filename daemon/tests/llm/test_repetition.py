"""Tests for llm.repetition — the line-repeat degeneration guard.

Covers both primitives (`trim_repeated_tail` for already-complete text,
`abort_on_repeated_lines` for a live delta stream), the "short list items
are legitimate repetition" tolerance, the cycle shapes (ABAB alternation,
period-4) that a naive "consecutive repeats only" rule would miss, and a
regression test wired through `llm.summary.stream_summarize` itself so a
degenerate stream is proven to never reach the caller's accumulated
(persisted) text.
"""

from __future__ import annotations

import pytest

from src.llm import client as llm_client
from src.llm import summary as summary_mod
from src.llm.repetition import abort_on_repeated_lines, trim_repeated_tail

LONG_LINE = "This exact bullet keeps coming back verbatim every single time."
OTHER_LINE = "A second, different bullet that the model alternates back to."
assert len(LONG_LINE) >= 20  # must count as "countable" for the default threshold
assert len(OTHER_LINE) >= 20


async def _async_iter(items: list[str]):
    for it in items:
        yield it


async def _collect(stream) -> str:
    parts: list[str] = []
    async for delta in stream:
        parts.append(delta)
    return "".join(parts)


# ---------------------------------------------------------------------------
# trim_repeated_tail (non-streaming)
# ---------------------------------------------------------------------------


def test_trim_repeated_tail_noop_on_clean_text() -> None:
    text = "## Overview\n\nSome intro.\n\n- Point one about a topic.\n- Point two, unrelated.\n"
    assert trim_repeated_tail(text) == text


def test_trim_repeated_tail_short_list_items_survive() -> None:
    """Short repeated list items ('- yes' style) must not be mistaken for a
    degenerate loop, however many times they repeat."""
    text = "\n".join(["- Да."] * 25)
    assert trim_repeated_tail(text) == text


def test_trim_repeated_tail_drops_back_to_back_repeat() -> None:
    """The period-1 (back-to-back) case: still caught as a special case of
    the general "occurs occurrence_threshold times anywhere" rule."""
    lines = ["## Overview", "", "Intro sentence here.", ""] + [LONG_LINE] * 10
    text = "\n".join(lines)
    trimmed = trim_repeated_tail(text)
    # occurrence_threshold - 1 (default 3 - 1 = 2) copies of the offending
    # line survive, then the text ends.
    assert trimmed == "\n".join(lines[:4] + [LONG_LINE, LONG_LINE])
    assert "Intro sentence here." in trimmed


def test_trim_repeated_tail_two_repeats_not_enough() -> None:
    """Two occurrences of a long line alone must not trigger the cut
    (default occurrence_threshold=3) — only a third occurrence does."""
    text = "\n".join(["Intro.", LONG_LINE, LONG_LINE])
    assert trim_repeated_tail(text) == text


def test_trim_repeated_tail_catches_abab_alternation() -> None:
    """Real incident shape: two bullets alternate, so no two occurrences of
    either line are ever adjacent — a consecutive-only rule would never
    flag this. The measured jobs (UBXw1EKb_Hc0, period ~2; LK-mGuWLJO1-,
    period ~4) are exactly this shape."""
    lines = ["Intro."]
    for _ in range(10):
        lines.append(LONG_LINE)
        lines.append(OTHER_LINE)
    text = "\n".join(lines)
    trimmed = trim_repeated_tail(text)
    # LONG_LINE hits its 3rd occurrence first (it appears first in each
    # A, B pair), so the cut lands right there: 2 A's, 2 B's survive.
    assert trimmed == "\n".join(["Intro.", LONG_LINE, OTHER_LINE, LONG_LINE, OTHER_LINE])


def test_trim_repeated_tail_catches_period_4_cycle() -> None:
    """A cycle over four distinct lines, matching the measured
    LK-mGuWLJO1- shape (period ~4) — none of the four lines is ever
    adjacent to itself, only every 4th line."""
    cycle = [f"Cycle line number {i} in the repeating loop pattern." for i in range(4)]
    for line in cycle:
        assert len(line) >= 20
    lines = ["Intro."] + cycle * 5
    text = "\n".join(lines)
    trimmed = trim_repeated_tail(text)
    # Each of the 4 cycle lines survives exactly occurrence_threshold - 1 = 2
    # times, then the text is cut at the first line reaching a 3rd occurrence
    # (the first line of the cycle's 3rd lap).
    assert trimmed == "\n".join(["Intro."] + cycle * 2)


# ---------------------------------------------------------------------------
# abort_on_repeated_lines (streaming)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_abort_on_repeated_lines_passes_clean_stream_through() -> None:
    deltas = ["## Over", "view\n\n", "Point one.\n", "Point two, different.\n"]
    out = await _collect(abort_on_repeated_lines(_async_iter(deltas)))
    assert out == "".join(deltas)


@pytest.mark.asyncio
async def test_abort_on_repeated_lines_tolerates_short_repeats() -> None:
    deltas = ["- Да.\n" for _ in range(20)]
    out = await _collect(abort_on_repeated_lines(_async_iter(deltas)))
    assert out == "".join(deltas)


@pytest.mark.asyncio
async def test_abort_on_repeated_lines_stops_degenerate_stream() -> None:
    """A stream that would repeat one long line forever must be cut off
    after occurrence_threshold repeats, not drained to exhaustion."""

    async def infinite_loop():
        # Legitimate lead-in, then the same long line forever — this must
        # never be fully consumed (would hang the test if it were).
        yield "Intro line.\n"
        while True:
            yield LONG_LINE + "\n"

    out = await _collect(abort_on_repeated_lines(infinite_loop()))
    assert "Intro line." in out
    # occurrence_threshold - 1 = 2 copies of the offending line survive, no more.
    assert out.count(LONG_LINE) == 2
    assert out.endswith(LONG_LINE + "\n")


@pytest.mark.asyncio
async def test_abort_on_repeated_lines_stops_abab_alternation() -> None:
    """The non-consecutive cycle shape must be caught live, not just by the
    non-streaming trim — this is the exact gap the consecutive-only rule
    had (jobs UBXw1EKb_Hc0 / LK-mGuWLJO1-)."""

    async def infinite_alternation():
        yield "Intro.\n"
        while True:
            yield LONG_LINE + "\n"
            yield OTHER_LINE + "\n"

    out = await _collect(abort_on_repeated_lines(infinite_alternation()))
    assert out.count(LONG_LINE) == 2
    assert out.count(OTHER_LINE) == 2
    assert out.endswith(OTHER_LINE + "\n")


@pytest.mark.asyncio
async def test_abort_on_repeated_lines_stops_period_4_cycle() -> None:
    cycle = [f"Cycle line number {i} in the repeating loop pattern." for i in range(4)]
    for line in cycle:
        assert len(line) >= 20

    async def infinite_cycle():
        yield "Intro.\n"
        while True:
            for line in cycle:
                yield line + "\n"

    out = await _collect(abort_on_repeated_lines(infinite_cycle()))
    for line in cycle:
        assert out.count(line) == 2
    # Cut lands on the first line of the cycle's 3rd lap.
    assert out.endswith(cycle[-1] + "\n")


@pytest.mark.asyncio
async def test_abort_on_repeated_lines_handles_arbitrary_chunk_boundaries() -> None:
    """The degenerate line arrives split across many small deltas, not
    conveniently aligned to line boundaries — the char-by-char state
    machine must still catch it."""
    whole = "Intro.\n" + (LONG_LINE + "\n") * 6
    # Split into 3-char chunks regardless of where lines fall.
    deltas = [whole[i : i + 3] for i in range(0, len(whole), 3)]
    out = await _collect(abort_on_repeated_lines(_async_iter(deltas)))
    assert out.count(LONG_LINE) == 2


@pytest.mark.asyncio
async def test_abort_on_repeated_lines_closes_underlying_generator() -> None:
    closed = False

    async def gen():
        nonlocal closed
        try:
            yield "Intro.\n"
            while True:
                yield LONG_LINE + "\n"
        finally:
            closed = True

    await _collect(abort_on_repeated_lines(gen()))
    assert closed, "underlying stream must be closed once degeneration is detected"


# ---------------------------------------------------------------------------
# Wired through llm.summary.stream_summarize (single-pass path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_summarize_never_persists_degenerate_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_stream(prompt: str, **kwargs: object):
        async def gen():
            yield "## Overview\n\n"
            yield "A normal opening point.\n"
            while True:
                yield LONG_LINE + "\n"

        return gen()

    monkeypatch.setattr(llm_client, "stream_complete", fake_stream)

    parts: list[str] = []
    async for delta in summary_mod.stream_summarize(
        "short source text" * 3, title="T", output_language="English"
    ):
        parts.append(delta)
    result = "".join(parts)

    assert "A normal opening point." in result
    assert result.count(LONG_LINE) == 2


@pytest.mark.asyncio
async def test_stream_summarize_never_persists_abab_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same regression, for the non-consecutive alternation shape."""

    def fake_stream(prompt: str, **kwargs: object):
        async def gen():
            yield "## Overview\n\n"
            while True:
                yield LONG_LINE + "\n"
                yield OTHER_LINE + "\n"

        return gen()

    monkeypatch.setattr(llm_client, "stream_complete", fake_stream)

    parts: list[str] = []
    async for delta in summary_mod.stream_summarize(
        "short source text" * 3, title="T", output_language="English"
    ):
        parts.append(delta)
    result = "".join(parts)

    assert result.count(LONG_LINE) == 2
    assert result.count(OTHER_LINE) == 2
