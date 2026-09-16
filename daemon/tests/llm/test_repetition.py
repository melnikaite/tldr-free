"""Tests for llm.repetition — the line-repeat degeneration guard.

Covers both primitives (`trim_repeated_tail` for already-complete text,
`abort_on_repeated_lines` for a live delta stream), the "short list items
are legitimate repetition" tolerance, the cycle shapes (ABAB alternation,
period-4) that a naive "consecutive repeats only" rule would miss, the
occurrence-2 suppression rule (a duplicate line is dropped even when no
cut ever happens), and a regression test wired through
`llm.summary.stream_summarize` itself so a degenerate stream is proven to
never reach the caller's accumulated (persisted) text.
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
    the general "occurs occurrence_threshold times anywhere" rule. Under
    the occurrence-2 suppression rule, the 2nd occurrence is dropped in
    place (not just the tail from the 3rd occurrence on), so only the 1st
    occurrence survives."""
    lines = ["## Overview", "", "Intro sentence here.", ""] + [LONG_LINE] * 10
    text = "\n".join(lines)
    trimmed = trim_repeated_tail(text)
    assert trimmed == "\n".join(lines[:4] + [LONG_LINE])
    assert "Intro sentence here." in trimmed


def test_trim_repeated_tail_suppresses_benign_single_duplicate() -> None:
    """A long line that legitimately appears exactly twice (no loop, no 3rd
    occurrence) must not trigger a cut — but its 2nd occurrence is still
    silently dropped, so no duplicate survives. The rest of the text is
    untouched."""
    text = "\n".join(["Intro.", LONG_LINE, LONG_LINE, "Conclusion line here."])
    trimmed = trim_repeated_tail(text)
    assert trimmed == "\n".join(["Intro.", LONG_LINE, "Conclusion line here."])
    assert trimmed.count(LONG_LINE) == 1


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
    # Each line's 2nd occurrence is suppressed in place, so by the time
    # LONG_LINE reaches what would be its 3rd occurrence (triggering the
    # cut), only the 1st occurrence of each of LONG_LINE and OTHER_LINE has
    # ever survived.
    assert trimmed == "\n".join(["Intro.", LONG_LINE, OTHER_LINE])


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
    # Each cycle line's 2nd occurrence (the whole 2nd lap) is suppressed in
    # place, so only the 1st lap survives before the cut triggered by the
    # first line to reach a 3rd occurrence.
    assert trimmed == "\n".join(["Intro."] + cycle)


def test_trim_repeated_tail_suppression_leaves_no_blank_line_behind() -> None:
    """Dropping a duplicate line in the middle of the text must not leave an
    empty line in its place — the lines before and after it become
    adjacent, exactly as if the duplicate had never been generated."""
    text = "\n".join(["Intro.", LONG_LINE, LONG_LINE, "Next point here, unrelated."])
    trimmed = trim_repeated_tail(text)
    assert trimmed == "\n".join(["Intro.", LONG_LINE, "Next point here, unrelated."])
    assert "\n\n" not in trimmed


def test_trim_repeated_tail_realistic_incident_fixture() -> None:
    """Regression fixture shaped like the real incident that motivated
    occurrence-2 suppression: a normal summary followed by the same long
    bullet line repeated ~15 times. Before this fix, `trim_repeated_tail`
    still let 2 copies of the bullet through; now exactly 1 must survive."""
    bullet = "- Рынок альткоинов — это не просто рост, а необходимый элемент для роста биткоина."
    assert len(bullet) >= 20
    lines = [
        "## Обзор",
        "",
        "Автор видео анализирует текущее состояние криптовалютного рынка.",
        "",
        "## Ключевые моменты",
        "",
        "- Биткоин находится в глобальном развороте. [01:48]",
        "- Кошельки, созданные в марте 2010 года, пробуждаются. [01:57]",
        "- Рост альткоинов — необходимое условие для эйфории. [06:26]",
    ] + [bullet] * 15
    text = "\n".join(lines)
    trimmed = trim_repeated_tail(text)
    assert trimmed.count(bullet) == 1
    assert "## Обзор" in trimmed
    assert "- Биткоин находится в глобальном развороте. [01:48]" in trimmed


def test_trim_repeated_tail_no_duplicate_survives_at_higher_threshold() -> None:
    """The no-duplicates rule must not be tied to the default threshold of 3.

    At `occurrence_threshold=4` a formulation like "suppress when the count
    reaches threshold - 1" would let the 2nd occurrence through (count 2,
    threshold - 1 == 3) and only drop the 3rd — a visible duplicate. Only
    the CUT depends on the threshold; suppression is "not the first".
    """
    tail = "A trailing sentence that must not survive the cut at all."
    assert len(tail) >= 20
    text = "\n".join([LONG_LINE] * 5 + [tail])
    trimmed = trim_repeated_tail(text, occurrence_threshold=4)
    assert trimmed.count(LONG_LINE) == 1
    # The 4th occurrence still triggers the cut, so anything after it is gone.
    assert tail not in trimmed


def test_trim_repeated_tail_keeps_first_occurrence_at_threshold_two() -> None:
    """`occurrence_threshold=2` must still keep the FIRST occurrence.

    Tying suppression to `threshold - 1` made that condition true on the
    very first occurrence here (count 1 == 2 - 1), silently eating lines
    that never repeated at all.
    """
    other = "A wholly distinct bullet that appears exactly once."
    assert len(other) >= 20
    text = "\n".join([LONG_LINE, other])
    trimmed = trim_repeated_tail(text, occurrence_threshold=2)
    assert trimmed.count(LONG_LINE) == 1
    assert other in trimmed



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
    # Only the 1st occurrence survives: the 2nd is suppressed in place, and
    # the 3rd both suppresses itself and aborts the stream.
    assert out.count(LONG_LINE) == 1
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
    # Each line's 2nd occurrence is suppressed in place, so only the 1st lap
    # (one of each) survives before LONG_LINE's 3rd occurrence aborts.
    assert out.count(LONG_LINE) == 1
    assert out.count(OTHER_LINE) == 1
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
    # Each cycle line's 2nd occurrence (the whole 2nd lap) is suppressed in
    # place, so only the 1st lap survives.
    for line in cycle:
        assert out.count(line) == 1
    # Cut is triggered by the first line of the cycle's 3rd lap, but the
    # last thing actually emitted is the 1st lap's last line.
    assert out.endswith(cycle[-1] + "\n")


@pytest.mark.asyncio
async def test_abort_on_repeated_lines_handles_arbitrary_chunk_boundaries() -> None:
    """The degenerate line arrives split across many small deltas, not
    conveniently aligned to line boundaries — the char-by-char state
    machine must still catch it, and the 2nd occurrence must never appear
    in any emitted delta even though its characters cross chunk boundaries
    that have nothing to do with the line boundaries."""
    whole = "Intro.\n" + (LONG_LINE + "\n") * 6
    # Split into 3-char chunks regardless of where lines fall.
    deltas = [whole[i : i + 3] for i in range(0, len(whole), 3)]
    emitted: list[str] = []
    async for delta in abort_on_repeated_lines(_async_iter(deltas)):
        emitted.append(delta)
    out = "".join(emitted)
    assert out.count(LONG_LINE) == 1
    # No individual emitted delta ever contains a 2nd copy of the line either
    # — the whole occurrence was held back and dropped, never dribbled out.
    joined_after_first = out.split(LONG_LINE, 1)[1]
    assert LONG_LINE not in joined_after_first


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
    assert result.count(LONG_LINE) == 1


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

    assert result.count(LONG_LINE) == 1
    assert result.count(OTHER_LINE) == 1


@pytest.mark.asyncio
async def test_abort_on_repeated_lines_suppresses_second_at_higher_threshold() -> None:
    """Streaming counterpart of the threshold-4 case: the 2nd occurrence
    must never appear in ANY emitted delta, even though the abort itself
    doesn't fire until the 4th."""
    deltas = [f"{LONG_LINE}\n" for _ in range(5)]
    emitted: list[str] = []
    async for delta in abort_on_repeated_lines(_async_iter(deltas), occurrence_threshold=4):
        emitted.append(delta)
    out = "".join(emitted)
    assert out.count(LONG_LINE) == 1
