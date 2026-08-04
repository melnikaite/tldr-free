"""Tests for workers.timecodes — the single source of truth for [MM:SS] markers."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

import pytest

from src.workers.timecodes import (
    build_marked_text,
    cap_markers_in_stream,
    cap_markers_per_line,
    collapse_repeated_segments,
    format_segments_as_marked_text,
    strip_all_timecodes,
    strip_bare_timecode_lines,
    strip_timecode_placeholders,
    strip_transcript_tail_noise,
)


def test_empty_segments_returns_empty_string() -> None:
    assert build_marked_text([], window_seconds=30) == ""


def test_zero_window_seconds_returns_empty_string() -> None:
    segs = [{"start": 0.0, "duration": 1.0, "text": "hi"}]
    assert build_marked_text(segs, window_seconds=0) == ""


def test_unpunctuated_cues_merge_into_one_line() -> None:
    # No sentence terminator and under the cap → one line, marked at the start.
    segs = [
        {"start": 0.0, "duration": 5.0, "text": "hello"},
        {"start": 5.0, "duration": 5.0, "text": "world"},
    ]
    out = build_marked_text(segs, window_seconds=30)
    assert out == "[00:00] hello world\n"


def test_one_line_per_sentence_with_exact_starts() -> None:
    # Sentence boundaries split the text; each line is marked with the exact
    # start of the cue its sentence begins in (not a rounded window).
    segs = [
        {"start": 0.0, "duration": 5.0, "text": "Hello there."},
        {"start": 7.0, "duration": 5.0, "text": "Next bit now."},
    ]
    out = build_marked_text(segs, window_seconds=30)
    assert out == "[00:00] Hello there.\n[00:07] Next bit now.\n"


def test_sentence_spanning_cues_is_one_line() -> None:
    segs = [
        {"start": 0.0, "duration": 5.0, "text": "It looks like"},
        {"start": 6.0, "duration": 5.0, "text": "today Fable reappeared."},
    ]
    out = build_marked_text(segs, window_seconds=30)
    assert out == "[00:00] It looks like today Fable reappeared.\n"


def test_decimal_point_is_not_a_sentence_break() -> None:
    # "5.6" must not split — the dot is followed by a digit, not whitespace.
    segs = [{"start": 12.0, "duration": 5.0, "text": "GPT 5.6 is out now."}]
    out = build_marked_text(segs, window_seconds=30)
    assert out == "[00:12] GPT 5.6 is out now.\n"


def test_long_unpunctuated_run_is_capped_at_window() -> None:
    # No punctuation → the window_seconds cap forces a break at a word boundary
    # so the whole track doesn't collapse into one [00:00] line.
    segs = [
        {"start": 0.0, "text": "aa"},
        {"start": 10.0, "text": "bb"},
        {"start": 20.0, "text": "cc"},
        {"start": 30.0, "text": "dd"},
        {"start": 40.0, "text": "ee"},
    ]
    out = build_marked_text(segs, window_seconds=30)
    assert out == "[00:00] aa bb cc\n[00:30] dd ee\n"


def test_switches_to_hh_mm_ss_at_one_hour() -> None:
    segs = [
        {"start": 0.0, "duration": 5.0, "text": "Alpha begins."},
        {"start": 3600.0, "duration": 5.0, "text": "Omega ends."},
    ]
    out = build_marked_text(segs, window_seconds=30)
    # use_hours kicks in because max start >= 3600 → all markers use HH:MM:SS.
    assert out == "[0:00:00] Alpha begins.\n[1:00:00] Omega ends.\n"


def test_below_one_hour_stays_mm_ss_with_exact_start() -> None:
    segs = [{"start": 3590.0, "duration": 5.0, "text": "almost there"}]
    out = build_marked_text(segs, window_seconds=30)
    assert out == "[59:50] almost there\n"


def test_segments_with_blank_text_skipped() -> None:
    segs = [
        {"start": 0.0, "duration": 1.0, "text": "real"},
        {"start": 5.0, "duration": 1.0, "text": "   "},
        {"start": 10.0, "duration": 1.0, "text": ""},
    ]
    out = build_marked_text(segs, window_seconds=30)
    assert out == "[00:00] real\n"


def test_deterministic_output_for_unsorted_input() -> None:
    segs_a = [
        {"start": 60.0, "duration": 5.0, "text": "Second."},
        {"start": 0.0, "duration": 5.0, "text": "First."},
        {"start": 30.0, "duration": 5.0, "text": "Middle."},
    ]
    segs_b = list(reversed(segs_a))
    out_a = build_marked_text(segs_a, window_seconds=30)
    out_b = build_marked_text(segs_b, window_seconds=30)
    assert out_a == out_b
    # Sorted by start; one sentence per cue, exact-start markers.
    assert out_a == "[00:00] First.\n[00:30] Middle.\n[01:00] Second.\n"


def test_whisper_segments_with_end_field_work_too() -> None:
    # Whisper verbose_json gives "end" instead of "duration"; both work because
    # we only read "start" + "text".
    segs = [
        {"start": 0.0, "end": 5.0, "text": "Hello."},
        {"start": 30.0, "end": 35.0, "text": "World."},
    ]
    out = build_marked_text(segs, window_seconds=30)
    assert out == "[00:00] Hello.\n[00:30] World.\n"


# ---------------------------------------------------------------------------
# format_segments_as_marked_text — one line per segment, no bucketing
# ---------------------------------------------------------------------------
# This is the sibling formatter used by the Transcript tab (no bucketing)
# and the translator (which feeds it to the LLM as the source). The key
# property is that the output has the SAME number of lines as the input
# has non-empty segments — that's the contract the translation prompt
# relies on for marker preservation.


def test_format_segments_emits_one_line_per_segment() -> None:
    """No bucketing — each segment is its own line, even if close in time."""
    segs = [
        {"start": 0.0, "end": 2.5, "text": "first"},
        {"start": 2.5, "end": 5.0, "text": "second"},
        {"start": 5.0, "end": 7.5, "text": "third"},
    ]
    out = format_segments_as_marked_text(segs)
    assert out == "[00:00] first\n[00:02] second\n[00:05] third\n"


def test_format_segments_empty_returns_empty() -> None:
    assert format_segments_as_marked_text([]) == ""


def test_format_segments_skips_empty_text() -> None:
    """Whitespace-only / blank segments are dropped — silence in audio."""
    segs = [
        {"start": 0.0, "text": "real"},
        {"start": 1.0, "text": "   "},
        {"start": 2.0, "text": ""},
        {"start": 3.0, "text": "also real"},
    ]
    out = format_segments_as_marked_text(segs)
    assert out == "[00:00] real\n[00:03] also real\n"


def test_format_segments_uses_hours_when_past_one_hour() -> None:
    """Same HH:MM:SS / MM:SS auto-detection as build_marked_text."""
    segs = [
        {"start": 0.0, "text": "intro"},
        {"start": 3605.0, "text": "much later"},
    ]
    out = format_segments_as_marked_text(segs)
    # Past 3600 s → hour-format used throughout for consistency.
    assert "[0:00:00] intro\n" in out
    assert "[1:00:05] much later\n" in out


def test_format_segments_preserves_input_order_per_segment() -> None:
    """Segments are emitted in input order. build_marked_text sorts by bucket
    index; this one preserves order (segments are already time-ordered by
    construction upstream, but we don't re-sort and don't merge)."""
    segs = [
        {"start": 0.0, "text": "a"},
        {"start": 0.5, "text": "b"},
        {"start": 1.0, "text": "c"},
    ]
    out = format_segments_as_marked_text(segs)
    assert out == "[00:00] a\n[00:00] b\n[00:01] c\n"


# --- strip_timecode_placeholders --------------------------------------------


def test_strip_removes_russian_placeholder() -> None:
    assert (
        strip_timecode_placeholders("- Главный вывод [Не указано]")
        == "- Главный вывод"
    )


def test_strip_removes_various_placeholders() -> None:
    for ph in ("[Not specified]", "[N/A]", "[—]", "[ ]", "[-]"):
        assert strip_timecode_placeholders(f"point {ph}") == "point"


def test_strip_keeps_real_timecodes() -> None:
    text = "- Key point [12:34]\n- Another [1:02:03]"
    assert strip_timecode_placeholders(text) == text


def test_strip_keeps_markdown_links() -> None:
    text = "See [the docs](https://example.com) for details"
    assert strip_timecode_placeholders(text) == text


def test_strip_drops_dangling_separator() -> None:
    assert strip_timecode_placeholders("Key point — [Не указано]") == "Key point"
    assert strip_timecode_placeholders("Key point - [N/A]") == "Key point"


def test_strip_collapses_inner_double_space() -> None:
    assert (
        strip_timecode_placeholders("before [Не указано] after") == "before after"
    )


def test_strip_noop_without_brackets() -> None:
    text = "plain summary with no markers"
    assert strip_timecode_placeholders(text) is text


def test_strip_mixed_lines() -> None:
    text = "- has time [00:30]\n- no time [Не указано]\n- link [x](y)"
    assert (
        strip_timecode_placeholders(text)
        == "- has time [00:30]\n- no time\n- link [x](y)"
    )


# --- strip_all_timecodes ----------------------------------------------------
# For non-transcript sources (web pages, PDFs) any [MM:SS] marker the model
# emitted is a hallucination — strip them all.


def test_strip_all_removes_mm_ss() -> None:
    assert strip_all_timecodes("- Key point [12:34]") == "- Key point"


def test_strip_all_removes_hh_mm_ss() -> None:
    assert strip_all_timecodes("- Key point [1:02:03]") == "- Key point"


def test_strip_all_removes_every_marker() -> None:
    text = "- First [00:30]\n- Second [12:34]\n- Third [1:02:03]"
    assert strip_all_timecodes(text) == "- First\n- Second\n- Third"


def test_strip_all_drops_dangling_separator() -> None:
    assert strip_all_timecodes("Key point — [12:34]") == "Key point"
    assert strip_all_timecodes("Key point - [00:30]") == "Key point"


def test_strip_all_collapses_inner_double_space() -> None:
    assert strip_all_timecodes("before [12:34] after") == "before after"


def test_strip_all_keeps_markdown_links() -> None:
    text = "See [the docs](https://example.com) for details"
    assert strip_all_timecodes(text) == text


def test_strip_all_noop_without_brackets() -> None:
    text = "plain summary with no markers"
    assert strip_all_timecodes(text) is text


def test_strip_all_noop_without_timecodes() -> None:
    # Brackets present but none are timecodes — left untouched.
    text = "See [the docs](https://example.com)"
    assert strip_all_timecodes(text) == text


# --- strip_bare_timecode_lines ----------------------------------------------
# A small local model sometimes ends a Q&A answer with a dump of bare [MM:SS]
# markers, one per line. Those lines are dropped; lines with real text stay.


def test_bare_drops_single_timecode_line() -> None:
    text = "Answer text here.\n[12:00]\n[5:00]\n[06:30]"
    assert strip_bare_timecode_lines(text) == "Answer text here."


def test_bare_drops_multiple_markers_on_one_line() -> None:
    text = "Real answer.\n[00:00] [00:30] [01:00]"
    assert strip_bare_timecode_lines(text) == "Real answer."


def test_bare_keeps_inline_timecode() -> None:
    text = "The GPU is mentioned [04:30] in the video."
    assert strip_bare_timecode_lines(text) == text


def test_bare_keeps_bullet_with_text_and_timecode() -> None:
    text = "* Homelab runs Linux [00:00]\n* RTX 4060 Ti [04:00]"
    assert strip_bare_timecode_lines(text) == text


def test_bare_drops_timecode_range_line() -> None:
    text = "Some answer.\n[00:00] - [23:00]"
    assert strip_bare_timecode_lines(text) == "Some answer."


def test_bare_noop_without_brackets() -> None:
    text = "plain answer, no markers"
    assert strip_bare_timecode_lines(text) is text


def test_bare_collapses_blank_runs() -> None:
    text = "Answer.\n[00:00]\n\n[01:00]\nMore text."
    assert strip_bare_timecode_lines(text) == "Answer.\n\nMore text."


# --- strip_transcript_tail_noise --------------------------------------------


def test_tail_noise_drops_trailing_hallucination() -> None:
    text = "[00:00] Real content here\n[02:28] Продолжение следует...\n"
    assert strip_transcript_tail_noise(text) == "[00:00] Real content here\n"


def test_tail_noise_drops_multiple_trailing() -> None:
    text = (
        "[00:00] Real\n"
        "[09:50] Спасибо за просмотр!\n"
        "[09:55] Подписывайтесь на канал\n"
    )
    assert strip_transcript_tail_noise(text) == "[00:00] Real\n"


def test_tail_noise_keeps_real_middle_content() -> None:
    # A phrase in the MIDDLE is left alone — only the tail is scanned.
    text = "[00:00] Спасибо за просмотр, говорит ведущий\n[00:30] Real ending\n"
    assert strip_transcript_tail_noise(text) == text


def test_tail_noise_english_phantoms() -> None:
    text = "[00:00] Actual talk\n[10:00] Thanks for watching!\n"
    assert strip_transcript_tail_noise(text) == "[00:00] Actual talk\n"


def test_tail_noise_noop_when_clean() -> None:
    text = "[00:00] line one\n[00:30] line two\n"
    assert strip_transcript_tail_noise(text) == text


def test_tail_noise_empty() -> None:
    assert strip_transcript_tail_noise("") == ""


# --- collapse_repeated_segments ----------------------------------------------
# Whisper hallucination-loop collapse. Fixtures mirror the real measured
# shapes: a 291-run of an identical long sentence, a 57-run of a short "Ja.",
# a near-duplicate drifting trio, and a short legitimate dialogue repeat that
# must survive.


def _segs(texts: list[str], *, step: float = 1.0, start: float = 0.0) -> list[dict]:
    return [
        {"start": start + i * step, "end": start + i * step + step, "text": t}
        for i, t in enumerate(texts)
    ]


def test_collapse_291_run_of_identical_long_sentence() -> None:
    text = "I'm not sure if I'm doing that right."
    segs = _segs([text] * 291)
    out = collapse_repeated_segments(segs)
    assert len(out) == 1
    assert out[0] == segs[0]


def test_collapse_57_run_of_short_ja() -> None:
    segs = _segs(["Ja."] * 57)
    out = collapse_repeated_segments(segs)
    assert len(out) == 1
    assert out[0] == segs[0]


def test_collapse_near_duplicate_drifting_trio() -> None:
    # Measured hallucination drift: the sentence mutates slightly each repeat
    # while remaining recognizably "the same" — exact-match would miss this.
    segs = _segs(
        [
            "He was a young man who was very interested in the world of "
            "science and technology.",
            "He was also a young man who was interested in the world of "
            "science and technology.",
            "He was also interested in the world of science and technology.",
        ]
    )
    out = collapse_repeated_segments(segs)
    assert len(out) == 1
    assert out[0] == segs[0]


def test_collapse_near_dup_pair_never_collapses() -> None:
    # Real measured false positive (job BeRoZoPrbhnT, youtube_api): a
    # two-speaker question/confirmation exchange scores 0.84 on
    # _NEAR_DUP_RATIO (just above the 0.82 threshold) purely because the
    # sentences are structurally similar — it is NOT a hallucination loop.
    # A near-dup run of length 2 must never collapse, unlike an exact-dup
    # pair of long segments (which does, per _LONG_SEGMENT_RUN_THRESHOLD).
    segs = _segs([">> That's locked in?", ">> It's locked in."])
    out = collapse_repeated_segments(segs)
    assert out == segs


def test_collapse_short_dialogue_survives() -> None:
    # 5 consecutive "Ja." is real call-and-response dialogue, not a
    # hallucination loop — short-segment threshold tolerates this.
    segs = _segs(["Ja."] * 5)
    out = collapse_repeated_segments(segs)
    assert out == segs


def test_collapse_noop_when_no_repetition() -> None:
    segs = _segs(["First sentence here.", "Second sentence here.", "Third one."])
    out = collapse_repeated_segments(segs)
    assert out == segs


def test_collapse_empty_segments() -> None:
    assert collapse_repeated_segments([]) == []


def test_collapse_non_consecutive_repeats_survive() -> None:
    # Same long text recurring later, separated by other content, is NOT a
    # consecutive run — collapse_repeated_segments never dedupes globally.
    text = "This sentence legitimately recurs in the transcript."
    segs = _segs([text, "Unrelated middle content here.", text])
    out = collapse_repeated_segments(segs)
    assert out == segs


# --- cap_markers_per_line -----------------------------------------------------


def test_cap_nine_markers_keeps_first() -> None:
    line = (
        "Key point [07:55] [08:05] [08:09] [08:29] [08:31] [08:33] "
        "[08:37] [08:40] [08:45]"
    )
    assert cap_markers_per_line(line) == "Key point [07:55]"


def test_cap_line_with_no_markers_unchanged() -> None:
    line = "Just a plain bullet with no timecodes."
    assert cap_markers_per_line(line) is line


def test_cap_line_with_exactly_max_markers_unchanged() -> None:
    line = "A point [01:00]"
    assert cap_markers_per_line(line, max_markers=1) is line


def test_cap_line_with_exactly_two_max_markers_unchanged() -> None:
    line = "A point [01:00] [02:00]"
    assert cap_markers_per_line(line, max_markers=2) == line


def test_cap_noop_on_document_with_no_markers_anywhere() -> None:
    # PDF/web-page safety: pure no-op on marker-less text.
    text = "# Title\n\n- First point.\n- Second point.\n- Third point.\n"
    assert cap_markers_per_line(text) is text


def test_cap_only_touches_lines_that_exceed_the_cap() -> None:
    text = (
        "- Fine line [00:10]\n"
        "- Overloaded line [01:00] [02:00] [03:00]\n"
        "- Another fine line, no marker\n"
    )
    out = cap_markers_per_line(text)
    assert out == (
        "- Fine line [00:10]\n"
        "- Overloaded line [01:00]\n"
        "- Another fine line, no marker\n"
    )


# --- cap_markers_in_stream ----------------------------------------------------
# Marker-granularity holdback: text is buffered ONLY long enough to decide
# whether it's a [MM:SS]-shaped bracket (bounded to ~10-11 chars), never
# until a whole line completes — see the module docstring on
# cap_markers_in_stream for why line-level buffering was rejected (it
# regressed real streaming latency on the longest, first-generated line of
# every summary — the Overview paragraph, measured 613-706 chars).


async def _fake_stream(chunks: list[str]) -> AsyncIterator[str]:
    for c in chunks:
        yield c


async def _collect(chunks: list[str], max_markers: int = 1) -> list[str]:
    return [
        d async for d in cap_markers_in_stream(_fake_stream(chunks), max_markers=max_markers)
    ]


async def test_stream_marker_split_across_three_deltas_stays_intact() -> None:
    chunks = ["before [0", "4:3", "0] after\n"]
    published = await _collect(chunks)
    assert "".join(published) == "before [04:30] after\n"


async def test_stream_markdown_link_split_across_deltas_untouched_and_uncounted() -> None:
    # A marker-shaped bracket immediately followed by "(" is a markdown link
    # (the (?!\() guard case), not a timecode — must survive verbatim and
    # must NOT count against the per-line marker cap.
    chunks = ["see [01", ":30](ur", "l) and [09:00] too\n"]
    published = await _collect(chunks)
    # The link is untouched AND uncounted, so the real marker after it
    # ([09:00]) still survives — if the link had been miscounted as the
    # first marker, [09:00] would have been dropped instead.
    assert "".join(published) == "see [01:30](url) and [09:00] too\n"


async def test_stream_non_marker_bracket_split_across_deltas_untouched() -> None:
    chunks = ["[Not sp", "ecified] and [te", "xt](url)\n"]
    published = await _collect(chunks)
    assert "".join(published) == "[Not specified] and [text](url)\n"


async def test_stream_excess_markers_on_one_line_dropped_surrounding_text_intact() -> None:
    chunks = ["A [00:01] B [00:02] C [00:03] D\n"]
    published = await _collect(chunks)
    joined = "".join(published)
    # First marker survives; the other two are dropped (their bracket text
    # only — the letters A/B/C/D around them all survive, in order).
    assert "[00:01]" in joined
    assert "[00:02]" not in joined
    assert "[00:03]" not in joined
    for token in ("A", "B", "C", "D"):
        assert token in joined
    assert joined.index("A") < joined.index("[00:01]") < joined.index("B")
    assert joined.index("B") < joined.index("C") < joined.index("D")


async def test_stream_marker_cap_resets_per_line() -> None:
    # The per-line marker allowance must reset on "\n" — a marker dropped on
    # line 1 for exceeding the cap must not carry over and suppress line 2's
    # own first marker, which gets its own fresh allowance.
    chunks = ["A [00:01] [00:02]\n", "B [00:03] [00:04]\n"]
    published = await _collect(chunks)
    joined = "".join(published)
    line1, line2 = joined.split("\n")[:2]
    assert "[00:01]" in line1
    assert "[00:02]" not in line1
    assert "[00:03]" in line2
    assert "[00:04]" not in line2


async def test_stream_long_line_with_markers_streams_in_small_pieces() -> None:
    # Regression guard for the bug being fixed: a long single line (matching
    # the measured real-world Overview-paragraph shape, 613-706 chars) that
    # ALSO carries several [MM:SS] markers must still stream progressively —
    # no yielded piece is anywhere near a whole-line size — while still
    # capping down to the first marker on the line.
    pad = "word " * 100  # 500 chars of marker-free padding
    line = f"{pad}[00:05] {pad}[00:12] {pad}[00:47] end\n"
    assert len(line) > 700
    chunks = [line[i : i + 3] for i in range(0, len(line), 3)]
    published = await _collect(chunks)
    joined = "".join(published)
    assert joined.count("word") == 300  # all the padding text survives
    assert len(re.findall(r"\[\d{2}:\d{2}\]", joined)) == 1
    assert "[00:05]" in joined
    assert "[00:12]" not in joined
    assert "[00:47]" not in joined
    assert all(len(piece) < 15 for piece in published)


async def test_stream_long_marker_free_line_streams_in_small_pieces() -> None:
    # Proves this doesn't regress back into line-level buffering: a ~700
    # char marker-free line (matching the real Overview-paragraph shape)
    # fed in small deltas must come back out in small pieces, not one big
    # chunk held until the trailing newline.
    line = ("word " * 140)[:700] + "\n"
    chunks = [line[i : i + 4] for i in range(0, len(line), 4)]
    published = await _collect(chunks)
    assert "".join(published) == line
    assert all(len(piece) < 15 for piece in published)


async def test_stream_caps_trailing_partial_line_without_newline() -> None:
    chunks = ["- Only line [00:01] [00:02] [00:03]"]  # no trailing "\n"
    published = await _collect(chunks)
    joined = "".join(published)
    assert "[00:01]" in joined
    assert "[00:02]" not in joined
    assert "[00:03]" not in joined


async def test_stream_passes_through_marker_less_text_unchanged() -> None:
    chunks = ["Hello ", "world, ", "no markers here.\n"]
    published = await _collect(chunks)
    assert "".join(published) == "Hello world, no markers here.\n"


# --- cap_markers_in_stream == cap_markers_per_line, for any chunking -------
# The two primitives must be byte-identical on the same input — a dropped
# marker's preceding whitespace run has to vanish in the streaming path
# exactly like ``_tidy_after_bracket_removal`` erases it in the non-streaming
# one. Two-or-more trailing spaces before a newline is a Markdown hard line
# break (`<br>`), so leaving one behind is a rendering bug, not cosmetics.
# Covers: a dropped marker at line-end (trailing-space case), a dropped
# marker sandwiched between real words, a markdown-link-shaped bracket next
# to a dropped marker, three consecutive markers with only the first kept,
# a marker cluster at the very start of a line (nothing precedes it), a
# second line getting its own cap allowance, and a line with no trailing
# newline at all.
_EQUIVALENCE_CASES = [
    "- Ежедневная сторона выросла на 20% [11:12] [11:24] [11:30]\n",
    "- A [00:01] B [00:02] C\n",
    "- см. [04:30](https://x/y) и маркер [05:00] [06:00]\n",
    "A [00:01] [00:02] [00:03] B\n",
    "[00:01] [00:02]\n",
    "text [Not specified] and [text](url) and [09:00] [10:00]\n",
    "Line one [00:01] [00:02]\nLine two [00:03] [00:04]\n",
    "- Only line [00:01] [00:02] [00:03]",
    "Just plain text with no markers at all.\n",
]


def _chunk(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


@pytest.mark.parametrize("chunk_size", [1, 3, 7])
@pytest.mark.parametrize("text", _EQUIVALENCE_CASES)
async def test_stream_matches_per_line_for_any_chunking(
    text: str, chunk_size: int
) -> None:
    published = await _collect(_chunk(text, chunk_size))
    assert "".join(published) == cap_markers_per_line(text)
