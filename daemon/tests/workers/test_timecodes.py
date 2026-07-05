"""Tests for workers.timecodes — the single source of truth for [MM:SS] markers."""

from __future__ import annotations

from src.workers.timecodes import (
    build_marked_text,
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
