"""Tests for workers.timecodes — the single source of truth for [MM:SS] markers."""

from __future__ import annotations

from src.workers.timecodes import (
    build_marked_text,
    format_segments_as_marked_text,
    strip_timecode_placeholders,
)


def test_empty_segments_returns_empty_string() -> None:
    assert build_marked_text([], window_seconds=30) == ""


def test_zero_window_seconds_returns_empty_string() -> None:
    segs = [{"start": 0.0, "duration": 1.0, "text": "hi"}]
    assert build_marked_text(segs, window_seconds=0) == ""


def test_single_bucket_mm_ss() -> None:
    segs = [
        {"start": 0.0, "duration": 5.0, "text": "hello"},
        {"start": 5.0, "duration": 5.0, "text": "world"},
    ]
    out = build_marked_text(segs, window_seconds=30)
    assert out == "[00:00] hello world\n"


def test_multiple_buckets_mm_ss() -> None:
    segs = [
        {"start": 0.0, "duration": 5.0, "text": "first"},
        {"start": 30.0, "duration": 5.0, "text": "second"},
        {"start": 65.0, "duration": 5.0, "text": "third"},
    ]
    out = build_marked_text(segs, window_seconds=30)
    assert out == "[00:00] first\n[00:30] second\n[01:00] third\n"


def test_switches_to_hh_mm_ss_at_one_hour() -> None:
    segs = [
        {"start": 0.0, "duration": 5.0, "text": "alpha"},
        {"start": 3600.0, "duration": 5.0, "text": "omega"},
    ]
    out = build_marked_text(segs, window_seconds=30)
    # use_hours kicks in because max_start >= 3600 → all markers use HH:MM:SS.
    assert out == "[0:00:00] alpha\n[1:00:00] omega\n"


def test_below_one_hour_stays_mm_ss() -> None:
    # 59:30 is well under an hour, marker stays MM:SS.
    segs = [{"start": 3590.0, "duration": 5.0, "text": "almost"}]
    out = build_marked_text(segs, window_seconds=30)
    assert out == "[59:30] almost\n"


def test_empty_buckets_skipped() -> None:
    segs = [
        {"start": 0.0, "duration": 5.0, "text": "first"},
        # 30..60 has no segments.
        {"start": 60.0, "duration": 5.0, "text": "third"},
    ]
    out = build_marked_text(segs, window_seconds=30)
    # No empty bucket, just the two non-empty.
    assert out == "[00:00] first\n[01:00] third\n"


def test_segments_with_blank_text_skipped() -> None:
    segs = [
        {"start": 0.0, "duration": 1.0, "text": "real"},
        {"start": 5.0, "duration": 1.0, "text": "   "},
        {"start": 10.0, "duration": 1.0, "text": ""},
    ]
    out = build_marked_text(segs, window_seconds=30)
    assert out == "[00:00] real\n"


def test_text_is_trimmed_per_bucket() -> None:
    segs = [
        {"start": 0.0, "duration": 5.0, "text": "  alpha  "},
        {"start": 5.0, "duration": 5.0, "text": "  beta  "},
    ]
    out = build_marked_text(segs, window_seconds=30)
    assert out == "[00:00] alpha beta\n"


def test_deterministic_output_for_unsorted_input() -> None:
    segs_a = [
        {"start": 60.0, "duration": 5.0, "text": "second"},
        {"start": 0.0, "duration": 5.0, "text": "first"},
        {"start": 30.0, "duration": 5.0, "text": "middle"},
    ]
    segs_b = list(reversed(segs_a))
    out_a = build_marked_text(segs_a, window_seconds=30)
    out_b = build_marked_text(segs_b, window_seconds=30)
    assert out_a == out_b
    # Buckets ordered by index: 0, 30, 60.
    assert out_a == "[00:00] first\n[00:30] middle\n[01:00] second\n"


def test_whisper_segments_with_end_field_work_too() -> None:
    # Whisper verbose_json gives "end" instead of "duration"; both should be
    # accepted because we only read "start" + "text".
    segs = [
        {"start": 0.0, "end": 5.0, "text": "hello"},
        {"start": 30.0, "end": 35.0, "text": "world"},
    ]
    out = build_marked_text(segs, window_seconds=30)
    assert out == "[00:00] hello\n[00:30] world\n"


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
