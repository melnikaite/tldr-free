"""Tests for workers.timecodes — the single source of truth for [MM:SS] markers."""

from __future__ import annotations

from src.workers.timecodes import build_marked_text


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
