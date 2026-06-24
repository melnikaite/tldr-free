"""Tests for languages.detect_language — best-effort transcript language guess."""

from __future__ import annotations

from src.llm.languages import detect_language


def test_detects_russian() -> None:
    text = (
        "[00:00] Сегодня мы поговорим о том, как работает железнодорожная "
        "система связи в Германии и почему поезда остановились."
    )
    assert detect_language(text) == "ru"


def test_detects_english() -> None:
    text = (
        "[00:00] Today we are going to talk about how the railway communication "
        "system works and why all the trains across the country stopped."
    )
    assert detect_language(text) == "en"


def test_strips_timecode_markers_before_detect() -> None:
    # Markers must not derail detection — plenty of real words remain.
    text = "[0:00:00] это [0:00:05] полностью [0:00:10] русский [0:00:15] текст про поезда"
    assert detect_language(text) == "ru"


def test_too_short_returns_none() -> None:
    assert detect_language("[00:00] hi") is None
    assert detect_language("") is None
    assert detect_language(None) is None


def test_markers_only_returns_none() -> None:
    assert detect_language("[00:00] [00:30] [01:00]") is None
