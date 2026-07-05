"""Tests for llm.qa.clean_answer — strips degenerate tail filler."""

from __future__ import annotations

from src.llm.qa import clean_answer


def test_drops_br_run_tail() -> None:
    text = "Real answer here.\n<br>\n<br>\n<br>\n<br>"
    assert clean_answer(text) == "Real answer here."


def test_drops_br_variants() -> None:
    text = "Answer.\n<br/>\n<br />\n<BR>"
    assert clean_answer(text) == "Answer."


def test_drops_inline_multi_br_line() -> None:
    text = "Answer.\n<br><br><br>"
    assert clean_answer(text) == "Answer."


def test_drops_bare_timecode_dump() -> None:
    text = "Answer text.\n[12:00]\n[05:00]\n[06:30]"
    assert clean_answer(text) == "Answer text."


def test_drops_both_failure_modes() -> None:
    text = "The GPU is a 4060 Ti. [04:30]\n[00:00]\n[01:00]\n<br>\n<br>"
    assert clean_answer(text) == "The GPU is a 4060 Ti. [04:30]"


def test_keeps_inline_timecode_and_text() -> None:
    text = "The card is mentioned [04:30] and has 8 GB."
    assert clean_answer(text) == text


def test_drops_stray_closing_tag_line() -> None:
    text = "Answer here.</blockquote>\nMore.\n</blockquote>"
    # The first line keeps its text; the lone closing tag line is dropped.
    assert clean_answer(text) == "Answer here.</blockquote>\nMore."


def test_keeps_link_line_with_label() -> None:
    text = 'Answer.\n<a href="https://x">Источник</a>'
    assert clean_answer(text) == text


def test_noop_on_clean_answer() -> None:
    text = "A clean answer with no junk at all."
    assert clean_answer(text) == text


def test_empty() -> None:
    assert clean_answer("") == ""
