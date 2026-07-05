"""Tests for the QA degeneration-tail cutoff regex in api.ai.

A small local model sometimes finishes its answer, then loops on filler
(``<br>`` runs, a stray closing tag, bare ``[MM:SS]`` markers) until it hits
``max_tokens``. ``_DEGEN_TAIL_RE`` detects that tail so the stream is cut.
"""

from __future__ import annotations

from src.api.ai import _DEGEN_TAIL_RE


def _hits(text: str) -> bool:
    # Mirrors the call site: match against the last 400 chars.
    return _DEGEN_TAIL_RE.search(text[-400:]) is not None


def test_six_br_run_is_detected() -> None:
    assert _hits("Real answer.\n" + "<br>\n" * 6)


def test_five_br_run_is_not_detected() -> None:
    # Below the 6-unit threshold — a couple of stray tags are not a loop.
    assert not _hits("Real answer.\n" + "<br>\n" * 5)


def test_bare_timecode_dump_is_detected() -> None:
    tail = "".join(f"[{m:02d}:00]\n" for m in range(6))
    assert _hits("Answer about the video.\n" + tail)


def test_mixed_tags_and_timecodes_detected() -> None:
    assert _hits("Answer.\n</blockquote>\n<br>\n[00:00]\n<br>\n[01:00]\n<br>")


def test_clean_answer_not_detected() -> None:
    assert not _hits("A complete, normal answer with no trailing junk.")


def test_inline_timecodes_not_detected() -> None:
    # Timecodes attached to real sentences must not trip the cutoff.
    text = "First point [00:30]. Second point [01:00]. Third [02:00]."
    assert not _hits(text)
