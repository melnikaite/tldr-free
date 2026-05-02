"""Tests for llm.chunking.split_for_summary."""

from __future__ import annotations

import re

from src.llm.chunking import split_for_summary
from src.llm.tokens import count_tokens

_TIMECODE_RE = re.compile(r"\[(\d{1,2}:)?\d{1,2}:\d{2}\]")


def _make_text(paragraphs: int, words_per_paragraph: int) -> str:
    """Generate plausible Russian-ish prose so the tokenizer behaves
    similarly to real input."""
    sentence = (
        "В этом параграфе рассказывается о важной теме, которую необходимо "
        "подробно рассмотреть, чтобы получить полное понимание материала. "
    )
    para = sentence * max(1, words_per_paragraph // 8)
    return "\n\n".join(f"Параграф {i + 1}. {para}" for i in range(paragraphs))


def test_short_text_returns_single_chunk() -> None:
    text = "Одно короткое предложение. Второе предложение."
    chunks = split_for_summary(text, target_tokens=4000, overlap_tokens=200)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_empty_text_returns_empty_list() -> None:
    assert split_for_summary("") == []
    assert split_for_summary("   \n  \n  ") == []


def test_chunks_respect_target_tokens() -> None:
    text = _make_text(paragraphs=40, words_per_paragraph=120)
    target = 800
    chunks = split_for_summary(text, target_tokens=target, overlap_tokens=80)
    assert len(chunks) > 1
    for ch in chunks:
        # Allow some headroom (1.5x): a single oversize paragraph or sentence
        # may exceed the target, but typical chunks should hover near it.
        assert count_tokens(ch) <= int(target * 1.5)
    # On average chunks should be reasonably full.
    avg = sum(count_tokens(ch) for ch in chunks) / len(chunks)
    assert avg > target * 0.3


def test_chunks_have_overlap() -> None:
    text = _make_text(paragraphs=30, words_per_paragraph=120)
    chunks = split_for_summary(text, target_tokens=600, overlap_tokens=120)
    assert len(chunks) >= 2
    # The tail of chunk N should appear at the head of chunk N+1.
    # Use last 30 chars of chunk N as a probe.
    for i in range(len(chunks) - 1):
        prev = chunks[i].rstrip()
        nxt = chunks[i + 1]
        if len(prev) < 30:
            continue
        # Look for a meaningful overlap: at least one >=10-char run shared.
        for span in range(60, 9, -10):
            if len(prev) < span:
                continue
            probe = prev[-span:]
            if probe in nxt:
                break
        else:
            raise AssertionError(
                f"No overlap detected between chunk {i} and {i + 1}"
            )


def test_timecode_markers_not_split() -> None:
    """Timecode markers like [12:34] and [01:23:45] must remain whole."""
    # Build a long transcript-style text with markers at every paragraph head.
    paragraphs = []
    for i in range(60):
        mm = f"{(i // 2):02d}:{(i % 2) * 30:02d}"
        paragraphs.append(
            f"[{mm}] Это сегмент номер {i + 1}, в котором обсуждается "
            "очень важная тема, повторяющаяся для увеличения длины. " * 3
        )
    # Sprinkle in a couple of [HH:MM:SS] markers
    paragraphs[15] = "[01:23:45] Длинное видео — здесь обсуждается важная тема, " + paragraphs[15]
    paragraphs[40] = "[02:03:04] Другой длинный таймкод, " + paragraphs[40]

    text = "\n\n".join(paragraphs)
    chunks = split_for_summary(text, target_tokens=600, overlap_tokens=80)

    # No chunk has unbalanced brackets.
    for ch in chunks:
        assert ch.count("[") == ch.count("]"), (
            f"Chunk has unbalanced brackets:\n{ch!r}"
        )

    # Every original marker appears at least once across all chunks.
    original_markers = _TIMECODE_RE.findall(text)
    # _TIMECODE_RE.findall returns tuples of groups; recapture full matches:
    full_markers = re.findall(r"\[\d{1,2}:\d{2}(?::\d{2})?\]", text)
    joined_chunks = "\n\n".join(chunks)
    for m in full_markers:
        assert m in joined_chunks, f"Timecode marker {m} lost in chunking"
    assert original_markers  # sanity: regex actually fired


def test_marker_kept_with_following_sentence() -> None:
    """A marker at the start of a sentence should never be orphaned across
    chunk boundaries (i.e. the chunk that closes does not end with `[12:34]`
    leaving the next chunk to start with the body)."""
    paragraphs = [
        f"[{i:02d}:00] Содержание сегмента {i}. " + "Очень длинный текст. " * 80
        for i in range(20)
    ]
    text = "\n\n".join(paragraphs)
    chunks = split_for_summary(text, target_tokens=500, overlap_tokens=50)
    for ch in chunks:
        stripped = ch.strip()
        # A chunk should never END with just a timecode marker (which would
        # mean the body got pushed to the next chunk).
        assert not re.search(r"\[\d{1,2}:\d{2}(?::\d{2})?\]\s*$", stripped), (
            f"Chunk ends with a stray marker:\n{ch!r}"
        )
