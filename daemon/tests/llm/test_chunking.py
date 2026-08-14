"""Tests for llm.chunking.split_for_summary and llm.chunking.pack_lines."""

from __future__ import annotations

import re

from src.llm.chunking import pack_lines, split_for_summary
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


# ---------------------------------------------------------------------------
# pack_lines
# ---------------------------------------------------------------------------


def test_pack_lines_empty_input() -> None:
    assert pack_lines([], target_tokens=100) == []


def test_pack_lines_never_splits_a_line() -> None:
    lines = [f"[{i:02d}:00] line number {i} with some words in it" for i in range(50)]
    groups = pack_lines(lines, target_tokens=50)
    # Every original line appears verbatim in exactly one group, in order.
    flat = [line for group in groups for line in group]
    assert flat == lines


def test_pack_lines_respects_budget() -> None:
    lines = [f"[{i:02d}:00] " + ("word " * 20) for i in range(30)]
    target = 100
    groups = pack_lines(lines, target_tokens=target)
    assert len(groups) > 1
    for group in groups[:-1]:
        # Each non-final group should be close to (not wildly under) budget —
        # the greedy packer stops adding once the NEXT line would overflow,
        # so a group is allowed to sit under budget by up to one line's
        # worth of tokens, but never over by more than a fraction.
        total = sum(count_tokens(line) for line in group)
        assert total <= target * 1.5


def test_pack_lines_oversized_single_line_gets_own_group() -> None:
    huge_line = "word " * 5000  # far bigger than target_tokens
    lines = ["[00:00] short", huge_line, "[00:02] short again"]
    groups = pack_lines(lines, target_tokens=50)
    assert huge_line in [line for group in groups for line in group]
    # The huge line must be alone in its own group — never merged with
    # neighbours, never split.
    huge_group = next(g for g in groups if huge_line in g)
    assert huge_group == [huge_line]


def test_pack_lines_preserves_blank_lines() -> None:
    lines = ["[00:00] a", "", "[00:01] b"]
    groups = pack_lines(lines, target_tokens=1000)
    flat = [line for group in groups for line in group]
    assert flat == lines


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


def _make_marked_transcript(lines: int) -> str:
    """Reproduce the real-world shape from ``workers/timecodes.build_marked_text``:
    one line per sentence, a ``[MM:SS]`` marker at the very start of each line,
    NO blank lines between them. This is the exact shape that made
    ``split_for_summary`` return a single 65k-token chunk in production —
    ``_split_into_paragraphs`` sees one giant paragraph (no blank lines), and
    ``_SENTENCE_RE``'s lookahead never matches because the character right
    after every ``. ``/``! ``/``? `` is always ``[`` (the next marker), not an
    uppercase letter."""
    out = []
    for i in range(lines):
        m, s = divmod(i * 5, 60)
        out.append(f"[{m:02d}:{s:02d}] This is sentence number {i + 1} in a long transcript.")
    return "\n".join(out)


def test_marked_transcript_without_blank_lines_splits_into_multiple_chunks() -> None:
    """Regression for the production incident: a marked transcript with no
    blank lines must still be split into chunks that respect target_tokens —
    NOT returned as a single oversized chunk."""
    text = _make_marked_transcript(2000)
    target = 4000
    overlap = 200
    assert count_tokens(text) > target * 5  # sanity: comfortably over budget

    chunks = split_for_summary(text, target_tokens=target, overlap_tokens=overlap)

    assert len(chunks) > 1, "marked transcript must be split into more than one chunk"
    for ch in chunks:
        # On this marked-transcript shape there are no unsplittable segments
        # (every line packs cleanly), so the only headroom above `target` is
        # the overlap prepended onto the next chunk — not the looser 1.5x
        # allowance other tests give ordinary prose (which can contain a
        # genuinely unsplittable sentence).
        assert count_tokens(ch) <= target + overlap, (
            f"chunk exceeds target_tokens+overlap budget ({count_tokens(ch)} > {target + overlap})"
        )


def test_marked_transcript_markers_not_torn_or_lost() -> None:
    """Every [MM:SS] marker in the marked-transcript shape survives whole,
    and none is dropped, across the waterfall split."""
    text = _make_marked_transcript(2000)
    target = 4000
    chunks = split_for_summary(text, target_tokens=target, overlap_tokens=200)

    for ch in chunks:
        assert ch.count("[") == ch.count("]"), f"unbalanced bracket in chunk:\n{ch[:200]!r}"

    original_markers = re.findall(r"\[\d{1,2}:\d{2}(?::\d{2})?\]", text)
    joined = "\n".join(chunks)
    for m in set(original_markers):
        assert m in joined, f"marker {m} lost during chunking"


def test_single_giant_line_without_spaces_splits_without_looping() -> None:
    """A single line with no spaces near it (one gigantic 'word') must still
    terminate — split by word/space as a last resort, or yielded whole if
    truly unsplittable, but never loop forever."""
    huge_word = "x" * 50_000
    text = f"[00:00] {huge_word}"
    target = 500

    chunks = split_for_summary(text, target_tokens=target, overlap_tokens=50)

    assert chunks  # terminates and returns something
    joined = "".join(chunks)
    assert huge_word in joined


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
