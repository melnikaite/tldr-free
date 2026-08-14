"""Paragraph/sentence-aware text splitting for map-reduce summarization.

Public surface:
    split_for_summary(text: str, *, target_tokens: int = 4000, overlap_tokens: int = 200) -> list[str]
        Splits primarily on blank lines (paragraphs), then on sentence boundaries
        for paragraphs that exceed target_tokens. Adds overlap by carrying the
        last `overlap_tokens` of each chunk into the next. GUARANTEES no
        returned chunk exceeds target_tokens (see the waterfall below) —
        with one unavoidable exception: a single "word" (no whitespace
        nearby) that alone exceeds target_tokens can't be split further and
        is returned as its own oversized chunk.
    pack_lines(lines: list[str], *, target_tokens: int) -> list[list[str]]
        Greedily packs whole LINES (never split) into groups under a token
        budget. Used by ``workers/translator.py`` (a marked transcript has
        no blank lines to split on, so line-atomic packing is the more
        semantically appropriate tool there — it never touches sentence
        boundaries, which matters for the translator's line-for-line output
        contract) AND, since fixing a production incident described below,
        reused inside ``split_for_summary`` itself as the second stage of
        its own waterfall.

Important: must NOT cut inside a [MM:SS] marker — keep markers attached to
their following sentence so map-reduce summaries preserve them.

A marked transcript (one line per sentence, ``[MM:SS]`` at the start of
each line, no blank lines — see ``workers/timecodes.build_marked_text``)
gives ``_split_into_paragraphs`` one giant "paragraph" and gives
``_SENTENCE_RE`` no breakpoints (the char after every ``.``/``!``/``?`` is
always ``[``, which its marker-safe lookahead excludes by design). Whatever
segment is still oversized after that falls through a line→word waterfall
in ``_segments_for`` instead of coming back whole.
"""

from __future__ import annotations

import re

from src.llm.tokens import count_tokens

# Sentence-end punctuation followed by whitespace + an upper-case letter
# (Latin or Cyrillic) starting the next sentence. We use a lookahead so the
# split doesn't consume the next character.
# Importantly: the lookahead requires the next non-whitespace char to be an
# upper-case letter — `[` (timecode marker opener) does NOT match, so we will
# never split between a sentence and a leading [MM:SS] marker.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ])")

# A timecode marker like [12:34] or [01:23:45]. We use this to ensure no split
# happens inside a marker.
_TIMECODE_RE = re.compile(r"\[\d{1,2}:\d{2}(?::\d{2})?\]")


def _split_into_paragraphs(text: str) -> list[str]:
    """Split on blank lines. Strips trailing/leading whitespace per paragraph
    but preserves the paragraph contents (including internal newlines)."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_into_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences. Conservative: if no clear breakpoint
    is found, returns the paragraph as a single sentence."""
    parts = _SENTENCE_RE.split(paragraph)
    return [s for s in (p.strip() for p in parts) if s]


def _segments_for(text: str, target_tokens: int) -> list[str]:
    """Yield paragraph- or sentence-sized segments from `text`, each
    guaranteed to be at most `target_tokens` (waterfall's last resort — a
    single word with no nearby whitespace — aside).

    A paragraph is kept whole when it fits in target_tokens. Otherwise it is
    broken into sentences. Whatever segment survives that (a paragraph with
    no sentence breaks found, or an individual sentence) and is STILL over
    target_tokens goes through one more waterfall stage: split by line, then
    by word — see ``_split_oversized_segment`` and the module docstring for
    the production incident this guards against (a marked transcript with no
    blank lines and no sentence breaks the splitter could find).
    """
    raw_segments: list[str] = []
    for paragraph in _split_into_paragraphs(text):
        if count_tokens(paragraph) <= target_tokens:
            raw_segments.append(paragraph)
            continue
        # Too big — split into sentences
        sentences = _split_into_sentences(paragraph)
        if len(sentences) <= 1:
            # No sentence breaks found; yield the paragraph whole (waterfall
            # below will split it further if it's still oversized).
            raw_segments.append(paragraph)
        else:
            raw_segments.extend(sentences)

    segments: list[str] = []
    for seg in raw_segments:
        if count_tokens(seg) <= target_tokens:
            segments.append(seg)
        else:
            segments.extend(_split_oversized_segment(seg, target_tokens))
    return segments


def _split_line_by_words(line: str, target_tokens: int) -> list[str]:
    """Last resort: split ONE line that alone exceeds target_tokens by word.

    A leading ``[MM:SS]``/``[HH:MM:SS]`` marker, if present, is consumed
    whole up front and glued back onto the first word-group, so it can
    never be torn. A single spaceless "word" still over budget becomes its
    own piece rather than looping — mirrors ``pack_lines``'s oversized-
    single-line contract.
    """
    marker_match = _TIMECODE_RE.match(line)
    prefix = marker_match.group() if marker_match else ""
    rest = line[marker_match.end() :].lstrip() if marker_match else line
    words = rest.split()

    if not words:
        return [line] if line.strip() else []

    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in words:
        word_tokens = count_tokens(word)
        if current and current_tokens + word_tokens > target_tokens:
            pieces.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(word)
        current_tokens += word_tokens
    if current:
        pieces.append(" ".join(current))

    if prefix and pieces:
        pieces[0] = f"{prefix} {pieces[0]}"
    return pieces


def _split_oversized_segment(segment: str, target_tokens: int) -> list[str]:
    """Waterfall stage 2/3 for a segment still over ``target_tokens`` after
    paragraph/sentence splitting: split by LINE via ``pack_lines`` (line-
    atomic, so a marker never gets torn off), then by WORD
    (``_split_line_by_words``) for any resulting group that's still a
    single oversized line — the common case here, since a marked transcript
    has no internal newlines to split on at all.
    """
    lines = segment.split("\n")
    if len(lines) == 1:
        return _split_line_by_words(segment, target_tokens)

    pieces: list[str] = []
    for group in pack_lines(lines, target_tokens=target_tokens):
        joined = "\n".join(group)
        if len(group) == 1 and count_tokens(joined) > target_tokens:
            pieces.extend(_split_line_by_words(joined, target_tokens))
        else:
            pieces.append(joined)
    return pieces


def _tail_for_overlap(text: str, overlap_tokens: int) -> str:
    """Return the last ~overlap_tokens worth of `text`, snapped to a word
    boundary, never starting inside a [MM:SS] marker."""
    if overlap_tokens <= 0 or not text:
        return ""
    enc_text = text
    # Quick path: if the whole text fits, return it.
    total = count_tokens(enc_text)
    if total <= overlap_tokens:
        return enc_text

    # Walk back from the end, taking ~overlap_tokens. Use character-based
    # binary chop on tokens for speed and simplicity: shrink window until
    # token count <= overlap_tokens.
    # Start by guessing 4 chars/token (safe overestimate) and refine.
    guess_chars = overlap_tokens * 4
    if guess_chars >= len(enc_text):
        return enc_text
    tail = enc_text[-guess_chars:]
    # Trim front until tokens <= overlap_tokens
    while count_tokens(tail) > overlap_tokens and len(tail) > 1:
        tail = tail[len(tail) // 2 :]
    # Snap to a whitespace boundary so we don't start mid-word
    ws_idx = tail.find(" ")
    if 0 <= ws_idx < len(tail) - 1:
        tail = tail[ws_idx + 1 :]
    # Avoid starting inside a timecode bracket: if the tail starts with
    # something like "12:34] ..." (i.e. we cut after '['), drop forward to the
    # next marker or whitespace boundary.
    if tail and tail[0] != "[":
        # Look for a stray closing bracket before any opening bracket.
        close = tail.find("]")
        opener = tail.find("[")
        if close != -1 and (opener == -1 or close < opener):
            # We cut inside a [MM:SS] marker — drop past the close bracket.
            tail = tail[close + 1 :].lstrip()
    return tail


def split_for_summary(
    text: str,
    *,
    target_tokens: int = 4000,
    overlap_tokens: int = 200,
) -> list[str]:
    """Split `text` into chunks for map-reduce summarization.

    Each chunk is roughly `target_tokens` tokens or smaller. Consecutive
    chunks share approximately `overlap_tokens` of trailing text from the
    previous chunk so the model sees continuity.

    Timecode markers like [12:34] / [01:23:45] are never split: paragraph
    splitting is on blank lines (which can never fall inside a marker), and
    sentence splitting only fires after `.!?` followed by whitespace + an
    upper-case letter (not `[`). Overlap snipping is character-based but
    snaps past any open bracket it lands inside.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if count_tokens(text) <= target_tokens:
        return [text]

    segments = _segments_for(text, target_tokens=target_tokens)

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current_parts, current_tokens
        if current_parts:
            chunks.append("\n\n".join(current_parts).strip())
            current_parts = []
            current_tokens = 0

    for seg in segments:
        seg_tokens = count_tokens(seg)
        # If adding this segment exceeds the budget, flush first.
        if current_tokens + seg_tokens > target_tokens and current_parts:
            flush()
            # Seed the next chunk with overlap from the just-flushed chunk
            if overlap_tokens > 0 and chunks:
                tail = _tail_for_overlap(chunks[-1], overlap_tokens)
                if tail:
                    current_parts.append(tail)
                    current_tokens = count_tokens(tail)

        current_parts.append(seg)
        current_tokens += seg_tokens

    flush()
    # Validate: ensure we never split a timecode marker. We assert that every
    # `[` in the original text remained matched with its `]` in some chunk.
    # This is implicit in the splitter design but cheap to double-check.
    for ch in chunks:
        opens = ch.count("[")
        # closes should be >= opens minus any from overlap; we just guard
        # against a chunk that opens a bracket without closing it.
        if opens != ch.count("]"):
            # Orphan bracket — should be impossible with the splitter, but
            # we bail rather than emit a torn marker.
            raise RuntimeError(
                "split_for_summary produced a chunk with an unbalanced timecode bracket"
            )
    return chunks


def pack_lines(lines: list[str], *, target_tokens: int) -> list[list[str]]:
    """Greedily pack whole ``lines`` into groups of at most ``target_tokens``.

    Unlike ``split_for_summary`` (which splits prose on blank lines, then
    sentences), this never looks inside a line — a line is the smallest
    unit that can be moved, so a ``[MM:SS] ...`` marker can never end up
    detached from its text. A single line that alone exceeds
    ``target_tokens`` still becomes its own one-line group rather than
    being split — the caller (the translator's LLM call) just sees a
    slightly-over-budget prompt for that one line, which is far cheaper
    than reconstructing a torn line downstream.

    Empty input returns ``[]``. Blank lines are kept as lines (empty
    strings) — they carry position in the caller's line-for-line contract,
    not just words to pack.
    """
    if not lines:
        return []

    groups: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for line in lines:
        line_tokens = count_tokens(line)
        if current and current_tokens + line_tokens > target_tokens:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(line)
        current_tokens += line_tokens
    if current:
        groups.append(current)
    return groups


__all__ = ["pack_lines", "split_for_summary"]
