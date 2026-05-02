"""Paragraph/sentence-aware text splitting for map-reduce summarization.

Public surface:
    split_for_summary(text: str, *, target_tokens: int = 4000, overlap_tokens: int = 200) -> list[str]
        Splits primarily on blank lines (paragraphs), then on sentence boundaries
        for paragraphs that exceed target_tokens. Adds overlap by carrying the
        last `overlap_tokens` of each chunk into the next.

Important: must NOT cut inside a [MM:SS] marker — keep markers attached to
their following sentence so map-reduce summaries preserve them.
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
    """Yield paragraph- or sentence-sized segments from `text`.

    A paragraph is kept whole when it fits in target_tokens. Otherwise it is
    broken into sentences. If a single sentence still exceeds target_tokens
    (rare — extremely long unbroken text), it is yielded as-is — the chunk
    packer will give it its own chunk.
    """
    segments: list[str] = []
    for paragraph in _split_into_paragraphs(text):
        if count_tokens(paragraph) <= target_tokens:
            segments.append(paragraph)
            continue
        # Too big — split into sentences
        sentences = _split_into_sentences(paragraph)
        if len(sentences) <= 1:
            # No sentence breaks found; yield the paragraph whole.
            segments.append(paragraph)
        else:
            segments.extend(sentences)
    return segments


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


__all__ = ["split_for_summary"]
