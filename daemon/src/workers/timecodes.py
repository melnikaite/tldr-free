"""Build raw_text with inline ``[MM:SS]`` / ``[HH:MM:SS]`` markers.

This is the SINGLE place where timecode markers are formatted. Both the
YouTube fast path (``youtube-transcript-api``) and the Whisper path
(``mlx-server``) go through ``build_marked_text`` so summaries and Q&A see
one uniform marker format.

Public surface:
    build_marked_text(segments: list[Segment], window_seconds: int) -> str

A ``Segment`` is a plain dict with ``start`` and ``text`` (we don't use
``duration``/``end`` except to decide HH:MM:SS vs MM:SS):
    {"start": float, "duration": float, "text": str}   # youtube-transcript-api
    {"start": float, "end":      float, "text": str}   # whisper (synthesised
                                                       #  as one whole-audio
                                                       #  segment — mlx-server
                                                       #  v1.8 stopped returning
                                                       #  per-segment timestamps)

We only need ``start`` and ``text``. The latest ``start`` decides MM:SS vs
HH:MM:SS for the whole text.

Algorithm — one line per SENTENCE, not per fixed time window:
1. Merge the short caption cues into one flowing string, remembering the
   cue start time behind every character.
2. Split that string on sentence boundaries (``. ? ! …`` followed by space).
3. Emit one line per sentence, marked with the EXACT start time of the cue
   the sentence begins in: ``"[MM:SS] sentence\n"``.
4. Safety cap: a sentence longer than ``window_seconds`` of speech is broken
   at a word boundary — auto-captions are often punctuation-free, and without
   the cap the whole transcript would collapse into one giant ``[00:00]`` line.
   So ``window_seconds`` is a MAX line length, not a fixed bucket size.

The output is deterministic and pure: same input → same output.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

# A bracketed inline marker the LLM emits where a timecode *would* go but none
# exists in the source — e.g. "[Не указано]", "[Not specified]", "[N/A]",
# "[—]". The only legitimate bracket markers in a summary are timecodes
# ([MM:SS] / [HH:MM:SS]), which always contain digits; a bracket with no digit
# is therefore a placeholder. We skip markdown links ("[text](url)") via the
# negative lookahead so genuine links survive untouched.
_PLACEHOLDER_BRACKET = re.compile(r"\[[^\]\d]*\](?!\()")

# A genuine inline timecode marker: ``[MM:SS]`` or ``[HH:MM:SS]``. The negative
# lookahead skips markdown links ("[text](url)") so they survive untouched. Used
# to scrub HALLUCINATED timecodes from a summary whose source has none (web
# pages, PDFs) — see ``strip_all_timecodes``.
_TIMECODE_MARKER = re.compile(r"\[\d{1,2}:\d{2}(?::\d{2})?\](?!\()")

# A sentence terminator (one or more of . ? ! …) followed by whitespace or the
# end of text. The trailing lookahead keeps decimals/version numbers intact
# ("GPT 5.6", "3.5") since their dot is followed by a digit, not a space.
_SENTENCE_END_RE = re.compile(r"[.!?…]+(?=\s|$)")

# Format strings for ``str.format(...)`` so callers / tests can refer to the
# exact formatting without parsing f-strings out of the source.
_MM_SS = "{m:02d}:{s:02d}"
_HH_MM_SS = "{h:d}:{m:02d}:{s:02d}"


def _format_marker(seconds: int, *, use_hours: bool) -> str:
    if use_hours:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return _HH_MM_SS.format(h=h, m=m, s=s)
    m, s = divmod(seconds, 60)
    return _MM_SS.format(m=m, s=s)


def _segment_start(seg: Mapping[str, Any]) -> float:
    start = seg.get("start")
    if start is None:
        return 0.0
    return float(start)


def _segment_text(seg: Mapping[str, Any]) -> str:
    text = seg.get("text") or ""
    return str(text).strip()


def build_marked_text(segments: list[dict[str, Any]], window_seconds: int) -> str:
    """Produce a flat text with one ``[MM:SS]`` marked line per SENTENCE.

    ``segments`` is a list of dicts (or any Mapping) with ``start`` and ``text``
    keys. Each line's marker is the EXACT start time (to the second) of the cue
    the sentence begins in. ``window_seconds`` is a SAFETY CAP — the maximum
    seconds of speech allowed on one line; a sentence longer than that (e.g. a
    punctuation-free auto-caption track) is split at a word boundary so the
    transcript never collapses into a single ``[00:00]`` line.
    """
    if not segments or window_seconds <= 0:
        return ""

    # Sort by start and drop empty cues, then merge into one string while
    # remembering the cue start time behind every character.
    cues = sorted(
        ((_segment_start(s), _segment_text(s)) for s in segments),
        key=lambda c: c[0],
    )
    cues = [(start, text) for start, text in cues if text]
    if not cues:
        return ""

    use_hours = cues[-1][0] >= 3600.0

    buf: list[str] = []
    starts: list[float] = []
    for start, text in cues:
        if buf:
            buf.append(" ")
            starts.append(start)
        buf.append(text)
        starts.extend([start] * len(text))
    merged = "".join(buf)
    n = len(merged)

    lines: list[str] = []
    i = 0
    while i < n:
        while i < n and merged[i].isspace():
            i += 1
        if i >= n:
            break
        start_idx = i
        line_start = starts[i]
        match = _SENTENCE_END_RE.search(merged, i)
        end = match.end() if match else n

        # Safety cap: keep at most window_seconds of speech on one line.
        cap = line_start + window_seconds
        if starts[end - 1] > cap:
            j = start_idx + 1
            while j < end and starts[j] <= cap:
                j += 1
            # Back up to a word boundary so we don't cut mid-word.
            k = j
            while k > start_idx and not merged[k - 1].isspace():
                k -= 1
            end = k if k > start_idx else j

        line = merged[start_idx:end].strip()
        if line:
            seconds = int(math.floor(max(line_start, 0.0)))
            marker = _format_marker(seconds, use_hours=use_hours)
            lines.append(f"[{marker}] {line}")
        i = end

    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def format_segments_as_marked_text(segments: list[dict[str, Any]]) -> str:
    """Emit one ``[MM:SS] text`` line per segment — no bucketing.

    Sibling of ``build_marked_text`` that preserves the source's native
    granularity (1-5 s per line for Whisper / yt-captions) rather than
    grouping into N-second buckets. Used by:
    - ``api/jobs.py`` ``_build_segments_text`` to serve the Transcript
      tab's body.
    - ``workers/translator.py`` to construct the translation source so
      the translated transcript inherits the same fine granularity as
      the original (not the coarse 30 s buckets that ``raw_text`` uses
      for summary / Q&A).

    Returns an empty string when the input is empty or every segment
    has empty text after stripping.
    """
    if not segments:
        return ""

    max_start = max((_segment_start(s) for s in segments), default=0.0)
    use_hours = max_start >= 3600.0

    lines: list[str] = []
    for seg in segments:
        text = _segment_text(seg)
        if not text:
            continue
        start = _segment_start(seg)
        if start < 0:
            start = 0.0
        marker = _format_marker(int(math.floor(start)), use_hours=use_hours)
        lines.append(f"[{marker}] {text}")

    if not lines:
        return ""
    return "\n".join(lines) + "\n"


# Phantom phrases Whisper invents over trailing silence / outro music — it has
# no audio to transcribe so it emits boilerplate it saw a lot in training. We
# strip these ONLY from the tail of the text fed to the summariser (the stored
# transcript keeps them); a phrase mid-text is left alone (could be real).
_TAIL_NOISE_PHRASES = (
    "продолжение следует",
    "спасибо за просмотр",
    "спасибо за внимание",
    "подписывайтесь",
    "ставьте лайк",
    "субтитры",
    "редактор субтитров",
    "до новых встреч",
    "до встречи",
    "всем пока",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "subscribe to",
    "subtitles by",
    "see you next time",
    "transcribed by",
    "amara.org",
)
# How many trailing lines to even consider — bounds the damage if a real line
# happens to contain a phrase.
_TAIL_SCAN_LINES = 4


def _is_tail_noise(line: str) -> bool:
    # Drop a leading ``[MM:SS]`` / ``[HH:MM:SS]`` marker, lowercase, strip
    # punctuation/space so "Продолжение следует..." matches.
    body = re.sub(r"^\s*\[\d{1,2}:\d{2}(?::\d{2})?\]\s*", "", line)
    body = body.strip().lower().strip(" .,!?…\"'»«-—")
    if not body:
        return False
    return any(p in body for p in _TAIL_NOISE_PHRASES)


def strip_transcript_tail_noise(text: str) -> str:
    """Remove Whisper's trailing hallucinated boilerplate from summary input.

    Operates on the LAST few lines only: walks up from the end dropping lines
    that are pure phantom phrases ("Продолжение следует…", "Thanks for
    watching"), stopping at the first real line. Pure (same in → same out) and
    conservative — never touches the middle of the transcript.
    """
    if not text:
        return text
    lines = text.rstrip("\n").split("\n")
    scanned = 0
    while lines and scanned < _TAIL_SCAN_LINES:
        if not lines[-1].strip():
            lines.pop()
            continue
        if _is_tail_noise(lines[-1]):
            lines.pop()
            scanned += 1
            continue
        break
    cleaned = "\n".join(lines)
    return cleaned + "\n" if text.endswith("\n") and cleaned else cleaned


def _tidy_after_bracket_removal(cleaned: str) -> str:
    """Tidy whitespace/separators orphaned by removing a bracket marker.

    A dangling space, a doubled space, or a now-orphaned " — "/"-" separator
    the model put between the text and the marker.
    """
    out_lines: list[str] = []
    for line in cleaned.split("\n"):
        # Collapse runs of spaces/tabs created by the removal.
        line = re.sub(r"[ \t]{2,}", " ", line)
        # Drop a separator the model left dangling before the removed marker,
        # e.g. "key point — " or "key point -".
        line = re.sub(r"[ \t]*[—–-][ \t]*$", "", line)
        out_lines.append(line.rstrip())
    return "\n".join(out_lines)


def strip_timecode_placeholders(text: str) -> str:
    """Remove empty bracket markers the LLM left where no timecode exists.

    Belt-and-suspenders for the summary prompts (which already forbid such
    placeholders): even instructed, a small local model occasionally writes
    "[Не указано]" / "[N/A]" next to a key point that has no timestamp. We
    strip those brackets, then tidy the leftover whitespace.

    Pure: same input → same output. Markdown links and real [MM:SS] markers
    are preserved (see ``_PLACEHOLDER_BRACKET``).
    """
    if not text or "[" not in text:
        return text

    cleaned = _PLACEHOLDER_BRACKET.sub("", text)
    if cleaned == text:
        return text
    return _tidy_after_bracket_removal(cleaned)


def strip_bare_timecode_lines(text: str) -> str:
    """Drop lines that are nothing but timecode markers.

    A small local model sometimes ends a Q&A answer with a dump of every
    ``[MM:SS]`` marker from the transcript — one bare timecode per line, no
    text. Those lines carry no information and look broken. We remove any line
    whose entire content, after removing ``[MM:SS]``/``[HH:MM:SS]`` markers, is
    empty or pure punctuation/whitespace. Lines with real words — including a
    timecode sitting inline next to a sentence — are kept untouched.

    Pure: same input → same output.
    """
    if not text or "[" not in text:
        return text

    out: list[str] = []
    changed = False
    for line in text.split("\n"):
        residue = _TIMECODE_MARKER.sub("", line)
        residue = re.sub(r"[\s\-–—,;:.]+", "", residue)
        if line.strip() and not residue:
            changed = True
            continue
        out.append(line)

    if not changed:
        return text
    # Collapse blank runs the removal may have opened up, then trim.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def strip_all_timecodes(text: str) -> str:
    """Remove ALL ``[MM:SS]`` / ``[HH:MM:SS]`` markers from a summary.

    For non-transcript sources (web pages, PDFs) the source carries no
    timecodes, so any marker the model emitted is hallucinated. The prompt
    already tells it not to, but a small local model occasionally invents
    "[00:42]" next to a key point anyway — this is the deterministic guarantee
    that none leak into the stored summary. We strip the markers, then tidy the
    leftover whitespace / dangling separators.

    Pure: same input → same output. Markdown links survive (see
    ``_TIMECODE_MARKER``).
    """
    if not text or "[" not in text:
        return text

    cleaned = _TIMECODE_MARKER.sub("", text)
    if cleaned == text:
        return text
    return _tidy_after_bracket_removal(cleaned)


__all__ = [
    "build_marked_text",
    "format_segments_as_marked_text",
    "strip_all_timecodes",
    "strip_bare_timecode_lines",
    "strip_timecode_placeholders",
    "strip_transcript_tail_noise",
]
