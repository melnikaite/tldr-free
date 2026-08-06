"""Build raw_text with inline ``[MM:SS]`` / ``[HH:MM:SS]`` markers.

This is the SINGLE place where timecode markers are formatted. Both the
YouTube fast path (``youtube-transcript-api``) and the Whisper path
(``mlx-server``) go through ``build_marked_text`` so summaries and Q&A see
one uniform marker format.

Public surface:
    build_marked_text(segments: list[Segment], window_seconds: int) -> str

Also in this module (not part of the marker-formatting algorithm above, but
kept alongside the other strip_*/cap_* transcript-hygiene helpers for the
same reason strip_transcript_tail_noise lives here):
    collapse_repeated_segments(segments) -> list[Segment]   # Whisper repetition-loop collapse
    cap_markers_per_line(text, max_markers=1) -> str        # cap markers per summary line
    cap_markers_in_stream(stream, max_markers=1) -> AsyncIterator[str]  # streaming wrapper for the above

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

import difflib
import math
import re
from collections.abc import AsyncIterator, Mapping
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


def format_timecode(seconds: float) -> str:
    """Render ONE timestamp as ``MM:SS`` (or ``H:MM:SS`` past the hour
    mark) — rounds to the nearest second and clamps negative input to 0.

    Public wrapper around ``_format_marker`` for callers that format a
    single, standalone timestamp outside of a full transcript (unlike
    ``build_marked_text``, which marks every line of one). Used by
    ``llm/qa.py`` (LOOK-step stage messages) and ``api/jobs.py`` (the
    ``GET /jobs/{id}/moments`` / ``POST /jobs/{id}/frames`` on-demand frame
    affordance) so both render a deixis moment's timestamp identically.
    """
    total = max(0, int(round(seconds)))
    return _format_marker(total, use_hours=total >= 3600)


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


# ---------------------------------------------------------------------------
# Whisper repetition-loop collapse
# ---------------------------------------------------------------------------
#
# Whisper-family models occasionally fall into a decoding loop over noisy /
# silent audio and emit the same sentence (or a slowly-drifting near-copy of
# it) over and over, one "segment" per repeat, each a second or so apart.
# Measured on real jobs in this DB: one job had 576 marked lines with 499
# duplicates (87%), including a single run of 291 CONSECUTIVE identical
# "I'm not sure if I'm doing that right." segments; another had a run of 57
# consecutive "Ja." segments. youtube-transcript-api captions never show this
# (0-5% duplicate lines, longest run 1-2) — it's a Whisper-only failure mode,
# so this collapse is only ever applied to the Whisper path (see
# ``transcribe.transcribe_audio``), never to YouTube caption segments.

# A segment whose text is at or under this length is "short" — a word or a
# brief phrase ("Ja.", "Yes.", "Mm-hmm."). Real dialogue legitimately repeats
# these several times in a row (call-and-response, stutter, emphasis), so
# short segments get a much more permissive run-length threshold than long
# ones before we suspect a hallucination loop.
_SHORT_SEGMENT_MAX_CHARS = 12

# A run of short segments survives up to this many consecutive repeats
# (a real "Ja." "Ja." "Ja." "Ja." "Ja." exchange, 3-5x, must not collapse)
# before it's treated as a hallucination loop. Chosen comfortably above the
# 3-5x legitimate-repeat range and far below the measured 57-repeat loop.
_SHORT_SEGMENT_RUN_THRESHOLD = 6

# A run of long segments (full sentences/phrases) collapses starting at just
# 2 consecutive EXACT repeats. Genuine speech essentially never repeats a
# full sentence verbatim back-to-back — the measured hallucination loops were
# 291 and 57 repeats, and even a "mere" 2-3x verbatim repeat of a long
# sentence in a row is essentially never real speech either — so unlike the
# short-segment case there's no meaningful legitimate-repeat range to
# protect, and we collapse aggressively. This threshold governs EXACT-match
# runs only; near-duplicate (non-exact) runs go through
# ``_NEAR_DUP_RUN_THRESHOLD`` instead (see below) — a lower bar here would
# also apply to those and start misfiring on ordinary dialogue.
_LONG_SEGMENT_RUN_THRESHOLD = 1

# difflib.SequenceMatcher ratio (0..1) above which two segment texts are
# considered "the same" for run-detection, on top of exact-match. This
# catches Whisper's incremental hallucination drift, e.g. (measured shape):
#   "He was a young man who was very interested in the world of science..."
#   "He was also a young man who was interested in the world of science..."
#   "He was also interested in the world of science and technology."
# Exact-only matching would silently miss this real, measured case — the
# text mutates slightly each repeat while the sentence stays recognizably
# the same. Measured on the drift trio above: 0.94 and 0.86. Two genuinely
# different but structurally-similar sentences ("First sentence here." vs.
# "Second sentence here.") can still land at ~0.73 purely from a shared
# trailing phrase — 0.82 sits comfortably above that false-positive case and
# comfortably below both measured drift-trio ratios, so it catches the real
# hallucination drift without merging coincidentally-similar prose.
_NEAR_DUP_RATIO = 0.82

# A near-duplicate (ratio-matched, non-exact) run must reach this many
# CONSECUTIVE segments before it's treated as a hallucination loop — a
# near-dup PAIR (run length 2) never collapses on its own. Measured
# real-world false positive this guards against: a genuine two-speaker
# exchange, ">> That's locked in?" followed by ">> It's locked in.", scores
# 0.84 on ``_NEAR_DUP_RATIO`` — just above the threshold — but is a question
# and its confirmation, not hallucination; collapsing it would delete real
# content. Applying ``_LONG_SEGMENT_RUN_THRESHOLD`` (1) to near-dup runs
# would collapse that pair immediately, which is why near-dup runs use this
# separate, stricter gate instead. The measured drift trio (see
# ``_NEAR_DUP_RATIO`` above) is 3 consecutive segments, so this threshold
# still catches it while leaving a mere pair alone.
_NEAR_DUP_RUN_THRESHOLD = 3


def _segments_are_near_duplicates(a: str, b: str) -> bool:
    if a == b:
        return True
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= _NEAR_DUP_RATIO


def collapse_repeated_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse consecutive Whisper hallucination-loop repeats to one segment.

    Walks the segments once, grouping CONSECUTIVE runs whose texts are
    exact-or-near-duplicates of their immediate predecessor (see
    ``_segments_are_near_duplicates``). A run collapses to its FIRST segment
    (start/end/text kept as-is; the rest of the run is dropped) once its
    length exceeds a threshold — which threshold depends on what kind of run
    it is:

    - A run where EVERY adjacent pair is an EXACT text match scales with
      segment length as before: short segments (``_SHORT_SEGMENT_MAX_CHARS``
      or under) tolerate up to ``_SHORT_SEGMENT_RUN_THRESHOLD`` consecutive
      repeats (real dialogue can legitimately repeat "Ja." a handful of
      times); longer segments collapse starting at
      ``_LONG_SEGMENT_RUN_THRESHOLD`` + 1 repeats, since a full sentence
      repeating verbatim back-to-back is essentially never real speech.
    - A run containing at least one NEAR-duplicate (non-exact,
      ratio-matched) adjacent pair instead requires
      ``_NEAR_DUP_RUN_THRESHOLD`` consecutive segments before it collapses,
      regardless of segment length. This is deliberately stricter than the
      exact-match rule: near-matching is fuzzy enough that a mere PAIR is
      often ordinary dialogue (a question immediately followed by a
      structurally-similar confirmation), not hallucination drift — see
      ``_NEAR_DUP_RUN_THRESHOLD`` for the measured real-world case this
      prevents from being deleted.

    Only CONSECUTIVE runs are ever collapsed — a phrase that recurs later in
    the transcript, separated by other content, is untouched; this is not a
    global dedup.

    Whisper-only: intentionally NOT applied to YouTube caption segments
    (measured clean, 0-5% duplicate lines) — see ``transcribe.transcribe_audio``
    for the hook point.

    Pure: same input -> same output. Segments are returned as-is (not copied)
    so identity-sensitive callers see the original dicts for kept segments.
    """
    if not segments:
        return list(segments)

    texts = [_segment_text(s) for s in segments]
    out: list[dict[str, Any]] = []
    n = len(segments)
    i = 0
    while i < n:
        run_end = i + 1
        all_exact = True
        while run_end < n and _segments_are_near_duplicates(
            texts[run_end - 1], texts[run_end]
        ):
            if texts[run_end - 1] != texts[run_end]:
                all_exact = False
            run_end += 1
        run_len = run_end - i
        if all_exact:
            threshold = (
                _SHORT_SEGMENT_RUN_THRESHOLD
                if len(texts[i]) <= _SHORT_SEGMENT_MAX_CHARS
                else _LONG_SEGMENT_RUN_THRESHOLD
            )
            collapse = run_len > threshold
        else:
            collapse = run_len >= _NEAR_DUP_RUN_THRESHOLD
        if collapse:
            out.append(segments[i])
        else:
            out.extend(segments[i:run_end])
        i = run_end
    return out


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


# ---------------------------------------------------------------------------
# Marker-per-line cap
# ---------------------------------------------------------------------------
#
# The LLM sometimes attaches every timecode it saw near a key point to one
# summary bullet instead of picking one, e.g. a line ending
# "[07:55] [08:05] [08:09] [08:29] [08:31] [08:33] [08:37] [08:40] [08:45]" —
# 9 clickable links with no way to tell which is relevant. Prompt wording is
# off the table here (two prior attempts backfired: fabricated timecodes on
# marker-less sources, or zero timecodes on marker-bearing ones), so this is
# enforced deterministically in code instead.

# Default cap: keep exactly one marker per line — the FIRST (earliest) one.
# Even capping at 2 leaves the reported worst case (9 markers) ambiguous
# ("which of these two?"); the earliest marker is always the single most
# defensible seek target (closest to where the point actually starts in the
# source), so we keep exactly one. Named constant so it's trivially
# adjustable without touching call sites.
_MAX_MARKERS_PER_LINE = 1


def cap_markers_per_line(text: str, max_markers: int = _MAX_MARKERS_PER_LINE) -> str:
    """Keep only the first ``max_markers`` ``[MM:SS]`` markers on each line.

    Splits ``text`` into lines; for any line carrying more than
    ``max_markers`` timecode markers, keeps the leftmost (earliest-occurring)
    ones and removes the rest, then tidies the whitespace/separators the
    removal leaves behind (mirrors ``strip_all_timecodes`` /
    ``strip_timecode_placeholders``, reusing ``_tidy_after_bracket_removal``).

    No-op by construction on:
    - text with no markers at all (PDFs/web pages are untouched), and
    - any line with ``<= max_markers`` markers (the overwhelmingly common
      case — most lines carry zero or one marker already).

    Pure: same input -> same output. Markdown links survive (see
    ``_TIMECODE_MARKER``).
    """
    if not text or "[" not in text:
        return text

    out_lines: list[str] = []
    changed = False
    for line in text.split("\n"):
        matches = list(_TIMECODE_MARKER.finditer(line))
        if len(matches) <= max_markers:
            out_lines.append(line)
            continue
        changed = True
        keep_end = matches[max_markers - 1].end() if max_markers > 0 else 0
        pieces: list[str] = [line[:keep_end] if max_markers > 0 else ""]
        last = keep_end
        for m in matches[max_markers:]:
            pieces.append(line[last : m.start()])
            last = m.end()
        pieces.append(line[last:])
        out_lines.append("".join(pieces))

    if not changed:
        return text
    return _tidy_after_bracket_removal("\n".join(out_lines))


# A complete timecode marker body, anchored (mirrors ``_TIMECODE_MARKER`` but
# without its trailing lookahead — the streaming state machine checks the
# "not followed by (" condition itself, one character at a time, since that
# next character may not have arrived yet).
_TIMECODE_BODY_FULL = re.compile(r"^\[\d{1,2}:\d{2}(?::\d{2})?\]$")

# A still-open (no closing "]" yet) prefix that COULD still grow into a full
# marker body: "[", digits, an optional ":"+digits repeated up to twice
# (MM:SS or HH:MM:SS). Anything that stops matching this is definitely not
# becoming a marker and the held-back text is flushed as plain literal text.
_TIMECODE_OPEN_PREFIX = re.compile(r"^\[\d{0,2}(:\d{0,2}(:\d{0,2})?)?$")

# Defensive cap on how long an unresolved candidate is allowed to grow before
# we give up on it — "[HH:MM:SS]" is 10 chars, so nothing legitimate is ever
# this long while still open. ``_TIMECODE_OPEN_PREFIX`` already bounds every
# real candidate to 9 chars before its closing "]", so this is redundant
# belt-and-suspenders, not load-bearing — it documents the "bounded by marker
# length, not line length" invariant explicitly rather than relying solely on
# the regex shape.
_MAX_MARKER_CANDIDATE_LEN = 11


class _MarkerCapState:
    """Per-stream state machine backing ``cap_markers_in_stream``.

    Holds back text only while it MIGHT be a timecode marker (bounded to
    ~10-11 chars — see ``_MAX_MARKER_CANDIDATE_LEN``), never a whole line.
    Ordinary text is returned from ``feed()`` immediately, unbuffered — with
    one exception: a run of spaces/tabs is held (``ws_hold``, itself bounded —
    it can only ever precede a ``[``, never grow across a whole line) just
    long enough to see what follows it. If a ``[`` follows and that bracket
    resolves to a marker that gets DROPPED (over the per-line cap), the held
    whitespace is discarded along with it — so a dropped marker never leaves
    a leftover double space (or worse, two-spaces-before-newline, which
    Markdown renders as a hard ``<br>``) behind. Any other outcome (kept
    marker, markdown link, disqualified non-marker bracket, or plain text)
    flushes the held whitespace verbatim first. This keeps
    ``cap_markers_in_stream`` and ``cap_markers_per_line`` producing
    byte-identical output for the same input — see the equivalence test in
    ``test_timecodes.py``.
    """

    __slots__ = (
        "max_markers",
        "line_count",
        "candidate",
        "awaiting_lookahead",
        "ws_hold",
    )

    def __init__(self, max_markers: int) -> None:
        self.max_markers = max_markers
        self.line_count = 0
        self.candidate = ""
        self.awaiting_lookahead = False
        self.ws_hold = ""

    def _take_ws(self) -> str:
        """Pop and return the held whitespace run, clearing it."""
        ws = self.ws_hold
        self.ws_hold = ""
        return ws

    def feed(self, char: str) -> str:
        """Feed one character; return the text (possibly empty) now safe to emit."""
        if self.awaiting_lookahead:
            # `candidate` holds a fully-formed "[MM:SS]"/"[HH:MM:SS]" body;
            # `char` is the ONE extra character needed to know whether it's
            # a markdown link ("[01:30](url)") or a genuine timecode marker
            # — mirrors ``_TIMECODE_MARKER``'s ``(?!\()`` negative lookahead.
            marker = self.candidate
            self.candidate = ""
            self.awaiting_lookahead = False
            if char == "(":
                # Markdown link: pass the held whitespace and the bracket
                # through untouched, uncounted, unstripped, then re-process
                # `char` normally.
                return self._take_ws() + marker + self.feed(char)
            if self.line_count >= self.max_markers:
                # Genuine marker, but the line is already at its cap — drop
                # the marker AND the whitespace that only served to separate
                # it from what came before; `char` still needs handling.
                self.ws_hold = ""
                return self.feed(char)
            self.line_count += 1
            return self._take_ws() + marker + self.feed(char)

        if self.candidate:
            trial = self.candidate + char
            if _TIMECODE_BODY_FULL.match(trial):
                self.candidate = trial
                self.awaiting_lookahead = True
                return ""
            if (
                len(trial) <= _MAX_MARKER_CANDIDATE_LEN
                and _TIMECODE_OPEN_PREFIX.match(trial)
            ):
                self.candidate = trial
                return ""
            # Disqualified — this was never becoming a marker. Flush the
            # held whitespace and the buffered text as literal, then
            # re-process `char` fresh (it may itself start a new candidate,
            # e.g. back-to-back brackets).
            flushed = self.candidate
            self.candidate = ""
            return self._take_ws() + flushed + self.feed(char)

        if char in (" ", "\t"):
            # Hold — we don't know yet whether this precedes a "[" whose
            # marker will be dropped (in which case this run must vanish
            # too). Bounded: flushed the instant anything but another
            # space/tab/"[" follows.
            self.ws_hold += char
            return ""
        if char == "[":
            self.candidate = char
            return ""
        if char == "\n":
            # New line — markers seen so far no longer count against it.
            self.line_count = 0
        return self._take_ws() + char

    def flush(self) -> str:
        """Resolve whatever's left buffered when the upstream stream ends."""
        if self.awaiting_lookahead:
            # Nothing more can follow, so the "(?!\\()" lookahead condition is
            # satisfied by default — this is a genuine, final marker.
            marker = self.candidate
            self.candidate = ""
            self.awaiting_lookahead = False
            if self.line_count >= self.max_markers:
                self.ws_hold = ""
                return ""
            self.line_count += 1
            return self._take_ws() + marker
        if self.candidate:
            # An incomplete/still-ambiguous candidate can never complete now.
            leftover = self.candidate
            self.candidate = ""
            return self._take_ws() + leftover
        # Nothing but possibly a trailing, never-resolved whitespace run.
        return self._take_ws()


async def cap_markers_in_stream(
    stream: AsyncIterator[str], *, max_markers: int = _MAX_MARKERS_PER_LINE,
) -> AsyncIterator[str]:
    """Wrap a raw LLM delta stream, capping ``[MM:SS]`` markers per line live.

    Both summary streaming call sites (``pipeline._summarize_and_finish`` and
    ``runner``'s inline whisper-worker loop) publish each delta AND accumulate
    it into the final stored ``summary_md`` — the same text. If capping only
    ran once on the fully-accumulated text at the end, a viewer watching the
    stream would see markers pile up past the cap in real time, then the
    display would visibly "snap" down to the capped version when the ``done``
    event fires. This wrapper avoids that: what's published as a delta and
    what's accumulated into the stored summary are always the literal same
    capped text, at every point in the stream, not just at the end.

    GRANULARITY — this holds text back only long enough to recognize a
    ``[MM:SS]``-shaped bracket (bounded to ~10-11 chars, see
    ``_MarkerCapState``/``_MAX_MARKER_CANDIDATE_LEN``), NOT until a whole
    line completes. An earlier version of this wrapper buffered whole lines,
    which regressed real streaming latency badly: measured real summaries
    have their LONGEST line as the very FIRST thing the model generates (the
    "## Обзор"/Overview paragraph, 613-706 chars in practice) — with
    line-level buffering the side panel sat empty for 3-5s (gemma) to 7-8s
    (qwen3-vl) at the most-watched moment of the whole UX, then dumped a wall
    of text at once. Marker-level holdback keeps ordinary text streaming
    char-by-char/token-by-token exactly as before; only text that might be a
    timecode marker is ever delayed, and only until it resolves (usually a
    handful of characters later). A markdown-link-shaped bracket
    ("[01:30](url)") is recognized via one extra lookahead character and
    passed through untouched, uncounted — mirroring ``_TIMECODE_MARKER``'s
    ``(?!\\()`` guard.

    INVARIANT — equivalent to ``cap_markers_per_line``: concatenating every
    piece this yields for a given input produces the SAME text
    ``cap_markers_per_line(text, max_markers=max_markers)`` would produce for
    that input as a whole. In particular, when a marker beyond the cap is
    dropped, the whitespace run immediately before it is dropped too (see
    ``_MarkerCapState.ws_hold``) — otherwise a trailing double-space before
    a newline would be a Markdown hard line break (``<br>``) injected right
    into the bullets this is supposed to be cleaning up. This equivalence is
    asserted directly in ``test_timecodes.py`` across several chunk sizes, so
    a line ending in dropped markers streams down to the exact same text a
    caller would get from running the non-streaming primitive on the
    fully-accumulated line.
    """
    state = _MarkerCapState(max_markers)
    async for delta in stream:
        out_parts: list[str] = []
        for char in delta:
            piece = state.feed(char)
            if piece:
                out_parts.append(piece)
        if out_parts:
            yield "".join(out_parts)
    tail = state.flush()
    if tail:
        yield tail


__all__ = [
    "build_marked_text",
    "cap_markers_in_stream",
    "cap_markers_per_line",
    "collapse_repeated_segments",
    "format_segments_as_marked_text",
    "format_timecode",
    "strip_all_timecodes",
    "strip_bare_timecode_lines",
    "strip_timecode_placeholders",
    "strip_transcript_tail_noise",
]
