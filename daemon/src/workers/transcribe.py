"""Whisper transcription via mlx-server's ``/v1/audio/transcriptions``.

    async def transcribe_audio(audio_path, *, total_duration) -> TranscribeResult

Non-streaming ``verbose_json``. The mlx-server install is patched (see
``scripts/mlx-patches/``) so the response actually carries the
per-segment timing + auto-detected language that ``mlx_whisper.transcribe``
produces internally — upstream's handler used to drop both.

Why non-streaming + verbose_json (not streaming + plain json)
-------------------------------------------------------------

The streaming endpoint only emits text deltas; no segment boundaries, no
language. With it we'd be back to "one giant bucket" — exactly the
problem the transcript-tab UI needs to solve. ``verbose_json`` returns
segments and language in one shot, so we make a single request and get
exactly what the downstream code needs.

Trade-off: we lose mid-transcription UI progress (the previous stream
form gave a chunk every 30 s of audio). Whisper-turbo on mlx is ~1×
realtime, so the daemon publishes a single ``transcribing`` stage and
the Library row sits on it until the call returns. The user explicitly
chose accuracy of timestamps over real-time progress; if that flips, we
synthesise an elapsed-vs-expected timer here without changing callers.

The mlx-server side timeout (``queue_timeout`` in
``~/.mlx-server/config.yaml``) must be ≥ expected transcription wall
time — for hour-long audio we leave it at the install default of an
hour. ``httpx`` here uses ``timeout=None`` to match.

Fallback if the server isn't patched
------------------------------------

Older / unpatched mlx-server responses lack ``segments`` and
``language``. We don't fail — we fabricate one segment spanning the
whole audio so downstream ``build_marked_text`` and summary still work
(same shape as the pre-patch behaviour). ``language`` ends up ``None``;
callers persist it as ``None`` and the UI falls back to the "Original"
label.

Coverage check: a "done" job whose transcript silently stops short
-------------------------------------------------------------------

Measured live TWICE, on the same ~21.5 min video, on two different
transcription attempts:

- Job ``3IXBfawKZrj7`` (chunked upload): Whisper decode-looped 205 s into
  a 648 s chunk, repeating one already-said sentence for the rest of the
  chunk. The repeat-run collapsed to one segment near the START of the
  gap — nothing noticed the run then continued to the chunk's own end,
  so ~6 minutes of real speech after it vanished.
- Job ``Y7odGFeN7agb`` (same video, re-run after the FIRST version of this
  fix, single-request upload — the file fit under the cap this time, no
  chunking at all): Whisper produced normal speech up to 728.9 s, then
  nothing until a single stray one-word segment ("2025.") at 1290.9 s —
  9m22s of real content missing from the MIDDLE of the transcript, with
  ONE trailing segment dragging the last segment's ``end`` to within a
  second of the audio's real 1291.6 s duration.

The second case is why an earlier version of this check — comparing only
the LAST kept segment's ``end`` against the known duration — is wrong:
that stray trailing segment makes the transcript look 100% covered by
that measure while 52% of the audio's actual runtime (669 s of 1292 s)
produced nothing. The hole is provably not always at the tail, and
whether chunking even happened has nothing to do with whether Whisper
decode-loops or drops a span — it happened at a DIFFERENT timestamp on a
DIFFERENT upload path on the exact same file. So this has to be checked
on both paths, and has to look for a gap ANYWHERE in the timeline, not
just after the last segment.

We know the audio's real duration (we compute it ourselves to decide how
to chunk — see ``_probe_duration``/``total_duration`` below), so instead
of trusting the model we look, after collapsing, for every suspicious
interval anywhere in ``[0, duration)``: the exact spans
``collapse_repeated_segments`` just discarded (``timecodes.DiscardedRun``
— the run's own ``[start, end)``, known precisely, not guessed) PLUS any
arithmetic gap between segments (``_find_gaps`` — before the first
segment, between any two consecutive ones, or after the last; this half
stays as a pure safety net for "the model returned nothing at all", the
one shape that leaves no run for collapse to have discarded anything
from).

Earlier versions of this check gated a gap's suspicion on its SIZE — over
some threshold it was "wrong", under it "fine". That conflates two
different questions: how much a false-positive check costs us (one extra,
bounded Whisper call — cheap) versus whether real content actually went
missing (which no size threshold can answer; it can only guess). So there
is no correctness threshold anymore. Every suspicious interval bigger
than ``_MIN_RECHECK_SECONDS`` — a pure COST regulator now, not a
correctness gate; see its own comment — gets re-transcribed (in bounded
``_MAX_RECHECK_SLICE_SECONDS`` slices; see below) and the question "was
this actually speech?" is answered by looking at what came back
(``_is_confirmed_silence``, plus a separate degenerate-run check), not by
how long the hole was.

This is also why ``collapse_repeated_segments`` still collapses a
repeat-run down to its FIRST occurrence (dropping the rest) instead of
extending the kept segment's ``end`` to the run's last occurrence: on the
FIRST measured failure, Whisper's raw (pre-collapse) segments advanced in
lock-step with real playback time even while hallucinating — the
repeated sentence's timestamps kept climbing all the way to the chunk's
true end. Collapsing-to-first is what manufactures the interval this
check reads; an "honest timeline" version of collapse (end = last
occurrence in the run) would hide that defect. On the SECOND measured
failure Whisper's own raw segments already had the gap (there was no
repeat-run to collapse at all — Whisper just produced nothing for that
span), so this particular argument doesn't even apply there, but the
conclusion is the same either way: check the COLLAPSED segments, every
suspicious interval, not just the worst one. See
``timecodes.collapse_repeated_segments`` and ``.claude/llm.md`` for more.

**Deciding "was that actually speech?" without trusting Whisper's
wording — and a hard-won distinction between "confirmed nothing" and "we
still don't know."** Whisper-family models emit non-speech markers
("*Dramatic music*", "*door slams*", "[Musik]") whose exact phrasing is a
side effect of what the training data's subtitles happened to say — it
is NOT stable across models, backends, or languages (measured live: 172
consecutive "*Musik*" lines from one backend where another said
"*Dramatic music*" once for the same kind of audio). Matching specific
phrases is therefore a losing, ever-growing blocklist.
``_is_confirmed_silence`` asks structurally instead: is there anything
left once whitespace is stripped; is what's left pure punctuation/
dashes; is it entirely made of bracket/asterisk/paren annotations. Only
these three are ever treated as confirmed non-speech — excluded from
``missing_seconds`` and logged as such.

A DEGENERATE REPEATED RUN (a hallucination loop reproducing within the
re-transcribed clip itself) is deliberately kept OUT of that list — this
is a fix, not the original design. An earlier version folded "the recheck
itself looped" into the same "not speech" verdict, on the theory that a
loop is never real content. That reasoning is backwards: a loop means the
recheck reproduced the SAME failure the original transcription had, i.e.
we STILL don't know what's there — not that we've confirmed there's
nothing. Treating it as confirmed-silent caused a real regression: a 562s
hole that measurably contained live dialogue (manually verified) got
reported as "not lost content" with `missing_seconds` silently going to
0 — a false "all clear" strictly worse than the honest gap it replaced.
So a degenerate run on a recheck now: (1) is NEVER added to the
confirmed-silent list and NEVER logged as "not lost", (2) still gets
whatever ``collapse_repeated_segments`` recovers — its own first-
occurrence rule keeps the (usually correct) content transcribed before
the loop kicked in, spliced in same as real speech, better than nothing —
and (3) leaves the remainder of the window exactly as suspicious as
before, so it's either re-sliced-and-retried within budget or, honestly,
still counted in ``missing_seconds``.

**The recheck's own ASK SIZE matters, not just whether one happens.** A
562s recheck reproduced the exact decode-loop that created the hole in
the first place — a long enough ask recreates the conditions Whisper
already failed under, so of course it fails the same way again. A manual
20s check of the identical audio at the identical offset came back with
real dialogue. So any suspicious interval longer than
``_MAX_RECHECK_SLICE_SECONDS`` is split into consecutive slices
(``_split_into_slices``) and rechecked slice by slice, never as one
oversized request — see that constant's own comment for the exact size
and why.

Rechecks are bounded (``_MAX_COVERAGE_RECHECKS`` — one shared counter
across every slice of every suspicious interval in the unit, so one huge
hole's slicing can't silently consume the whole budget and starve
everything else) and memoized (a slice already re-transcribed this call
is never asked again — same discipline ``workers/translator.py`` uses
for its bisection retries) so the loop provably terminates regardless of
how many suspicious intervals — or how large any one of them is — a
pathological transcript produces. A slice that comes back as real speech
(or a degenerate run's recovered first-occurrence portion) is spliced in
using the same mechanism as before (backoff context on the trailing
edge, a wider leading-edge distrust window ONLY on a suspicious
interval's true leading edge — an artificial internal split point
between two slices of the same oversized interval does not get that
widening, since it isn't a real gap edge — clipped to avoid a duplicate
seam; see ``_ensure_coverage``'s own docstring). If a slice can't be
re-transcribed at all (ffmpeg unavailable), still reads as an
unresolved degenerate loop, or the budget runs out before a slice is
checked, we don't know what's there — conservatively, that still counts
toward ``missing_seconds`` rather than silently assuming it's fine.
Callers persist the residual on the job
(``Job.transcript_missing_seconds``) so the UI can tell the user this
"done" job's transcript is known-incomplete, instead of it looking
exactly like a full one.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from src.config import get_config
from src.workers.timecodes import collapse_repeated_segments

log = logging.getLogger(__name__)


@dataclass
class TranscribeResult:
    """Per-segment timing + detected language from a Whisper transcription.

    ``segments`` is a list of dicts with ``start`` / ``end`` / ``text`` —
    the canonical shape ``timecodes.build_marked_text`` consumes. When
    the server didn't return real segments (unpatched mlx-server) we
    construct a single all-encompassing segment so the rest of the
    pipeline behaves normally.

    ``language`` is an ISO-639-1 code (e.g. ``"en"``, ``"ru"``) or
    ``None`` if the server didn't surface it.

    ``duration_seconds`` mirrors the upstream ``duration`` field — handy
    for synthesised progress and as a sanity check vs yt-dlp's metadata.

    ``missing_seconds`` means seconds of SPEECH we failed to recover, not
    seconds of uncovered timeline — it's 0.0 when every suspicious
    interval either covered fully (immediately, or after the bounded
    rechecks in ``_ensure_coverage``) or was CONFIRMED as non-speech
    (empty/punctuation/bracket-annotation — see ``_is_confirmed_silence``;
    a degenerate repeated run on a recheck is deliberately never confirmed
    this way, since it means "still unknown", not "confirmed silent"; see
    the module docstring's regression writeup); otherwise it's the
    residual length of whatever's still open once the recheck budget ran
    out, real content Whisper never reliably produced. See the module
    docstring's "Coverage check" section.
    """

    segments: list[dict[str, Any]]
    language: str | None
    duration_seconds: float | None
    missing_seconds: float = 0.0


# How long a suspicious interval (a collapse-discarded run, or an arithmetic
# gap — see _ensure_coverage) has to be before we bother re-transcribing it.
# This is NOT a correctness threshold anymore — there is no size past which
# a hole is "probably fine" and no size under which it's "probably lost
# content"; only actually asking Whisper again (_is_confirmed_silence, plus
# the collapse-based degenerate-run check) answers that. This constant
# exists purely to regulate COST: every interval above it costs at least one
# extra, bounded Whisper call. A value here that's a little too small just
# means a few more cheap rechecks of ordinary short pauses (seconds of
# wall-clock, not content); a value too large would mean going back to
# guessing on short-but-real intervals instead of checking them — so this
# errs small. ~5s is comfortably above a normal breath/pause (nothing to
# check) and comfortably below any interval worth an actual answer.
_MIN_RECHECK_SECONDS = 5.0

# The maximum span of audio ONE recheck request is allowed to cover. A
# suspicious interval longer than this is split into consecutive slices
# (see _split_into_slices) instead of being cut and sent to Whisper whole.
#
# This exists because of a live regression: a single recheck over a 562s
# hole reproduced the EXACT decode-loop that created the hole in the first
# place — asking about a long span reconstructs the same conditions Whisper
# already failed under. A manual, by-hand check of the SAME audio using a
# 20s window at the same offset came back with real, correct dialogue. So
# the size of the ASK, not just whether we ask, determines whether the
# recheck can actually see something different from the original attempt.
#
# 90s is chosen as comfortably above that 20s probe (real margin for
# sentence/paragraph context, not a bare minimum) while staying far below
# both measured failure sizes — 205s (a chunk-internal decode loop) and
# 562s (the regression above) — roughly 2x and 6x clear of them
# respectively. It also divides the worst measured hole (562s) into a
# manageable ~7 slices, keeping the recheck budget below predictable and
# small.
_MAX_RECHECK_SLICE_SECONDS = 90.0

# How many seconds a retry's re-cut extends PAST THE TRAILING edge (after
# ``gap_end``) of the gap it's trying to fill — decode context only, not
# distrust: nothing measured suggests Whisper drifts on the way OUT of a
# gap the way it drifts on the way IN (see ``_PREFIX_DISTRUST_SECONDS``
# below for the leading edge, which needs a much bigger margin for a
# different reason). Resending identical bytes to a deterministic decoder
# that just dropped a span has no reason to produce a different result; a
# cut boundary landing mid-phrase is a known trigger for this failure mode,
# so shifting the edge changes the audio content and the boundary the model
# sees, which is the one thing actually likely to change the outcome. Long
# enough to span a few words of context, short enough that we never throw
# away much confirmed-good trailing coverage to get it (this margin
# overlaps the head of the next confirmed segment, which gets re-derived
# from the retry rather than kept from the original — see ``_clip_to_gap``).
_RETRY_BACKOFF_SECONDS = 5.0

# How many seconds BEFORE a gap's start the prefix is presumed to have
# ALREADY been drifting out of sync, and gets re-transcribed along with the
# gap itself rather than trusted as-is. This is a distinct, much larger
# margin than _RETRY_BACKOFF_SECONDS above, because it addresses a
# different failure: Whisper doesn't only drop content once it falls into
# a hallucination loop, it can drift for a while BEFORE the loop actually
# starts, misattributing real speech to the wrong (earlier) timestamps as
# it loses sync. Measured live: the five segments immediately before a
# real 728.9s gap were each marked as EXACTLY 1.000s long — real Whisper
# output is never that round — and a re-cut of that same span (728.9s
# onward) produced the SAME dialogue with normal, live-sounding durations
# (1.4s, 2.3s, 2.4s, ...) instead. So the drift was already visible via
# those suspiciously-round durations a full 10s before the gap it preceded
# — and 10s is just where it happened to become visible by that symptom,
# not necessarily where it started, so the window needs real margin above
# that observed minimum rather than exactly covering it. 30s is 3x that
# 10s floor. Applied only to the LEADING edge of a gap: nothing measured
# shows the same drift happening on the way out the other side, hence the
# asymmetry with _RETRY_BACKOFF_SECONDS on the trailing edge. Content
# inside this window is never lost, only re-transcribed — the retry's cut
# covers it, so whatever's really there comes back with corrected timing.
_PREFIX_DISTRUST_SECONDS = 30.0

# Hard cap on RECHECKS per unit of work (one chunk, or the whole file on the
# single-request path) — the budget the module docstring promises. Bounded
# the same way translator.py bounds its bisection retries — each recheck
# only re-transcribes ONE SLICE (see _MAX_RECHECK_SLICE_SECONDS — never a
# whole oversized interval, never the whole unit), and a slice already
# rechecked this call is never asked again (see the ``checked`` set in
# ``_ensure_coverage``), so together this guarantees the loop terminates and
# total added cost per unit stays bounded regardless of how many distinct
# suspicious intervals — or how large any one of them is — a pathological
# transcript produces.
#
# One shared counter across every slice of every suspicious interval in the
# unit (not a separate budget per interval) is deliberate: it's what stops
# ONE oversized hole's slicing from silently consuming the ENTIRE budget and
# starving every other suspicious interval in the same unit — once an
# interval's own slices are all in ``checked``, it stops competing for
# budget, freeing the rest for whatever else is pending.
#
# 12 comes directly from the slice size above: the worst measured hole
# (562s) needs ceil(562 / 90) = 7 slices to fully re-cover; 12 leaves
# headroom for a second, smaller hole in the same unit, or for a slice that
# only partially resolves (a degenerate repeat on the recheck itself, see
# below) and needs a follow-up on its own remainder — while still being a
# small, fixed, auditable number rather than "however many it takes".
_MAX_COVERAGE_RECHECKS = 12


async def transcribe_audio(
    audio_path: Path,
    *,
    total_duration: float | None,
) -> TranscribeResult:
    """Transcribe the audio, splitting it first if it exceeds the upload cap.

    Most Whisper backends reject large bodies (LocalAI ~15 MB, OpenAI 25 MB).
    For audio over ``whisper.max_upload_mb`` we split it into time-based chunks
    with ffmpeg, transcribe each, and stitch the segments back together with
    their original timestamps. Small audio takes the single-request path.

    Raises ``httpx.HTTPStatusError`` on server-side failure (caller turns that
    into a friendly error). Returns an empty-segments result when the server
    reports success but didn't transcribe anything (e.g. silent audio).

    Before returning, collapses consecutive Whisper repetition-loop segments
    (see ``timecodes.collapse_repeated_segments``) — hallucination loops over
    noisy/silent audio can otherwise leave hundreds of duplicate lines in
    every downstream consumer of ``segments``: ``build_marked_text`` (feeds
    the summary), the persisted ``raw_segments_json`` (Transcript tab via
    ``api/jobs._build_segments_text``), and ``workers/translator.py``
    (translation source). Applying it here, once, after chunked transcription
    has already merged all chunks back together, means a repetition loop
    spanning a chunk boundary is still caught, and every consumer downstream
    of this function is automatically clean — no changes needed in
    ``runner.py`` or elsewhere. This is Whisper-only by construction: the
    YouTube caption fast path (``pipeline.py``) builds its segments directly
    from ``youtube-transcript-api``/yt-dlp and never calls this function.

    Before the final collapse, each unit of work (the single request, or
    each chunk independently) has already been coverage-checked and
    retried if short — see the module docstring's "Coverage check"
    section and ``_ensure_coverage``. ``missing_seconds`` on the result
    surfaces whatever shortfall survived retries, summed across chunks.
    """
    cfg = get_config().whisper
    max_bytes = max(1, cfg.max_upload_mb) * 1024 * 1024
    size = audio_path.stat().st_size

    if size <= max_bytes:
        result = await _transcribe_whole(audio_path, total_duration=total_duration)
    else:
        result = await _transcribe_chunked(
            audio_path, total_duration=total_duration, max_bytes=max_bytes
        )

    final_segments, _discarded = collapse_repeated_segments(result.segments)
    return TranscribeResult(
        segments=final_segments,
        language=result.language,
        duration_seconds=result.duration_seconds,
        missing_seconds=result.missing_seconds,
    )


async def _transcribe_whole(
    audio_path: Path, *, total_duration: float | None
) -> TranscribeResult:
    """Transcribe ``audio_path`` in one request, checking + retrying coverage
    when ``total_duration`` is known. Shared by the single-request path and
    both of ``_transcribe_chunked``'s "can't actually chunk" fallbacks."""
    payload = await _post_audio(audio_path)
    result = _parse_payload(payload, total_duration=total_duration)
    if total_duration is None or total_duration <= 0:
        # Nothing to compare against — can't tell a short transcript from a
        # short recording. Same trust-the-backend behaviour as before this
        # feature existed.
        return result
    segments, missing = await _ensure_coverage(
        result.segments, source_path=audio_path, window_duration=total_duration,
    )
    return TranscribeResult(
        segments=segments,
        language=result.language,
        duration_seconds=result.duration_seconds,
        missing_seconds=missing,
    )


async def _transcribe_chunked(
    audio_path: Path,
    *,
    total_duration: float | None,
    max_bytes: int,
) -> TranscribeResult:
    """Split oversized audio with ffmpeg, transcribe parts, merge segments."""
    duration = total_duration if total_duration and total_duration > 0 else None
    if duration is None:
        duration = await asyncio.to_thread(_probe_duration, audio_path)
    if duration is None or duration <= 0:
        # Can't time-slice without a duration — fall back to one shot and let
        # the backend's own error surface if it really is too big. No known
        # duration also means no coverage check is possible here either.
        log.warning("transcribe: unknown duration, cannot chunk; trying single upload")
        return await _transcribe_whole(audio_path, total_duration=total_duration)

    size = audio_path.stat().st_size
    # Target 90% of the cap for VBR headroom; at least 2 chunks since we're here.
    target = max(1, int(max_bytes * 0.9))
    num_chunks = max(2, math.ceil(size / target))
    chunk_seconds = duration / num_chunks
    log.info(
        "transcribe: audio %.1f MB > cap → %d chunks of ~%.0f s",
        size / 1024 / 1024,
        num_chunks,
        chunk_seconds,
    )

    chunks = await asyncio.to_thread(
        _split_audio, audio_path, num_chunks, chunk_seconds
    )
    if not chunks:
        log.warning("transcribe: ffmpeg split produced nothing; trying single upload")
        return await _transcribe_whole(audio_path, total_duration=duration)

    all_segments: list[dict[str, Any]] = []
    language: str | None = None
    missing_total = 0.0
    try:
        for idx, (chunk_path, offset) in enumerate(chunks):
            payload = await _post_audio(chunk_path)
            part = _parse_payload(payload, total_duration=chunk_seconds)
            # The last chunk may be shorter than chunk_seconds if the file
            # doesn't divide evenly (ffmpeg's -t just stops at EOF) — use
            # whatever's actually left of the known total as this chunk's
            # expected coverage, not the nominal per-chunk length.
            expected_local_duration = min(chunk_seconds, duration - offset)
            segments, missing = await _ensure_coverage(
                part.segments,
                source_path=chunk_path,
                window_duration=expected_local_duration,
            )
            if missing > 0:
                missing_total += missing
                log.warning(
                    "transcribe: chunk %d/%d still short by ~%.0fs of audio "
                    "after retries — transcript may be missing content there",
                    idx + 1, len(chunks), missing,
                )
            for seg in segments:
                all_segments.append(
                    {
                        "start": seg["start"] + offset,
                        "end": seg["end"] + offset,
                        "text": seg["text"],
                    }
                )
            if language is None:
                language = part.language
            log.info("transcribe: chunk %d/%d done", idx + 1, len(chunks))
    finally:
        for chunk_path, _ in chunks:
            chunk_path.unlink(missing_ok=True)
        # chunks share one mkdtemp dir; remove it once emptied.
        with contextlib.suppress(OSError):
            chunks[0][0].parent.rmdir()

    return TranscribeResult(
        segments=all_segments,
        language=language,
        duration_seconds=duration,
        missing_seconds=missing_total,
    )


def _segment_start(seg: dict[str, Any]) -> float:
    try:
        return float(seg.get("start", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _segment_end(seg: dict[str, Any]) -> float:
    try:
        return float(seg.get("end", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _find_gaps(
    segments: list[dict[str, Any]], window_duration: float
) -> list[tuple[float, float]]:
    """Uncovered ``[start, end)`` intervals in ``segments`` against
    ``window_duration`` — before the first segment, between any two
    consecutive segments, and after the last one, uniformly (no special
    casing for "the tail" vs "the middle": both are just gaps). Callers
    pass ALREADY-COLLAPSED segments — see the module docstring for why a
    repeat-run has to be collapsed first for this to see the real gap it
    leaves. ``segments`` need not be pre-sorted.

    An empty ``segments`` list produces one gap spanning the whole window
    — "nothing was transcribed at all" is the largest possible gap, not a
    special case.

    This is the SAFETY-NET half of ``_ensure_coverage``'s suspicious-window
    detection — the other half is ``timecodes.DiscardedRun``, which is
    exact and directly known rather than arithmetic. A gap catches the
    shape a discarded run can't: Whisper returning nothing at all for a
    span, with no repeat-run for collapse to have discarded anything from.
    """
    if not segments:
        return [(0.0, window_duration)] if window_duration > 0 else []

    ordered = sorted(segments, key=_segment_start)
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for seg in ordered:
        start = _segment_start(seg)
        end = _segment_end(seg)
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if window_duration > cursor:
        gaps.append((cursor, window_duration))
    return gaps


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sort and merge overlapping/touching ``[start, end)`` intervals.

    ``timecodes.DiscardedRun`` spans and ``_find_gaps``'s arithmetic gaps
    frequently describe the SAME underlying stretch from two different
    angles (a collapsed run that also happens to leave a timeline gap
    behind it) — merging first means that overlap is treated as ONE
    suspicious window, not two (which would otherwise mean asking Whisper
    about the same audio twice, and double-counting it if it turns out to
    be missing).
    """
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged: list[tuple[float, float]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _suspicious_windows(
    segments: list[dict[str, Any]], window_duration: float
) -> list[tuple[float, float]]:
    """Every interval worth asking Whisper to re-check: collapse's exactly-
    known discarded runs, plus the arithmetic-gap safety net, merged so an
    overlap between the two isn't double-counted. Does NOT filter by
    ``_MIN_RECHECK_SECONDS`` — that's a caller decision (how much budget is
    left, which window is worth spending it on), not a property of what's
    suspicious."""
    collapsed, discarded = collapse_repeated_segments(segments)
    return _merge_intervals(
        [(run.start, run.end) for run in discarded]
        + _find_gaps(collapsed, window_duration)
    )


def _window_key(start: float, end: float) -> tuple[float, float]:
    """Dedup key for a suspicious window (or slice of one) — rounded
    defensively against float jitter, though in practice the SAME
    unchanged window recurs bit-for-bit across loop iterations of
    ``_ensure_coverage`` when nothing about it was touched (e.g. confirmed
    non-speech)."""
    return (round(start, 3), round(end, 3))


def _pending_slices(
    working: list[dict[str, Any]],
    window_duration: float,
    checked: set[tuple[float, float]],
) -> list[tuple[float, float, bool]]:
    """Every re-checkable ``(slice_start, slice_end, is_leading)`` still
    worth asking Whisper about: suspicious windows above the cost cutoff
    (``_MIN_RECHECK_SECONDS``), split into ``_MAX_RECHECK_SLICE_SECONDS``
    slices so no single recheck request ever covers more than that, minus
    whatever's already in ``checked`` this call.
    """
    pending: list[tuple[float, float, bool]] = []
    for start, end in _suspicious_windows(working, window_duration):
        if (end - start) <= _MIN_RECHECK_SECONDS:
            continue
        for slice_start, slice_end, leading in _split_into_slices(
            start, end, _MAX_RECHECK_SLICE_SECONDS
        ):
            if _window_key(slice_start, slice_end) not in checked:
                pending.append((slice_start, slice_end, leading))
    return pending


def _uncovered_by_confirmed_silence(
    window: tuple[float, float], confirmed_silent: list[tuple[float, float]]
) -> float:
    """Length of ``window`` still unaccounted for after subtracting any
    overlap with intervals a recheck already confirmed are non-speech
    (empty/punctuation/annotation-only — see ``_is_confirmed_silence``; a
    degenerate repeated run is deliberately NEVER added here, since it
    means "still unknown", not "confirmed non-speech").

    Interval subtraction rather than an exact key lookup: ``_merge_intervals``
    can fold a confirmed-silent window together with an adjacent, still-
    unresolved one into a single wider final window whose own key matches
    neither original — subtracting the actual overlapping SPAN handles that
    correctly where matching by key would not.
    """
    start, end = window
    remaining = end - start
    for silent_start, silent_end in confirmed_silent:
        overlap = min(end, silent_end) - max(start, silent_start)
        if overlap > 0:
            remaining -= overlap
    return max(remaining, 0.0)


# --- "did a re-checked window come back as CONFIRMED non-speech?" -----------
#
# Whisper-family models emit non-speech markers over music/noise/silence, but
# the exact WORDING is a side effect of training data, not a stable contract —
# measured live: one backend produced 172 CONSECUTIVE "*Musik*" segments over
# a musical stretch where a differently-trained backend said "*Dramatic
# music*" once for the same kind of audio. Matching known phrases is an
# ever-growing, backend/model/language-specific blocklist that will always be
# behind whatever a new model was trained to say. So this classifies
# STRUCTURALLY instead: is there text left after stripping whitespace; is
# what's left pure punctuation/dashes; is it made ENTIRELY of bracket/
# asterisk/paren annotations.
#
# Deliberately NOT included here: a degenerate repeated run. That's a
# SEPARATE verdict, checked by the caller (_ensure_coverage) via
# collapse_repeated_segments directly, and it is NEVER treated as confirmed
# silence — see the module docstring's regression writeup for why folding
# "the recheck looped" into "confirmed nothing" is wrong (a loop means we
# still don't know what's there, not that we've confirmed there's nothing).

# Punctuation/dash-family characters explicitly enumerated (no unicode range
# syntax — a typo'd range silently swallowing letters is a much worse failure
# mode than missing one obscure dash variant). Matches text that, once
# whitespace is stripped, is composed only of these — e.g. a bare "-" a
# Whisper backend sometimes emits over a short pause with no actual words.
# Real speech always contains at least one letter/digit.
_PUNCTUATION_CHARS = "-‐‑‒–—―.,!?;:'\"«»„“”‘’…"
_PUNCTUATION_ONLY_RE = re.compile(rf"^[\s{re.escape(_PUNCTUATION_CHARS)}]*$")

# One bracket/asterisk/paren-delimited annotation, e.g. "*Dramatic music*",
# "[Musik]", "(laughs)" — the shapes Whisper backends use for non-speech
# sound description. Each alternative requires its OWN matching delimiter
# pair (no cross-matching a "*" with a "]"), and forbids nesting the same
# delimiter inside itself, so it can't accidentally swallow real prose that
# merely contains a stray bracket character.
_STAR_ANNOTATION = r"\*[^*\n]+\*"
_SQUARE_ANNOTATION = r"\[[^\[\]\n]+\]"
_PAREN_ANNOTATION = r"\([^()\n]+\)"

# The text as a WHOLE is nothing but one or more such annotations (optionally
# whitespace-separated) — "*Music* *Applause*" matches, but "Text (aside)
# more text" does not, since the leading/trailing prose falls outside any
# annotation group and the anchored ``^...$`` match fails.
_ANNOTATION_ONLY_RE = re.compile(
    rf"^(?:\s*(?:{_STAR_ANNOTATION}|{_SQUARE_ANNOTATION}|{_PAREN_ANNOTATION})\s*)+$"
)


def _is_confirmed_silence(segments: list[dict[str, Any]]) -> bool:
    """Decide whether a re-transcribed suspicious window is CONFIRMED
    non-speech — the ONLY verdict allowed to exclude a window from
    ``missing_seconds`` and be logged as "not lost content".

    Checked in order, each a no-op pass-through to the next when it
    doesn't apply:

    1. Empty or whitespace-only (Whisper returned nothing) -> confirmed
       silence.
    2. Only punctuation/dash characters (e.g. a bare "-") -> confirmed
       silence.
    3. The WHOLE text is composed of bracket/asterisk/paren annotations
       (e.g. "*Dramatic music*", "[Musik]") -> confirmed silence. Real
       dialogue containing an incidental parenthetical aside does NOT
       match this — the check requires annotations to account for the
       ENTIRE text, not just be present somewhere in it.
    4. Otherwise -> NOT confirmed silence (i.e. "unknown or real speech";
       the caller decides which by checking for a degenerate repeated run
       separately — see ``_ensure_coverage``. A loop is deliberately NOT
       folded into this function: it means we still don't know what's
       there, which is a different, weaker claim than "confirmed no
       speech", and conflating the two caused a real regression — see the
       module docstring).
    """
    texts = [str(seg.get("text") or "").strip() for seg in segments]
    joined = " ".join(t for t in texts if t).strip()
    if not joined:
        return True
    if _PUNCTUATION_ONLY_RE.match(joined):
        return True
    return bool(_ANNOTATION_ONLY_RE.match(joined))


def _split_into_slices(
    start: float, end: float, max_len: float
) -> list[tuple[float, float, bool]]:
    """Split ``[start, end)`` into consecutive slices, each at most
    ``max_len`` long, so a single recheck request never covers more audio
    than that (see ``_MAX_RECHECK_SLICE_SECONDS`` for why).

    Returns ``(slice_start, slice_end, is_leading)`` triples. Only the
    FIRST slice is marked ``is_leading=True`` — it sits at the suspicious
    interval's real leading edge, the one place ``_PREFIX_DISTRUST_SECONDS``
    is meant to widen. An internal split point between two slices of the
    SAME oversized interval is an artificial chop point we introduced, not
    a genuine gap edge Whisper actually drifted before — widening the
    distrust window there would just re-transcribe extra already-good
    audio for no reason. A single slice that already fits within
    ``max_len`` is still leading (the whole interval IS its own leading
    edge).
    """
    if end - start <= max_len:
        return [(start, end, True)]
    slices: list[tuple[float, float, bool]] = []
    cursor = start
    leading = True
    while cursor < end:
        slice_end = min(end, cursor + max_len)
        slices.append((cursor, slice_end, leading))
        cursor = slice_end
        leading = False
    return slices


def _clip_to_gap(
    segments: list[dict[str, Any]], gap_start: float, gap_end: float
) -> list[dict[str, Any]]:
    """Trim a retry's own segments (already offset to absolute local time)
    down to the GAP's own boundaries, dropping anything that lives entirely
    inside the ``_RETRY_BACKOFF_SECONDS`` context margin on either side and
    clamping the start/end of anything straddling an edge.

    The backoff extension exists only to give the decoder context; without
    this clip, a retry segment sitting inside that margin duplicates
    whatever the confirmed neighbor already covers (the same audio
    transcribed twice) and — worse — its own un-clamped ``start`` can sit
    BEFORE an already-confirmed segment's ``start``, breaking the
    non-decreasing-start invariant every downstream consumer relies on
    (``build_marked_text``'s timecodes, the translator's forward-only
    marker alignment in ``_align_translation``). Measured live: a retry cut
    5s before/after a gap left a duplicate line and a segment starting 4s
    earlier than the one immediately before it in the list.

    Also sorts by ``start`` before returning. The backend is generally
    well-ordered, but nothing guarantees it, and a single out-of-order pair
    surviving into the splice would violate the same invariant — cheaper
    and more robust to sort here once than to assume backend ordering and
    let ``_ensure_coverage``'s final assertion be the only thing standing
    between that and a crashed job.
    """
    clipped: list[dict[str, Any]] = []
    for seg in segments:
        start = _segment_start(seg)
        end = _segment_end(seg)
        if end <= gap_start or start >= gap_end:
            continue  # entirely inside the backoff margin — drop, don't keep
        clipped.append(
            {
                "start": max(start, gap_start),
                "end": min(end, gap_end),
                "text": seg["text"],
            }
        )
    clipped.sort(key=_segment_start)
    return clipped


def _is_monotonic_by_start(segments: list[dict[str, Any]]) -> bool:
    """True iff ``segments`` never goes backward in time by ``start``.

    Cheap invariant, checked wherever ``_ensure_coverage`` returns: every
    downstream consumer of these segments (timecodes, the translator's
    forward-only alignment) assumes this holds. A violation means a bug in
    the splice logic above, not bad input data from the backend.
    """
    starts = [_segment_start(s) for s in segments]
    return all(a <= b for a, b in zip(starts, starts[1:], strict=False))


async def _ensure_coverage(
    segments: list[dict[str, Any]],
    *,
    source_path: Path,
    window_duration: float,
) -> tuple[list[dict[str, Any]], float]:
    """Check ``segments`` (a LOCAL timeline starting at 0) against
    ``window_duration`` seconds of ``source_path``: re-transcribe every
    suspicious interval bigger than ``_MIN_RECHECK_SECONDS`` — sliced to
    ``_MAX_RECHECK_SLICE_SECONDS`` at most per request — and decide, from
    what comes back, whether it was real speech, confirmed non-speech, or
    still unresolved.

    Returns ``(segments, missing_seconds)`` — the (possibly patched-up)
    segment list to use, and the summed length of every suspicious
    interval that's STILL open once the budget (``_MAX_COVERAGE_RECHECKS``)
    runs out, MINUS whatever length was CONFIRMED as non-speech along the
    way (that doesn't count as missing — see ``_is_confirmed_silence``). A
    degenerate repeated run on a recheck is deliberately never subtracted
    this way — see the module docstring's regression writeup. Segments are
    returned UNCOLLAPSED (the caller's final ``collapse_repeated_segments``
    pass — run once over the whole merged transcript — still applies);
    only the accept/recheck DECISION is made on a locally-collapsed view.

    There is deliberately no size-based correctness gate here anymore (see
    the module docstring): every interval above the cost cutoff gets
    rechecked, in descending size order, up to the budget. A slice
    already rechecked THIS call is never asked again (the ``checked`` set)
    — a deterministic decoder given the exact same audio has no reason to
    answer differently — which combined with the budget guarantees this
    loop terminates regardless of how many distinct suspicious intervals —
    or how large any one of them is — a pathological transcript produces.

    Each recheck re-cuts ``source_path`` over a window that is ASYMMETRIC
    around the slice: ``_PREFIX_DISTRUST_SECONDS`` (30s) before
    ``gap_start``, but only ``_RETRY_BACKOFF_SECONDS`` (5s) after
    ``gap_end`` — and the leading widening applies ONLY when this slice is
    the suspicious interval's true leading edge (``is_leading`` from
    ``_pending_slices``/``_split_into_slices``); an artificial internal
    split point between two slices of the same oversized interval isn't a
    real gap edge Whisper actually drifted before, so it doesn't get that
    treatment. The two edges answer different questions — the trailing
    margin is pure decode context (nothing measured shows drift on the way
    OUT of a gap), while the leading margin (when it applies) actively
    DISTRUSTS that stretch of the prefix: Whisper measurably drifts out of
    sync for a while before it actually falls into the hallucination loop
    that produces a gap, misattributing real speech to earlier, wrong
    timestamps, so the prefix immediately before a gap can't be trusted
    just because it technically has no gap of its own.

    What comes back is classified three ways, not two:

    - CONFIRMED non-speech (``_is_confirmed_silence`` — empty, punctuation-
      only, or entirely bracket/asterisk/paren annotations): nothing is
      spliced, the interval is recorded as confirmed-silent so the final
      ``missing_seconds`` accounting excludes it, logged as "not lost
      content".
    - A DEGENERATE REPEATED RUN within the recheck's OWN segments (reusing
      ``collapse_repeated_segments`` directly): this is NOT confirmed
      non-speech — it means the recheck reproduced the same kind of
      failure the original transcription had, so we still don't know
      what's really there. Never added to the confirmed-silent list, never
      logged as "not lost". What DOES get spliced in is
      ``collapse_repeated_segments``'s own first-occurrence survivor from
      this recheck — partial, but real, recognized content is better than
      discarding the whole slice — and the remainder of the slice is left
      open, exactly as suspicious as before.
    - Otherwise (real speech): spliced in in full, same as the degenerate
      case's partial splice below.

    Splicing (for both the real-speech and degenerate-partial cases) goes
    in between whatever was on either side of ``[distrust_start,
    gap_end)`` (``distrust_start = max(0, gap_start -
    _PREFIX_DISTRUST_SECONDS)`` if leading, else ``gap_start`` itself).
    Two boundary decisions, both keyed on this same interval rather than
    the (slightly wider) cut window: which CONFIRMED segments survive the
    splice (``prefix``/``suffix`` below — a confirmed segment survives
    whole as long as it ends at/before ``distrust_start`` or starts
    at/after ``gap_end``; keying this on the cut window instead would drop
    an entire long confirmed segment just because its last few seconds
    fall inside the trailing backoff margin), and how much of the RETRY's
    OWN output survives (``_clip_to_gap`` — anything the retry produced
    entirely inside the trailing backoff margin is dropped, and anything
    straddling an edge is clamped to it, rather than every retried segment
    surviving unclipped). Skipping the second half would leave the
    trailing margin's context audio transcribed twice — once by the
    confirmed neighbor, once by the retry — and, worse, let a retried
    segment's own unclamped ``start`` land before an already-confirmed
    one, which is exactly what breaks the non-decreasing-``start``
    invariant this function guarantees on every return (see
    ``_is_monotonic_by_start``).
    """
    if window_duration <= 0:
        return segments, 0.0

    working = list(segments)
    assert _is_monotonic_by_start(working), (
        "transcribe: input segments went backward in time by start "
        "(bug upstream of _ensure_coverage, not this function)"
    )
    checked: set[tuple[float, float]] = set()
    confirmed_silent: list[tuple[float, float]] = []
    rechecks = 0

    while rechecks < _MAX_COVERAGE_RECHECKS:
        pending = _pending_slices(working, window_duration, checked)
        if not pending:
            break

        gap_start, gap_end, leading = max(pending, key=lambda w: w[1] - w[0])
        checked.add(_window_key(gap_start, gap_end))
        rechecks += 1

        # The retry TARGET is wider than the slice itself on the leading
        # edge, but ONLY when this slice is the suspicious interval's real
        # leading edge (see this function's own docstring and
        # _split_into_slices): the prefix immediately before it is
        # presumed to already be drifted (see _PREFIX_DISTRUST_SECONDS),
        # so it's re-transcribed along with the gap rather than trusted
        # as-is. The trailing edge always gets only the small decode-
        # context margin (_RETRY_BACKOFF_SECONDS) — nothing measured shows
        # the same drift on the way out of a gap. distrust_start is the
        # SPLICE boundary (source of truth for what survives, see
        # _clip_to_gap / prefix below); the actual AUDIO cut extends a
        # little further still, past distrust_start, purely for decode
        # context — same split the trailing edge already has between
        # gap_end (splice boundary) and cut_end (audio boundary).
        distrust_start = (
            max(0.0, gap_start - _PREFIX_DISTRUST_SECONDS) if leading else gap_start
        )
        cut_start = max(0.0, distrust_start - _RETRY_BACKOFF_SECONDS)
        cut_end = min(window_duration, gap_end + _RETRY_BACKOFF_SECONDS)
        cut_duration = cut_end - cut_start
        if cut_duration <= 1.0:
            # Nothing meaningful left to re-transcribe; leave it unresolved
            # (counted in the final accounting below) and move on to
            # whatever else is pending.
            continue

        cut_path = await asyncio.to_thread(
            _cut_audio_segment, source_path, cut_start, cut_duration
        )
        if cut_path is None:
            log.warning(
                "transcribe: coverage recheck %d/%d couldn't re-cut audio for "
                "a ~%.0fs slice at %.0fs; leaving it unresolved",
                rechecks, _MAX_COVERAGE_RECHECKS, gap_end - gap_start, gap_start,
            )
            continue

        try:
            payload = await _post_audio(cut_path)
        finally:
            cut_path.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                cut_path.parent.rmdir()

        retry_result = _parse_payload(payload, total_duration=cut_duration)

        if _is_confirmed_silence(retry_result.segments):
            confirmed_silent.append((gap_start, gap_end))
            log.info(
                "transcribe: coverage recheck %d/%d at %.0fs-%.0fs came back "
                "confirmed non-speech (empty/punctuation/annotation) — not "
                "lost content",
                rechecks, _MAX_COVERAGE_RECHECKS, gap_start, gap_end,
            )
            continue

        # NOT confirmed silence. Check for a degenerate repeated run — a
        # hallucination loop reproducing on the recheck itself. This is
        # deliberately NOT treated as confirmed non-speech (see this
        # function's and the module's docstrings): it means we still don't
        # know what's really there, so nothing is added to
        # confirmed_silent and nothing is logged as "not lost". What DOES
        # survive is collapse_repeated_segments's own first-occurrence
        # rule — usually the real content recognized before the loop took
        # over — spliced in exactly like real speech, rather than throwing
        # the whole slice away. Whatever's left unrecovered stays exactly
        # as suspicious as before (picked up again next iteration if
        # budget allows, else honestly counted as missing).
        collapsed_retry, retry_discarded = collapse_repeated_segments(
            retry_result.segments
        )
        if retry_discarded:
            log.warning(
                "transcribe: coverage recheck %d/%d at %.0fs-%.0fs hit a "
                "decode-loop again — kept the recognized portion up to the "
                "loop; remainder stays unresolved, NOT confirmed silent",
                rechecks, _MAX_COVERAGE_RECHECKS, gap_start, gap_end,
            )
            retry_segments_source = collapsed_retry
        else:
            retry_segments_source = retry_result.segments
            log.info(
                "transcribe: coverage recheck %d/%d recovered real speech at "
                "%.0fs-%.0fs",
                rechecks, _MAX_COVERAGE_RECHECKS, gap_start, gap_end,
            )

        retried_segments_raw = [
            {
                "start": seg["start"] + cut_start,
                "end": seg["end"] + cut_start,
                "text": seg["text"],
            }
            for seg in retry_segments_source
        ]
        # Clip the retry's own output down to [distrust_start, gap_end)
        # BEFORE splicing — the trailing backoff margin is decode context
        # only, never meant to survive into the final list (see
        # _clip_to_gap); the leading distrust window (when it applies) is
        # DELIBERATELY wider than the gap itself, so the retry's version of
        # that stretch is what survives, not the (presumed-drifted)
        # original. Splitting the SPLICE on these same two boundaries
        # (below) keeps confirmed neighbors whole outside them.
        retried_segments = _clip_to_gap(retried_segments_raw, distrust_start, gap_end)
        # A confirmed segment survives whole as long as it ends at/before
        # distrust_start (not gap_start — the distrusted prefix stretch is
        # discarded here, not just the gap) or starts at/after gap_end.
        # Content in the distrust window isn't lost: the retry re-covers
        # the same audio, just with corrected timing.
        prefix = [s for s in working if _segment_end(s) <= distrust_start]
        suffix = [s for s in working if _segment_start(s) >= gap_end]
        working = prefix + retried_segments + suffix
        assert _is_monotonic_by_start(working), (
            f"transcribe: coverage splice produced a non-monotonic segment "
            f"list (gap {gap_start:.1f}-{gap_end:.1f}) — bug in the clip/"
            f"splice logic, not the transcript"
        )

    # Budget exhausted (or nothing left pending): report every suspicious
    # interval still over the cost cutoff, minus whatever's confirmed
    # non-speech along the way — ordinary short pauses stay invisible,
    # confirmed music/noise/silence doesn't count as missing, and anything
    # else still open (never rechecked, or rechecked but not fully closed)
    # conservatively does, since we don't actually know what's there.
    assert _is_monotonic_by_start(working), (
        "transcribe: coverage splice produced a non-monotonic segment list "
        "(bug in the clip/splice logic, not the transcript)"
    )
    final_windows = _suspicious_windows(working, window_duration)
    missing = sum(
        _uncovered_by_confirmed_silence((start, end), confirmed_silent)
        for start, end in final_windows
        if (end - start) > _MIN_RECHECK_SECONDS
    )
    return working, missing


def _cut_audio_segment(src_path: Path, start: float, duration: float) -> Path | None:
    """Cut one time-slice ``[start, start+duration)`` out of ``src_path`` via
    ffmpeg, codec-copied, for a coverage retry.

    Lands in its own fresh temp dir (never the ``_split_audio`` chunk
    directory, whose lifecycle is unrelated) — the caller unlinks the file
    and removes the dir right after transcribing it. Returns ``None`` if
    ffmpeg is unavailable or the cut fails; the caller treats that as "give
    up this retry", never as a reason to fail the whole transcription.
    """
    ffmpeg = _ffmpeg_bin("ffmpeg")
    if not ffmpeg:
        return None
    tmp_dir = Path(tempfile.mkdtemp(prefix="tldr-recut-", dir=src_path.parent))
    suffix = src_path.suffix or ".opus"
    out = tmp_dir / f"retry{suffix}"
    try:
        subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error",
                "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
                "-i", str(src_path), "-c", "copy", str(out),
            ],
            check=True, capture_output=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        log.warning("transcribe: coverage-retry ffmpeg cut failed (%s)", exc)
        with contextlib.suppress(OSError):
            out.unlink(missing_ok=True)
            tmp_dir.rmdir()
        return None
    if not out.is_file() or out.stat().st_size == 0:
        with contextlib.suppress(OSError):
            out.unlink(missing_ok=True)
            tmp_dir.rmdir()
        return None
    return out


async def _post_audio(audio_path: Path) -> dict[str, Any]:
    """POST one audio file to the transcription endpoint, return parsed JSON."""
    cfg = get_config().whisper
    endpoint = f"{cfg.base_url.rstrip('/')}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {cfg.effective_api_key}"}

    with audio_path.open("rb") as fh:
        files = {"file": (audio_path.name, fh, "application/octet-stream")}
        data = {"model": cfg.model, "response_format": "verbose_json"}
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(endpoint, headers=headers, data=data, files=files)
            r.raise_for_status()
            result: dict[str, Any] = r.json()
            return result


def _parse_payload(
    payload: dict[str, Any], *, total_duration: float | None
) -> TranscribeResult:
    """Turn one transcription response into a TranscribeResult."""
    segments = _normalise_segments(payload.get("segments"))
    if not segments:
        # Unpatched server, or model returned text only. Construct a
        # single segment so build_marked_text still produces something
        # usable. raw_text loses fine-grained markers but summary works.
        full_text = str(payload.get("text") or "").strip()
        if full_text:
            end = float(total_duration) if total_duration and total_duration > 0 else 0.0
            segments = [{"start": 0.0, "end": end, "text": full_text}]
            log.warning(
                "transcribe: server returned no segments — using one-bucket "
                "fallback (no per-segment timing from this backend).",
            )

    raw_lang = payload.get("language")
    language: str | None = None
    if isinstance(raw_lang, str) and raw_lang.strip():
        # Some Whisper backends use full names ("english"); we normalise
        # the casing but leave the value as-is — the LLM language helper
        # later canonicalises to ISO-639-1.
        language = raw_lang.strip().lower()

    raw_duration = payload.get("duration")
    duration_seconds: float | None = None
    if isinstance(raw_duration, (int, float)) and raw_duration > 0:
        duration_seconds = float(raw_duration)
    elif total_duration and total_duration > 0:
        duration_seconds = float(total_duration)

    return TranscribeResult(
        segments=segments,
        language=language,
        duration_seconds=duration_seconds,
    )


def _ffmpeg_bin(name: str) -> str | None:
    """Path to ffmpeg/ffprobe from the resolver, or None if unavailable."""
    from src.workers.ffmpeg import resolve_ffmpeg_dir

    directory = resolve_ffmpeg_dir()
    if not directory:
        return None
    exe = f"{name}.exe" if os.name == "nt" else name
    candidate = Path(directory) / exe
    return str(candidate) if candidate.is_file() else None


def _probe_duration(audio_path: Path) -> float | None:
    """Audio duration in seconds via ffprobe, or None if it can't be read."""
    ffprobe = _ffmpeg_bin("ffprobe")
    if not ffprobe:
        return None
    try:
        out = subprocess.run(
            [
                ffprobe, "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(audio_path),
            ],
            check=True, capture_output=True, text=True,
        )
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, OSError, ValueError) as exc:
        log.warning("transcribe: ffprobe duration failed (%s)", exc)
        return None


def _split_audio(
    audio_path: Path, num_chunks: int, chunk_seconds: float
) -> list[tuple[Path, float]]:
    """Cut ``audio_path`` into ``num_chunks`` time slices, codec-copied.

    Returns ``[(chunk_path, start_offset_seconds), ...]``. Chunks land in a
    temp dir next to the source; the caller unlinks them. Returns ``[]`` when
    ffmpeg is unavailable or every cut fails.
    """
    ffmpeg = _ffmpeg_bin("ffmpeg")
    if not ffmpeg:
        log.warning("transcribe: no ffmpeg to split audio")
        return []

    tmp_dir = Path(tempfile.mkdtemp(prefix="tldr-chunks-", dir=audio_path.parent))
    suffix = audio_path.suffix or ".opus"
    chunks: list[tuple[Path, float]] = []
    for i in range(num_chunks):
        offset = i * chunk_seconds
        out = tmp_dir / f"chunk{i:03d}{suffix}"
        try:
            subprocess.run(
                [
                    ffmpeg, "-y", "-loglevel", "error",
                    "-ss", f"{offset:.3f}", "-t", f"{chunk_seconds:.3f}",
                    "-i", str(audio_path), "-c", "copy", str(out),
                ],
                check=True, capture_output=True,
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            log.warning("transcribe: ffmpeg chunk %d failed (%s)", i, exc)
            continue
        if out.is_file() and out.stat().st_size > 0:
            chunks.append((out, offset))
    return chunks


def _normalise_segments(raw: Any) -> list[dict[str, Any]]:
    """Coerce server's segment list into the ``build_marked_text`` shape.

    Drops malformed entries quietly rather than failing the whole
    transcription — one corrupt segment shouldn't kill an hour of work.
    """
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for seg in raw:
        if not isinstance(seg, dict):
            continue
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        out.append({"start": start, "end": end, "text": text})
    return out


__all__ = ["transcribe_audio", "TranscribeResult"]
