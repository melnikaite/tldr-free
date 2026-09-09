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

Fairness gating: worker-pool size vs. request concurrency
-----------------------------------------------------------

``runner.py``'s whisper worker now runs as a small pool
(``whisper.max_concurrent_jobs``), so multiple jobs' calls into this module
can be in flight at once. Two separate ``asyncio.Semaphore``s bound that:
``_global_whisper_lock()`` (sized by ``whisper.max_concurrent_requests``)
caps simultaneous Whisper HTTP requests across every job, and a fresh
``asyncio.Semaphore(1)`` created once per call to ``transcribe_audio()``
caps a single job's OWN concurrent requests at 1 — its chunk loop and
coverage-recheck loop are strictly sequential, so this is never actually
contended, but it's held explicitly rather than relied on as an accident of
that sequencing. Every ``_post_audio`` call site goes through
``_call_whisper``, which acquires both, per-job lock first. The global cap
is a FAIRNESS knob, not a throughput one — see ``_global_whisper_lock``'s
own docstring and ``WhisperConfig.max_concurrent_requests`` for the
measured backend behaviour that makes this true.

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

Rechecks are bounded per unit of work (one shared counter across every
slice of every suspicious interval IN THAT UNIT, so one huge hole's
slicing can't silently consume the whole budget and starve everything
else) and memoized so the loop provably terminates regardless of how many
suspicious intervals — or how large any one of them is — a pathological
transcript produces. The bound itself now SCALES with the unit's own
duration (``_coverage_recheck_budget``, config'd via
``whisper.coverage_recheck_budget_factor`` /
``min_coverage_rechecks`` / ``max_coverage_rechecks``) rather than being one
fixed number shared by a 30s chunk and a 300s one alike — see that
function's docstring. Memoisation is overlap-based, not exact-key
(``_overlaps_checked`` / ``_WINDOW_OVERLAP_DEDUP_FRACTION``): splicing a
retry's result back into the timeline shifts the REMAINING gap's edges by
a fraction of a second between outer-loop iterations, so an exact-key
"already asked this" check under-catches — measured live, three
consecutive rechecks (``469s-490s``, ``468s-490s``, ``467s-490s``) were
spent on what was really one hole, each overlapping the last by >90%. A
slice that comes back as real speech (or a degenerate run's recovered
first-occurrence portion) is spliced in using the same mechanism as before
(backoff context on the trailing edge, a wider leading-edge distrust
window ONLY on a suspicious interval's true leading edge — an artificial
internal split point between two slices of the same oversized interval
does not get that widening, since it isn't a real gap edge — clipped to
avoid a duplicate seam; see ``_ensure_coverage``'s own docstring). If a
slice can't be re-transcribed at all (ffmpeg unavailable), still reads as
an unresolved degenerate loop, or the budget runs out before a slice is
checked, we don't know what's there — conservatively, that still counts
toward ``missing_seconds`` rather than silently assuming it's fine.
Callers persist the residual on the job
(``Job.transcript_missing_seconds``) so the UI can tell the user this
"done" job's transcript is known-incomplete, instead of it looking
exactly like a full one.

**Language is detected once per call, then pinned.** Every Whisper request
independently auto-detects language when none is supplied — fine for a
short, unambiguous clip, but on a chunked long-form transcription each
chunk (and each chunk's own coverage rechecks) used to auto-detect
SEPARATELY. Measured live: the opening ~3 minutes of a German episode came
back transcribed in English while ``transcript_language`` was recorded as
``"de"`` — the first chunk's own detection wobbled on a noisy/musical
opening. The first request of a call to ``transcribe_audio`` (the whole
file, or a chunked path's first chunk) still auto-detects with no
``language`` sent; every request after that — the rest of that unit's
coverage rechecks, and every subsequent chunk plus ITS rechecks — pins
whatever language got bootstrapped from that first request (see
``_bootstrap_language``), rather than re-asking Whisper to guess again on
smaller, more ambiguous slices where a wrong guess is more likely.
Confirmed empirically before relying on this: the LocalAI/whisper.cpp
backend measurably CONSUMES the ``language`` form field for decoding (an
intentionally garbage value measurably changed the output), not merely
echoing it back in the response.

**The bootstrap has two tiers, because the backend that motivated fix 4
turned out not to REPORT the field it consumes.** Measured live against
the actual LocalAI instance and a real long-form episode: the response to
every ``POST /v1/audio/transcriptions`` carried no ``language`` field at
all — top-level keys were just ``duration``/``segments``/``text`` — even
though the same backend genuinely uses ``language`` when it IS sent on the
request (see ``_post_audio``'s docstring). Pinning "whatever the first
response reported" is therefore inert on this backend: the field being
pinned from is always absent, so ``pinned_language`` stays ``None`` for
the whole call and every request keeps auto-detecting independently —
precisely the failure this feature exists to eliminate, just never
actually engaging. So ``_bootstrap_language`` prefers the backend's own
report when present, and otherwise derives a language from the first
request's own transcribed text via ``llm.languages.detect_language`` —
the same function ``runner.py`` already falls back to, but applied here
BEFORE the rest of the call instead of after the whole transcription,
early enough to actually pin something. Provenance
(``"reported_by_backend"`` / ``"detected_from_text"`` / ``"unpinned"``) is
recorded on ``TranscribeDiagnostics`` — this incident is the reason that
matters.
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
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from src.config import get_config
from src.llm import languages
from src.workers.timecodes import collapse_repeated_segments

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _global_whisper_lock() -> asyncio.Semaphore:
    """Global cap on simultaneous Whisper HTTP calls across every job.

    See ``WhisperConfig.max_concurrent_requests`` — this is a fairness knob,
    not a throughput knob. Measured against a real backend (LocalAI,
    whisper-large, single GPU): 1 concurrent request took 8.0s wall, 2
    concurrent took 16.5s, 3 concurrent took 22.9s — the backend serialises
    requests FIFO internally, so concurrency here buys zero extra
    throughput. The only reason to allow more than 1 slot is fairness:
    letting a second job's chunk slip into the backend's FIFO instead of
    waiting for the first job's entire multi-chunk sequence to finish. See
    the config comment and ``.claude/workers.md`` for the full writeup.

    ``@lru_cache`` (mirroring ``llm.client._llm_lock``'s exact pattern) so
    the semaphore is created lazily and binds to the running event loop,
    rather than at import time.
    """
    n = max(1, get_config().whisper.max_concurrent_requests)
    return asyncio.Semaphore(n)


async def _call_whisper(
    per_job_lock: asyncio.Semaphore,
    audio_path: Path,
    *,
    language: str | None = None,
) -> dict[str, Any]:
    """Gate one ``_post_audio`` call through both semaphores: the per-job
    lock first (never contended in practice — see below), then the global
    fairness lock (this is the one that may actually wait on other jobs).

    The per-job lock is acquired first and is effectively free: a single
    job's own chunk loop (``_transcribe_chunked``) and its coverage-recheck
    loop (``_ensure_coverage``) are strictly sequential awaits with no
    ``gather``/``create_task`` between them, so at most one coroutine per
    job ever calls this at a time. It still has to be explicit and held,
    not just relied on as an accident of the current sequential code —
    that's what guards against a future change (e.g. someone parallelizing
    chunk processing for speed) silently breaking the fairness guarantee.

    ``language`` is passed straight through to ``_post_audio`` — ``None``
    (the default) means "let Whisper auto-detect", used for exactly one
    request per call to ``transcribe_audio`` (the first one). Every caller
    after that pins whatever language got detected — see the module
    docstring's "Language is detected once per call, then pinned" section.
    """
    async with per_job_lock, _global_whisper_lock():
        return await _post_audio(audio_path, language=language)


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
    # Non-sensitive record of how this transcription went — chunking
    # decision, every coverage-recheck verdict, backend/model, yt-dlp
    # version. None only when this TranscribeResult was constructed
    # somewhere that never populated it (defensive default; every real
    # transcribe_audio() call fills it in). See TranscribeDiagnostics.
    diagnostics: TranscribeDiagnostics | None = None


@dataclass
class ChunkingDiagnostics:
    """Non-sensitive record of the chunking decision for one
    ``transcribe_audio()`` call: was the audio split before upload, and
    why. Part of ``TranscribeDiagnostics`` — see its docstring for the
    privacy rationale and where this ends up persisted."""

    chunked: bool = False
    audio_size_bytes: int | None = None
    max_upload_bytes: int | None = None
    num_chunks: int | None = None
    chunk_seconds: float | None = None
    # How many chunks EACH bound independently implied — populated only on
    # the chunked path, where num_chunks = max(chunks_by_bytes,
    # chunks_by_seconds). Whichever is larger determined num_chunks; both are
    # kept (not just the winner) so a diagnostic reader can see how close the
    # other bound came, not just which one won.
    chunks_by_bytes: int | None = None
    chunks_by_seconds: int | None = None
    # The whisper.max_chunk_seconds value in effect for this call — config
    # can change between runs, so the raw number alone (chunks_by_seconds)
    # isn't reconstructable without this.
    max_chunk_seconds_cap: float | None = None
    reason: str = ""


@dataclass
class TranscribeDiagnostics:
    """A persistable, privacy-safe record of one ``transcribe_audio()``
    call, built up as the call proceeds — the chunking decision
    (``ChunkingDiagnostics``), every coverage-recheck window and its
    verdict (``_ensure_coverage``), the Whisper backend/model in use, the
    yt-dlp version, and the final ``missing_seconds``. ``runner.py``
    JSON-serialises this (``dataclasses.asdict``) onto
    ``Job.diagnostics_json`` (migration v10) so a bad transcription can be
    diagnosed without reading rotating logs by hand — see
    ``api/jobs.py``'s ``GET /jobs/{id}/diagnostics``.

    Deliberately excludes, by construction (never even threaded in here to
    begin with): cookies (this module never receives them), the audio
    bytes, and any transcript TEXT — ``coverage_rechecks`` entries carry
    only timestamps, counts, and a closed-set verdict string, never the
    text a recheck actually produced. ``whisper_backend_base_url`` /
    ``whisper_model`` are the same values already surfaced verbatim by
    ``GET /config`` — configuration, not secrets.
    """

    chunking: ChunkingDiagnostics = field(default_factory=ChunkingDiagnostics)
    coverage_rechecks: list[dict[str, Any]] = field(default_factory=list)
    # The CONFIGURED ceiling (whisper.max_coverage_rechecks) in effect for
    # this call — a summary/reference value. The budget actually enforced
    # for any one unit (chunk, or the whole file) is usually smaller and
    # scales with that unit's own duration; see each coverage_rechecks
    # entry's own "of" field for the real per-unit number, and
    # _coverage_recheck_budget for the formula.
    max_coverage_rechecks: int = 0
    # Suspicious windows that were SKIPPED because they overlapped an
    # already-checked window beyond the dedup threshold (see
    # _WINDOW_OVERLAP_DEDUP_FRACTION / _overlaps_checked) rather than
    # because they were resolved — evidence that memoisation actually did
    # something, distinct from a normal recheck outcome.
    overlap_suppressed: list[dict[str, Any]] = field(default_factory=list)
    final_missing_seconds: float = 0.0
    # The language pinned for every request after this call's first one
    # (see _bootstrap_language / the module docstring's "Language is
    # detected once per call, then pinned" section), and how it was
    # obtained: "reported_by_backend" (the first request's own response
    # carried a language), "detected_from_text" (the backend never reports
    # one at all — measured live against LocalAI/whisper.cpp — so it was
    # derived from the first request's transcribed text via
    # llm.languages.detect_language), or "unpinned" (neither source gave a
    # confident answer, so every request in this call still auto-detects
    # independently, same as before fix 4 existed). A code alone
    # ("de") is not sensitive — the same value already lands in
    # Job.transcript_language.
    pinned_language: str | None = None
    language_pin_source: str = "unpinned"
    # Phase 6 (owner ruling, 2026-08-28 — "a wrong transcript beats a
    # hole"): every span where the recheck budget ran out unresolved but
    # the FIRST PASS had produced something, so the original (low-
    # confidence) text was restored instead of leaving the span empty —
    # see _restore_unresolved_windows. Per-span detail (unit + window,
    # never text — same privacy posture as coverage_rechecks) plus the two
    # summary numbers a reader actually wants at a glance: how many spans,
    # and how many seconds of the transcript are now backfilled rather
    # than either missing outright or fully trustworthy.
    backfilled_spans: list[dict[str, Any]] = field(default_factory=list)
    backfill_count: int = 0
    backfill_total_seconds: float = 0.0
    whisper_backend_base_url: str | None = None
    whisper_model: str | None = None
    yt_dlp_version: str | None = None

    def record_recheck(
        self, *, unit: str, index: int, of: int, start: float, end: float, verdict: str,
    ) -> None:
        """Append one coverage-recheck outcome. ``verdict`` is one of
        "recovered_real_speech" / "confirmed_non_speech" /
        "decode_loop_again" / "recut_unavailable" / "skipped_too_small" /
        "recovered_unpinned" / "confirmed_non_speech_both_ways" — the last
        two only occur when a language was pinned for this call AND the
        pinned decode came back confirmed-silent, triggering the one-shot
        unpinned (auto-detect) retry of the SAME cut (see the "language and
        _is_confirmed_silence" branch in ``_ensure_coverage``): the pin
        itself either cost us content ("recovered_unpinned" — the
        unpinned retry found real/degenerate text the pinned decode
        missed) or didn't ("confirmed_non_speech_both_ways" — both decodes
        agree there's nothing there). See the call sites in
        ``_ensure_coverage``, each of which already logs the same event;
        this just gives it a durable, structured twin."""
        self.coverage_rechecks.append(
            {
                "unit": unit,
                "index": index,
                "of": of,
                "window_start": round(start, 1),
                "window_end": round(end, 1),
                "verdict": verdict,
            }
        )

    def record_overlap_suppressed(self, *, unit: str, start: float, end: float) -> None:
        """Append one window that memoisation skipped as a near-duplicate
        of an already-checked one — see _overlaps_checked."""
        self.overlap_suppressed.append(
            {
                "unit": unit,
                "window_start": round(start, 1),
                "window_end": round(end, 1),
            }
        )

    def record_backfill(self, *, unit: str, start: float, end: float) -> None:
        """Append one span that Phase 6 restored original first-pass text
        into (marked low-confidence) instead of leaving it empty — see
        _restore_unresolved_windows. Updates the two summary counters too,
        so a reader doesn't have to sum backfilled_spans by hand."""
        self.backfilled_spans.append(
            {
                "unit": unit,
                "window_start": round(start, 1),
                "window_end": round(end, 1),
            }
        )
        self.backfill_count += 1
        self.backfill_total_seconds += end - start


def _yt_dlp_version() -> str | None:
    """Best-effort yt-dlp version string for the diagnostic record.
    Imported lazily (module-load shouldn't pay this cost — same rationale
    as ``workers/youtube.py``'s lazy ``yt_dlp`` imports) and swallows every
    failure to ``None``; a missing/unreadable version must never block
    transcription."""
    try:
        from yt_dlp.version import __version__ as yt_dlp_version

        return str(yt_dlp_version)
    except Exception:
        return None


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

# Cap on RECHECKS per unit of work (one chunk, or the whole file on the
# single-request path) — the budget the module docstring promises. Bounded
# the same way translator.py bounds its bisection retries — each recheck
# only re-transcribes ONE SLICE (see _MAX_RECHECK_SLICE_SECONDS — never a
# whole oversized interval, never the whole unit), and a slice already
# checked this call (or overlapping one closely enough, see
# _overlaps_checked) is never asked again, so together this guarantees the
# loop terminates and total added cost per unit stays bounded regardless of
# how many distinct suspicious intervals — or how large any one of them is —
# a pathological transcript produces.
#
# One shared counter across every slice of every suspicious interval in the
# unit (not a separate budget per interval) is deliberate: it's what stops
# ONE oversized hole's slicing from silently consuming the ENTIRE budget and
# starving every other suspicious interval in the same unit — once an
# interval's own slices are all in ``checked``, it stops competing for
# budget, freeing the rest for whatever else is pending.
#
# UNLIKE the original version of this cap, the budget is no longer one fixed
# number shared by every unit regardless of size — see _coverage_recheck_budget
# just below. A fixed 12 was derived from ONE incident's worst hole (562s
# inside an unbounded ~748s chunk); once whisper.max_chunk_seconds bounds how
# long a first-pass chunk can even BE (see WhisperConfig), a short chunk
# doesn't need the same budget as a much longer one, and a first-pass unit
# that's still long (the single-request path has no chunk-size bound at all)
# should get more than 12 rather than being stuck with a number sized for a
# different, smaller unit.
def _coverage_recheck_budget(window_duration: float) -> int:
    """How many rechecks THIS unit of work (one chunk, or the whole file on
    the single-request path) gets — scaled by the unit's OWN duration
    instead of one fixed number shared by every unit regardless of size.

    Formula: how many _MAX_RECHECK_SLICE_SECONDS-sized slices it would take
    to recheck the unit's ENTIRE window (i.e. the worst case — the whole
    unit turns out to be one giant hole), times
    ``whisper.coverage_recheck_budget_factor`` for headroom (a second,
    smaller hole in the same unit; a slice that only partially resolves via
    a degenerate-repeat splice and needs a follow-up on its remainder — see
    ``_ensure_coverage``). Clamped to
    ``[whisper.min_coverage_rechecks, whisper.max_coverage_rechecks]`` so
    neither a very short unit nor a very long one escapes a small, bounded,
    auditable budget — "scaled" must never mean "unbounded".

    Worked example matching the original fixed value's derivation: a 300s
    chunk (the default whisper.max_chunk_seconds) needs
    ceil(300 / 90) = 4 slices to recheck end to end; with the default
    factor 2.0 that's 8 — comfortably more than the 4 needed if the ENTIRE
    chunk turned out to be one hole, same shape as the original 12-from-7
    reasoning, scaled down because the unit itself is now bounded smaller.
    """
    cfg = get_config().whisper
    slices_needed = max(1, math.ceil(window_duration / _MAX_RECHECK_SLICE_SECONDS))
    scaled = math.ceil(slices_needed * cfg.coverage_recheck_budget_factor)
    return min(cfg.max_coverage_rechecks, max(cfg.min_coverage_rechecks, scaled))


async def transcribe_audio(
    audio_path: Path,
    *,
    total_duration: float | None,
    metadata_language: str | None = None,
) -> TranscribeResult:
    """Transcribe the audio, splitting it first if it exceeds the upload cap.

    ``metadata_language`` is the PUBLISHER's own declared language — e.g.
    yt-dlp's ``language``/``original_language`` metadata field
    (``workers/youtube.fetch_video_metadata``), threaded in by
    ``workers/runner.py``. Raw, un-normalised (``"deu"``, ``"de"``,
    ``"German"`` all accepted) — normalised once here via
    ``_normalize_metadata_language`` before use. When present, it is the
    HIGHEST-priority source for the language pinned across every request
    in this call (see ``_bootstrap_language``): authoritative, free, and
    immune to the Whisper-mis-detection failure mode text-based bootstrap
    fell into (see that function's docstring). ``None`` when the caller
    doesn't have one — falls through to the backend's own report, then
    (chunked path only) a two-independent-chunk-agreement text detection,
    then unpinned per-request auto-detect, exactly as before this
    parameter existed.

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

    Every Whisper HTTP call made on behalf of THIS job — the initial
    request(s) and any coverage recheck — goes through one
    ``asyncio.Semaphore(1)`` created here, fresh per call to this function
    (i.e. per job's whole transcription unit of work), guaranteeing at most
    one in-flight Whisper request per job regardless of how many jobs are
    running concurrently in the worker pool. See ``_call_whisper`` and
    ``_global_whisper_lock`` for the rest of the gating.
    """
    per_job_lock = asyncio.Semaphore(1)
    cfg = get_config().whisper
    max_bytes = max(1, cfg.max_upload_mb) * 1024 * 1024
    size = audio_path.stat().st_size
    normalized_metadata_language = _normalize_metadata_language(metadata_language)

    # Non-sensitive diagnostic record for this call — see
    # TranscribeDiagnostics. Built up as the call proceeds and handed back
    # on the result so runner.py can persist it (Job.diagnostics_json,
    # migration v10) without this module knowing anything about storage.
    diagnostics = TranscribeDiagnostics(
        max_coverage_rechecks=cfg.max_coverage_rechecks,
        whisper_backend_base_url=cfg.base_url,
        whisper_model=cfg.model,
        yt_dlp_version=_yt_dlp_version(),
    )

    if size <= max_bytes:
        diagnostics.chunking = ChunkingDiagnostics(
            chunked=False,
            audio_size_bytes=size,
            max_upload_bytes=max_bytes,
            reason=(
                f"{size / 1024 / 1024:.1f} MB <= {max_bytes / 1024 / 1024:.0f} MB "
                "cap — single request"
            ),
        )
        result = await _transcribe_whole(
            audio_path,
            total_duration=total_duration,
            per_job_lock=per_job_lock,
            diagnostics=diagnostics,
            metadata_language=normalized_metadata_language,
        )
    else:
        result = await _transcribe_chunked(
            audio_path,
            total_duration=total_duration,
            max_bytes=max_bytes,
            per_job_lock=per_job_lock,
            metadata_language=normalized_metadata_language,
            diagnostics=diagnostics,
        )

    diagnostics.final_missing_seconds = result.missing_seconds
    final_segments, _discarded = collapse_repeated_segments(result.segments)
    return TranscribeResult(
        segments=final_segments,
        language=result.language,
        duration_seconds=result.duration_seconds,
        missing_seconds=result.missing_seconds,
        diagnostics=diagnostics,
    )


def _normalize_metadata_language(raw: object) -> str | None:
    """Normalise a publisher-declared language (yt-dlp's ``language`` /
    ``original_language`` metadata field — e.g. ``"deu"``, ``"de"``,
    ``"German"``) to the ISO-639-1 code Whisper's ``language`` request
    field expects.

    Reuses ``llm.languages.normalize_lang`` — the SAME canonicalisation
    already applied to user-typed translation targets, whose alias table
    already maps common ISO-639-2 codes (``"deu"`` -> ``"de"``) — rather
    than duplicating that table here. Anything ``normalize_lang`` doesn't
    recognise (a code outside the ~20 languages it knows, garbage, empty)
    comes back ``None`` rather than being passed through raw: sending
    Whisper a code it doesn't understand is worse than sending none at all
    (falls back to auto-detect / the text-agreement fallback below,
    exactly as if metadata had never reported anything).
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return languages.normalize_lang(raw).code
    except languages.UnknownLanguageError:
        return None


def _bootstrap_language(
    payload: dict[str, Any],
    result: TranscribeResult,
    *,
    metadata_language: str | None,
) -> tuple[str | None, str]:
    """Decide the language to PIN starting from this call's very FIRST
    request, and how that decision was reached — the second value is a
    provenance tag (``"from_metadata"`` / ``"reported_by_backend"`` /
    ``"unpinned"``) recorded on ``TranscribeDiagnostics`` so a future
    incident doesn't need to rediscover this by hand.

    Priority, highest first:

    1. ``metadata_language`` — the PUBLISHER's own declaration (yt-dlp's
       ``language``/``original_language`` metadata field, already
       normalised by ``_normalize_metadata_language``; threaded in from
       ``workers/runner.py``, which extracts it via
       ``workers/youtube.fetch_video_metadata``). Authoritative, free (the
       metadata probe already happens for other reasons), and immune to
       the failure below — it doesn't depend on Whisper's own output at
       all, so it can't be corrupted by Whisper mis-detecting.
    2. Whatever the backend itself reported (``result.language``, from
       ``_parse_payload``) — a single sample, but a self-reported model
       decision over the actual audio, not a guess from already-possibly-
       wrong text.

    Deliberately does NOT fall back to text-detection here any more. A
    live regression (job ``r56sj0_o1ihv``, a 24:56 German episode) showed
    why: ``llm.languages.detect_language`` on chunk 0's own transcribed
    text returned a CONFIDENT ``"en"`` for chunk 0's musical, sparse-
    dialogue opening (the exact stretch Whisper's own auto-detect already
    wobbles on) — detect_language's short-input guard did not save this,
    because the input was long enough, just wrong. That "en" then got
    PINNED and forced onto every subsequent chunk, corrupting a transcript
    that was 96% correct without any pinning at all (9.6 German-marker-
    word ratio before this feature existed, 0.0 after). The perverse
    dynamic: text detection was being applied to text produced by the very
    mis-detection it was supposed to correct — a single low-confidence
    sample can never safely be the PRIMARY signal for something that then
    gets forced onto every other sample. See ``_transcribe_chunked`` for
    the fallback this function no longer performs: text-detection is only
    trusted there when TWO INDEPENDENT chunks agree, never from one.
    """
    if metadata_language:
        return metadata_language, "from_metadata"
    if result.language:
        return result.language, "reported_by_backend"
    return None, "unpinned"


async def _transcribe_whole(
    audio_path: Path,
    *,
    total_duration: float | None,
    per_job_lock: asyncio.Semaphore,
    diagnostics: TranscribeDiagnostics | None = None,
    unit_label: str = "whole",
    metadata_language: str | None = None,
) -> TranscribeResult:
    """Transcribe ``audio_path`` in one request, checking + retrying coverage
    when ``total_duration`` is known. Shared by the single-request path and
    both of ``_transcribe_chunked``'s "can't actually chunk" fallbacks —
    ``unit_label`` distinguishes those in the diagnostic record (default
    ``"whole"``, since a fallback callsite still means "we ended up doing
    ONE request", even if chunking was attempted first).

    Language: when ``metadata_language`` (the publisher's own declaration —
    see ``_bootstrap_language``) is known, THIS request is sent with it
    already (never auto-detects at all); otherwise it auto-detects (no
    ``language`` sent — see ``_call_whisper``'s docstring). Either way,
    whatever gets PINNED (``_bootstrap_language`` — metadata, or the
    backend's own report) is used for every coverage-recheck request
    ``_ensure_coverage`` makes below, so a recheck never re-guesses on a
    smaller, more ambiguous slice. Deliberately never falls back to
    single-sample text-detection here — this is the ONLY request in the
    unit, so there is no second, independent sample to require agreement
    from (see ``_transcribe_chunked`` for where that fallback lives); if
    metadata is absent and the backend doesn't self-report, this stays
    unpinned and every recheck auto-detects independently, exactly as
    before language pinning existed at all."""
    payload = await _call_whisper(per_job_lock, audio_path, language=metadata_language)
    result = _parse_payload(payload, total_duration=total_duration)
    pinned_language, provenance = _bootstrap_language(
        payload, result, metadata_language=metadata_language
    )
    if diagnostics is not None:
        diagnostics.pinned_language = pinned_language
        diagnostics.language_pin_source = provenance
    if total_duration is None or total_duration <= 0:
        # Nothing to compare against — can't tell a short transcript from a
        # short recording. Same trust-the-backend behaviour as before this
        # feature existed.
        return result
    segments, missing = await _ensure_coverage(
        result.segments,
        source_path=audio_path,
        window_duration=total_duration,
        per_job_lock=per_job_lock,
        diagnostics=diagnostics,
        unit_label=unit_label,
        language=pinned_language,
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
    per_job_lock: asyncio.Semaphore,
    diagnostics: TranscribeDiagnostics | None = None,
    metadata_language: str | None = None,
) -> TranscribeResult:
    """Split oversized audio with ffmpeg, transcribe parts, merge segments.

    ``metadata_language`` (see ``_bootstrap_language``) is used from chunk
    0's OWN first request onward when known; when it's absent, language
    pinning here additionally supports a text-detection fallback that
    requires TWO independent chunks (0 and 1) to agree before pinning
    anything — see the loop below and the module docstring's language
    section for why a single chunk's text-based guess is never trusted."""
    duration = total_duration if total_duration and total_duration > 0 else None
    if duration is None:
        duration = await asyncio.to_thread(_probe_duration, audio_path)
    if duration is None or duration <= 0:
        # Can't time-slice without a duration — fall back to one shot and let
        # the backend's own error surface if it really is too big. No known
        # duration also means no coverage check is possible here either.
        log.warning("transcribe: unknown duration, cannot chunk; trying single upload")
        if diagnostics is not None:
            diagnostics.chunking = ChunkingDiagnostics(
                chunked=False,
                reason="unknown audio duration — cannot chunk; single upload",
            )
        return await _transcribe_whole(
            audio_path,
            total_duration=total_duration,
            per_job_lock=per_job_lock,
            diagnostics=diagnostics,
            metadata_language=metadata_language,
        )

    size = audio_path.stat().st_size
    # Target 90% of the cap for VBR headroom; at least 2 chunks since we're here.
    target = max(1, int(max_bytes * 0.9))
    chunks_by_bytes = math.ceil(size / target)
    # Bound chunk DURATION too, not just size — see WhisperConfig.
    # max_chunk_seconds's own comment for why the byte cap alone isn't
    # enough (well-compressed audio can pack many minutes under it, and a
    # first-pass request that long reproduces the exact decode-loop failure
    # _MAX_RECHECK_SLICE_SECONDS already works around for rechecks).
    max_chunk_seconds = get_config().whisper.max_chunk_seconds
    chunks_by_seconds = (
        math.ceil(duration / max_chunk_seconds) if max_chunk_seconds > 0 else 1
    )
    num_chunks = max(2, chunks_by_bytes, chunks_by_seconds)
    chunk_seconds = duration / num_chunks
    bound = "seconds" if chunks_by_seconds > chunks_by_bytes else "bytes"
    log.info(
        "transcribe: audio %.1f MB > cap → %d chunks of ~%.0f s (bound: %s; "
        "%d by size, %d by %.0fs/chunk cap)",
        size / 1024 / 1024,
        num_chunks,
        chunk_seconds,
        bound,
        chunks_by_bytes,
        chunks_by_seconds,
        max_chunk_seconds,
    )
    if diagnostics is not None:
        diagnostics.chunking = ChunkingDiagnostics(
            chunked=True,
            audio_size_bytes=size,
            max_upload_bytes=max_bytes,
            num_chunks=num_chunks,
            chunk_seconds=round(chunk_seconds, 1),
            chunks_by_bytes=chunks_by_bytes,
            chunks_by_seconds=chunks_by_seconds,
            max_chunk_seconds_cap=max_chunk_seconds,
            reason=(
                f"{size / 1024 / 1024:.1f} MB > {max_bytes / 1024 / 1024:.0f} MB "
                f"cap -> {chunks_by_bytes} chunks by size; {duration:.0f}s / "
                f"{max_chunk_seconds:.0f}s cap -> {chunks_by_seconds} chunks by "
                f"duration -> using {num_chunks} chunks of ~{chunk_seconds:.0f}s "
                f"(bound: {bound})"
            ),
        )

    chunks = await asyncio.to_thread(
        _split_audio, audio_path, num_chunks, chunk_seconds
    )
    if not chunks:
        log.warning("transcribe: ffmpeg split produced nothing; trying single upload")
        if diagnostics is not None:
            diagnostics.chunking = ChunkingDiagnostics(
                chunked=False,
                audio_size_bytes=size,
                max_upload_bytes=max_bytes,
                reason="ffmpeg split produced nothing — single upload",
            )
        return await _transcribe_whole(
            audio_path,
            total_duration=duration,
            per_job_lock=per_job_lock,
            diagnostics=diagnostics,
            metadata_language=metadata_language,
        )

    all_segments: list[dict[str, Any]] = []
    language: str | None = None
    # Pinned language for every request from chunk 0 onward — see the
    # module docstring's "Language is detected once per call, then pinned"
    # section. Decided from chunk 0's own response via _bootstrap_language
    # (metadata, or the backend's own report), so chunk 0's own coverage
    # rechecks (right below) already get the pinned value, not just chunk
    # 1 onward.
    #
    # When NEITHER metadata nor the backend gives an answer, this ALSO
    # supports one more fallback that _bootstrap_language deliberately does
    # NOT perform on its own: text-detection, but ONLY pinned once a
    # SECOND, independent chunk (chunk 1) agrees with chunk 0's own
    # detected guess. ``chunk0_text_lang`` holds that first guess as a mere
    # CANDIDATE — never assigned to ``pinned_language`` on its own — until
    # chunk 1 either confirms it (pin, provenance "detected_agreed") or
    # disagrees / can't be detected either (stays unpinned for the rest of
    # the call, falling back to today's per-request auto-detect). This is
    # the fix for a live regression (job r56sj0_o1ihv): chunk 0's own text
    # detection alone returned a confident WRONG answer ("en" for a German
    # episode) that then got forced onto every other chunk, corrupting an
    # otherwise-96%-correct transcript. See _bootstrap_language's own
    # docstring for the measured before/after.
    #
    # Seeded with metadata_language (already normalised), not None: unlike
    # a backend self-report or text-detection, metadata is known BEFORE
    # any Whisper call happens at all, so chunk 0's own FIRST request
    # should already be pinned when we have it, not just requests after
    # it.
    pinned_language: str | None = metadata_language
    language_provenance = "from_metadata" if metadata_language else "unpinned"
    chunk0_text_lang: str | None = None
    missing_total = 0.0
    try:
        for idx, (chunk_path, offset) in enumerate(chunks):
            # Sequential await, one chunk at a time — no gather/create_task
            # here — is what makes the per-job semaphore never contended in
            # practice (see _call_whisper's docstring); it's still acquired
            # explicitly on every call rather than relied on implicitly.
            payload = await _call_whisper(per_job_lock, chunk_path, language=pinned_language)
            part = _parse_payload(payload, total_duration=chunk_seconds)
            if idx == 0:
                pinned_language, language_provenance = _bootstrap_language(
                    payload, part, metadata_language=metadata_language
                )
                if pinned_language is None:
                    # No metadata, no backend report — hold a CANDIDATE
                    # from chunk 0's own text, but do NOT pin it: see this
                    # block's own comment above and _bootstrap_language's
                    # docstring for why a single chunk's text-based guess
                    # is never trusted on its own any more.
                    chunk0_text_lang = languages.detect_language(
                        str(payload.get("text") or "")
                    )
            elif idx == 1 and pinned_language is None and chunk0_text_lang is not None:
                chunk1_text_lang = languages.detect_language(str(payload.get("text") or ""))
                if chunk1_text_lang is not None and chunk1_text_lang == chunk0_text_lang:
                    pinned_language = chunk0_text_lang
                    language_provenance = "detected_agreed"
                # Disagreement (or chunk 1 undetectable too) — stays
                # unpinned deliberately; do not average, do not prefer
                # either guess, do not retry with a third chunk. Two
                # independent samples disagreeing is exactly the signal
                # that text-detection isn't reliable for THIS episode.
            if diagnostics is not None:
                diagnostics.pinned_language = pinned_language
                diagnostics.language_pin_source = language_provenance
            # The last chunk may be shorter than chunk_seconds if the file
            # doesn't divide evenly (ffmpeg's -t just stops at EOF) — use
            # whatever's actually left of the known total as this chunk's
            # expected coverage, not the nominal per-chunk length.
            expected_local_duration = min(chunk_seconds, duration - offset)
            segments, missing = await _ensure_coverage(
                part.segments,
                source_path=chunk_path,
                window_duration=expected_local_duration,
                per_job_lock=per_job_lock,
                diagnostics=diagnostics,
                unit_label=f"chunk {idx + 1}/{len(chunks)}",
                language=pinned_language,
            )
            if missing > 0:
                missing_total += missing
                log.warning(
                    "transcribe: chunk %d/%d still short by ~%.0fs of audio "
                    "after retries — transcript may be missing content there",
                    idx + 1, len(chunks), missing,
                )
            for seg in segments:
                # dict(seg, ...) — NOT a fresh {"start"/"end"/"text"}
                # literal — preserves any OTHER key the segment carries
                # (e.g. Phase 6's low_confidence flag on a restored
                # segment) instead of silently stripping it. Measured
                # regression: the original 3-key literal here dropped
                # low_confidence on every chunked job's backfilled spans,
                # even though _restore_unresolved_windows set it correctly
                # per chunk — this loop discarded it on the very next line
                # merging chunks back together.
                all_segments.append(
                    dict(seg, start=seg["start"] + offset, end=seg["end"] + offset)
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


def _segment_ending_at(segments: list[dict[str, Any]], t: float) -> dict[str, Any] | None:
    """The segment in ``segments`` whose ``end`` exactly equals ``t``, or
    ``None``. Used by ``_restore_unresolved_windows`` to find whatever
    already sits immediately before a restored span — ``t`` there is
    always a cursor value ``_find_gaps`` derived directly from some
    segment's own unmodified ``end`` (``max(cursor, end)``), so exact
    float equality is safe here, not a rounding gamble."""
    for seg in segments:
        if _segment_end(seg) == t:
            return seg
    return None


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
    """Rounding helper for logging/diagnostics only — NOT used for dedup
    membership any more (see ``_overlaps_checked``): exact-key equality
    under-catches, because splicing a retry's result back into the timeline
    shifts the REMAINING gap's edges by a fraction of a second on the very
    next outer-loop iteration (see that function's docstring for the
    measured example), so the "same" unresolved hole almost never recurs
    bit-for-bit."""
    return (round(start, 3), round(end, 3))


# How much of a CANDIDATE window must already be covered by something in
# ``checked`` — as a fraction of the CANDIDATE's own length — before it's
# treated as "already checked" rather than a genuinely new recheck target.
#
# Exact-boundary equality (the ORIGINAL memoisation rule) is the trivial
# 100%-overlap case but isn't enough on its own: splicing a retry's result
# back into the timeline shifts the REMAINING gap's edges by a fraction of
# a second (a decode-loop leaves a slightly different confirmed segment on
# either side depending on exactly where the retry's own output got
# clipped — see ``_clip_to_gap``), so the "same" unresolved hole reappears
# as a slightly different ``(start, end)`` pair on the next outer-loop
# iteration and an exact key match never fires again. Measured live in job
# ``r56sj0_o1ihv``: THREE separate rechecks spent on what was really one
# ~22s hole — ``469s-490s``, then ``468s-490s``, then ``467s-490s`` — each
# consecutive pair overlapping the other by more than 90% of the shorter
# window's length, burning 3 slots of a 12-slot budget on a hole that
# needed exactly ONE.
#
# 0.8 sits below that measured >90% floor with real margin, while still
# requiring MOST of a CANDIDATE window to coincide with something already
# checked before two genuinely distinct suspicious intervals could ever be
# confused for the same one — and two such intervals that touched or
# overlapped at all would already have been folded into one by
# ``_merge_intervals`` upstream, so two windows surviving as separate
# ``_suspicious_windows`` entries in the first place are already either
# non-overlapping or overlapping only slightly.
_WINDOW_OVERLAP_DEDUP_FRACTION = 0.8


def _overlaps_checked(
    window: tuple[float, float], checked: list[tuple[float, float]]
) -> bool:
    """True when at least ``_WINDOW_OVERLAP_DEDUP_FRACTION`` of ``window``
    (the CANDIDATE, about to become a new recheck) is already covered by
    some interval in ``checked`` — see that constant's own docstring for
    why exact-key equality isn't enough.

    Deliberately measured against the CANDIDATE's own length, NOT
    ``min(candidate, checked)`` — that symmetric form has a real coverage
    hole in the opposite direction from the one this function exists to
    fix: a LARGE new candidate that merely *contains* a small
    already-checked interval (e.g. checked has a 10s slice at
    ``(400, 410)``, and splicing produces a fresh 90s candidate
    ``(400, 490)``) would read as "already checked" under ``min()``
    (overlap 10s / min(90, 10) = 1.0), silently skipping the 80s that were
    NEVER actually examined — the same class of bug (memoisation answering
    the wrong question) this function was written to fix, just inverted.
    Measuring against the candidate's own length instead means a small
    candidate fully inside a large checked interval still suppresses
    (overlap equals the candidate's whole length, ratio 1.0), while a large
    candidate that only partially overlaps a small checked interval does
    NOT (10s / 90s = 0.11, well under the threshold) — the asymmetry is the
    point, since the question is "is this candidate mostly already
    covered", which is a property of the candidate, not of whichever
    interval happens to be shorter.
    """
    start, end = window
    length = end - start
    if length <= 0:
        return True
    for c_start, c_end in checked:
        overlap = min(end, c_end) - max(start, c_start)
        if overlap <= 0:
            continue
        if overlap / length >= _WINDOW_OVERLAP_DEDUP_FRACTION:
            return True
    return False


def _pending_slices(
    working: list[dict[str, Any]],
    window_duration: float,
    checked: list[tuple[float, float]],
) -> tuple[list[tuple[float, float, bool]], list[tuple[float, float]]]:
    """Every re-checkable ``(slice_start, slice_end, is_leading)`` still
    worth asking Whisper about: suspicious windows above the cost cutoff
    (``_MIN_RECHECK_SECONDS``), split into ``_MAX_RECHECK_SLICE_SECONDS``
    slices so no single recheck request ever covers more than that, minus
    whatever overlaps something already in ``checked`` beyond
    ``_WINDOW_OVERLAP_DEDUP_FRACTION`` (see ``_overlaps_checked``).

    Returns ``(pending, suppressed)`` — the still-open slices, and the
    ones filtered out specifically because they matched something already
    checked (as opposed to being below the cost cutoff, which isn't
    "suppressed", just never a recheck candidate at all). The caller uses
    ``suppressed`` only for diagnostics — it plays no role in the budget or
    the loop's termination.
    """
    pending: list[tuple[float, float, bool]] = []
    suppressed: list[tuple[float, float]] = []
    for start, end in _suspicious_windows(working, window_duration):
        if (end - start) <= _MIN_RECHECK_SECONDS:
            continue
        for slice_start, slice_end, leading in _split_into_slices(
            start, end, _MAX_RECHECK_SLICE_SECONDS
        ):
            if _overlaps_checked((slice_start, slice_end), checked):
                suppressed.append((slice_start, slice_end))
            else:
                pending.append((slice_start, slice_end, leading))
    return pending, suppressed


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


def _subtract_confirmed_silent(
    window: tuple[float, float], confirmed_silent: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """``window`` minus every overlapping interval in ``confirmed_silent``,
    as a list of the ``[start, end)`` pieces still genuinely UNRESOLVED (0,
    1, or more pieces, depending on how many confirmed-silent intervals cut
    into the middle of ``window``).

    Used ONLY to decide what Phase 6's restoration
    (``_restore_unresolved_windows``) is allowed to backfill —
    ``missing_seconds`` itself is computed with the length-only
    ``_uncovered_by_confirmed_silence`` above and is NOT affected by this
    function. The two answer different questions: that one asks "how much
    of this window is still uncertain" (a number); this one asks "which
    exact sub-ranges are still uncertain" (so restoration never touches a
    span a recheck already independently confirmed has no speech in).
    """
    pieces = [window]
    for silent_start, silent_end in confirmed_silent:
        next_pieces: list[tuple[float, float]] = []
        for start, end in pieces:
            if silent_end <= start or silent_start >= end:
                next_pieces.append((start, end))
                continue
            if silent_start > start:
                next_pieces.append((start, silent_start))
            if silent_end < end:
                next_pieces.append((silent_end, end))
        pieces = next_pieces
    return [(start, end) for start, end in pieces if end - start > 0]


def _restore_unresolved_windows(
    working: list[dict[str, Any]],
    original_segments: list[dict[str, Any]],
    final_windows: list[tuple[float, float]],
    confirmed_silent: list[tuple[float, float]],
    *,
    diagnostics: TranscribeDiagnostics | None,
    unit_label: str,
) -> list[dict[str, Any]]:
    """Owner ruling, 2026-08-28: **"a wrong transcript beats a hole."**
    Never let a span end up with literally ZERO text once the recheck
    budget for this unit is exhausted, as long as the FIRST PASS
    (``original_segments`` — captured in ``_ensure_coverage`` before any
    recheck splice ran, since a splice can itself remove the original
    content for a span it only partially resolved) actually produced
    SOMETHING for it.

    Applies only to the sub-ranges of each still-open ``final_windows``
    entry that are NOT confirmed non-speech (``_subtract_confirmed_silent``)
    — a CONFIRMED-silent span is a stronger, different claim (an
    independent recheck positively established there's no speech there),
    and restoring the original, already-discarded hallucination over it
    would reintroduce KNOWN-WRONG content over a span we've actually
    verified is silent. Those are deliberately left exactly as before this
    feature existed.

    Does NOT change ``missing_seconds``: the caller computes that from the
    SAME ``final_windows``/``confirmed_silent`` BEFORE calling this, and it
    stays exactly as it was before Phase 6 — this only changes which TEXT
    the caller gets back, never what still counts as uncertain. Where
    ``original_segments`` has nothing overlapping a sub-range (the first
    pass genuinely produced nothing there), nothing is restored — there is
    nothing TO restore, and the span stays open exactly as before.

    Guards against reintroducing a full collapsed run (the ruling's second
    constraint): the original segments for each sub-range are run back
    through ``collapse_repeated_segments`` UNCHANGED — the same function,
    same thresholds, used everywhere else in this module — before anything
    is spliced in, so a 40x hallucination loop still collapses to its one
    first-occurrence survivor. Whatever ``collapse_repeated_segments``
    keeps for the sub-range (normally that one instance; occasionally more
    than one, e.g. a short repeated line under its own SHORT-run tolerance
    that never triggered collapse in the first place) is what gets
    restored, each copy marked ``low_confidence: True`` on the segment
    dict (a plain, additive key — nothing downstream currently reads it,
    and JSON-serialises/persists fine alongside ``start``/``end``/
    ``text``). The LAST restored segment's own ``end`` is extended to the
    sub-range's end purely so the reader sees continuous, if uncertain,
    coverage instead of the text stopping short with nothing after it: the
    timing past that first kept instance was never trustworthy anyway
    (that is the whole reason the interval was suspicious in the first
    place), so stretching it doesn't discard anything that was actually
    known-good.

    One more guard the ruling's second constraint requires in practice:
    the segment collapse already kept immediately BEFORE this sub-range
    (its survivor from the SAME run) is usually still sitting in
    ``working``, untouched, with the identical text we're about to
    restore. Leaving both would hand a LATER ``collapse_repeated_segments``
    pass over the whole merged transcript (``transcribe_audio`` runs one,
    once, at the very end) two adjacent identical-text segments — which is
    exactly what that function collapses, re-discarding this restoration
    and silently recreating the hole it exists to prevent. So when that
    neighbour's text matches, this widens the restored segment's ``start``
    back to the neighbour's own start (absorbing it into ONE segment)
    instead of leaving two — the only case this function ever touches a
    restored segment's ``start``, and it never touches any OTHER kept
    segment's bounds.
    """
    restored: list[dict[str, Any]] = []
    restored_ranges: list[tuple[float, float]] = []
    for w_start, w_end in final_windows:
        if (w_end - w_start) <= _MIN_RECHECK_SECONDS:
            continue
        for sub_start, sub_end in _subtract_confirmed_silent(
            (w_start, w_end), confirmed_silent
        ):
            clipped = _clip_to_gap(original_segments, sub_start, sub_end)
            if not clipped:
                continue  # first pass genuinely produced nothing here
            collapsed, _discarded = collapse_repeated_segments(clipped)
            if not collapsed:
                continue
            marked = [dict(seg, low_confidence=True) for seg in collapsed]
            last = marked[-1]
            if _segment_end(last) < sub_end:
                last["end"] = sub_end

            # Whatever already sits in ``working`` immediately before
            # ``sub_start`` (collapse's own kept survivor of the SAME run,
            # in the common case) must not be left standing right next to
            # our restored segment if the two share identical text — a
            # LATER collapse_repeated_segments pass over the whole merged
            # transcript operates purely on adjacent text equality and
            # would treat that pair as the run continuing, re-discarding
            # our restoration and recreating the exact hole this function
            # exists to prevent. So absorb it instead: widen this
            # restored segment's OWN start back to the neighbour's start
            # (same text either way, just one segment instead of two) and
            # mark the neighbour's span as replaced too.
            range_start = sub_start
            left_neighbor = _segment_ending_at(working, sub_start)
            if left_neighbor is not None and str(
                left_neighbor.get("text") or ""
            ) == str(marked[0].get("text") or ""):
                range_start = _segment_start(left_neighbor)
                marked[0] = dict(marked[0], start=range_start)

            restored.extend(marked)
            restored_ranges.append((range_start, sub_end))
            if diagnostics is not None:
                diagnostics.record_backfill(unit=unit_label, start=sub_start, end=sub_end)
            log.warning(
                "transcribe: %s backfilled ~%.0fs at %.0fs-%.0fs with "
                "low-confidence first-pass text instead of leaving a hole "
                "(recheck never resolved this span)",
                unit_label, sub_end - sub_start, sub_start, sub_end,
            )
    if not restored:
        return working

    # A sub-range this function restores may still hold its OWN untouched
    # original content in ``working`` (e.g. recut_unavailable / budget-
    # exhausted never spliced anything, so the raw — possibly still-
    # looping — first-pass segments are still sitting there): drop
    # anything overlapping a restored range before adding the collapsed +
    # low_confidence replacement, or the raw duplicate run would coexist
    # right alongside its own cleaned-up stand-in. A range whose splice
    # already ran (decode_loop_again) has nothing left in ``working`` for
    # this exact sub-range in the first place (that's WHY it's still a gap
    # here), so this is a no-op for that case — never a double-removal.
    def _overlaps_any_restored_range(seg: dict[str, Any]) -> bool:
        s, e = _segment_start(seg), _segment_end(seg)
        return any(s < r_end and e > r_start for r_start, r_end in restored_ranges)

    pruned = [seg for seg in working if not _overlaps_any_restored_range(seg)]
    combined = sorted(pruned + restored, key=_segment_start)
    assert _is_monotonic_by_start(combined), (
        "transcribe: Phase 6 backfill produced a non-monotonic segment "
        "list — bug in _restore_unresolved_windows, not the transcript"
    )
    return combined


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


def transcript_is_unusable(segments: list[dict[str, Any]]) -> bool:
    """True when segments carry no usable content — reused (not
    reimplemented) by runner.py's media page-text fallback gate: empty/
    punctuation-only/annotation-only (``_is_confirmed_silence``), OR a
    degenerate repeated run (``collapse_repeated_segments`` discarded
    something, e.g. a Whisper hallucination loop over near-silent audio).

    Deliberately checked against the raw segment list, NOT
    ``timecodes.build_marked_text``'s ``[MM:SS]``-marked string — the
    marked-up text would make ``_is_confirmed_silence``'s bracket-annotation
    check misfire on a real, non-garbage transcript (every marked line
    starts with a `[MM:SS]` bracket).
    """
    if _is_confirmed_silence(segments):
        return True
    _collapsed, discarded = collapse_repeated_segments(segments)
    return bool(discarded)


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
        # dict(seg, ...) — NOT a fresh {"start"/"end"/"text"} literal —
        # preserves any OTHER key the segment carries (e.g. Phase 6's
        # low_confidence flag on a restored segment fed back through this
        # function) instead of silently stripping it. Measured regression:
        # the original {"start":...,"end":...,"text":...} literal here
        # dropped low_confidence on every segment that passed through a
        # coverage-recheck splice.
        clipped.append(dict(seg, start=max(start, gap_start), end=min(end, gap_end)))
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
    per_job_lock: asyncio.Semaphore,
    diagnostics: TranscribeDiagnostics | None = None,
    unit_label: str = "whole",
    language: str | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Check ``segments`` (a LOCAL timeline starting at 0) against
    ``window_duration`` seconds of ``source_path``: re-transcribe every
    suspicious interval bigger than ``_MIN_RECHECK_SECONDS`` — sliced to
    ``_MAX_RECHECK_SLICE_SECONDS`` at most per request — and decide, from
    what comes back, whether it was real speech, confirmed non-speech, or
    still unresolved.

    ``language``, when given, is pinned on every recheck request this call
    makes (see ``_call_whisper``/``_post_audio``) instead of letting each
    recheck auto-detect independently on its own small, more ambiguous
    slice — the caller already knows the language from the unit's own
    first-pass request; see the module docstring's "Language is detected
    once per call, then pinned" section.

    Returns ``(segments, missing_seconds)`` — the (possibly patched-up)
    segment list to use, and the summed length of every suspicious
    interval that's STILL open once this call's budget
    (``_coverage_recheck_budget(window_duration)``) runs out, MINUS
    whatever length was CONFIRMED as non-speech along the way (that
    doesn't count as missing — see ``_is_confirmed_silence``). A
    degenerate repeated run on a recheck is deliberately never subtracted
    this way — see the module docstring's regression writeup. Segments are
    returned UNCOLLAPSED (the caller's final ``collapse_repeated_segments``
    pass — run once over the whole merged transcript — still applies);
    only the accept/recheck DECISION is made on a locally-collapsed view.

    There is deliberately no size-based correctness gate here anymore (see
    the module docstring): every interval above the cost cutoff gets
    rechecked, in descending size order, up to the budget. A slice that
    overlaps one already rechecked THIS call beyond
    ``_WINDOW_OVERLAP_DEDUP_FRACTION`` is never asked again (the
    ``checked`` list, tested via ``_overlaps_checked`` — NOT exact-key
    equality; see that constant's docstring for why) — a deterministic
    decoder given the exact same audio has no reason to answer differently
    — which combined with the budget guarantees this loop terminates
    regardless of how many distinct suspicious intervals — or how large
    any one of them is — a pathological transcript produces.

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

    When this call has a PINNED language (``language`` above), a
    CONFIRMED-non-speech verdict is not trusted on its own first: the pin
    itself can be the reason nothing usable came back — measured live
    (job r56sj0_o1ihv, forced ``language="de"``): a stretch with real
    dialogue decoded under the pin collapsed to bare punctuation, while an
    auto-detected pass over the EXACT SAME audio recovered real text. So a
    confirmed-silent verdict under a pin triggers ONE extra Whisper call —
    the SAME cut, re-asked with ``language=None`` — before concluding
    anything (owner ruling: "a wrong transcript beats a hole"). This is
    not a new recheck slot (bounded by the SAME per-unit budget as every
    other recheck, never a separate/unbounded pass) and is skipped
    entirely when there's no pin to begin with (re-asking with the
    identical language would just reproduce the identical request). See
    ``TranscribeDiagnostics.record_recheck``'s docstring for the two
    verdicts this can produce (``"recovered_unpinned"`` /
    ``"confirmed_non_speech_both_ways"``).

    What comes back is classified three ways, not two:

    - CONFIRMED non-speech (``_is_confirmed_silence`` — empty, punctuation-
      only, or entirely bracket/asterisk/paren annotations — under the
      pinned language AND, if one was pinned, again under auto-detect):
      nothing is spliced, the interval is recorded as confirmed-silent so
      the final ``missing_seconds`` accounting excludes it, logged as "not
      lost content".
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
    # Snapshot of what the FIRST PASS actually produced, before any recheck
    # splice touches ``working`` — kept immutable for the rest of this call
    # so the Phase 6 restoration pass at the end (``_restore_unresolved_windows``)
    # can recover it even for a window whose splice already replaced the
    # original content with a partial retry recovery (see that function's
    # own docstring for why ``working`` alone isn't enough).
    original_segments = list(segments)
    assert _is_monotonic_by_start(working), (
        "transcribe: input segments went backward in time by start "
        "(bug upstream of _ensure_coverage, not this function)"
    )
    budget = _coverage_recheck_budget(window_duration)
    checked: list[tuple[float, float]] = []
    # Suppressed windows already reported to diagnostics THIS call — a
    # window that keeps overlapping ``checked`` shows up in ``suppressed``
    # on every outer-loop iteration until ``pending`` is empty (see
    # _pending_slices), so this avoids logging the same suppression
    # repeatedly while other, still-open windows keep the loop going.
    logged_suppressions: set[tuple[float, float]] = set()
    confirmed_silent: list[tuple[float, float]] = []
    rechecks = 0

    while rechecks < budget:
        pending, suppressed = _pending_slices(working, window_duration, checked)
        if diagnostics is not None:
            for s_start, s_end in suppressed:
                key = _window_key(s_start, s_end)
                if key in logged_suppressions:
                    continue
                logged_suppressions.add(key)
                diagnostics.record_overlap_suppressed(
                    unit=unit_label, start=s_start, end=s_end
                )
        if not pending:
            break

        gap_start, gap_end, leading = max(pending, key=lambda w: w[1] - w[0])
        checked.append((gap_start, gap_end))
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
            if diagnostics is not None:
                diagnostics.record_recheck(
                    unit=unit_label, index=rechecks, of=budget,
                    start=gap_start, end=gap_end, verdict="skipped_too_small",
                )
            continue

        cut_path = await asyncio.to_thread(
            _cut_audio_segment, source_path, cut_start, cut_duration
        )
        if cut_path is None:
            log.warning(
                "transcribe: coverage recheck %d/%d couldn't re-cut audio for "
                "a ~%.0fs slice at %.0fs; leaving it unresolved",
                rechecks, budget, gap_end - gap_start, gap_start,
            )
            if diagnostics is not None:
                diagnostics.record_recheck(
                    unit=unit_label, index=rechecks, of=budget,
                    start=gap_start, end=gap_end, verdict="recut_unavailable",
                )
            continue

        try:
            # Sequential await inside a while-loop, one recheck slice at a
            # time — see _call_whisper's docstring for why this keeps the
            # per-job semaphore uncontended in practice while still holding
            # it explicitly on every call.
            payload = await _call_whisper(per_job_lock, cut_path, language=language)
            retry_result = _parse_payload(payload, total_duration=cut_duration)

            # A PINNED language can itself cost us content: measured live
            # (job r56sj0_o1ihv, forced language="de") — a stretch with
            # real dialogue decoded under the pinned language collapsed to
            # bare punctuation, which _is_confirmed_silence correctly read
            # as "nothing here" for THAT decode, while an auto-detected
            # pass over the EXACT SAME audio recovered real text. Owner
            # ruling: "a wrong transcript beats a hole" — before concluding
            # anything, retry this SAME cut once with auto-detect
            # (language=None). Only when a language was actually pinned
            # (retrying with the identical language would just reproduce
            # the identical request) and only when the pinned attempt
            # itself came back confirmed-silent (a normal recovery/
            # decode-loop verdict below is not second-guessed — the pin
            # already produced something usable). Reuses cut_path — the
            # AUDIO is unaffected by the language parameter, only the
            # DECODE differs, so this costs one extra Whisper call, never
            # a second ffmpeg cut. Not a new recheck slot: it verifies
            # THIS slot's verdict rather than opening a new suspicious
            # window, so it is bounded by the SAME per-unit budget as
            # every other recheck (at most one extra call per recheck that
            # happens to land on confirmed-silence under a pin) — never an
            # unbounded second pass.
            used_unpinned_retry = False
            if language and _is_confirmed_silence(retry_result.segments):
                unpinned_payload = await _call_whisper(per_job_lock, cut_path, language=None)
                unpinned_result = _parse_payload(unpinned_payload, total_duration=cut_duration)
                if not _is_confirmed_silence(unpinned_result.segments):
                    retry_result = unpinned_result
                    used_unpinned_retry = True
                # else: confirmed silent under BOTH the pinned AND the
                # auto-detected decode — genuinely nothing there, not a
                # language artifact of the pin.
        finally:
            cut_path.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                cut_path.parent.rmdir()

        if _is_confirmed_silence(retry_result.segments):
            confirmed_silent.append((gap_start, gap_end))
            verdict = "confirmed_non_speech_both_ways" if language else "confirmed_non_speech"
            log.info(
                "transcribe: coverage recheck %d/%d at %.0fs-%.0fs came back "
                "confirmed non-speech (empty/punctuation/annotation)%s — not "
                "lost content",
                rechecks, budget, gap_start, gap_end,
                " even after an unpinned retry" if language else "",
            )
            if diagnostics is not None:
                diagnostics.record_recheck(
                    unit=unit_label, index=rechecks, of=budget,
                    start=gap_start, end=gap_end, verdict=verdict,
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
            verdict = "recovered_unpinned" if used_unpinned_retry else "decode_loop_again"
            log.warning(
                "transcribe: coverage recheck %d/%d at %.0fs-%.0fs hit a "
                "decode-loop again%s — kept the recognized portion up to "
                "the loop; remainder stays unresolved, NOT confirmed silent",
                rechecks, budget, gap_start, gap_end,
                " (via an unpinned retry)" if used_unpinned_retry else "",
            )
            if diagnostics is not None:
                diagnostics.record_recheck(
                    unit=unit_label, index=rechecks, of=budget,
                    start=gap_start, end=gap_end, verdict=verdict,
                )
            retry_segments_source = collapsed_retry
        else:
            retry_segments_source = retry_result.segments
            verdict = "recovered_unpinned" if used_unpinned_retry else "recovered_real_speech"
            log.info(
                "transcribe: coverage recheck %d/%d recovered real speech at "
                "%.0fs-%.0fs%s",
                rechecks, budget, gap_start, gap_end,
                " (via an unpinned retry)" if used_unpinned_retry else "",
            )
            if diagnostics is not None:
                diagnostics.record_recheck(
                    unit=unit_label, index=rechecks, of=budget,
                    start=gap_start, end=gap_end, verdict=verdict,
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
    # Phase 6 (owner ruling, 2026-08-28): "a wrong transcript beats a
    # hole" — missing (above) is the internal uncertainty signal and is
    # UNCHANGED by this step (computed from the exact same final_windows/
    # confirmed_silent, before any restoration). What follows only changes
    # which SEGMENTS get returned to the caller, never what still counts as
    # uncertain — see _restore_unresolved_windows's own docstring.
    working = _restore_unresolved_windows(
        working,
        original_segments,
        final_windows,
        confirmed_silent,
        diagnostics=diagnostics,
        unit_label=unit_label,
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


async def _post_audio(audio_path: Path, *, language: str | None = None) -> dict[str, Any]:
    """POST one audio file to the transcription endpoint, return parsed JSON.

    ``language`` (an ISO-639-1-ish code, e.g. ``"de"``) is sent as the
    request's ``language`` form field when given, omitted entirely when
    ``None`` (auto-detect). Confirmed empirically against the LocalAI/
    whisper.cpp backend this project targets: the field is genuinely
    CONSUMED for decoding, not just echoed back in the response — sending
    an intentionally-garbage value measurably changed the transcribed
    text, which a pure echo could not do. See the module docstring's
    "Language is detected once per call, then pinned" section for why this
    is only sent on requests AFTER the first one for a given call.
    """
    cfg = get_config().whisper
    endpoint = f"{cfg.base_url.rstrip('/')}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {cfg.effective_api_key}"}

    with audio_path.open("rb") as fh:
        files = {"file": (audio_path.name, fh, "application/octet-stream")}
        data = {"model": cfg.model, "response_format": "verbose_json"}
        if language:
            data["language"] = language
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


async def probe_duration(path: Path) -> float | None:
    """Async wrapper around ``_probe_duration`` for a LOCAL file path.

    Mirrors the internal ``asyncio.to_thread(_probe_duration, audio_path)``
    call already used by ``_transcribe_chunked`` — exposed publicly so
    ``runner.py`` can probe a just-downloaded file's real duration before
    deciding whether to call Whisper at all. This is the required fallback
    tier for ``kind=media`` jobs whose yt-dlp metadata probe came back with
    no duration (e.g. a plain static-asset URL — yt-dlp's generic extractor
    doesn't report ``duration`` for those via ``extract_info(download=
    False)``, so the pre-download probe alone is not sufficient): download
    proceeds as normal (cheap), then this runs on the real file, before the
    (expensive, and — if the clip turns out to be a UI sound effect —
    fabrication-prone) Whisper call.
    """
    return await asyncio.to_thread(_probe_duration, path)


def _probe_duration_url(url: str, *, timeout: float = 5.0) -> float | None:
    """Best-effort duration probe directly against ``url``, no download.

    ffmpeg's http(s) protocol supports byte-range requests, so for a normal
    seekable static file ffprobe typically needs only 1-2 small requests to
    read the container's duration metadata — genuinely cheaper than
    downloading first. Optional/bonus tier, tried before yt-dlp's download
    step for a ``kind=media`` job whose yt-dlp metadata probe came back with
    no duration (see ``runner.py``).

    Deliberately conservative:
    - A hard wall-clock ``timeout`` via ``subprocess.run(..., timeout=...)``
      is protocol-agnostic — it kills the ffprobe subprocess regardless of
      what it's doing internally (DNS, TLS handshake, a slow/non-seekable/
      hostile server holding the connection open), so this can never hang
      the single Whisper worker.
    - EVERY failure mode (missing ffprobe, non-zero exit, timeout,
      unparseable output) swallows to ``None`` and falls through to the
      normal download path with zero behavior change on failure.
    - No cookies are forwarded to ffprobe (real added complexity for an
      optional path) — an authenticated URL simply fails this probe, an
      acceptable, disclosed limitation, and falls through to the normal
      (cookie-aware) yt-dlp download, which is NOT a reliability regression.
    """
    ffprobe = _ffmpeg_bin("ffprobe")
    if not ffprobe:
        return None
    try:
        out = subprocess.run(
            [
                ffprobe, "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", url,
            ],
            check=True, capture_output=True, text=True, timeout=timeout,
        )
        return float(out.stdout.strip())
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        OSError,
        ValueError,
    ) as exc:
        log.info("transcribe: ffprobe URL duration probe failed (%s)", exc)
        return None


async def probe_url_duration(url: str, *, timeout: float = 5.0) -> float | None:
    """Async wrapper around ``_probe_duration_url``. See its docstring."""
    return await asyncio.to_thread(_probe_duration_url, url, timeout=timeout)


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


__all__ = [
    "transcribe_audio",
    "ChunkingDiagnostics",
    "TranscribeDiagnostics",
    "TranscribeResult",
    "transcript_is_unusable",
    "probe_duration",
    "probe_url_duration",
]
