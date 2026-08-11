"""Background transcript translation.

Translates a job's transcript into a target language and stores the
result in ``transcript_translation``. Reused by:

- ``POST /jobs/{id}/transcript/translate {lang}`` — user explicitly
  asked for a translation.
- ``re_enqueue_pending`` on daemon startup — any row left in
  ``status="running"`` gets picked up so the spinner the user saw
  before the restart actually finishes (we have the source text cached,
  nothing external is needed).

Design constraints (from the project plan):

- **Dedup**: a second ``translate`` call while one is already running for
  ``(job_id, language_code)`` is a no-op. We do this with a row-level
  lock — the first POST inserts a ``queued/running`` row; the second
  sees the row, returns the current status.
- **Pause-aware**: between groups (and via ``respect_pause=True`` on
  ``llm_client.stream_complete``) so the global pause flag works the
  same way as it does for summaries.
- **Progress in percent**: derived from how many SOURCE lines have been
  resolved (translated or, on failure, kept verbatim), plus an in-flight
  estimate for whichever group is currently streaming. No partial text
  is leaked (the UI shows only "Translating N%").
- **Never trust the model's line alignment — verify it deterministically
  and repair by narrowing the window.** The transcript is packed into
  groups with ``llm.chunking.pack_lines`` (never tears a line, unlike
  the summary splitter — see that module's docstring for why
  ``split_for_summary`` is wrong for this job). Each group's output is
  checked line-for-line against the input by ``_align_translation``: a
  mismatch bisects the group and retries the halves, down to a single
  line, before giving up and keeping that line in the source language.
  Losing input is structurally impossible — the worst case is a
  ``status="partial"`` row with some lines left untranslated, never a
  chunk silently dropped.

State life-cycle:

    queued ─ (worker picked up) → running ─ (all lines aligned) → done
                                    ├ (some lines fell back)     → partial
                                    └ (exception)                → failed

A ``failed`` or ``partial`` row stays until the user clicks Retry-all (or
deletes the parent job, which CASCADEs). We never auto-retry on failure
to avoid runaway LLM bursts on a buggy prompt.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlmodel import select

from src.llm import client as llm_client
from src.llm.chunking import pack_lines
from src.llm.languages import Language, UnknownLanguageError, normalize_lang
from src.llm.tokens import count_tokens
from src.storage import repo
from src.storage.db import Job, TranscriptTranslation, session_scope
from src.workers import timecodes
from src.workers.broker import get_broker, get_event_broker, job_event
from src.workers.control import get_control

log = logging.getLogger(__name__)


# We pack the input transcript LINE-BY-LINE (never splitting a line) to keep
# a single LLM call within the model's context window. This is the packing
# budget, not the per-call max_tokens (see ``_max_tokens_for``).
_TRANSCRIPT_CHUNK_TOKENS = 2000

# Streaming loop guard + post-hoc degeneration check: a run of MORE than
# this many consecutive identical non-empty output lines is treated as a
# model gone into a repetition loop. Mirrors the QA path's
# ``api.ai._DEGEN_TAIL_RE`` but line-granular (translation output is
# line-shaped by contract, unlike free-prose QA answers).
#
# This is a FLOOR, not an absolute cap — see ``_degenerate_run_threshold``.
# Real transcripts legitimately repeat a line many times in a row
# (measured live: 28 consecutive ``[05:28] Ja.`` lines from Whisper before
# ``timecodes.collapse_repeated_segments`` runs on a re-imported/legacy
# job), and this module must not assume that cleanup already happened —
# the effective threshold floats up to match whatever repetition the
# INPUT of a given call already contains.
_MAX_REPEATED_LINES = 6

# How many times a mismatching group may be bisected before we give up and
# fall back the whole (sub-)group to source text. Bounds recursion depth.
#
# Must be deep enough to actually REACH single-line granularity on a real
# group, or the single-line retry path (where the code is designed to
# bottom out) is dead code. ``pack_lines`` produces much bigger groups
# than the old sentence-shredding chunker did — measured live on a
# 656-line transcript: 4 groups of 145-177 lines, not 6 ragged ones — so
# depth must track that. At depth 4, a ~170-line group only narrows to
# ~11 lines before hitting the cap (170 → 85 → 42 → 21 → 11), which is
# exactly what was measured: an 11-line contiguous fallback block where a
# single mistranslated line should have cost 1 line, not 11. At depth 8 a
# ~200-line group bottoms out at ~1 line. If you raise this further,
# re-derive ``_CALL_BUDGET_FLOOR``/``_CALL_BUDGET_PER_BAD_REGION`` below —
# they assume this value.
_MAX_BISECT_DEPTH = 8

# Per-job ceiling on LLM calls across the whole translation (initial
# attempts + every bisection + single-line retries). Bounds a pathological
# model that mismatches on every single call from fanning out unboundedly.
#
# Derived, not guessed: must comfortably absorb TWO fully-isolated bad
# regions (real transcripts rarely have more than a couple of distinct
# trouble spots) reaching max depth, plus one initial call per group.
# Isolating one contiguous bad region via bisection costs
# ``2 * _MAX_BISECT_DEPTH + 1`` calls — the root call, plus at each of
# `_MAX_BISECT_DEPTH` further splits along the bad half, one call for the
# half that clears immediately and one continuing into the still-bad
# half. Measured on the live 656-line/4-group job this fires on: 14 calls
# actually made against a budget of 24 — comfortable headroom, not a wall
# the run bumps into.
_CALL_BUDGET_FLOOR = 24
_CALL_BUDGET_PER_BAD_REGION = 2 * _MAX_BISECT_DEPTH + 1

# A group is "markerless" (raw_text fallback source: PDF/HTML, legacy jobs)
# when fewer than this fraction of its input lines carry a leading
# ``[MM:SS]``/``[HH:MM:SS]`` marker. Marker-based verification is
# impossible there, so alignment degrades to an emptiness/degeneration
# check only (see ``_align_translation``).
_MARKERLESS_MARKER_FRACTION = 0.9

# Leading ``[MM:SS]`` / ``[HH:MM:SS]`` marker, captured with the rest of the
# line so we can rebuild the line from the INPUT's marker (never trust the
# model's copy of it) plus the OUTPUT's translated text.
_LEADING_MARKER_RE = re.compile(r"^(\[\d{1,2}:\d{2}(?::\d{2})?\])(.*)$")


# In-flight task tracking. The translation coroutine is held in a
# module-level set so Python's GC doesn't kill it before completion
# (same trick the API uses for the main pipeline tasks).
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def enqueue_translation(job_id: str, lang_input: str) -> dict[str, Any]:
    """Ensure a translation task for ``(job_id, lang_input)`` is running.

    Dedup contract:
      - If a row already exists with status ``running`` / ``queued`` /
        ``done`` / ``partial`` for this ``(job_id, language_code)`` —
        return its current status, don't spawn a new task. ``partial`` is
        treated like ``done`` here: it already has usable text, so
        selecting the language again must not silently restart the work.
      - If a row exists with status ``failed`` — clear the error, reset
        status to ``queued``, spawn the task. Manual retry path.
      - No row → insert ``queued``, spawn the task.

    Works even when the source transcript isn't ready yet (job still in
    extraction phase). The spawned worker polls for the source text and
    starts translating as soon as it appears. UI sees an immediate chip
    with a spinner — no error.

    Raises ``KeyError`` if the parent job is missing, ``UnknownLanguageError``
    if ``lang_input`` doesn't normalise to a supported code.
    """
    lang = normalize_lang(lang_input)
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise KeyError(f"job {job_id} not found")
        # en→en (or other matching source) — return the original directly
        # rather than spinning Gemma for an identity translation. Only
        # short-circuit when the source language IS known; if it's null
        # (PDF / HTML / pre-extraction), let the worker run — Gemma will
        # detect and translate.
        source_lang = getattr(job, "transcript_language", None)
        if source_lang and source_lang == lang.code:
            return {
                "language_code": lang.code,
                "status": "done",
                "progress_percent": 100,
                "is_source": True,
            }

        existing = session.get(TranscriptTranslation, (job_id, lang.code))
        if existing is not None:
            if existing.status in ("queued", "running", "done", "partial"):
                return _row_summary(existing)
            # failed — reset and re-spawn
            existing.status = "queued"
            existing.progress_percent = 0
            existing.error = None
            existing.text = None
            existing.updated_at = datetime.utcnow()
            session.add(existing)
        else:
            row = TranscriptTranslation(
                job_id=job_id,
                language_code=lang.code,
                status="queued",
                progress_percent=0,
            )
            session.add(row)

    _spawn(job_id, lang)
    _publish_translation_event(job_id, lang.code, "queued", 0, None)
    return {
        "language_code": lang.code,
        "status": "queued",
        "progress_percent": 0,
        "is_source": False,
    }


def _reset_failed_rows(job_id: str) -> list[dict[str, Any]]:
    """DB-only: flip every ``failed``/``partial`` translation row for
    ``job_id`` back to ``queued`` (clearing error / text / progress).
    Returns the row summaries so the async caller can spawn workers for
    each.

    Raises ``KeyError`` when the parent job doesn't exist.
    """
    out: list[dict[str, Any]] = []
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise KeyError(f"job {job_id} not found")
        rows = session.exec(
            select(TranscriptTranslation).where(
                TranscriptTranslation.job_id == job_id,
                TranscriptTranslation.status.in_(("failed", "partial")),  # type: ignore[attr-defined]
            )
        ).all()
        for row in rows:
            row.status = "queued"
            row.progress_percent = 0
            row.error = None
            row.text = None
            row.updated_at = datetime.utcnow()
            session.add(row)
            out.append(_row_summary(row))
    return out


async def retry_all_failed(job_id: str) -> list[dict[str, Any]]:
    """Re-enqueue every ``failed`` OR ``partial`` translation row for
    ``job_id`` — a partial row has SOME lines still in the source
    language, so it's as much a "retry me" candidate as a fully failed
    one; the existing "Retry all" button covers both.

    Async because ``_spawn`` calls ``asyncio.create_task`` which requires
    a running event loop. Callers (the API endpoint, tests) must be in
    an async context — sync FastAPI route handlers run on the threadpool
    where create_task would raise ``RuntimeError``.

    Returns the list of summaries that were re-enqueued (each as a dict
    matching ``TranscriptTranslationSummary``). Idempotent: rows in any
    other status are left alone.
    """
    out = _reset_failed_rows(job_id)
    # Spawn coroutines outside the session scope so the worker sees the
    # committed row state.
    for row_summary in out:
        try:
            lang = normalize_lang(row_summary["language_code"])
            _spawn(job_id, lang)
            _publish_translation_event(
                job_id, lang.code, "queued", 0, None,
            )
        except UnknownLanguageError:
            # Stored language is no longer in our supported list — leave
            # the row queued but the worker will fail when it tries to
            # normalise. Edge case (we removed a language).
            log.warning(
                "translator: cannot retry job %s lang %s — unknown",
                job_id, row_summary["language_code"],
            )
    return out


def re_enqueue_running_on_startup() -> int:
    """Pick up translation rows still in ``running`` after a daemon restart.

    Called from ``main.lifespan`` so spinners the user saw before the
    restart actually finish. Resets ``progress_percent`` to 0 (we don't
    checkpoint partial group output, so the worker starts from scratch).

    Returns the count of rows re-enqueued.
    """
    count = 0
    with session_scope() as session:
        rows = session.exec(
            select(TranscriptTranslation).where(
                TranscriptTranslation.status.in_(("running", "queued"))  # type: ignore[attr-defined]
            )
        ).all()
        for row in rows:
            row.status = "queued"
            row.progress_percent = 0
            row.updated_at = datetime.utcnow()
            session.add(row)
            count += 1

    if count == 0:
        return 0

    # Spawn after commit so the workers see the new row state.
    with session_scope() as session:
        rows = session.exec(
            select(TranscriptTranslation).where(
                TranscriptTranslation.status == "queued"
            )
        ).all()
        targets = [(r.job_id, r.language_code) for r in rows]

    for job_id, code in targets:
        try:
            lang = normalize_lang(code)
        except UnknownLanguageError:
            log.warning(
                "translator: startup re-enqueue skipped job %s lang %s — unsupported",
                job_id, code,
            )
            continue
        _spawn(job_id, lang)
    log.info("translator: re-enqueued %d translation task(s) at startup", count)
    return count


# ---------------------------------------------------------------------------
# Internals — call budget
# ---------------------------------------------------------------------------


class _CallBudget:
    """Per-job ceiling on LLM calls, shared across a whole ``_run``.

    A pathological model that mismatches on every attempt must not be
    able to fan out bisection unboundedly — once the budget is spent,
    every further ``_translate_group`` call falls back to source text
    immediately instead of making another call.
    """

    __slots__ = ("limit", "used")

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def try_consume(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


# ---------------------------------------------------------------------------
# Internals — marker-based alignment
# ---------------------------------------------------------------------------


def _extract_marker(line: str) -> tuple[str, str] | None:
    """Split a line into ``(marker, rest)`` or ``None`` if it has no
    leading ``[MM:SS]``/``[HH:MM:SS]`` marker."""
    m = _LEADING_MARKER_RE.match(line)
    if not m:
        return None
    return m.group(1), m.group(2)


def _marker_seconds(marker: str) -> int:
    """Parse a ``[MM:SS]``/``[HH:MM:SS]`` marker into total seconds.

    Matching markers by parsed VALUE rather than raw string means a model
    that copies the marker with a cosmetic slip — dropping the leading
    zero (``[1:02]`` for ``[01:02]``), say — still counts as the same
    marker for alignment purposes. The reconstructed line always uses the
    INPUT's own marker text regardless (see ``_align_translation``), so
    this tolerance never lets a mangled marker leak into stored output.
    """
    parts = [int(p) for p in marker.strip("[]").split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts[-3], parts[-2], parts[-1]
    return h * 3600 + m * 60 + s


def _longest_repeat_run(lines: list[str]) -> int:
    """Length of the longest run of CONSECUTIVE identical non-empty lines.

    Shared by both degeneration checks below. A blank line breaks a run
    (it can't be "the same text repeated" across a gap).
    """
    longest = 0
    run = 0
    prev: str | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            run = 0
            prev = None
            continue
        run = run + 1 if stripped == prev else 1
        longest = max(longest, run)
        prev = stripped
    return longest


def _degenerate_run_threshold(input_lines: list[str]) -> int:
    """The repeat-run length beyond which OUTPUT is treated as a runaway
    loop, for a call whose input is ``input_lines``.

    Deliberately NOT a fixed constant: some real transcripts legitimately
    repeat a line many times in a row (measured on a live job: 28
    consecutive ``[05:28] Ja.`` lines from Whisper) — collapsing those is
    ``timecodes.collapse_repeated_segments``'s job upstream, not this
    module's, and the translator must not assume that pass has already
    run. A faithful translation of a genuinely-repetitive INPUT is
    exactly as repetitive, so the threshold floats up to match whatever
    repetition the input itself already contains; only a run that goes
    BEYOND what the input justifies is flagged.
    """
    return max(_MAX_REPEATED_LINES, _longest_repeat_run(input_lines))


def _has_degenerate_run(lines: list[str], threshold: int) -> bool:
    """True if ``lines`` contains a run of more than ``threshold``
    CONSECUTIVE identical non-empty lines — a post-hoc catch for a
    repetition loop the streaming guard (``_stream_group``) missed,
    e.g. because the whole loop arrived in one delta."""
    return _longest_repeat_run(lines) > threshold


def _align_translation(
    input_lines: list[str], output_lines: list[str],
) -> list[str] | None:
    """Verify and rebuild ``output_lines`` against ``input_lines``.

    Returns ``None`` on any of:
    - a degenerate repeated run in the output,
    - (marker-bearing input) an input marker that can't be found, forward
      of the cursor, anywhere in the output,
    - (markerless input) an empty output.

    On success for marker-bearing input, returns EXACTLY
    ``len(input_lines)`` lines, each rebuilt as the INPUT's own marker
    (never the model's copy of it) followed by the matched output line's
    text, with any unmarked output lines that immediately follow it
    (up to the next marked line) absorbed into that text — this is what
    lets a model that split one input line into two still align cleanly.

    Markerless input (the ``raw_text`` fallback source for PDF/HTML/legacy
    jobs — no ``[MM:SS]`` markers to verify against) has no way to check
    line-for-line correctness, so we accept the output as-is once it's
    non-empty and free of a degenerate run; the returned list is whatever
    the model produced, not necessarily ``len(input_lines)`` long.

    The degeneration threshold is relative to THIS input (see
    ``_degenerate_run_threshold``) — a faithful translation of a
    genuinely-repetitive source (real Whisper transcripts can legitimately
    repeat one line dozens of times) must not be rejected just because it
    repeats too.
    """
    if _has_degenerate_run(output_lines, _degenerate_run_threshold(input_lines)):
        return None

    if not input_lines:
        return []

    parsed_in = [_extract_marker(line) for line in input_lines]
    marker_count = sum(1 for p in parsed_in if p is not None)
    if marker_count / len(input_lines) < _MARKERLESS_MARKER_FRACTION:
        joined = "\n".join(output_lines).strip()
        return list(output_lines) if joined else None

    if marker_count != len(input_lines):
        # Predominantly marked, but at least one line in THIS particular
        # window has no marker of its own — verification would silently
        # skip it. Shouldn't happen given how marked transcripts are
        # built, but fail closed rather than guess.
        return None

    parsed_out = [_extract_marker(line) for line in output_lines]
    n_out = len(output_lines)
    cursor = 0
    result: list[str] = []
    for parsed in parsed_in:
        assert parsed is not None
        marker, _input_rest = parsed
        marker_secs = _marker_seconds(marker)
        match_idx = None
        matched_rest = ""
        for j in range(cursor, n_out):
            out_parsed = parsed_out[j]
            if out_parsed is not None and _marker_seconds(out_parsed[0]) == marker_secs:
                match_idx = j
                matched_rest = out_parsed[1]
                break
        if match_idx is None:
            return None

        text = matched_rest
        k = match_idx + 1
        absorbed: list[str] = []
        while k < n_out and parsed_out[k] is None:
            stripped = output_lines[k].strip()
            if stripped:
                absorbed.append(stripped)
            k += 1
        if absorbed:
            text = text.rstrip() + " " + " ".join(absorbed)

        result.append(f"{marker}{text}")
        cursor = k

    return result


# ---------------------------------------------------------------------------
# Internals — LLM call + streaming loop guard
# ---------------------------------------------------------------------------


def _max_tokens_for(text: str) -> int:
    """Dynamic ``max_tokens`` for one translation call.

    Fixed 4000 was measured thin: Russian output runs ~1.5x the source
    token count, so a 2000-token source chunk can legitimately need ~3000
    tokens of RU output alone, before counting any repair/bisection
    overhead. ``count_tokens(text) * 2.5`` gives generous headroom;
    floor/ceiling keep tiny and huge groups sane.
    """
    return max(512, min(8000, int(count_tokens(text) * 2.5)))


async def _stream_group(
    lines: list[str],
    lang: Language,
    prompt_template: str,
    *,
    on_delta: Callable[[str], None] | None = None,
) -> list[str]:
    """Stream one LLM call translating ``lines`` (joined with ``\\n``).

    Applies the streaming loop guard: once the consecutive identical
    non-empty output line count goes beyond ``_degenerate_run_threshold(lines)``
    — which floats up to match whatever repetition ``lines`` itself
    already contains, see that function — stops reading the stream
    immediately. This exists to stop burning tokens on a runaway model
    early, not as the primary defence (``_align_translation``'s
    degeneration check on the final text is that; this just
    short-circuits the input side of it).

    Returns the raw output split on ``\\n`` — NOT yet verified; the caller
    runs it through ``_align_translation``.
    """
    text = "\n".join(lines)
    prompt = prompt_template.format(
        target_language_name=lang.english_name,
        transcript=text,
    )
    max_tokens = _max_tokens_for(text)
    threshold = _degenerate_run_threshold(lines)

    buf: list[str] = []
    pending_line = ""
    last_line: str | None = None
    repeat_run = 0
    async for delta in llm_client.stream_complete(
        prompt,
        max_tokens=max_tokens,
        temperature=0.2,  # low but non-zero to avoid robotic phrasing
        respect_pause=True,
    ):
        buf.append(delta)
        if on_delta is not None:
            on_delta("".join(buf))
        pending_line += delta
        while "\n" in pending_line:
            line, pending_line = pending_line.split("\n", 1)
            stripped = line.strip()
            if stripped:
                repeat_run = repeat_run + 1 if stripped == last_line else 1
                last_line = stripped
            else:
                repeat_run = 0
                last_line = None
            if repeat_run > threshold:
                # Runaway repetition loop — stop reading now rather than
                # burning the rest of max_tokens on filler.
                return "".join(buf).split("\n")
    return "".join(buf).split("\n")


async def _translate_group(
    lines: list[str],
    lang: Language,
    prompt_template: str,
    *,
    depth: int,
    budget: _CallBudget,
    on_delta: Callable[[str], None] | None = None,
) -> tuple[list[str], int]:
    """Translate ``lines`` (a group from ``pack_lines``, or a bisected
    slice of one), verifying alignment and repairing by narrowing the
    window on mismatch.

    Returns ``(output_lines, fallback_count)`` — ``fallback_count`` is how
    many of ``lines`` ended up copied verbatim from the source because no
    amount of retrying/bisecting produced an aligned translation for them.

    Order of attempts:
    1. One LLM call on the whole group; aligned → done.
    2. Single line, still mismatched → retry once more; still mismatched
       → fall back that one line to source (counted).
    3. Multi-line, still mismatched, depth/budget allow → bisect in half,
       recurse into both halves independently.
    4. Depth limit or call budget exhausted → fall back the WHOLE
       (sub-)group to source, counted.
    """
    if not lines:
        return [], 0

    if not budget.try_consume():
        return list(lines), len(lines)

    output_lines = await _stream_group(lines, lang, prompt_template, on_delta=on_delta)
    aligned = _align_translation(lines, output_lines)
    if aligned is not None:
        return aligned, 0

    if len(lines) == 1:
        if not budget.try_consume():
            return list(lines), 1
        retry_output = await _stream_group(lines, lang, prompt_template, on_delta=on_delta)
        retry_aligned = _align_translation(lines, retry_output)
        if retry_aligned is not None:
            return retry_aligned, 0
        return list(lines), 1

    if depth >= _MAX_BISECT_DEPTH:
        return list(lines), len(lines)

    # Progress must not freeze for the whole duration of a bisecting
    # group — that's the SLOWEST case (several LLM calls deep) — so keep
    # reporting through the recursion. A sub-call's fraction is computed
    # against the ORIGINAL group's line count (the closure `_run` built),
    # so it under-reports relative to what's already been published; that
    # is harmless because ``_publish_pct`` only ever moves forward.
    mid = len(lines) // 2
    left_out, left_fallback = await _translate_group(
        lines[:mid], lang, prompt_template, depth=depth + 1, budget=budget, on_delta=on_delta,
    )
    right_out, right_fallback = await _translate_group(
        lines[mid:], lang, prompt_template, depth=depth + 1, budget=budget, on_delta=on_delta,
    )
    return left_out + right_out, left_fallback + right_fallback


# ---------------------------------------------------------------------------
# Internals — the worker loop
# ---------------------------------------------------------------------------


def _spawn(job_id: str, lang: Language) -> None:
    """Create the background task. Kept module-level so GC won't kill it."""
    task = asyncio.create_task(
        _run(job_id, lang),
        name=f"translate:{job_id}:{lang.code}",
    )
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


async def _run(job_id: str, lang: Language) -> None:
    """Translate the transcript into ``lang``, line-group by line-group.

    Source preference: when fine-grained Whisper / YouTube segments are
    available (``Job.raw_segments_json``), we translate THOSE so the
    output keeps the same 1-5 s granularity as the original Transcript
    tab. Falls back to the 30-second-bucketed ``Job.raw_text`` for
    legacy jobs and unpatched-mlx fallbacks where segments weren't
    captured.

    May spend time at the start waiting for the source transcript to be
    extracted — if the user requested a translation BEFORE the pipeline
    had finished Whisper. We poll every 3 s; this is cheap (a single DB
    read) and lets the chip show "running 0%" with a spinner during the
    wait.

    Result status: ``done`` when every source line ended up translated,
    ``partial`` when some lines fell back to source text (see
    ``_translate_group``), ``failed`` only on an outright exception (e.g.
    the backend unreachable) — a model that merely mangles its output no
    longer fails the whole job, it degrades to ``partial``.
    """
    source_text = await _wait_for_source_text(job_id, lang.code)
    if source_text is None:
        # Job failed during extraction or was deleted; status already
        # recorded by ``_wait_for_source_text``.
        return

    _update_status(job_id, lang.code, status="running", progress_percent=0)
    _publish_translation_event(job_id, lang.code, "running", 0, None)

    input_lines = source_text.splitlines()
    total_lines = len(input_lines)
    if total_lines == 0:
        _update_status(
            job_id, lang.code,
            status="failed",
            progress_percent=0,
            error="transcript is empty",
        )
        _publish_translation_event(
            job_id, lang.code, "failed", 0, "transcript is empty",
        )
        return

    groups = pack_lines(input_lines, target_tokens=_TRANSCRIPT_CHUNK_TOKENS)
    prompt_template = _load_prompt()
    # See ``_CALL_BUDGET_FLOOR``/``_CALL_BUDGET_PER_BAD_REGION`` for the
    # derivation: one call per group, plus room for two fully-isolated bad
    # regions to each reach max bisection depth.
    call_budget_limit = max(
        _CALL_BUDGET_FLOOR,
        len(groups) + 2 * _CALL_BUDGET_PER_BAD_REGION,
    )
    budget = _CallBudget(call_budget_limit)

    # Progress reporting: percent is round(100 * resolved_lines /
    # total_lines), where a line counts as "resolved" only once a group
    # finishes (translated or fallen back) — never decreases, and gets an
    # in-flight nudge while the current group is still streaming so the
    # chip doesn't sit frozen through a long group. Throttled to ≤ one
    # publish per 100 ms so a fast LLM doesn't spam /events with a frame
    # per token.
    publish_min_interval = 0.10
    last_published_at = 0.0
    last_published_pct = -1

    def _publish_pct(pct: int) -> None:
        """Persist + broadcast a progress update. Never decreases."""
        nonlocal last_published_pct, last_published_at
        pct = max(pct, last_published_pct)
        if pct == last_published_pct:
            return
        last_published_pct = pct
        last_published_at = time.monotonic()
        log.info(
            "translator publish job=%s lang=%s pct=%d",
            job_id, lang.code, pct,
        )
        _update_status(
            job_id, lang.code,
            status="running",
            progress_percent=pct,
        )
        _publish_translation_event(
            job_id, lang.code, "running", pct, None,
        )

    resolved_lines = 0
    all_output_lines: list[str] = []
    total_fallback = 0

    try:
        for group in groups:
            # Park here if the user paused the queue. The LLM call below
            # also re-checks via respect_pause=True so we don't sneak
            # past a flip that happened between iterations.
            await _checkpoint_pause_translation()

            group_start_pct = int(round(100 * resolved_lines / total_lines))
            if group_start_pct != last_published_pct:
                _publish_pct(group_start_pct)

            group_len = len(group)
            group_resolved_base = resolved_lines

            def _on_delta(
                received_text: str,
                _group_len: int = group_len,
                _base: int = group_resolved_base,
            ) -> None:
                lines_seen = received_text.count("\n")
                fraction = min(0.99, lines_seen / max(1, _group_len))
                inflight = _base + fraction * _group_len
                pct = int(round(100 * inflight / total_lines))
                now = time.monotonic()
                if pct > last_published_pct and now - last_published_at >= publish_min_interval:
                    _publish_pct(pct)

            output_lines, fallback_count = await _translate_group(
                group, lang, prompt_template,
                depth=0, budget=budget, on_delta=_on_delta,
            )
            all_output_lines.extend(output_lines)
            total_fallback += fallback_count
            resolved_lines += group_len

            group_done_pct = int(round(100 * resolved_lines / total_lines))
            if group_done_pct != last_published_pct:
                _publish_pct(group_done_pct)
    except Exception as exc:
        log.exception("translation failed for job %s lang %s", job_id, lang.code)
        _update_status(
            job_id, lang.code,
            status="failed",
            progress_percent=0,
            error=f"translation failed: {exc}",
        )
        _publish_translation_event(
            job_id, lang.code, "failed", 0, str(exc),
        )
        return

    full = "\n".join(all_output_lines).strip()
    if not full:
        _update_status(
            job_id, lang.code,
            status="failed",
            progress_percent=0,
            error="LLM returned empty translation",
        )
        _publish_translation_event(
            job_id, lang.code, "failed", 0, "LLM returned empty translation",
        )
        return

    if total_fallback > 0:
        status = "partial"
        error = (
            f"{total_fallback} of {total_lines} lines could not be "
            "translated and are shown in the original language"
        )
    else:
        status = "done"
        error = None

    _update_status(
        job_id, lang.code,
        status=status,
        progress_percent=100,
        text=full,
        error=error,
    )
    _publish_translation_event(job_id, lang.code, status, 100, error)


async def _checkpoint_pause_translation() -> None:
    """Park if the global pause flag is set, then return when resumed.

    Same contract as ``pipeline._checkpoint_pause`` but doesn't touch
    Job.progress_stage (translations don't share that column). We just
    publish a single ``stage("paused")`` for any /events listener that
    cares, then loop on the control flag.
    """
    control = get_control()
    if not control.paused:
        return
    log.info("translator: paused — waiting for resume")
    while control.paused:
        await asyncio.sleep(0.2)
    log.info("translator: resumed")


async def _wait_for_source_text(job_id: str, language_code: str) -> str | None:
    """Poll until source transcript text is available, then return it.

    Source preference (in order):
    1. ``Job.raw_segments_json`` rendered as one-line-per-segment via
       ``timecodes.format_segments_as_marked_text`` — gives the translator
       the same fine-grained granularity as the Transcript tab UI, so
       translated transcripts inherit it.
    2. ``Job.raw_text`` (the 30 s-bucketed shape) as a fallback for
       legacy jobs and pre-mlx-patch runs where segments weren't saved.

    Returns ``None`` (and records a failed translation row) when:
    - the parent job no longer exists,
    - the parent job entered ``failed`` status before producing text.

    Polling cadence: 3 s. Whisper transcription of a long podcast takes
    minutes; one DB read every 3 s is negligible. We could swap this for
    a broker subscription on the per-job channel waiting for
    ``stage("ready")`` — but polling keeps this module independent of
    that plumbing and is plenty fast for the user-facing UX.
    """
    while True:
        with session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                log.info("translator: job %s gone while waiting for source", job_id)
                return None
            # Prefer fine-grained segments when present.
            text = _source_text_from_job(job)
            if text:
                return text
            if job.status == "failed":
                err = "parent job failed during extraction"
                _update_status(
                    job_id, language_code,
                    status="failed",
                    progress_percent=0,
                    error=err,
                )
                _publish_translation_event(
                    job_id, language_code, "failed", 0, err,
                )
                return None
        await asyncio.sleep(3.0)


def _source_text_from_job(job: Any) -> str:
    """Best source text for translation. Empty string when neither is ready.

    Mirrors ``api.jobs._build_segments_text`` semantics — fine-grained
    when ``raw_segments_json`` is set, ``raw_text`` otherwise.
    """
    raw_segments_json = getattr(job, "raw_segments_json", None)
    if raw_segments_json:
        try:
            segments = json.loads(raw_segments_json)
        except (TypeError, ValueError):
            segments = None
        if isinstance(segments, list) and segments:
            fine_text = timecodes.format_segments_as_marked_text(segments)
            if fine_text.strip():
                return fine_text
    return job.raw_text or ""


def _update_status(
    job_id: str,
    language_code: str,
    *,
    status: str,
    progress_percent: int,
    text: str | None = None,
    error: str | None = None,
) -> None:
    """Single source of truth for writing to a translation row.

    Always sets ``updated_at`` and only writes ``text`` / ``error`` when
    explicitly provided (so transient progress updates don't accidentally
    clear them). Callers that need to CLEAR a stale ``error`` from a
    previous ``partial``/``failed`` run must reset the row explicitly
    (see ``_reset_failed_rows``) — a completed ``done`` run passes
    ``error=None`` and relies on that prior reset, not on this function.
    """
    with session_scope() as session:
        row = session.get(TranscriptTranslation, (job_id, language_code))
        if row is None:
            log.warning(
                "translator: row gone for %s/%s before status=%s",
                job_id, language_code, status,
            )
            return
        row.status = status
        row.progress_percent = progress_percent
        if text is not None:
            row.text = text
        if error is not None:
            row.error = error
        row.updated_at = datetime.utcnow()
        session.add(row)


def _row_summary(row: TranscriptTranslation) -> dict[str, Any]:
    return {
        "language_code": row.language_code,
        "status": row.status,
        "progress_percent": int(row.progress_percent),
        "error": row.error,
    }


def _publish_translation_event(
    job_id: str,
    language_code: str,
    status: str,
    progress_percent: int,
    error: str | None,
) -> None:
    """Emit a ``transcript_translation`` event on both the per-job broker
    (so /ai/stream subscribers see it for the current job) AND the global
    event broker (so /events subscribers — including the sidepanel — get
    the update for chips).
    """
    payload = {
        "kind": "transcript_translation",
        "job_id": job_id,
        "language_code": language_code,
        "status": status,
        "progress_percent": progress_percent,
        "error": error,
    }
    # Per-job broker (in-flight Q&A / summary subscribers).
    get_broker().publish(job_id, payload)
    # Also re-use the job_event channel as a coarse "something changed
    # on this job" signal so the sidepanel's existing subscription picks
    # it up without needing a new event type.
    try:
        get_event_broker().publish(
            job_event("translation_updated", {"id": job_id, **payload})
        )
    except Exception:
        # Broker hiccup must not roll back the translation write itself.
        log.debug("translator: failed to publish translation event", exc_info=True)
    # Repo helpers also exist to refresh JobDetails on subscribers that
    # poll — but the broker fan-out is enough for the sidepanel.
    _ = repo  # silence unused-import lint when repo isn't called here


def _load_prompt() -> str:
    p = Path(__file__).resolve().parent.parent / "prompts" / "transcript_translate.txt"
    return p.read_text(encoding="utf-8")


__all__ = [
    "enqueue_translation",
    "retry_all_failed",
    "re_enqueue_running_on_startup",
]
