"""Background transcript translation.

Translates a job's ``raw_text`` into a target language and stores the
result in ``transcript_translation``. Reused by:

- ``POST /jobs/{id}/transcript/translate {lang}`` — user explicitly
  asked for a translation.
- ``re_enqueue_pending`` on daemon startup — any row left in
  ``status="running"`` gets picked up so the spinner the user saw
  before the restart actually finishes (we have ``raw_text`` cached,
  nothing external is needed).

Design constraints (from the project plan):

- **Dedup**: a second ``translate`` call while one is already running for
  ``(job_id, language_code)`` is a no-op. We do this with a row-level
  lock — the first POST inserts a ``queued/running`` row; the second
  sees the row, returns the current status.
- **Pause-aware**: between chunks (and via ``respect_pause=True`` on
  ``llm_client.stream_complete``) so the global pause flag works the
  same way as it does for summaries.
- **Progress in percent**: we update ``progress_percent`` after each
  chunk and ~halfway through each chunk so the chip moves visibly.
  No partial text is leaked (the UI shows only "Translating N%").
- **Marker integrity**: the chunker for summary already validates each
  chunk has balanced ``[…]`` markers; we reuse it. The translation
  prompt instructs Gemma to copy the marker verbatim — see
  ``prompts/transcript_translate.txt``.

State life-cycle:

    queued ─ (worker picked up) → running ─ (success) → done
                                    └ (exception) → failed

A ``failed`` row stays until the user clicks Retry-all (or deletes the
parent job, which CASCADEs). We never auto-retry on failure to avoid
runaway LLM bursts on a buggy prompt.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlmodel import select

from src.llm import client as llm_client
from src.llm.chunking import split_for_summary
from src.llm.languages import Language, UnknownLanguageError, normalize_lang
from src.storage import repo
from src.storage.db import Job, TranscriptTranslation, session_scope
from src.workers import timecodes
from src.workers.broker import get_broker, get_event_broker, job_event
from src.workers.control import get_control

log = logging.getLogger(__name__)


# We chunk the input transcript to keep a single LLM call within the
# model's context window. Translation is roughly text-in / text-out at
# the same token budget, so we leave headroom by halving the summary
# chunk size (which itself targets ~prompt + 1.5× output).
_TRANSCRIPT_CHUNK_TOKENS = 2000
# Overlap so words/sentences at chunk boundaries don't get mistranslated
# in isolation. Empty overlap on the line-anchored translation pattern
# would also work but tiny overlap is cheap insurance.
_TRANSCRIPT_OVERLAP_TOKENS = 0


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
        ``done`` for this ``(job_id, language_code)`` — return its current
        status, don't spawn a new task. UI sees instant feedback.
      - If a row exists with status ``failed`` — clear the error, reset
        status to ``queued``, spawn the task. Manual retry path.
      - No row → insert ``queued``, spawn the task.

    Works even when ``Job.raw_text`` isn't ready yet (job still in
    extraction phase). The spawned worker polls for raw_text and starts
    translating as soon as it appears. UI sees an immediate chip with a
    spinner — no error.

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
            if existing.status in ("queued", "running", "done"):
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
    """DB-only: flip every ``failed`` translation row for ``job_id`` back
    to ``queued`` (clearing error / text / progress). Returns the row
    summaries so the async caller can spawn workers for each.

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
                TranscriptTranslation.status == "failed",
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
    """Re-enqueue every ``failed`` translation row for ``job_id``.

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
    checkpoint partial chunk output, so the worker starts from scratch).

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
# Internals
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
    """Translate the transcript into ``lang`` chunk-by-chunk.

    Source preference: when fine-grained Whisper / YouTube segments are
    available (``Job.raw_segments_json``), we translate THOSE so the
    output keeps the same 1-5 s granularity as the original Transcript
    tab. Falls back to the 30-second-bucketed ``Job.raw_text`` for
    legacy jobs and unpatched-mlx fallbacks where segments weren't
    captured. Either way the prompt's "one input line → one output
    line with the [MM:SS] marker preserved" contract gives the
    translated body the same line shape as the source.

    May spend time at the start waiting for the source transcript to be
    extracted — if the user requested a translation BEFORE the pipeline
    had finished Whisper. We poll every 3 s; this is cheap (a single DB
    read) and lets the chip show "running 0%" with a spinner during the
    wait.
    """
    source_text = await _wait_for_source_text(job_id, lang.code)
    if source_text is None:
        # Job failed during extraction or was deleted; status already
        # recorded by ``_wait_for_source_text``.
        return

    _update_status(job_id, lang.code, status="running", progress_percent=0)
    _publish_translation_event(job_id, lang.code, "running", 0, None)

    chunks = split_for_summary(
        source_text,
        target_tokens=_TRANSCRIPT_CHUNK_TOKENS,
        overlap_tokens=_TRANSCRIPT_OVERLAP_TOKENS,
    )
    if not chunks:
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

    prompt_template = _load_prompt()
    translated_parts: list[str] = []

    try:
        # Progress reporting.
        #
        # Combined signal: MAX(lines_done / total_lines, bytes_received /
        # chunk_len). Lines is exact per the prompt contract; bytes is a
        # fallback when Gemma emits the whole translation as one block
        # with no intermediate newlines. Throttled to ≤ one publish per
        # 100 ms so a fast LLM doesn't spam /events with a frame per
        # token, but loose enough that progress stays visibly smooth.
        publish_min_interval = 0.10
        last_published_at = 0.0
        last_published_pct = -1

        def _publish_pct(pct: int) -> None:
            """Persist + broadcast a progress update. Caller is responsible
            for throttle / dedup; this just writes to DB + broker."""
            nonlocal last_published_pct, last_published_at
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

        for i, chunk in enumerate(chunks):
            # Park here if the user paused the queue. The LLM call below
            # also re-checks via respect_pause=True so we don't sneak
            # past a flip that happened between iterations.
            await _checkpoint_pause_translation()

            # Always publish at the start of each chunk so the chip
            # advances to the chunk-boundary % even if the chunk runs
            # near-instantly (small input, fast LLM).
            chunk_start_pct = int(round((i / len(chunks)) * 100))
            if chunk_start_pct != last_published_pct:
                _publish_pct(chunk_start_pct)

            prompt = prompt_template.format(
                target_language_name=lang.english_name,
                transcript=chunk,
            )
            # Two progress signals, we take the MAX of the two so the
            # chip moves no matter how the LLM phrases its output:
            #
            # - **lines**: prompt instructs one output line per input
            #   line, so lines emitted vs lines expected is the most
            #   accurate measure (1:1 by construction). Capped at
            #   ``total_lines - 1`` so we don't claim done mid-stream.
            # - **bytes**: a robust fallback when Gemma emits the whole
            #   translation as one blob (no intermediate ``\n``). The
            #   translation is roughly the same length as the source
            #   chunk, so ``received_chars / len(chunk)`` is a decent
            #   approximation, capped at 0.99.
            total_lines = max(1, chunk.count("\n") + 1)
            chunk_budget = max(1, len(chunk))
            log.info(
                "translator chunk %d/%d: budget=%d chars, total_lines=%d",
                i + 1, len(chunks), chunk_budget, total_lines,
            )
            buf: list[str] = []
            delta_count = 0
            async for delta in llm_client.stream_complete(
                prompt,
                max_tokens=4000,    # translations are roughly same length as input
                temperature=0.2,    # low but non-zero to avoid robotic phrasing
                respect_pause=True,
            ):
                buf.append(delta)
                delta_count += 1
                received_text = "".join(buf)
                lines_done = min(total_lines - 1, received_text.count("\n"))
                lines_fraction = (
                    lines_done / total_lines if total_lines > 1 else 0.0
                )
                bytes_fraction = len(received_text) / chunk_budget
                fraction = max(0.0, min(0.99, max(lines_fraction, bytes_fraction)))
                pct = int(round(((i + fraction) / len(chunks)) * 100))
                # Diagnostic: log every 50th delta so we can see what's
                # actually arriving and what fraction / pct it computes.
                if delta_count <= 5 or delta_count % 50 == 0:
                    log.info(
                        "translator delta #%d: len=%d, received=%d, "
                        "lines=%d, lines_f=%.4f, bytes_f=%.4f, pct=%d",
                        delta_count, len(delta), len(received_text),
                        lines_done, lines_fraction, bytes_fraction, pct,
                    )
                now = time.monotonic()
                if (
                    pct != last_published_pct
                    and now - last_published_at >= publish_min_interval
                ):
                    _publish_pct(pct)
            translated_parts.append("".join(buf).strip())

            # End-of-chunk: jump to the chunk-completion %.
            chunk_done_pct = int(round(((i + 1) / len(chunks)) * 100))
            if chunk_done_pct != last_published_pct:
                _publish_pct(chunk_done_pct)
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

    full = "\n".join(translated_parts).strip()
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

    _update_status(
        job_id, lang.code,
        status="done",
        progress_percent=100,
        text=full,
    )
    _publish_translation_event(job_id, lang.code, "done", 100, None)


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
    clear them).
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
