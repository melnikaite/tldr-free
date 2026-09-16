"""Job pipeline — orchestrates extraction + summarization and publishes events.

A "pipeline" is a coroutine spawned by ``api/jobs.create_job`` for each new
job. It owns the row's status transitions and broadcasts AIStageEvent /
AIDeltaEvent / AIDoneEvent / AIErrorEvent via the broker so SSE subscribers
(``/ai/stream``) see live progress.

Two distinct pipelines live here:

- ``run_fast_pipeline`` — for kind=page and kind=youtube where the transcript
  API works. Runs extraction inline, then streams the summary. Always
  finishes the job (status=done | failed) before returning.

- ``defer_to_whisper`` — for kind=youtube where the transcript fetch failed
  permanently or exhausted retries. Marks the job queued, enqueues a
  WhisperTask, and returns. The whisper worker (``runner.whisper_worker``)
  picks up from there and continues the same event stream via the broker.

Both paths converge on the same broker channel for a job_id, so subscribers
don't have to know which path the job took.

Stage names are coordinated with the schema's AIStageEvent docs:
"queued", "extracting", "transcribing", "ready", "summarizing".
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.api.schemas import (
    AUDIO_TRANSCRIPT_SOURCES,
    DeferredReason,
    JobKind,
    JobStatus,
    TranscriptSource,
)
from src.config import get_config
from src.llm import languages
from src.llm import summary as llm_summary
from src.llm import vision as llm_vision
from src.storage import repo
from src.workers import deixis, page, timecodes, youtube
from src.workers import frames as frames_mod
from src.workers import pdf as pdf_worker
from src.workers.broker import (
    delta_event,
    done_event,
    error_event,
    get_broker,
    stage_event,
)
from src.workers.control import get_control
from src.workers.deixis import DeixisCandidate, DeixisCategory
from src.workers.errors import (
    ExhaustedRetriesError,
    PermanentTranscriptError,
)
from src.workers.log_context import reset_job_id, set_job_id
from src.workers.queue import WhisperTask, get_queue

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pause checkpoint — call between pipeline steps. If the user has paused,
# parks here, surfaces ``progress_stage="paused"`` to the Library, then
# restores the previous stage on resume so the row picks up where it was.
# ---------------------------------------------------------------------------


async def _checkpoint_pause(job_id: str, broker: Any, on_resume_stage: str) -> None:
    """If paused, wait for resume and surface a ``paused`` progress stage.

    The current step always finishes — we only park BETWEEN steps. This is
    the soft-pause contract: in-flight work runs to completion, the next
    step blocks. After resume we restore ``progress_stage=on_resume_stage``
    so the Library row goes back to e.g. ``transcribing`` instead of
    silently sitting at ``paused``.
    """
    control = get_control()
    if not control.paused:
        return
    repo.update_status(job_id, status=JobStatus.RUNNING.value, progress_stage="paused")
    broker.publish(job_id, stage_event("paused"))
    await control.wait_if_paused()
    repo.update_status(
        job_id, status=JobStatus.RUNNING.value, progress_stage=on_resume_stage,
    )
    broker.publish(job_id, stage_event(on_resume_stage))


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


async def run_pipeline(
    job_id: str,
    *,
    kind: JobKind,
    url: str,
    page_text: str | None,
    page_title: str | None,
    media_url: str | None,
    pdf_bytes: bytes | None,
    cookies: list[Any],
) -> None:
    """Top-level pipeline runner. Decides the path based on kind + extraction.

    Spawned via ``asyncio.create_task`` from ``POST /jobs``. Never raises —
    all failures are swallowed into ``mark_failed`` + ``error_event``.
    """
    broker = get_broker()
    # Bind job_id for every worker log line emitted while this job is being
    # processed (page/youtube/media/pdf branches below, plus anything they
    # call into) — see workers/log_context.py. Reset in `finally` so the
    # binding never leaks into whatever this task (or thread pool) does
    # next.
    token = set_job_id(job_id)
    try:
        if kind == JobKind.PAGE:
            await _run_page(
                job_id,
                url=url,
                page_text=page_text,
                page_title=page_title,
                cookies=cookies,
            )
        elif kind == JobKind.MEDIA:
            if not media_url:
                # Defensive: api/jobs validates this upstream, but the
                # invariant lives here too so callers don't accidentally
                # call us with kind=MEDIA + no URL and get a silent hang.
                repo.mark_failed(job_id, error="media kind requires media_url")
                broker.publish(job_id, error_event("media kind requires media_url"))
                return
            await _run_media(
                job_id,
                media_url=media_url,
                page_title=page_title,
                page_text=page_text,
                cookies=cookies,
            )
        elif kind == JobKind.PDF:
            await _run_pdf(
                job_id,
                url=url,
                pdf_bytes=pdf_bytes,
                page_title=page_title,
                cookies=cookies,
            )
        else:
            await _run_youtube(job_id, url=url, page_title=page_title, cookies=cookies)
    except Exception as exc:
        log.exception("pipeline crashed for job %s", job_id)
        try:
            repo.mark_failed(job_id, error=f"pipeline error: {exc}")
        except Exception:
            log.exception("repo.mark_failed also failed for %s", job_id)
        broker.publish(job_id, error_event(f"pipeline error: {exc}"))
    finally:
        reset_job_id(token)


# ---------------------------------------------------------------------------
# Page path
# ---------------------------------------------------------------------------


async def _run_page(
    job_id: str,
    *,
    url: str,
    page_text: str | None,
    page_title: str | None,
    cookies: list[Any],
) -> None:
    broker = get_broker()
    cfg = get_config()

    repo.update_status(job_id, status=JobStatus.RUNNING.value, progress_stage="extracting")
    broker.publish(job_id, stage_event("extracting"))

    text = (page_text or "").strip()
    title = page_title
    transcript_source = TranscriptSource.PAGE_EXTRACT

    # Pause checkpoint before any slow network work.
    await _checkpoint_pause(job_id, broker, "extracting")

    if not text:
        try:
            extracted_title, extracted_text = await page.extract_with_trafilatura(url)
        except Exception as exc:
            log.exception("trafilatura failed for %s", url)
            repo.mark_failed(job_id, error=f"page extraction failed: {exc}")
            broker.publish(job_id, error_event(f"page extraction failed: {exc}"))
            return
        text = (extracted_text or "").strip()
        if not title and extracted_title:
            title = extracted_title
        transcript_source = TranscriptSource.TRAFILATURA

    if not text:
        repo.mark_failed(job_id, error="failed to extract page text")
        broker.publish(job_id, error_event("failed to extract page text"))
        return

    # Pause checkpoint before persist + summary so resume picks up at "ready".
    await _checkpoint_pause(job_id, broker, "extracting")

    # Extraction done — persist raw_text immediately so /ai/stream replay
    # works even if the user disconnects before summary completes.
    _persist_extracted(
        job_id,
        raw_text=text,
        title=title,
        transcript_source=transcript_source,
    )
    broker.publish(job_id, stage_event("ready"))

    await _summarize_and_finish(
        job_id,
        text=text,
        title=title,
        transcript_source=transcript_source,
        video_id=None,
        cfg=cfg,
        cookies=cookies,
    )


# ---------------------------------------------------------------------------
# YouTube path
# ---------------------------------------------------------------------------


async def _run_youtube(
    job_id: str,
    *,
    url: str,
    page_title: str | None,
    cookies: list[Any],
) -> None:
    broker = get_broker()
    cfg = get_config()

    repo.update_status(job_id, status=JobStatus.RUNNING.value, progress_stage="extracting")
    broker.publish(job_id, stage_event("extracting"))

    try:
        video_id = youtube.extract_video_id(url)
    except ValueError as exc:
        repo.mark_failed(job_id, error=f"invalid youtube url: {exc}")
        broker.publish(job_id, error_event(f"invalid youtube url: {exc}"))
        return

    # Pause checkpoint before fetching captions / transcript.
    await _checkpoint_pause(job_id, broker, "extracting")

    transcript_source: TranscriptSource | None = None
    segments: list[dict[str, Any]] | None = None

    try:
        segments = await youtube.fetch_transcript_with_retry(
            video_id=video_id,
            cookies=cookies,
            max_attempts=cfg.youtube.fast_path_max_attempts,
            backoff_seconds=cfg.youtube.fast_path_backoff_seconds,
            preferences=cfg.youtube.subtitle_lang_preferences,
            output_language=cfg.output.language,
        )
        transcript_source = TranscriptSource.YOUTUBE_API
    except (PermanentTranscriptError, ExhaustedRetriesError) as exc:
        try:
            reason = DeferredReason(exc.code)
        except ValueError:
            reason = DeferredReason.NETWORK_ERROR
        log.info(
            "job %s: youtube-transcript-api unavailable (%s); trying yt-dlp captions",
            job_id, reason.value,
        )
        # Pause checkpoint before the second slow yt-dlp call.
        await _checkpoint_pause(job_id, broker, "fetching_captions")
        broker.publish(job_id, stage_event("fetching_captions"))
        try:
            yt_segments = await youtube.download_subtitles(
                url=url,
                cookies=cookies,
                dir=_subtitles_dir(),
                lang_preferences=cfg.youtube.subtitle_lang_preferences,
                output_language=cfg.output.language,
                max_attempts=cfg.youtube.caption_fallback_max_attempts,
                backoff_seconds=cfg.youtube.caption_fallback_backoff_seconds,
            )
        except Exception:
            log.exception("yt-dlp subtitle fallback failed for %s", job_id)
            yt_segments = None

        if yt_segments:
            log.info("job %s: fetched %d caption segments via yt-dlp", job_id, len(yt_segments))
            segments = yt_segments
            transcript_source = TranscriptSource.YOUTUBE_AUTO_CAPTIONS
        else:
            # Both fast paths failed → defer to Whisper.
            try:
                await get_queue().put(
                    WhisperTask(job_id=job_id, url=url, cookies=cookies)
                )
            except Exception as queue_exc:
                log.exception("failed to enqueue %s", job_id)
                repo.mark_failed(job_id, error=f"queue error: {queue_exc}")
                broker.publish(job_id, error_event(f"queue error: {queue_exc}"))
                return

            repo.update_status(
                job_id,
                status=JobStatus.QUEUED.value,
                progress_stage="queued",
                queued_reason=reason.value,
            )
            broker.publish(job_id, stage_event("queued", detail=reason.value))
            return

    # Fast path success (either source) — shared tail with the generic media
    # captions path below.
    assert segments is not None and transcript_source is not None
    await _finish_caption_fast_path(
        job_id,
        url=url,
        cookies=cookies,
        segments=segments,
        transcript_source=transcript_source,
        video_id=video_id,
        page_title=page_title,
        cfg=cfg,
    )


# ---------------------------------------------------------------------------
# Shared: the tail of every caption-based fast path (YouTube API, YouTube's
# own yt-dlp captions, and generic site captions below)
# ---------------------------------------------------------------------------


async def _finish_caption_fast_path(
    job_id: str,
    *,
    url: str,
    cookies: list[Any],
    segments: list[dict[str, Any]],
    transcript_source: TranscriptSource,
    video_id: str | None,
    page_title: str | None,
    cfg: Any,
) -> None:
    """Build [MM:SS]-marked text, resolve title/language, persist, summarize.

    Common to any path that already has caption segments in hand (``{start,
    duration, text}``) and just needs to turn them into a finished job —
    whether those segments came from youtube-transcript-api, yt-dlp captions
    on YouTube, or yt-dlp captions on a generic site. ``video_id`` is
    YouTube-only (``None`` for generic media); nothing here branches on it.
    """
    broker = get_broker()

    raw_text = timecodes.build_marked_text(
        segments,
        window_seconds=cfg.youtube.segment_window_seconds,
    )
    # Also serialise the fine-grained segments themselves so the Transcript
    # tab can render one line per ~2-5 s caption cue (vs the 30 s buckets
    # ``raw_text`` uses for summary). Caption sources hand us
    # {start, duration, text}; we normalise into the {start, end, text}
    # shape ``_build_segments_text`` in api/jobs.py consumes.
    raw_segments_json: str | None = None
    if len(segments) > 1:
        normalised = [
            {
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("start", 0.0))
                + float(s.get("duration", s.get("end", 0.0) - s.get("start", 0.0))),
                "text": str(s.get("text") or ""),
            }
            for s in segments
        ]
        raw_segments_json = json.dumps(
            normalised, ensure_ascii=False, separators=(",", ":"),
        )

    # Pause checkpoint before another yt-dlp probe (metadata) + persist + summary.
    await _checkpoint_pause(job_id, broker, "extracting")

    # Authoritative title from yt-dlp metadata. The extension scrapes
    # ``document.title`` / ``h1`` from a possibly stale DOM (especially a
    # YouTube SPA injected into a backgrounded tab, but also true of generic
    # media pages), so its guess can be wrong. Fall back to the extension's
    # title only if the probe fails.
    metadata = await youtube.fetch_video_metadata(
        url=url, cookies=cookies, scratch_dir=_subtitles_dir(),
    )
    title = metadata.get("title") or page_title
    # yt-dlp's metadata probe already returns the video's primary language
    # (or original_language for dubbed videos). Use that as the transcript
    # source language — it's the closest signal we have on the fast path,
    # because the caption track we picked may differ (e.g. auto-translated
    # captions in another language). Best-effort only — None falls through
    # cleanly and the UI shows "Original".
    transcript_language = languages.short_lang_code(metadata.get("language"))
    # Last resort when metadata carries no language: guess from the captions.
    if transcript_language is None:
        transcript_language = languages.detect_language(raw_text)

    _persist_extracted(
        job_id,
        raw_text=raw_text,
        title=title,
        transcript_source=transcript_source,
        video_id=video_id,
        transcript_language=transcript_language,
        raw_segments_json=raw_segments_json,
    )
    broker.publish(job_id, stage_event("ready"))

    await _summarize_and_finish(
        job_id,
        text=raw_text,
        title=title,
        transcript_source=transcript_source,
        video_id=video_id,
        transcript_language=transcript_language,
        raw_segments_json=raw_segments_json,
        cfg=cfg,
        cookies=cookies,
    )


# ---------------------------------------------------------------------------
# Generic media path (non-YouTube): direct mp4/webm, HLS/DASH, iframe embeds
# Vimeo/Dailymotion/Twitch/Bunny/Brightcove/JW/Wistia/Streamable/SoundCloud/…
#
# Same caption-first, Whisper-fallback shape as the YouTube path above: many
# non-YouTube sites publish real subtitle tracks yt-dlp can list and fetch
# (ZDF, ARD, Vimeo, TED, Coursera, …) — confirmed concretely against a ZDF
# episode whose site captions covered 24:54 of a 24:56 runtime while our
# Whisper run on the same audio left ~5 minutes of gaps. ``download_subtitles``
# (workers/youtube.py) already probes/downloads/parses generically for any
# yt-dlp-supported URL; it only needed WebVTT format negotiation alongside
# YouTube's json3. An empty/failed probe is NOT an error — it falls straight
# through to the Whisper queue, which already handles yt-dlp audio download →
# transcribe → summarize for any URL yt-dlp can extract from. The runner is
# URL-agnostic — ``WhisperTask.url`` becomes the argument to
# ``youtube.download_audio`` regardless of the original kind.
# ---------------------------------------------------------------------------


async def _run_media(
    job_id: str,
    *,
    media_url: str,
    page_title: str | None,
    page_text: str | None,
    cookies: list[Any],
) -> None:
    broker = get_broker()
    cfg = get_config()

    repo.update_status(job_id, status=JobStatus.RUNNING.value, progress_stage="extracting")
    broker.publish(job_id, stage_event("extracting"))

    # Pause checkpoint before the caption probe.
    await _checkpoint_pause(job_id, broker, "extracting")

    broker.publish(job_id, stage_event("fetching_captions"))
    try:
        segments = await youtube.download_subtitles(
            url=media_url,
            cookies=cookies,
            dir=_subtitles_dir(),
            lang_preferences=cfg.youtube.subtitle_lang_preferences,
            output_language=cfg.output.language,
            max_attempts=cfg.youtube.caption_fallback_max_attempts,
            backoff_seconds=cfg.youtube.caption_fallback_backoff_seconds,
            # Unlike YouTube (where an empty result is often a throttled/
            # bot-checked session — see download_subtitles's docstring), a
            # generic site either has a subtitle track or it doesn't: a
            # clean "no track" result reproduces identically on every
            # attempt, so retrying it here would only add 2 more
            # extract_info probes + backoff sleeps to the common captionless
            # case for nothing. Real transient failures (exceptions) still
            # retry up to max_attempts regardless of this flag.
            retry_on_no_track=False,
        )
    except Exception:
        log.exception("caption probe failed for media job %s", job_id)
        segments = None

    if segments:
        log.info(
            "job %s: fetched %d caption segments from site subtitles",
            job_id, len(segments),
        )
        await _finish_caption_fast_path(
            job_id,
            url=media_url,
            cookies=cookies,
            segments=segments,
            transcript_source=TranscriptSource.SITE_CAPTIONS,
            video_id=None,
            page_title=page_title,
            cfg=cfg,
        )
        return

    # No usable site captions (or the probe itself failed) — Whisper, same
    # as always.
    try:
        await get_queue().put(
            WhisperTask(job_id=job_id, url=media_url, cookies=cookies, page_text=page_text)
        )
    except Exception as exc:
        log.exception("failed to enqueue media job %s", job_id)
        repo.mark_failed(job_id, error=f"queue error: {exc}")
        broker.publish(job_id, error_event(f"queue error: {exc}"))
        return

    repo.update_status(
        job_id,
        status=JobStatus.QUEUED.value,
        progress_stage="queued",
    )
    broker.publish(job_id, stage_event("queued"))


# ---------------------------------------------------------------------------
# PDF path — text via pypdf (fast path), vision OCR fallback for scanned PDFs
# ---------------------------------------------------------------------------


async def _run_pdf(
    job_id: str,
    *,
    url: str,
    pdf_bytes: bytes | None,
    page_title: str | None,
    cookies: list[Any],
) -> None:
    broker = get_broker()
    cfg = get_config()

    repo.update_status(job_id, status=JobStatus.RUNNING.value, progress_stage="extracting")
    broker.publish(job_id, stage_event("extracting"))

    await _checkpoint_pause(job_id, broker, "extracting")

    try:
        text, transcript_source = await pdf_worker.process_pdf(
            job_id=job_id, url=url, pdf_bytes=pdf_bytes, cookies=cookies,
        )
    except Exception as exc:
        log.exception("pdf extraction failed for %s", job_id)
        repo.mark_failed(job_id, error=f"pdf extraction failed: {exc}")
        broker.publish(job_id, error_event(f"pdf extraction failed: {exc}"))
        return
    finally:
        # Release the raw PDF bytes (tens of MB for big files) before the
        # LLM summary step starts — extraction is the only step that needs
        # them. Without this, a 30 MB PDF stays resident through minutes
        # of streaming summary, multiplied by every concurrent job.
        pdf_bytes = None

    text = text.strip()
    if not text:
        repo.mark_failed(
            job_id,
            error="pdf produced no extractable text (try OCRing it first)",
        )
        broker.publish(job_id, error_event("pdf produced no extractable text"))
        return

    await _checkpoint_pause(job_id, broker, "extracting")

    _persist_extracted(
        job_id,
        raw_text=text,
        title=page_title,
        transcript_source=transcript_source,
    )
    broker.publish(job_id, stage_event("ready"))

    await _summarize_and_finish(
        job_id,
        text=text,
        title=page_title,
        transcript_source=transcript_source,
        video_id=None,
        cfg=cfg,
        cookies=cookies,
    )


# ---------------------------------------------------------------------------
# Shared: persist raw_text (mid-pipeline), summarize, mark done
# ---------------------------------------------------------------------------


def _persist_extracted(
    job_id: str,
    *,
    raw_text: str,
    title: str | None,
    transcript_source: TranscriptSource,
    video_id: str | None = None,
    transcript_language: str | None = None,
    raw_segments_json: str | None = None,
) -> None:
    """Set raw_text + transcript_source + video_id mid-pipeline (no status change).

    Done BEFORE summarization so /ai/stream subscribers can fall back to
    raw_text on summary failure or restart. ``repo.set_extracted`` itself
    publishes ``job_event("updated")`` — that's the path that surfaces the
    canonical YouTube title to the Library before the summary lands.
    """
    repo.set_extracted(
        job_id,
        raw_text=raw_text,
        transcript_source=transcript_source.value,
        title=title,
        video_id=video_id,
        transcript_language=transcript_language,
        raw_segments_json=raw_segments_json,
    )


# ---------------------------------------------------------------------------
# Summary-time frame analysis — runs BEFORE summarization for jobs with a
# timestamped speech transcript, so the summary prompt can say what the
# video's picture actually shows instead of only what the words say.
# ---------------------------------------------------------------------------


async def _fetch_moment_frames_safe(
    job: Any, candidate: DeixisCandidate, job_id: str, cookies: list[Any]
) -> list[Path]:
    """`llm.vision.fetch_moment_frames`, degraded: any failure (download
    error, budget spent) logs a warning and returns ``[]`` rather than
    raising — this step must never fail the job over one bad moment. Kept
    as its own function (rather than inlined in the conveyor loop below) so
    it can be handed to ``asyncio.create_task`` for the prefetch.

    ``cookies`` is the ingestion-time cookie list still in scope via
    ``run_pipeline`` → ``_summarize_and_finish`` → ``_run_frame_analysis``
    — see ``llm.vision.fetch_moment_frames``'s docstring for why this step
    (unlike the QA LOOK step) has cookies to forward at all.
    """
    try:
        return await llm_vision.fetch_moment_frames(job=job, candidate=candidate, cookies=cookies)
    except Exception:
        log.warning(
            "frame analysis: frame fetch failed for job %s at %.1fs",
            job_id, candidate.timestamp, exc_info=True,
        )
        return []


async def _run_frame_analysis(
    job_id: str, broker: Any, cookies: list[Any]
) -> list[dict[str, Any]]:
    """Summary-time counterpart to the QA LOOK step: inspect a handful of
    moments where the transcript's speech points at the video's picture,
    and return the ones a vision model judged as actually adding something
    (see ``llm.vision.analyze_summary_frames`` / ``prompts/summary_frames.
    txt``). Never raises — any failure anywhere in this function degrades
    to ``[]``, and the caller (``_summarize_and_finish``) proceeds to
    summarize normally either way.

    No-op (returns ``[]`` immediately) for a job that doesn't qualify for
    deixis candidates at all — ``workers.deixis.candidates_for_job``
    already encodes that rule (pages/PDFs/caption-less jobs), so it isn't
    re-derived here.

    EXTERNAL candidates are dropped up front: there is no frame to fetch
    for a "look in the description" reference (see ``DeixisCategory``).

    Selection is CHRONOLOGICAL, FIRST-FIT: walk the surviving candidates in
    transcript order and keep taking moments until the next one would push
    the per-job frame budget (``workers.frames.MAX_FRAMES_PER_JOB``) over
    the edge — roughly 4 moments at ``DEFAULT_NUM_FRAMES`` (5) frames each.
    This is deliberately different from the QA LOOK step, which ranks
    candidates by what the model's PLAN call picked as relevant to a
    specific question — there is no question here to rank moments
    against, so "first N in the video" is the only ordering that doesn't
    require guessing which moments matter most.

    Conveyor: while awaiting moment i's (slow) vision call, moment i+1's
    frames are already downloading — prefetch depth exactly 1 (never more,
    to keep disk usage and yt-dlp concurrency bounded). A prefetch task
    that ends up never consumed (an unexpected early exit from the loop —
    a bug, cancellation, or a raised exception that escapes the per-moment
    guards below) is always awaited-or-cancelled in the ``finally`` block,
    never left dangling.

    ``cookies`` is the SAME list ``run_pipeline`` received on the original
    job-creation request, threaded down through ``_summarize_and_finish``
    — forwarded to every frame fetch below so a cookie-gated video's frames
    can actually be downloaded here, unlike the QA LOOK step which has none
    left to forward by the time a question is asked (see
    ``llm.vision.fetch_moment_frames``'s docstring).
    """
    try:
        job = repo.get_job(job_id)
    except Exception:
        log.warning("frame analysis: failed to load job %s; skipping", job_id, exc_info=True)
        return []
    if job is None:
        return []

    candidates = [
        c for c in deixis.candidates_for_job(job) if c.category != DeixisCategory.EXTERNAL
    ]
    if not candidates:
        return []

    selected: list[DeixisCandidate] = []
    frames_budget_used = 0
    for candidate in candidates:
        if frames_budget_used + frames_mod.DEFAULT_NUM_FRAMES > frames_mod.MAX_FRAMES_PER_JOB:
            break
        selected.append(candidate)
        frames_budget_used += frames_mod.DEFAULT_NUM_FRAMES
    if not selected:
        return []

    await _checkpoint_pause(job_id, broker, "analyzing_frames")

    output_language = get_config().output.language_name
    findings: list[dict[str, Any]] = []
    prefetch_task: asyncio.Task[list[Path]] | None = None
    try:
        for i, candidate in enumerate(selected):
            frame_paths = (
                await prefetch_task if prefetch_task is not None
                else await _fetch_moment_frames_safe(job, candidate, job_id, cookies)
            )
            prefetch_task = None

            # Kick off the NEXT moment's download now, before this moment's
            # (slow) vision call below — that overlap is the whole point of
            # the conveyor.
            if i + 1 < len(selected):
                prefetch_task = asyncio.create_task(
                    _fetch_moment_frames_safe(job, selected[i + 1], job_id, cookies)
                )

            if not frame_paths:
                continue

            timecode = timecodes.format_timecode(candidate.timestamp)
            broker.publish(
                job_id,
                stage_event(
                    "analyzing_frames", detail=f"{timecode} — {candidate.phrase}"
                ),
            )
            try:
                result = await llm_vision.analyze_summary_frames(
                    frame_paths,
                    candidate=candidate,
                    output_language=output_language,
                    job_id=job_id,
                )
            except Exception:
                log.warning(
                    "frame analysis: vision call failed for job %s at %.1fs",
                    job_id, candidate.timestamp, exc_info=True,
                )
                continue

            # Only relevant:true moments are worth storing — a generic shot
            # with nothing to add would just clutter the summary prompt and
            # the client's thumbnail row for no benefit (see
            # prompts/summary_frames.txt's meaning of `relevant` here).
            if not result.relevant:
                continue

            frame_path = (
                frame_paths[result.best_frame_index - 1]
                if result.best_frame_index is not None
                else None
            )
            findings.append(
                {
                    "seconds": candidate.timestamp,
                    "timecode": timecode,
                    "phrase": candidate.phrase,
                    "category": candidate.category.value,
                    "finding": result.finding,
                    "frame_url": (
                        f"/jobs/{job_id}/frames/{frame_path.parent.name}/{frame_path.name}"
                        if frame_path is not None
                        else None
                    ),
                }
            )
    finally:
        # A prefetch task nobody consumed (budget exhausted mid-loop is
        # impossible here since `selected` is pre-sized, but an unexpected
        # early exit — cancellation, a bug escaping the guards above — can
        # still leave one pending). Cancel and await it rather than
        # abandoning it, so its yt-dlp subprocess/ffmpeg call always winds
        # down cleanly instead of outliving this function.
        if prefetch_task is not None and not prefetch_task.done():
            prefetch_task.cancel()
        if prefetch_task is not None:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await prefetch_task

    return findings


async def _summarize_and_finish(
    job_id: str,
    *,
    text: str,
    title: str | None,
    transcript_source: TranscriptSource,
    video_id: str | None,
    transcript_language: str | None = None,
    raw_segments_json: str | None = None,
    cfg: Any,
    cookies: list[Any],
) -> None:
    """Run streaming summarization and mark the job done.

    Publishes stage("summarizing"), then a stream of delta events, then
    a done event. On exception, marks the job failed and publishes error.

    Honours the global pause flag before kicking off the LLM call so a
    paused user doesn't pay a fresh ML burst on a fresh job. In-flight
    streaming completes normally — pause is checkpoint-based, not preemptive.

    ``cookies`` is the same list ``run_pipeline`` received on the original
    job-creation request, still in scope this deep because ingestion hasn't
    finished yet — passed straight through to ``_run_frame_analysis`` (see
    its docstring). All three callers below have it in scope; page/PDF jobs
    pass it too for consistency even though frame analysis is a no-op for
    them (``deixis.candidates_for_job`` returns ``[]`` for non-transcript
    kinds), so nothing behavioural changes there.
    """
    broker = get_broker()

    # Summary-time frame analysis, BEFORE the summarization call — see
    # _run_frame_analysis's docstring. Degrades to [] on any failure and
    # never raises, so this can never fail the job. Persisted immediately
    # (rather than only at mark_done) so the record survives even if the
    # summary call that follows fails partway through — same reasoning as
    # diagnostics_json being written at set_extracted time.
    visual_findings: list[timecodes.VisualFinding] = []
    try:
        moment_findings = await _run_frame_analysis(job_id, broker, cookies)
    except Exception:
        log.warning("frame analysis step failed for job %s; continuing", job_id, exc_info=True)
        moment_findings = []
    if moment_findings:
        try:
            repo.set_moment_findings(
                job_id,
                moment_findings_json=json.dumps(moment_findings, ensure_ascii=False),
            )
        except Exception:
            log.warning(
                "failed to persist moment findings for job %s", job_id, exc_info=True,
            )
        visual_findings = [
            timecodes.VisualFinding(seconds=m["seconds"], text=m["finding"])
            for m in moment_findings
        ]

    # Park here while the user has the global queue paused.
    await _checkpoint_pause(job_id, broker, "summarizing")

    repo.update_status(job_id, status=JobStatus.RUNNING.value, progress_stage="summarizing")
    broker.publish(job_id, stage_event("summarizing"))

    parts: list[str] = []
    # Batch delta publishes — without this an LLM that emits 50-100 tokens/sec
    # floods the broker (and the SSE event loop) so badly that concurrent
    # /jobs and /events readers stall waiting for a slot. 100ms / 64 chars
    # keeps the stream visually fluid while letting the loop schedule work.
    buf: list[str] = []
    last_flush = asyncio.get_event_loop().time()
    FLUSH_INTERVAL = 0.1
    FLUSH_CHARS = 64

    def _flush() -> None:
        nonlocal last_flush
        if not buf:
            return
        broker.publish(job_id, delta_event("".join(buf)))
        buf.clear()
        last_flush = asyncio.get_event_loop().time()

    # Transcript-derived sources (Whisper / captions) may carry speech-to-text
    # artefacts: trailing outro hallucinations and misheard terms. Feed the
    # summariser a tail-cleaned copy and flag the source so it corrects obvious
    # ASR errors. The stored transcript (`text` → mark_done) is untouched.
    from_audio = transcript_source in AUDIO_TRANSCRIPT_SOURCES
    summary_input = timecodes.strip_transcript_tail_noise(text) if from_audio else text
    # Weave any summary-time on-screen findings into the material itself —
    # see timecodes.inject_visual_findings for why this replaced a separate
    # appended VISUAL FINDINGS block. `text` (-> mark_done's stored raw_text)
    # is never touched by this; only this local, LLM-facing copy is.
    if visual_findings:
        summary_input = timecodes.inject_visual_findings(summary_input, visual_findings)

    try:
        # cap_markers_in_stream holds text back only long enough to resolve a
        # [MM:SS]-shaped bracket (never a whole line), capping markers per
        # line as each one resolves, so what's published here and what's
        # accumulated into `parts` (-> stored summary_md) are always the same
        # capped text at every point in the stream — not just once at the
        # end. See timecodes.cap_markers_in_stream for the full rationale
        # (incl. why this is a code-level fix, not a prompt change).
        async for delta in timecodes.cap_markers_in_stream(
            llm_summary.stream_summarize(
                summary_input,
                title=title,
                output_language=cfg.output.language_name,
                from_audio_transcript=from_audio,
            )
        ):
            parts.append(delta)
            buf.append(delta)
            now = asyncio.get_event_loop().time()
            if (
                sum(len(s) for s in buf) >= FLUSH_CHARS
                or (now - last_flush) >= FLUSH_INTERVAL
            ):
                _flush()
        _flush()  # tail
    except Exception as exc:
        log.exception("summary failed for job %s", job_id)
        repo.mark_failed(job_id, error=f"summarization failed: {exc}")
        broker.publish(job_id, error_event(f"summarization failed: {exc}"))
        return

    summary_md = timecodes.strip_timecode_placeholders("".join(parts).strip())
    # Non-transcript sources (web pages, PDFs) have no timecodes in the source,
    # so any [MM:SS] marker here is a model hallucination — strip them all. For
    # audio sources real markers must survive, so this only runs for the rest.
    if not from_audio:
        summary_md = timecodes.strip_all_timecodes(summary_md)
    if not summary_md:
        repo.mark_failed(job_id, error="LLM returned empty summary")
        broker.publish(job_id, error_event("LLM returned empty summary"))
        return

    repo.mark_done(
        job_id,
        raw_text=text,
        summary_md=summary_md,
        transcript_source=transcript_source.value,
        title=title,
        video_id=video_id,
        transcript_language=transcript_language,
        raw_segments_json=raw_segments_json,
    )
    broker.publish(job_id, done_event(summary_md))

    # Optional cooldown to give the host a breather before the next pipeline
    # task grabs the LLM lock.
    cooldown = max(0, cfg.workers.cooldown_seconds)
    if cooldown:
        log.info("pipeline %s: cooldown for %ds", job_id, cooldown)
        try:
            await asyncio.sleep(cooldown)
        except asyncio.CancelledError:
            raise


# ---------------------------------------------------------------------------
# Helpers used by api/jobs
# ---------------------------------------------------------------------------


def _subtitles_dir() -> Any:
    """Scratch directory for yt-dlp's transient subtitle downloads."""
    from pathlib import Path
    p = Path(get_config().storage.data_dir) / "subtitles"
    p.mkdir(parents=True, exist_ok=True)
    return p


_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}


def infer_kind(
    url: str,
    declared: str,
    media_url: str | None = None,
    *,
    pdf_bytes_present: bool = False,
) -> JobKind:
    """Map ``kind="auto"`` to the concrete enum based on the URL + hints.

    Resolution order for ``declared == "auto"``:
      1. ``media_url`` set → ``MEDIA`` (extension found a transcribable
         media element on the page; takes priority over host inference so
         a YouTube *embed* on a third-party site still flows through the
         generic media path with the iframe URL, not through the YouTube
         fast path on the page URL).
      2. ``pdf_bytes_present`` or URL path ends in ``.pdf`` → ``PDF``.
      3. URL host in ``_YOUTUBE_HOSTS`` → ``YOUTUBE``.
      4. Otherwise → ``PAGE``.
    """
    if declared in (
        JobKind.PAGE.value,
        JobKind.YOUTUBE.value,
        JobKind.MEDIA.value,
        JobKind.PDF.value,
    ):
        return JobKind(declared)
    if media_url:
        return JobKind.MEDIA
    parsed = urlparse(url)
    if pdf_bytes_present or (parsed.path or "").lower().endswith(".pdf"):
        return JobKind.PDF
    host = (parsed.hostname or "").lower()
    if host in _YOUTUBE_HOSTS:
        return JobKind.YOUTUBE
    return JobKind.PAGE


__all__ = ["infer_kind", "run_pipeline"]
