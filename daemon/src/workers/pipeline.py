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
import json
import logging
from typing import Any
from urllib.parse import urlparse

from src.api.schemas import (
    DeferredReason,
    JobKind,
    JobStatus,
    TranscriptSource,
)
from src.config import get_config
from src.llm import languages
from src.llm import summary as llm_summary
from src.storage import repo
from src.workers import page, timecodes, youtube
from src.workers import pdf as pdf_worker
from src.workers.broker import (
    delta_event,
    done_event,
    error_event,
    get_broker,
    stage_event,
)
from src.workers.control import get_control
from src.workers.errors import (
    ExhaustedRetriesError,
    PermanentTranscriptError,
)
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
    try:
        if kind == JobKind.PAGE:
            await _run_page(job_id, url=url, page_text=page_text, page_title=page_title)
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


# ---------------------------------------------------------------------------
# Page path
# ---------------------------------------------------------------------------


async def _run_page(
    job_id: str,
    *,
    url: str,
    page_text: str | None,
    page_title: str | None,
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
            )
            broker.publish(job_id, stage_event("queued", detail=reason.value))
            return

    # Fast path success (either source) — produce raw_text with [MM:SS] markers.
    assert segments is not None and transcript_source is not None
    raw_text = timecodes.build_marked_text(
        segments,
        window_seconds=cfg.youtube.segment_window_seconds,
    )
    # Also serialise the fine-grained segments themselves so the Transcript
    # tab can render one line per ~2-5 s caption cue (vs the 30 s buckets
    # ``raw_text`` uses for summary). youtube-transcript-api hands us
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

    # Authoritative title from YouTube via yt-dlp metadata. The extension
    # scrapes ``document.title`` / ``h1`` from a possibly stale SPA DOM
    # (especially when injected into a backgrounded tab), so its guess can
    # belong to the previous video. Fall back to the extension's title only
    # if the probe fails.
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
    transcript_language = _normalise_lang_code(metadata.get("language"))
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
    )


# ---------------------------------------------------------------------------
# Generic media path (non-YouTube): direct mp4/webm, HLS/DASH, iframe embeds
# Vimeo/Dailymotion/Twitch/Bunny/Brightcove/JW/Wistia/Streamable/SoundCloud/…
#
# No subtitle fast path (most non-YouTube sites don't expose machine-readable
# captions). Drop straight onto the Whisper queue, which already handles
# yt-dlp audio download → transcribe → summarize for any URL yt-dlp can
# extract from. The runner is URL-agnostic — ``WhisperTask.url`` becomes the
# argument to ``youtube.download_audio`` regardless of the original kind.
# ---------------------------------------------------------------------------


async def _run_media(
    job_id: str,
    *,
    media_url: str,
    page_title: str | None,  # noqa: ARG001  — title is read from the DB row by the worker
    cookies: list[Any],
) -> None:
    broker = get_broker()
    try:
        await get_queue().put(
            WhisperTask(job_id=job_id, url=media_url, cookies=cookies)
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
    )


# ---------------------------------------------------------------------------
# Shared: persist raw_text (mid-pipeline), summarize, mark done
# ---------------------------------------------------------------------------


def _normalise_lang_code(raw: Any) -> str | None:
    """Lowercase + strip a language code from yt-dlp / whisper output.

    yt-dlp's ``info.get("language")`` sometimes returns ``"en-US"`` or a
    full name; we keep the value short here (just lowercase + first two
    chars when it looks like ``xx-YY``) and leave full canonicalisation
    to the Phase 3 language helper. ``None`` and empty strings come back
    as ``None`` so the column stays null for "we don't know".
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower()
    if not s:
        return None
    # ``en-US`` / ``ru-ru`` style — keep the first segment.
    if "-" in s and len(s.split("-", 1)[0]) == 2:
        s = s.split("-", 1)[0]
    return s


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


# Sources whose text comes from speech (so it may contain ASR artefacts the
# summariser should clean up). Page/PDF sources are excluded.
_AUDIO_TRANSCRIPT_SOURCES = frozenset(
    {
        TranscriptSource.WHISPER,
        TranscriptSource.YOUTUBE_AUTO_CAPTIONS,
        TranscriptSource.YOUTUBE_API,
    }
)


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
) -> None:
    """Run streaming summarization and mark the job done.

    Publishes stage("summarizing"), then a stream of delta events, then
    a done event. On exception, marks the job failed and publishes error.

    Honours the global pause flag before kicking off the LLM call so a
    paused user doesn't pay a fresh ML burst on a fresh job. In-flight
    streaming completes normally — pause is checkpoint-based, not preemptive.
    """
    broker = get_broker()

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
    from_audio = transcript_source in _AUDIO_TRANSCRIPT_SOURCES
    summary_input = timecodes.strip_transcript_tail_noise(text) if from_audio else text

    try:
        async for delta in llm_summary.stream_summarize(
            summary_input,
            title=title,
            output_language=cfg.output.language_name,
            from_audio_transcript=from_audio,
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
