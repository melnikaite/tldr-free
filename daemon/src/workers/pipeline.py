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
from src.llm import summary as llm_summary
from src.storage import repo
from src.workers import page, timecodes, youtube
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

    _persist_extracted(
        job_id,
        raw_text=raw_text,
        title=title,
        transcript_source=transcript_source,
        video_id=video_id,
    )
    broker.publish(job_id, stage_event("ready"))

    await _summarize_and_finish(
        job_id,
        text=raw_text,
        title=title,
        transcript_source=transcript_source,
        video_id=video_id,
        cfg=cfg,
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
    )


async def _summarize_and_finish(
    job_id: str,
    *,
    text: str,
    title: str | None,
    transcript_source: TranscriptSource,
    video_id: str | None,
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

    try:
        async for delta in llm_summary.stream_summarize(
            text,
            title=title,
            output_language=cfg.output.language_name,
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

    summary_md = "".join(parts).strip()
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


def infer_kind(url: str, declared: str) -> JobKind:
    """Map ``kind="auto"`` to the concrete enum based on the URL host."""
    if declared in (JobKind.PAGE.value, JobKind.YOUTUBE.value):
        return JobKind(declared)
    host = (urlparse(url).hostname or "").lower()
    if host in _YOUTUBE_HOSTS:
        return JobKind.YOUTUBE
    return JobKind.PAGE


__all__ = ["infer_kind", "run_pipeline"]
