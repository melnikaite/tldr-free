"""Background coroutine that consumes the deferred-Whisper queue.

Started from ``main.lifespan`` as ``asyncio.create_task(whisper_worker(queue, repo))``.
Single worker, sequential processing. Each item:
    1. yt-dlp audio download → broker stage("transcribing", "downloading")
    2. mlx /v1/audio/transcriptions (streaming, chunk-by-chunk)
       → broker stage("transcribing", "<percent>%") per ~30 s of audio
    3. assemble raw_text → broker stage("ready")
    4. summarize (streaming) → broker delta(...) per token
    5. mark_done → broker done(content)
finally: delete the audio file.

All progress flows through ``workers.broker``, so any /ai/stream subscriber
sees the same event stream regardless of whether the job came in via the
fast path or the deferred whisper queue.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from src.api.schemas import JobStatus, TranscriptSource
from src.config import get_config
from src.llm import summary as llm_summary
from src.workers import timecodes, transcribe, youtube
from src.workers.broker import (
    delta_event,
    done_event,
    error_event,
    get_broker,
    stage_event,
)
from src.workers.control import get_control
from src.workers.queue import WhisperQueue

log = logging.getLogger(__name__)


def _audio_dir() -> Path:
    """Subdirectory of ``config.storage.data_dir`` for tmp audio files."""
    p = Path(get_config().storage.data_dir) / "audio"
    p.mkdir(parents=True, exist_ok=True)
    return p


async def _checkpoint_pause(
    job_id: str, repo_module: object, on_resume_stage: str,
) -> None:
    """Same contract as ``pipeline._checkpoint_pause`` but threads
    ``repo_module`` (DI for tests). Park here while paused; restore the
    given stage on resume so the Library row goes back to e.g.
    ``transcribing`` instead of staying stuck at ``paused``."""
    control = get_control()
    if not control.paused:
        return
    update_status = repo_module.update_status  # type: ignore[attr-defined]
    broker = get_broker()
    update_status(job_id, status=JobStatus.RUNNING.value, progress_stage="paused")
    broker.publish(job_id, stage_event("paused"))
    await control.wait_if_paused()
    update_status(job_id, status=JobStatus.RUNNING.value, progress_stage=on_resume_stage)
    broker.publish(job_id, stage_event(on_resume_stage))


async def _process_one(
    task_url: str,
    task_cookies: list[Any],
    task_job_id: str,
    repo_module: object,
) -> None:
    """Process a single task. Raises on any failure — caller handles ``mark_failed``.

    State-changing ``repo_module`` calls (update_status / mark_done /
    set_extracted) publish ``job_event("updated", …)`` themselves, so the
    Library reacts in real time without us emitting anything extra here.
    """
    cfg = get_config()
    broker = get_broker()

    get_job = repo_module.get_job  # type: ignore[attr-defined]
    update_status = repo_module.update_status  # type: ignore[attr-defined]
    mark_done = repo_module.mark_done  # type: ignore[attr-defined]
    set_extracted = repo_module.set_extracted  # type: ignore[attr-defined]
    set_audio = repo_module.set_audio  # type: ignore[attr-defined]

    job = get_job(task_job_id)
    title = getattr(job, "title", None) if job is not None else None

    # Reuse a previously-downloaded audio file when retrying a job that failed
    # mid-pipeline (after download, before mark_done). Avoids hitting yt-dlp
    # again, which is by far the slowest non-Whisper step.
    cached_audio_str = getattr(job, "audio_path", None) if job is not None else None
    cached_audio = Path(cached_audio_str) if cached_audio_str else None
    cached_duration = getattr(job, "audio_duration_seconds", None) if job is not None else None

    audio_path: Path | None = None
    audio_duration: float | None = None
    download_succeeded = False
    transcribe_done = False

    try:
        if cached_audio is not None and cached_audio.exists():
            log.info("runner: reusing cached audio for %s: %s", task_job_id, cached_audio)
            audio_path = cached_audio
            audio_duration = cached_duration
            download_succeeded = True
            broker.publish(task_job_id, stage_event("downloading", detail="cached"))
        else:
            # Pause checkpoint before yt-dlp.
            await _checkpoint_pause(task_job_id, repo_module, "downloading")
            update_status(
                task_job_id,
                status=JobStatus.RUNNING.value,
                progress_stage="downloading",
            )
            broker.publish(task_job_id, stage_event("downloading"))
            audio_path, audio_duration = await youtube.download_audio(
                url=task_url,
                cookies=task_cookies,
                dir=_audio_dir(),
            )
            download_succeeded = True
            try:
                set_audio(
                    task_job_id,
                    audio_path=str(audio_path),
                    audio_duration_seconds=audio_duration,
                )
            except Exception:
                log.exception("set_audio failed for %s; continuing", task_job_id)

        # Pause checkpoint before mlx /v1/audio/transcriptions (long ML call).
        await _checkpoint_pause(task_job_id, repo_module, "transcribing")
        update_status(
            task_job_id,
            status=JobStatus.RUNNING.value,
            progress_stage="transcribing",
        )
        broker.publish(task_job_id, stage_event("transcribing", detail="0%"))

        def _on_transcribe_progress(fraction: float) -> None:
            pct = int(round(min(1.0, max(0.0, fraction)) * 100))
            broker.publish(
                task_job_id,
                stage_event("transcribing", detail=f"{pct}%"),
            )

        segments = await transcribe.transcribe_stream(
            audio_path,
            total_duration=audio_duration,
            on_progress=_on_transcribe_progress,
        )
        transcribe_done = True

        raw_text = timecodes.build_marked_text(
            segments,
            window_seconds=cfg.youtube.segment_window_seconds,
        )

        # Persist raw_text mid-pipeline so re-subscribers can see context if
        # the summary fails or the daemon restarts.
        try:
            set_extracted(
                task_job_id,
                raw_text=raw_text,
                transcript_source=TranscriptSource.WHISPER.value,
            )
        except Exception:
            log.exception("set_extracted failed for %s; continuing", task_job_id)

        broker.publish(task_job_id, stage_event("ready"))

        # Pause checkpoint before the LLM stream.
        await _checkpoint_pause(task_job_id, repo_module, "summarizing")
        update_status(
            task_job_id,
            status=JobStatus.RUNNING.value,
            progress_stage="summarizing",
        )
        broker.publish(task_job_id, stage_event("summarizing"))

        parts: list[str] = []
        async for delta in llm_summary.stream_summarize(
            raw_text,
            title=title,
            output_language=cfg.output.language_name,
        ):
            parts.append(delta)
            broker.publish(task_job_id, delta_event(delta))
        summary = "".join(parts).strip()

        if not summary:
            broker.publish(task_job_id, error_event("LLM returned empty summary"))
            raise RuntimeError("LLM returned empty summary")

        # video_id best-effort.
        video_id: str | None = None
        try:
            video_id = youtube.extract_video_id(task_url)
        except ValueError:
            video_id = None

        mark_done(
            task_job_id,
            raw_text=raw_text,
            summary_md=summary,
            transcript_source=TranscriptSource.WHISPER.value,
            title=title,
            video_id=video_id,
        )
        broker.publish(task_job_id, done_event(summary))
    finally:
        # Cleanup policy:
        # - On full success (mark_done reached) → unlink audio + clear DB ref.
        # - On failure AFTER successful download → KEEP the audio + DB ref so a
        #   retry can skip the (slow) yt-dlp step.
        # - On failure DURING download (no audio yet) → nothing to clean.
        full_success = transcribe_done and audio_path is not None
        if full_success and audio_path is not None:
            try:
                if audio_path.exists():
                    audio_path.unlink()
            except OSError:
                log.warning("runner: failed to unlink audio file %s", audio_path)
            try:
                set_audio(task_job_id, audio_path=None, audio_duration_seconds=None)
            except Exception:
                log.exception("set_audio(None) failed for %s", task_job_id)
        elif download_succeeded and audio_path is not None:
            log.info(
                "runner: keeping audio for %s at %s (failure after download — retry can reuse it)",
                task_job_id,
                audio_path,
            )


async def whisper_worker(queue: WhisperQueue, repo_module: object) -> None:
    """Consume queue items forever. Cancellation propagates from the lifespan."""
    log.info("whisper worker started")
    control = get_control()
    while True:
        # Honour a global pause before pulling the next item. Already-running
        # work isn't interrupted; pause only gates the *next* task pickup.
        await control.wait_if_paused()
        task = await queue.get()
        queue.mark_running(True)
        try:
            await _process_one(
                task_url=task.url,
                task_cookies=list(task.cookies),
                task_job_id=task.job_id,
                repo_module=repo_module,
            )
        except asyncio.CancelledError:
            log.info("whisper worker cancelled")
            raise
        except Exception as exc:
            log.exception("whisper job %s failed", task.job_id)
            mark_failed = getattr(repo_module, "mark_failed", None)
            if mark_failed is not None:
                try:
                    mark_failed(task.job_id, error=str(exc))
                except Exception:  # pragma: no cover — defensive
                    log.exception("failed to mark job %s as failed", task.job_id)
            get_broker().publish(task.job_id, error_event(str(exc)))
        finally:
            queue.mark_running(False)
            queue.task_done()
            # Optional cooldown between consecutive jobs to let the box cool.
            cooldown = max(0, get_config().workers.cooldown_seconds)
            if cooldown:
                log.info("whisper worker: cooldown for %ds before next job", cooldown)
                try:
                    await asyncio.sleep(cooldown)
                except asyncio.CancelledError:
                    log.info("whisper worker cancelled during cooldown")
                    raise


__all__ = ["whisper_worker"]
