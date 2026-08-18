"""Background coroutine that consumes the deferred-Whisper queue.

Started from ``main.lifespan`` as ``asyncio.create_task(whisper_worker(queue, repo))``.
Single worker, sequential processing. Each item:
    1. yt-dlp audio download → broker stage("downloading")
    2. mlx /v1/audio/transcriptions (verbose_json, one HTTP call)
       → broker stage("transcribing") once; the call returns when the
       whole file is done. See workers/transcribe.py for why we don't
       stream chunks any more (we need real segments + language for the
       transcript-tab UI and the streaming endpoint doesn't expose them).
    3. assemble raw_text from segments → broker stage("ready")
    4. summarize (streaming) → broker delta(...) per token
    5. mark_done with the Whisper-detected language → broker done(content)
finally: delete the audio file.

All progress flows through ``workers.broker``, so any /ai/stream subscriber
sees the same event stream regardless of whether the job came in via the
fast path or the deferred whisper queue.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from src.api.schemas import JobStatus, TranscriptSource
from src.config import get_config
from src.llm import languages
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

# Mirrors extension/src/content/extract.js's MIN_MEDIA_DURATION_SECONDS.
# Probed duration below this many seconds means the "media" the extension
# found almost certainly isn't speech — a UI notification ding, not a
# podcast/lecture — so we skip Whisper transcription entirely and fall back
# to page text. Gated to kind=media only; the youtube fast/deferred paths
# are untouched. Duration is established, in order, by: (1) yt-dlp metadata
# (free, no download — but plain static-asset URLs never report `duration`
# via yt-dlp's generic extractor, so this routinely comes back unknown for
# exactly the case this constant exists for); (2) optionally, ffprobe read
# directly off the URL (no download, best-effort); (3) ffprobe on the
# actual downloaded file — the required, always-correct fallback, since by
# this point the file is real and local. See ``_process_one`` below.
MEDIA_MIN_DURATION_SECONDS = 12.0


def _known_duration(value: Any) -> float | None:
    """Coerce a probe's ``duration`` field to ``float``, or ``None`` if it's
    missing / not a real number (e.g. ``bool`` — ``isinstance(True, int)``
    is true in Python, so it's excluded explicitly)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


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


async def _fallback_to_page_text(
    job_id: str,
    repo_module: object,
    page_text: str | None,
    title: str | None,
    *,
    no_text_error: str,
) -> None:
    """Summarize extension-supplied page text instead of audio.

    Triggered from ``_process_one`` for ``kind=media`` jobs in three cases:
    a known duration below ``MEDIA_MIN_DURATION_SECONDS`` from the
    pre-download probes (skips download+transcribe entirely), a known
    duration below threshold from the post-download local ffprobe (skips
    only Whisper — the download already happened), or
    ``transcribe.transcript_is_unusable`` finding Whisper's output has no
    usable content (empty/annotation-only/degenerate repeat). Either way we
    have no usable audio content, but the extension may have captured the
    page's text alongside the media candidate — summarize that instead of
    failing outright.

    Persists via ``set_extracted``/``mark_done`` with
    ``transcript_source=TranscriptSource.PAGE_EXTRACT`` — the same value
    used for extension-supplied Readability text on the ``kind=page`` path,
    which is exactly what this is. The combination of ``kind=media`` +
    ``transcript_source=page_extract`` on the row IS the honest signal that
    page text, not audio, was summarized.

    Raises ``RuntimeError`` (caller — ``whisper_worker`` — marks the job
    failed) when there's no page text to fall back to, or when the
    resulting summary is empty. Never calls ``mark_failed`` itself, per
    ``_process_one``'s "raises on any failure" contract.
    """
    broker = get_broker()
    cfg = get_config()
    update_status = repo_module.update_status  # type: ignore[attr-defined]
    mark_done = repo_module.mark_done  # type: ignore[attr-defined]
    set_extracted = repo_module.set_extracted  # type: ignore[attr-defined]

    text = (page_text or "").strip()
    if not text:
        raise RuntimeError(no_text_error)

    log.info(
        "runner: %s falling back to page text instead of audio (transcript_source=%s)",
        job_id, TranscriptSource.PAGE_EXTRACT.value,
    )

    try:
        set_extracted(
            job_id,
            raw_text=text,
            transcript_source=TranscriptSource.PAGE_EXTRACT.value,
            title=title,
        )
    except Exception:
        log.exception("set_extracted (page_text fallback) failed for %s; continuing", job_id)

    broker.publish(job_id, stage_event("ready"))

    await _checkpoint_pause(job_id, repo_module, "summarizing")
    update_status(job_id, status=JobStatus.RUNNING.value, progress_stage="summarizing")
    broker.publish(job_id, stage_event("summarizing"))

    parts: list[str] = []
    async for delta in timecodes.cap_markers_in_stream(
        llm_summary.stream_summarize(
            text,
            title=title,
            output_language=cfg.output.language_name,
            from_audio_transcript=False,
        )
    ):
        parts.append(delta)
        broker.publish(job_id, delta_event(delta))
    summary = timecodes.strip_timecode_placeholders("".join(parts).strip())
    # Not audio-derived — any [MM:SS] marker here is a model hallucination,
    # same rationale as pipeline._summarize_and_finish's non-audio branch.
    summary = timecodes.strip_all_timecodes(summary)

    if not summary:
        broker.publish(job_id, error_event("LLM returned empty summary"))
        raise RuntimeError("LLM returned empty summary")

    mark_done(
        job_id,
        raw_text=text,
        summary_md=summary,
        transcript_source=TranscriptSource.PAGE_EXTRACT.value,
        title=title,
    )
    broker.publish(job_id, done_event(summary))


async def _process_one(
    task_url: str,
    task_cookies: list[Any],
    task_job_id: str,
    repo_module: object,
    task_page_text: str | None = None,
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
    kind = getattr(job, "kind", None) if job is not None else None
    is_media = kind == "media"

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
    # Populated by the early duration probe below (kind=media only) so the
    # post-transcription title/language backfill can reuse it instead of
    # probing yt-dlp metadata twice.
    metadata: dict[str, Any] = {}
    metadata_fetched = False

    try:
        if cached_audio is not None and cached_audio.exists():
            log.info("runner: reusing cached audio for %s: %s", task_job_id, cached_audio)
            audio_path = cached_audio
            audio_duration = cached_duration
            download_succeeded = True
            broker.publish(task_job_id, stage_event("downloading", detail="cached"))
        else:
            # Duration probe before download — kind=media only. A cheap
            # yt-dlp metadata-only call (no download) tells us the clip's
            # real length; if it's a known finite number below threshold,
            # this "media" is almost certainly a UI sound effect, not
            # speech — skip the (slow) download + transcribe and fall back
            # to page text instead. A failed/unknown probe (metadata == {}
            # or no duration) falls through to the existing, unmodified
            # download path — never blocks anything.
            #
            # This alone is NOT sufficient: yt-dlp's generic extractor does
            # not report `duration` via extract_info(download=False) for a
            # plain static-asset URL (e.g. /assets/notification.mp3 on a
            # real web app) — confirmed live against both a 3s and a 40s
            # file, both came back `duration: None`. So for exactly the
            # case this exists to catch, this probe alone never fires and
            # the job would otherwise proceed straight to download +
            # Whisper every time. Two more tiers below close that gap.
            known_duration: float | None = None
            if is_media:
                metadata = await youtube.fetch_video_metadata(
                    url=task_url, cookies=task_cookies, scratch_dir=_audio_dir(),
                )
                metadata_fetched = True
                known_duration = _known_duration(metadata.get("duration"))

                if known_duration is None:
                    # Bonus tier: try reading duration directly off the URL
                    # via ffprobe, no download. ffmpeg's http(s) protocol
                    # supports byte-range requests, so for a normal
                    # seekable static file this is typically 1-2 small
                    # requests, not a full download. Best-effort — see
                    # transcribe._probe_duration_url's docstring for the
                    # hard-timeout + swallow-everything guarantees that
                    # make this safe to attempt unconditionally. Does NOT
                    # forward cookies (an authenticated URL will simply
                    # fail this probe and fall through to the normal
                    # cookie-aware download below — a disclosed
                    # limitation, not a regression).
                    known_duration = await transcribe.probe_url_duration(task_url)

                if (
                    known_duration is not None
                    and known_duration < MEDIA_MIN_DURATION_SECONDS
                ):
                    log.info(
                        "runner: %s probed duration %.1fs < %.1fs threshold — "
                        "skipping download, falling back to page text",
                        task_job_id, known_duration, MEDIA_MIN_DURATION_SECONDS,
                    )
                    await _fallback_to_page_text(
                        task_job_id,
                        repo_module,
                        task_page_text,
                        title,
                        no_text_error=(
                            "media clip too short to contain speech and no "
                            "page text to fall back to"
                        ),
                    )
                    return

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

        # Post-download, pre-Whisper local ffprobe gate — kind=media only,
        # fires only when duration is STILL unknown after both probes
        # above (or this job reused cached audio from a prior run whose
        # duration was never recorded). This is the tier that actually
        # catches the case that motivated all of this: a plain static-asset
        # URL where yt-dlp never reports a duration. The file is real and
        # local now, so ffprobe on it is authoritative — if it reveals a
        # duration below threshold, we skip Whisper entirely (never call
        # transcribe_audio) and fall back to page text, same as the
        # pre-download reject above.
        if is_media and audio_duration is None and audio_path is not None:
            probed_local_duration = await transcribe.probe_duration(audio_path)
            if probed_local_duration is not None:
                audio_duration = probed_local_duration
                try:
                    set_audio(
                        task_job_id,
                        audio_path=str(audio_path),
                        audio_duration_seconds=audio_duration,
                    )
                except Exception:
                    log.exception("set_audio failed for %s; continuing", task_job_id)

            if (
                probed_local_duration is not None
                and probed_local_duration < MEDIA_MIN_DURATION_SECONDS
            ):
                log.info(
                    "runner: %s post-download ffprobe duration %.1fs < %.1fs "
                    "threshold — skipping whisper, falling back to page text",
                    task_job_id, probed_local_duration, MEDIA_MIN_DURATION_SECONDS,
                )
                # Whisper never runs on this path, but the outcome ahead
                # (_fallback_to_page_text -> mark_done) is a genuine
                # terminal success, not a failure a retry could improve on
                # — so the downloaded file must be deleted, not kept. The
                # `finally` block's cleanup policy keys off
                # `transcribe_done` to distinguish "delete" from "keep for
                # retry"; setting it True here repurposes its meaning from
                # literally "Whisper ran" to "no further use will be made
                # of this audio file, delete it" — both this branch and
                # the empty-transcript fallback below reach mark_done
                # through _fallback_to_page_text, so both should get the
                # same cleanup outcome even though only one of them
                # actually called Whisper.
                transcribe_done = True
                await _fallback_to_page_text(
                    task_job_id,
                    repo_module,
                    task_page_text,
                    title,
                    no_text_error=(
                        "media clip too short to contain speech and no "
                        "page text to fall back to"
                    ),
                )
                return

        # Pause checkpoint before mlx /v1/audio/transcriptions (long ML call).
        await _checkpoint_pause(task_job_id, repo_module, "transcribing")
        update_status(
            task_job_id,
            status=JobStatus.RUNNING.value,
            progress_stage="transcribing",
        )
        # No mid-transcription percent any more — verbose_json is one HTTP
        # roundtrip per file (server side mlx_whisper handles batching
        # internally) so we publish the stage once and let the Library row
        # sit on it until the call returns. See workers/transcribe.py.
        broker.publish(task_job_id, stage_event("transcribing"))

        whisper_result = await transcribe.transcribe_audio(
            audio_path,
            total_duration=audio_duration,
        )
        transcribe_done = True

        # transcribe_audio already retried a short chunk/file before giving
        # up — a nonzero missing_seconds here means the transcript is known
        # to stop short of the material's real length even though the job
        # is about to complete normally (status=done, no error). Log it
        # loudly and persist it so the job doesn't look silently complete.
        if whisper_result.missing_seconds:
            log.warning(
                "runner: %s transcript may be missing ~%.0fs of trailing "
                "audio — coverage retries were exhausted",
                task_job_id, whisper_result.missing_seconds,
            )
        transcript_missing_seconds = whisper_result.missing_seconds or None

        raw_text = timecodes.build_marked_text(
            whisper_result.segments,
            window_seconds=cfg.youtube.segment_window_seconds,
        )

        # Unusable-transcript fallback — kind=media only. Whisper ran (and
        # the download stays subject to the normal success/cleanup policy
        # below) but produced nothing usable; fall back to page text rather
        # than summarizing garbage into a fabricated-looking result. Checked
        # against the raw segment list, NOT `raw_text` — `raw_text` is
        # already `[MM:SS]`-marked by build_marked_text above, and
        # transcript_is_unusable's annotation check would misfire on those
        # brackets for a real transcript. This replaces a naive
        # `not raw_text.strip()` check, which only caught the literal-empty
        # case — Whisper hallucinating a few plausible-looking characters
        # over silence/noise (e.g. "[chime]") is NOT empty but is exactly as
        # unusable, and produced a real fabricated summary in production.
        if is_media and transcribe.transcript_is_unusable(whisper_result.segments):
            log.info(
                "runner: %s whisper transcript is unusable (empty/annotation-"
                "only/degenerate repeat) — falling back to page text",
                task_job_id,
            )
            await _fallback_to_page_text(
                task_job_id,
                repo_module,
                task_page_text,
                title,
                no_text_error=(
                    "whisper transcript is unusable and no page text to fall back to"
                ),
            )
            return

        # Whisper's auto-detect. ``None`` if the backend doesn't report it
        # (e.g. LocalAI returns ``language: null``); we backfill from yt-dlp
        # metadata below, else the column stays null and the UI shows
        # "Original".
        whisper_language = whisper_result.language

        # The DB row's title is whatever the extension scraped from a possibly
        # stale SPA DOM — often just the video id. Fetch YouTube's own title
        # (and language as a fallback), the same probe the caption fast path
        # uses. Best-effort: a metadata hiccup must not break the summary.
        # Reuse the early duration-probe's metadata when we already have it
        # (kind=media, no cached audio) instead of probing yt-dlp twice.
        if not metadata_fetched:
            metadata = await youtube.fetch_video_metadata(
                url=task_url, cookies=task_cookies, scratch_dir=_audio_dir(),
            )
        meta_title = metadata.get("title")
        if isinstance(meta_title, str) and meta_title.strip():
            title = meta_title.strip()
        if whisper_language is None:
            whisper_language = languages.short_lang_code(metadata.get("language"))
        # Last resort: guess from the transcript text itself (LocalAI Whisper
        # returns no language, and some videos carry no metadata language).
        if whisper_language is None:
            whisper_language = languages.detect_language(raw_text)

        # Serialize the fine-grained Whisper segments so the Transcript tab
        # can render one line per ~1-5 s instead of the 30 s buckets
        # ``raw_text`` uses. Only persist when we got real segments — if the
        # mlx-server patch isn't applied we got a single all-encompassing
        # segment and there's no fine-grained detail to surface.
        raw_segments_json: str | None = None
        if len(whisper_result.segments) > 1:
            raw_segments_json = json.dumps(
                whisper_result.segments, ensure_ascii=False, separators=(",", ":"),
            )

        # Persist raw_text + language mid-pipeline so re-subscribers can see
        # context if the summary fails or the daemon restarts. ``transcript_
        # language`` is set on the same write so the column is filled even
        # if the LLM stream below dies.
        try:
            set_extracted(
                task_job_id,
                raw_text=raw_text,
                transcript_source=TranscriptSource.WHISPER.value,
                transcript_language=whisper_language,
                raw_segments_json=raw_segments_json,
                transcript_missing_seconds=transcript_missing_seconds,
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

        # Summarise a tail-cleaned copy (drops Whisper's "Продолжение
        # следует…" outro hallucinations) — the stored transcript keeps them.
        summary_input = timecodes.strip_transcript_tail_noise(raw_text)
        parts: list[str] = []
        # cap_markers_in_stream holds text back only long enough to resolve a
        # [MM:SS]-shaped bracket (never a whole line), capping markers as
        # each one resolves, so the published delta and the accumulated
        # `parts` (-> stored summary) are always the same capped text — see
        # timecodes.cap_markers_in_stream for the full rationale.
        async for delta in timecodes.cap_markers_in_stream(
            llm_summary.stream_summarize(
                summary_input,
                title=title,
                output_language=cfg.output.language_name,
                from_audio_transcript=True,
            )
        ):
            parts.append(delta)
            broker.publish(task_job_id, delta_event(delta))
        summary = timecodes.strip_timecode_placeholders("".join(parts).strip())

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
            transcript_language=whisper_language,
            raw_segments_json=raw_segments_json,
            transcript_missing_seconds=transcript_missing_seconds,
        )
        broker.publish(task_job_id, done_event(summary))
    finally:
        # Cleanup policy:
        # - On full success (mark_done reached) → unlink audio + clear DB ref.
        # - On failure AFTER successful download → KEEP the audio + DB ref so a
        #   retry can skip the (slow) yt-dlp step.
        # - On failure DURING download (no audio yet) → nothing to clean.
        #
        # NOTE: `transcribe_done` is no longer literally "Whisper ran" — the
        # post-download, pre-Whisper duration reject above (kind=media, a
        # too-short clip discovered only after downloading it) also sets it
        # True before falling back to page text and returning, even though
        # transcribe_audio was never called. That path still ends in
        # mark_done (a genuine terminal success — there's no retry that
        # would help), so it needs the SAME "delete, don't keep" outcome as
        # the real Whisper-ran case, and reusing this flag (rather than
        # adding a parallel cleanup branch) is how it gets it.
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
                task_page_text=task.page_text,
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
