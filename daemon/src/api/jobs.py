"""POST/GET/DELETE /jobs + chat-message endpoints.

POST /jobs is now ASYNC. The route persists the row, kicks off the pipeline
(workers/pipeline.run_pipeline) as a background task, and returns 202 with
the new job id. The client follows progress via POST /ai/stream {job_id}.

Chat history per job lives in `Message`; GET /jobs/{id}/messages returns
the saved bubbles. There's no DELETE today — clearing chat means deleting
the job (FK cascade drops the messages with it).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from src.api.schemas import (
    DeixisMoment,
    FrameFetchRequest,
    FrameFetchResponse,
    FrameRef,
    JobCreateRequest,
    JobCreateResponse,
    JobDetails,
    JobExportRequest,
    JobImportResponse,
    JobKind,
    JobListResponse,
    JobStatus,
    JobSummary,
    MessagesListResponse,
    MomentsListResponse,
    TranscriptSource,
)
from src.api.schemas import (
    Message as MessageModel,
)
from src.storage import bundle, repo
from src.workers import deixis, frames, pipeline, timecodes, youtube
from src.workers.broker import get_stream_buffer
from src.workers.deixis import DeixisCategory
from src.workers.errors import FrameExtractionError

log = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


# ---------------------------------------------------------------------------
# Track background pipeline tasks so they aren't garbage-collected mid-run.
# asyncio.create_task only holds a weak reference to the coroutine; if no
# strong reference exists the task can be cancelled by the GC.
# ---------------------------------------------------------------------------


_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


def _spawn(coro: Any) -> asyncio.Task[Any]:
    task = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return task


# Serialise POST /jobs so the dedup-then-create check is atomic. Without
# this two near-simultaneous clicks on the same URL both see "no existing
# job", both call create_job, and we end up with two rows for one URL.
# The handler is fast (one SELECT + maybe one INSERT) so the lock never
# becomes a real contention point.
_create_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_status_filter(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or None


def _parse_since(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"`since` must be ISO-8601, got {raw!r}: {exc}",
        ) from exc


def _to_summary(job: Any) -> JobSummary:
    return JobSummary(
        id=job.id,
        url=job.url,
        kind=JobKind(job.kind),
        status=JobStatus(job.status),
        title=job.title,
        duration_seconds=job.duration_seconds,
        progress_stage=job.progress_stage,
        transcript_source=(
            TranscriptSource(job.transcript_source) if job.transcript_source else None
        ),
        created_at=job.created_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
    )


def _build_segments_text(raw_segments_json: str | None) -> str | None:
    """Render fine-grained Whisper segments as ``[MM:SS] text`` lines.

    Returns ``None`` when no usable segments are present (legacy jobs,
    pre-patch fallbacks). The Transcript UI parses each line back into a
    cue via the same ``[MM:SS]`` regex it uses for the summary view, so
    the format is compatible — only the granularity differs.
    """
    if not raw_segments_json:
        return None
    try:
        segments = json.loads(raw_segments_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(segments, list) or not segments:
        return None
    # Delegate to the single timecode formatter so the marker format stays
    # identical to the summary / transcript views (one source of truth).
    dict_segments = [s for s in segments if isinstance(s, dict)]
    return timecodes.format_segments_as_marked_text(dict_segments) or None


def _to_details(job: Any) -> JobDetails:
    # Include in-flight buffer for running jobs so reconnecting clients can
    # replay buffered content without waiting for future delta events.
    partial = get_stream_buffer(job.id) if JobStatus(job.status) == JobStatus.RUNNING else None
    # Pull cached translations for the transcript-tab language switcher.
    # Empty list when nobody's translated this job yet; Phase 3 populates
    # via POST /jobs/{id}/transcript/translate.
    translations = repo.list_translations(job.id)
    # Parse the alt-media-candidates JSON blob. Malformed payloads (should
    # never happen — we wrote it ourselves) get logged and dropped rather
    # than failing the whole job detail response; the picker just shows
    # nothing in that case.
    alt_candidates: list[Any] = []
    raw_alts = getattr(job, "alt_media_candidates_json", None)
    if raw_alts:
        try:
            parsed = json.loads(raw_alts)
            if isinstance(parsed, list):
                alt_candidates = parsed
        except (ValueError, TypeError):
            log.warning("api: malformed alt_media_candidates_json for job %s", job.id)
    return JobDetails(
        id=job.id,
        url=job.url,
        kind=JobKind(job.kind),
        status=JobStatus(job.status),
        title=job.title,
        duration_seconds=job.duration_seconds,
        progress_stage=job.progress_stage,
        transcript_source=(
            TranscriptSource(job.transcript_source) if job.transcript_source else None
        ),
        created_at=job.created_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
        summary_md=job.summary_md,
        error=job.error,
        video_id=job.video_id,
        partial_summary=partial or None,
        transcript_language=getattr(job, "transcript_language", None),
        transcript_translations=translations,
        alt_media_candidates=alt_candidates,
    )


def _parse_frame_refs(raw: str | None, *, message_id: Any) -> list[Any]:
    """Parse ``Message.frame_refs_json`` back into a list of dicts.

    Malformed payloads (should never happen — we wrote it ourselves in
    ``repo.add_message``) are logged and dropped rather than failing the
    whole message list, same defensive style as
    ``_to_details``'s ``alt_media_candidates_json`` parsing above.
    """
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        log.warning("api: malformed frame_refs_json for message %s", message_id)
        return []
    if not isinstance(parsed, list):
        log.warning("api: frame_refs_json for message %s is not a list", message_id)
        return []
    return parsed


def _to_message(row: Any) -> MessageModel:
    return MessageModel(
        id=row.id,
        job_id=row.job_id,
        role=row.role,
        content=row.content,
        created_at=row.created_at,
        frame_refs=_parse_frame_refs(
            getattr(row, "frame_refs_json", None), message_id=row.id
        ),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", status_code=202, response_model=JobCreateResponse)
async def create_job(req: JobCreateRequest) -> JSONResponse:
    """Persist the row, kick off the background pipeline, return 202.

    Dedup: if there's already a job for this URL in ``queued`` / ``running``
    / ``done``, return that one instead of starting a new pipeline. Only
    ``failed`` jobs are bypassed — clicking the toolbar after a failure is
    treated as an explicit retry.

    The pipeline (workers.pipeline.run_pipeline) runs extraction +
    summarization and broadcasts events via the broker. The client
    subscribes via POST /ai/stream {job_id} to watch / replay.

    The dedup-check + create is wrapped in ``_create_lock`` so two parallel
    POSTs for the same URL can't both miss the existing-row check and
    create duplicates.
    """
    async with _create_lock:
        existing_rows, _ = repo.list_jobs(url=req.url, limit=1)
        if existing_rows:
            existing = existing_rows[0]
            if existing.status in (
                JobStatus.QUEUED.value,
                JobStatus.RUNNING.value,
                JobStatus.DONE.value,
            ):
                body = JobCreateResponse(
                    id=existing.id,
                    kind=JobKind(existing.kind),
                    status=JobStatus(existing.status),
                )
                return JSONResponse(status_code=202, content=body.model_dump(mode="json"))

        # Decode the optional PDF payload up front so we can fail fast on
        # bad base64 rather than crashing the pipeline coroutine later.
        pdf_bytes: bytes | None = None
        if req.pdf_bytes_b64:
            try:
                pdf_bytes = base64.b64decode(req.pdf_bytes_b64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    f"pdf_bytes_b64 is not valid base64: {exc}",
                ) from exc

        kind = pipeline.infer_kind(
            req.url, req.kind, req.media_url, pdf_bytes_present=pdf_bytes is not None,
        )

        # For YouTube the extension's scraped title is unreliable (SPA, fast
        # tab switching), so seed with the video id. The pipeline overwrites
        # it with yt-dlp's canonical title within a couple of seconds.
        # For MEDIA we trust the extension's scrape — yt-dlp on arbitrary
        # media URLs doesn't reliably surface a human title (the URL might
        # be a CDN .mp4), and the extension's og:title / h1 / document.title
        # is the best signal we have.
        initial_title = req.page_title
        if kind == JobKind.YOUTUBE:
            with contextlib.suppress(ValueError):
                initial_title = youtube.extract_video_id(req.url)

        # Pre-serialise the alt-media-candidates list so the storage layer
        # stays Pydantic-free. Empty list and None both stored as NULL —
        # the picker won't render with either, no need to distinguish.
        alt_candidates_json: str | None = None
        if req.alt_media_candidates:
            alt_candidates_json = json.dumps(
                [c.model_dump() for c in req.alt_media_candidates],
                ensure_ascii=False,
                separators=(",", ":"),
            )

        # Single create with progress_stage='extracting' avoids a needless
        # second update_status (and a second job event) right after creation.
        # repo.create_job emits job_event("created") for us.
        job = repo.create_job(
            url=req.url,
            kind=kind.value,
            title=initial_title,
            progress_stage="extracting",
            alt_media_candidates_json=alt_candidates_json,
        )

    # Spawn outside the lock — pipeline runs for minutes, holding the lock
    # would block every other POST /jobs.
    _spawn(
        pipeline.run_pipeline(
            job.id,
            kind=kind,
            url=req.url,
            page_text=req.page_text,
            page_title=req.page_title,
            media_url=req.media_url,
            pdf_bytes=pdf_bytes,
            cookies=list(req.cookies or []),
        )
    )

    body = JobCreateResponse(id=job.id, kind=kind, status=JobStatus.RUNNING)
    return JSONResponse(status_code=202, content=body.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Export / import — moving a library between machines. Declared BEFORE the
# `/{job_id}` routes below: FastAPI/Starlette matches routes in registration
# order, and `/{job_id}` would otherwise swallow `/export`/`/import` as if
# `job_id == "export"`.
# ---------------------------------------------------------------------------


@router.post("/export")
def export_jobs(req: JobExportRequest) -> FileResponse:
    """Pack the ``status == "done"`` subset of ``req.ids`` into a zip and
    return it as a download. Declared as plain ``def`` (not ``async def``)
    so FastAPI runs this blocking, disk-bound work in a threadpool instead
    of on the event loop — see ``storage.bundle.export_jobs``.

    The zip is written to a tempfile rather than built in memory (bundles
    can carry megabytes of transcript text plus frame JPEGs) and unlinked
    via ``BackgroundTask`` once the response finishes streaming it out.

    No custom response headers beyond ``Content-Disposition`` (set by
    ``FileResponse``'s ``filename=``) — the extension can't read custom
    headers through CORS, so nothing else is added.

    400 if nothing in ``req.ids`` turned out exportable (unknown ids and
    non-``done`` jobs are silently skipped, not individually erred on —
    see ``bundle.export_jobs``).
    """
    try:
        zip_path, count = bundle.export_jobs(req.ids)
    except bundle.BundleError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"tldr-export-{count}-jobs.zip",
        background=BackgroundTask(zip_path.unlink, missing_ok=True),
    )


def _reject_oversized_content_length(request: Request) -> None:
    """Reject up front when the client's own ``Content-Length`` header
    already declares more than the upload cap — no point reading a single
    byte in that case. Not the primary guard (a client can omit or lie
    about this header); ``_stream_body_to_file`` enforces the real cap
    against bytes actually received."""
    declared = request.headers.get("content-length")
    if declared is None:
        return
    try:
        declared_bytes = int(declared)
    except ValueError:
        return
    if declared_bytes > bundle.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"upload declares {declared_bytes} bytes, over the "
            f"{bundle.MAX_UPLOAD_BYTES}-byte limit",
        )


async def _stream_body_to_file(request: Request, fd: int) -> None:
    """Stream the request body into the already-open file descriptor
    ``fd``, enforcing the upload size cap against bytes ACTUALLY received.

    ``await request.body()`` would buffer the whole upload into memory
    before any size check ever ran — a 512 MiB (or larger, lying-header)
    upload would already have cost the memory by the time the cap fired.
    Streaming via ``request.stream()`` into a tempfile means the cap is
    enforced as bytes arrive, aborting mid-upload instead of after.
    """
    written = 0
    with os.fdopen(fd, "wb") as out:
        async for chunk in request.stream():
            written += len(chunk)
            if written > bundle.MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    f"upload exceeded the {bundle.MAX_UPLOAD_BYTES}-byte limit "
                    "while streaming",
                )
            out.write(chunk)


@router.post("/import", response_model=JobImportResponse)
async def import_jobs(request: Request) -> JobImportResponse:
    """Import a zip bundle produced by ``POST /jobs/export`` (or a
    compatible one). Body is the raw zip bytes — deliberately NOT
    ``UploadFile``/multipart (``python-multipart`` is not a dependency of
    this daemon), so the bytes are streamed straight off the Starlette
    ``Request`` into a tempfile (see ``_stream_body_to_file``) rather than
    buffered into memory via ``request.body()``.

    The actual unzip/validate/insert work runs off the event loop via
    ``asyncio.to_thread`` — a large upload's worth of zip validation and
    frame-file copying would otherwise block every other request for the
    duration.

    400 (nothing written) for: an upload over the size cap (rejected by
    ``Content-Length`` up front when present, or mid-stream otherwise), a
    non-zip payload, a missing/malformed manifest, an unsupported
    format/version, or any member name that doesn't match the bundle's own
    strict shape (zip-slip guard) or blows the size limits (zip-bomb
    guard). Per-job problems (duplicate URL, malformed job entry, invalid
    kind) are NOT request-level failures — they land in the response's
    ``skipped``/``failed`` lists instead, see ``storage.bundle.import_bundle``.
    """
    _reject_oversized_content_length(request)

    fd, tmp_name = tempfile.mkstemp(prefix="tldr-import-", suffix=".zip")
    tmp_path = Path(tmp_name)
    try:
        await _stream_body_to_file(request, fd)
        try:
            return await asyncio.to_thread(bundle.import_bundle, tmp_path)
        except bundle.BundleError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


# NB: read endpoints below are plain `def`, not `async def`. FastAPI runs
# `def` handlers in a threadpool, which keeps our synchronous SQLAlchemy
# calls off the event loop. Without this, a sustained burst of SSE delta
# events from a running pipeline can starve a concurrent /jobs poll —
# every sync SQL call inside an async handler holds the loop for its
# whole duration.


@router.get("", response_model=JobListResponse)
def list_jobs(
    status: str | None = Query(default=None, description="Comma-separated statuses"),
    kind: str | None = Query(default=None, description="page | youtube"),
    since: str | None = Query(default=None, description="ISO-8601 datetime"),
    url: str | None = Query(
        default=None,
        description="Exact URL match (used by the extension to look up whether "
        "the current tab has already been summarized)",
    ),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> JobListResponse:
    statuses = _parse_status_filter(status)
    since_dt = _parse_since(since)

    rows, total = repo.list_jobs(
        status=statuses,
        kind=kind,
        since=since_dt,
        url=url,
        limit=limit,
        offset=offset,
    )

    items = [_to_summary(row) for row in rows]
    return JobListResponse(items=items, total=total)


@router.get("/{job_id}", response_model=JobDetails)
def get_job(job_id: str) -> JobDetails:
    job = repo.get_job(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"job {job_id} not found")
    return _to_details(job)


@router.post("/{job_id}/transcript/translate", status_code=202)
async def translate_transcript(job_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Enqueue a transcript translation. Returns the current status of the row.

    Body: ``{"lang": "ru"}`` (ISO-639-1 code, ISO-639-2, or English name;
    see ``llm.languages.normalize_lang``). The call is **dedup**:

    - Existing row in ``queued`` / ``running`` / ``done`` → no new task,
      return the existing status.
    - Existing row in ``failed`` → reset, re-spawn task.
    - No row → insert ``queued``, spawn task.

    Returns ``{language_code, status, progress_percent, is_source}``.
    ``is_source=true`` means the requested language equals the
    transcript's source language — we serve the original directly via
    ``GET /transcript`` without running the LLM.
    """
    from src.llm.languages import UnknownLanguageError
    from src.workers import translator

    lang_input = (body or {}).get("lang")
    if not isinstance(lang_input, str) or not lang_input.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "missing 'lang' field")

    try:
        return await translator.enqueue_translation(job_id, lang_input)
    except KeyError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e
    except UnknownLanguageError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e


@router.post("/{job_id}/transcript/retry-all", status_code=202)
async def retry_all_transcript_translations(job_id: str) -> dict[str, Any]:
    """Re-enqueue every ``failed`` translation row for this job.

    Returns ``{retried: [TranscriptTranslationSummary, ...]}``. Empty
    list when nothing was failed (idempotent). MUST be async because
    spawning the translator workers calls ``asyncio.create_task`` under
    the hood — that requires a running event loop, which a sync FastAPI
    handler (run on a threadpool) doesn't have.
    """
    from src.workers import translator
    try:
        retried = await translator.retry_all_failed(job_id)
    except KeyError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e
    return {"retried": retried}


@router.get("/{job_id}/transcript")
def get_transcript(job_id: str, lang: str | None = None) -> dict[str, Any]:
    """Return the full transcript text for a job in the requested language.

    Three response shapes, all ``200``:

    - **Ready** ``{text: "...", language_code, is_original, is_pending: false}``
      — the text is available, render it.
    - **Pending** ``{text: null, ..., is_pending: true}`` — job exists but
      the transcript isn't ready yet (extraction still running for source,
      or translation in flight for a non-source language). The sidepanel
      shows a placeholder + refetches on the next ``job_event`` for this
      job. We do NOT 404 here because the Transcript tab opens as soon as
      the user clicks it, often before the daemon has finished extraction
      on a freshly-submitted job.
    - **No-such-translation** ``404`` — the caller asked for a language
      that has no row in ``transcript_translation`` at all. UI should
      POST to ``/translate`` to start one.

    The job itself missing is still a ``404`` (different message).

    Lazy-loaded by the sidepanel's Transcript tab — the raw_text payload
    can be megabytes for hour-long podcasts, so we keep it off JobDetails
    and only serve it here when the user opens the tab.
    """
    job = repo.get_job(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"job {job_id} not found")

    source_lang = getattr(job, "transcript_language", None)
    requested = (lang or "").strip().lower() or None

    is_original = requested is None or (
        source_lang is not None and requested == source_lang
    )
    if is_original:
        if not job.raw_text:
            # Job exists but extraction hasn't finished yet (or failed
            # before producing text). Return a "pending" sentinel rather
            # than a 404 — the sidepanel re-asks on the next job event.
            return {
                "text": None,
                "language_code": source_lang,
                "is_original": True,
                "is_pending": True,
            }
        # Prefer the fine-grained Whisper segments for the Transcript UI
        # so each line corresponds to a real ~1-5 s utterance rather than
        # the 30 s buckets ``raw_text`` uses for summary. Falls back to
        # raw_text for legacy jobs and unpatched-mlx fallbacks where
        # segments weren't captured.
        text = _build_segments_text(job.raw_segments_json) or job.raw_text
        return {
            "text": text,
            "language_code": source_lang,  # may be None for PDF / HTML
            "is_original": True,
            "is_pending": False,
        }

    # Cached translation lookup. ``requested`` is non-None by virtue of
    # the ``is_original`` branch above (None or matching source already
    # returned). Mypy doesn't track that flow — narrow explicitly.
    assert requested is not None
    translation = repo.get_translation(job_id, requested)
    if translation is None:
        # No row at all — caller must POST /translate first.
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"no cached {requested!r} translation for job {job_id}",
        )
    if translation.get("status") != "done":
        # Row exists, work in flight (queued / running / failed). Pending
        # sentinel so the UI shows a "translating…" placeholder for the
        # body. Chip status is the canonical source for UI; this response
        # is just the body for the currently-selected language.
        return {
            "text": None,
            "language_code": requested,
            "is_original": False,
            "is_pending": True,
        }
    return {
        "text": translation["text"],
        "language_code": requested,
        "is_original": False,
        "is_pending": False,
    }


@router.get("/{job_id}/frames/{rel_path:path}")
def get_frame(job_id: str, rel_path: str) -> FileResponse:
    """Serve one JPEG frame fetched via ``workers/frames.py`` — either by
    the QA LOOK step (``llm/qa.py``, backs the thumbnail under an answer
    that looked at the video) or by ``POST /jobs/{id}/frames`` (the
    on-demand "look" affordance next to a summary line). Both write into
    the same ``<data_dir>/frames/<job_id>/`` layout, so one route serves
    either (see ``FrameRef``).

    ``rel_path`` is the ``t<second>/frame_NN.jpg`` shape ``fetch_frames``
    writes — validated hard by ``frames.resolve_frame_path`` (job must
    exist here first; the resolved path must land inside that job's OWN
    frame directory, no traversal). Single-user localhost daemon, so no
    auth beyond that path check.

    404 for an unknown job OR a missing/escaping file — deliberately the
    same status for both so this endpoint doesn't leak which job ids exist
    independently of whether they have frames.
    """
    if repo.get_job(job_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"job {job_id} not found")
    path = frames.resolve_frame_path(job_id, rel_path)
    if path is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "frame not found")
    return FileResponse(path, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# On-demand "look" affordance — GET the job's deixis moments, POST to fetch
# (or reuse) that moment's frames. Distinct from the QA LOOK step: no vision
# call here, the user looks themselves. See FrameRef / DeixisMoment.
# ---------------------------------------------------------------------------


@router.get("/{job_id}/moments", response_model=MomentsListResponse)
def list_moments(job_id: str) -> MomentsListResponse:
    """The deixis moments for a job — feeds the sidepanel's on-demand
    "look" affordance next to a summary line's ``[MM:SS]`` marker.

    Reuses ``workers.deixis.candidates_for_job`` — the EXACT same
    job-qualification + candidate-finding logic the QA LOOK step uses — so
    this list is byte-for-byte the set of moments QA could ever look at for
    this job. EXTERNAL candidates are dropped: they point outside the video
    (a link, an article number in the description) and must never be
    offered for a frame fetch, same rule the QA path enforces daemon-side.

    Empty list (never an error) for: jobs that don't qualify at all (page/
    PDF, or any transcript_source not in AUDIO_TRANSCRIPT_SOURCES), jobs
    with no raw_segments_json, or a transcript with no deixis moments.
    """
    job = repo.get_job(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"job {job_id} not found")
    candidates = deixis.candidates_for_job(job)
    items = [
        DeixisMoment(
            seconds=c.timestamp,
            timecode=timecodes.format_timecode(c.timestamp),
            phrase=c.phrase,
            category=c.category.value,  # type: ignore[arg-type]
        )
        for c in candidates
        if c.category != DeixisCategory.EXTERNAL
    ]
    return MomentsListResponse(items=items)


# Category -> section download resolution for the on-demand frame fetch.
# Small enough (2 entries) that it isn't worth sharing across modules — see
# workers/frames.py's own module docstring: mapping a deixis category to a
# resolution is deliberately the CALLER's job, not that module's. Mirrors
# llm/qa.py's `_HEIGHT_BY_CATEGORY` (OBJECT is worth reading a label off;
# ACTION only needs to be seen; EXTERNAL is absent on purpose — it can never
# reach here, see the guard in `fetch_moment_frames` below).
_MOMENT_HEIGHT_BY_CATEGORY: dict[DeixisCategory, int] = {
    DeixisCategory.OBJECT: frames.SECTION_MAX_HEIGHT_READABLE_PX,
    DeixisCategory.ACTION: frames.SECTION_MAX_HEIGHT_PX,
}

# Tolerance for matching the request's `seconds` back to one of this job's
# own DeixisMoment entries. The client echoes back the exact
# `DeixisMoment.seconds` value GET /moments handed it, unmodified — this
# only needs to absorb float round-tripping through JSON, NOT reconcile a
# rendered [MM:SS] marker against a moment (that fuzzier matching is the
# extension's own job — see sidepanel/app.js's own, much wider, tolerance
# constant). A tight window here is a deliberate defence: it means a
# request can't "aim" at a nearby moment it wasn't actually shown.
_MOMENT_MATCH_TOLERANCE_SECONDS = 1.0


@router.post("/{job_id}/frames", response_model=FrameFetchResponse)
async def fetch_moment_frames(job_id: str, req: FrameFetchRequest) -> FrameFetchResponse:
    """Fetch (or reuse already-downloaded) frames for one of this job's own
    deixis moments, for the user to look at directly. NO vision/LLM call
    here — the user clicked because they want to see the picture
    themselves; involving a model would only add latency and a chance to
    invent something.

    Validates ``seconds`` against this job's own moments (the same source
    ``GET /moments`` uses) before ever touching the network: a value that
    doesn't correspond to a real moment 404s, and a match that turns out to
    be EXTERNAL is rejected the same way the QA LOOK step's daemon-side
    guard rejects an EXTERNAL pick regardless of what a caller sends.

    On success, frames already on disk for this exact moment are returned
    without re-downloading (``fetch_frames(reuse_existing=True)``).
    Failure is always a clear HTTP error, never a silently-empty body:
    - 409 if this job's per-job frame budget is already spent.
    - 502 if the section download failed (after its own internal retries
      — see ``workers.frames``'s "Section download retries").
    """
    job = repo.get_job(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"job {job_id} not found")

    candidates = deixis.candidates_for_job(job)
    match = min(
        (
            c for c in candidates
            if abs(c.timestamp - req.seconds) <= _MOMENT_MATCH_TOLERANCE_SECONDS
        ),
        key=lambda c: abs(c.timestamp - req.seconds),
        default=None,
    )
    if match is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"no deixis moment near {req.seconds}s for job {job_id}",
        )
    if match.category == DeixisCategory.EXTERNAL:
        # Defence in depth: GET /moments never offers an EXTERNAL entry in
        # the first place, but this guard is what actually GUARANTEES one
        # never reaches fetch_frames, not the client's compliance.
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "this moment points outside the video (a link or description "
            "reference) — there is no frame to fetch for it",
        )

    url = getattr(job, "url", None)
    if not url:
        raise HTTPException(status.HTTP_409_CONFLICT, f"job {job_id} has no source url")

    max_height = _MOMENT_HEIGHT_BY_CATEGORY.get(match.category, frames.SECTION_MAX_HEIGHT_PX)
    try:
        frame_paths = await frames.fetch_frames(
            job_id=job_id,
            url=url,
            timestamp_seconds=match.timestamp,
            max_height_px=max_height,
            # No cookies: cookies only ever arrive on the original
            # job-creation request and are never persisted on Job (see
            # llm/qa.py's `_inspect_moment` — same story here, long after
            # ingestion there is nothing stored to forward).
            cookies=None,
            reuse_existing=True,
        )
    except FrameExtractionError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"could not fetch the video frame: {exc}",
        ) from exc

    if not frame_paths:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"this job's frame budget ({frames.MAX_FRAMES_PER_JOB} frames) is "
            "already spent; no new frames can be fetched",
        )

    timecode = timecodes.format_timecode(match.timestamp)
    items = [
        FrameRef(
            seconds=match.timestamp,
            timecode=timecode,
            phrase=match.phrase,
            frame_url=f"/jobs/{job_id}/frames/{p.parent.name}/{p.name}",
        )
        for p in frame_paths
    ]
    return FrameFetchResponse(items=items)


@router.delete("/{job_id}", status_code=204)
async def delete_job(job_id: str) -> Response:
    # async because repo.delete_job publishes a job_event via the broker, and
    # asyncio.Queue.put_nowait is unsafe from a threadpool thread.
    deleted = repo.delete_job(job_id)
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"job {job_id} not found")
    return Response(status_code=204)


@router.post("/{job_id}/retry", status_code=202, response_model=JobCreateResponse)
async def retry_job(job_id: str) -> JSONResponse:
    """Re-run the pipeline for a failed job, preserving its id and any cached audio.

    Only ``failed`` jobs are accepted. The pipeline restarts from extraction;
    if a previous Whisper download left an ``audio_path`` on the row, the
    worker reuses that file instead of running yt-dlp again.
    """
    job = repo.get_job(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"job {job_id} not found")
    if job.status != JobStatus.FAILED.value:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"job {job_id} is in status {job.status!r}; only failed jobs can be retried",
        )

    kind = JobKind(job.kind)
    # MEDIA jobs can't be retried server-side: the media_url discovered on
    # the page lives only on the extension side (and can be a signed/expiring
    # CDN URL). Tell the caller explicitly so the UI can prompt the user to
    # re-summarize from the extension tab where the URL is fresh.
    if kind == JobKind.MEDIA:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "media jobs cannot be retried from the daemon — the media URL is "
            "discovered per-page and may be signed/expiring. Open the source "
            "tab and click Summarize again.",
        )
    # PDF jobs from ``file://`` URLs had their bytes shipped in the original
    # POST and we don't persist them. http(s) PDFs retry fine because the
    # daemon refetches the URL itself.
    if kind == JobKind.PDF and job.url.startswith("file://"):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "local (file://) PDF jobs cannot be retried from the daemon — the "
            "uploaded bytes aren't persisted. Open the PDF tab and click "
            "Summarize again.",
        )

    # repo.reset_for_retry emits job_event("updated") so the Library refreshes.
    repo.reset_for_retry(job_id)

    _spawn(
        pipeline.run_pipeline(
            job_id,
            kind=kind,
            url=job.url,
            page_text=None,        # extension may no longer be on this page; trafilatura will refetch
            page_title=job.title,
            media_url=None,        # not persisted; media retry path was rejected above
            pdf_bytes=None,        # not persisted; file:// retry was rejected above
            cookies=[],            # cookies aren't persisted on the job row
        )
    )

    body = JobCreateResponse(id=job_id, kind=kind, status=JobStatus.RUNNING)
    return JSONResponse(status_code=202, content=body.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------


@router.get("/{job_id}/messages", response_model=MessagesListResponse)
def list_messages(job_id: str) -> MessagesListResponse:
    if repo.get_job(job_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"job {job_id} not found")
    rows = repo.list_messages(job_id)
    return MessagesListResponse(items=[_to_message(r) for r in rows])
