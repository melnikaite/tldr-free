"""POST/GET/DELETE /jobs + chat-message endpoints.

POST /jobs is now ASYNC. The route persists the row, kicks off the pipeline
(workers/pipeline.run_pipeline) as a background task, and returns 202 with
the new job id. The client follows progress via POST /ai/stream {job_id}.

Chat history per job lives in `Message`; this module exposes
GET /jobs/{id}/messages and DELETE /jobs/{id}/messages.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.responses import JSONResponse

from src.api.schemas import (
    JobCreateRequest,
    JobCreateResponse,
    JobDetails,
    JobKind,
    JobListResponse,
    JobStatus,
    JobSummary,
    MessagesListResponse,
    TranscriptSource,
)
from src.api.schemas import (
    Message as MessageModel,
)
from src.storage import repo
from src.workers import pipeline, youtube

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


def _to_details(job: Any) -> JobDetails:
    raw_text_length = len(job.raw_text) if job.raw_text is not None else None
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
        raw_text_length=raw_text_length,
        error=job.error,
        video_id=job.video_id,
    )


def _to_message(row: Any) -> MessageModel:
    return MessageModel(
        id=row.id,
        job_id=row.job_id,
        role=row.role,
        content=row.content,
        created_at=row.created_at,
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

        kind = pipeline.infer_kind(req.url, req.kind)

        # For YouTube the extension's scraped title is unreliable (SPA, fast
        # tab switching), so seed with the video id. The pipeline overwrites
        # it with yt-dlp's canonical title within a couple of seconds.
        initial_title = req.page_title
        if kind == JobKind.YOUTUBE:
            with contextlib.suppress(ValueError):
                initial_title = youtube.extract_video_id(req.url)

        # Single create with progress_stage='extracting' avoids a needless
        # second update_status (and a second job event) right after creation.
        # repo.create_job emits job_event("created") for us.
        job = repo.create_job(
            url=req.url,
            kind=kind.value,
            title=initial_title,
            progress_stage="extracting",
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
            cookies=list(req.cookies or []),
        )
    )

    body = JobCreateResponse(id=job.id, kind=kind, status=JobStatus.RUNNING)
    return JSONResponse(status_code=202, content=body.model_dump(mode="json"))


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
    # repo.reset_for_retry emits job_event("updated") so the Library refreshes.
    repo.reset_for_retry(job_id)

    _spawn(
        pipeline.run_pipeline(
            job_id,
            kind=kind,
            url=job.url,
            page_text=None,        # extension may no longer be on this page; trafilatura will refetch
            page_title=job.title,
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
