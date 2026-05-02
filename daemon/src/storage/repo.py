"""High-level CRUD over Job + Message.

Public surface:

    create_job(*, url, kind, title=None) -> Job
    update_status(job_id, *, status, progress_stage=None, error=None) -> None
    mark_done(job_id, *, raw_text, summary_md, transcript_source, ...) -> None
    mark_failed(job_id, *, error) -> None
    get_job(job_id) -> Job | None
    list_jobs(*, status=None, kind=None, since=None, limit, offset)
        -> tuple[list[Job], int]
    delete_job(job_id) -> None                  # cascades into Message
    find_pending_for_restart() -> list[Job]     # status in {queued, running}

All functions open their own short-lived session through ``session_scope`` so
callers (FastAPI handlers, workers) don't have to thread a Session around.

Datetimes use ``datetime.utcnow()`` for default values, matching the SQLModel
pattern in ``db.py``.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from nanoid import generate as _nanoid_generate
from sqlalchemy import func
from sqlalchemy.orm import defer
from sqlmodel import select

from src.storage.db import Job, Message, session_scope

log = logging.getLogger(__name__)

# nanoid: URL-safe alphabet, 12 chars (~71 bits — collision-safe at our scale).
_ID_LENGTH = 12


def _new_id() -> str:
    return str(_nanoid_generate(size=_ID_LENGTH))


# ---------------------------------------------------------------------------
# Create / update
# ---------------------------------------------------------------------------


def create_job(
    *,
    url: str,
    kind: str,
    title: str | None = None,
    progress_stage: str | None = None,
) -> Job:
    """Insert a fresh Job row in ``status='running'`` and return it.

    Emits ``job_event("created", …)`` so the Library renders the row instantly
    without polling.
    """
    now = datetime.utcnow()
    job = Job(
        id=_new_id(),
        url=url,
        kind=kind,
        status="running",
        title=title,
        progress_stage=progress_stage,
        created_at=now,
        updated_at=now,
    )
    with session_scope() as session:
        session.add(job)
        session.flush()
        session.refresh(job)
        # Detach so the returned object is usable after the session closes.
        session.expunge(job)
    _emit_created(job.id)
    return job


def update_status(
    job_id: str,
    *,
    status: str,
    progress_stage: str | None = None,
    error: str | None = None,
) -> None:
    """Update status (and optionally progress_stage / error) on an existing job.

    Emits ``job_event("updated", …)`` for the Library/sidebar.
    """
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise KeyError(f"Job {job_id} not found")
        job.status = status
        if progress_stage is not None:
            job.progress_stage = progress_stage
        if error is not None:
            job.error = error
        job.updated_at = datetime.utcnow()
        session.add(job)
    _emit_updated(job_id)


def mark_done(
    job_id: str,
    *,
    raw_text: str,
    summary_md: str,
    transcript_source: str,
    title: str | None = None,
    duration_seconds: int | None = None,
    video_id: str | None = None,
) -> None:
    """Finalise a job with status=done, persisting all extracted fields.

    Emits ``job_event("updated", …)`` so the Library row flips to done with the
    final title in one event.
    """
    now = datetime.utcnow()
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise KeyError(f"Job {job_id} not found")
        job.status = "done"
        job.raw_text = raw_text
        job.summary_md = summary_md
        job.transcript_source = transcript_source
        if title is not None:
            job.title = title
        if duration_seconds is not None:
            job.duration_seconds = duration_seconds
        if video_id is not None:
            job.video_id = video_id
        job.completed_at = now
        job.updated_at = now
        job.error = None
        job.progress_stage = None
        session.add(job)
    _emit_updated(job_id)


def set_extracted(
    job_id: str,
    *,
    raw_text: str,
    transcript_source: str,
    title: str | None = None,
    video_id: str | None = None,
) -> None:
    """Persist extraction output mid-pipeline (before the summary call).

    Used by both the synchronous fast-path pipeline and the Whisper runner so
    that raw_text + transcript_source + video_id are saved on the row even if
    the summary call later fails or the daemon restarts. Does NOT touch
    ``status`` — that's the caller's responsibility (typically remains
    ``running`` with ``progress_stage='ready'`` or ``'summarizing'``).

    ``title`` overwrites the existing value when provided — the caller is
    expected to pass a more authoritative source (e.g. yt-dlp metadata) than
    whatever the extension guessed at job-creation time.

    Emits ``job_event("updated", …)`` — this is the path that surfaces the
    canonical YouTube title to the Library mid-pipeline.
    """
    now = datetime.utcnow()
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise KeyError(f"Job {job_id} not found")
        job.raw_text = raw_text
        job.transcript_source = transcript_source
        if title is not None:
            job.title = title
        if video_id is not None:
            job.video_id = video_id
        job.updated_at = now
        session.add(job)
    _emit_updated(job_id)


def mark_failed(job_id: str, *, error: str) -> None:
    """Move a job into status=failed with an error message.

    Emits ``job_event("updated", …)``.
    """
    now = datetime.utcnow()
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise KeyError(f"Job {job_id} not found")
        job.status = "failed"
        job.error = error
        job.updated_at = now
        job.completed_at = now
        session.add(job)
    _emit_updated(job_id)


def reset_for_retry(job_id: str) -> None:
    """Move a failed job back into ``status=running`` with a clean error/progress.

    ``audio_path`` and ``audio_duration_seconds`` are intentionally preserved
    so the Whisper worker can skip re-downloading on retry.

    Emits ``job_event("updated", …)``.
    """
    now = datetime.utcnow()
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise KeyError(f"Job {job_id} not found")
        job.status = "running"
        job.progress_stage = "extracting"
        job.error = None
        job.completed_at = None
        job.updated_at = now
        session.add(job)
    _emit_updated(job_id)


def set_audio(
    job_id: str,
    *,
    audio_path: str | None,
    audio_duration_seconds: float | None = None,
) -> None:
    """Persist (or clear) the locally cached audio file path for a job.

    Set after a successful yt-dlp download so a later retry of the same job
    can skip re-downloading. Cleared (with ``audio_path=None``) by the
    Whisper worker after ``mark_done`` and by ``delete_job``.

    Does NOT emit a job event — ``audio_path`` is internal plumbing the UI
    doesn't render. Skipping the publish keeps the global stream quiet.
    """
    now = datetime.utcnow()
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise KeyError(f"Job {job_id} not found")
        job.audio_path = audio_path
        if audio_duration_seconds is not None or audio_path is None:
            job.audio_duration_seconds = audio_duration_seconds
        job.updated_at = now
        session.add(job)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def get_job(job_id: str) -> Job | None:
    """Return the Job row by id, or None if missing. Detached from any session."""
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return None
        session.expunge(job)
        return job


def job_summary_dict(job: Job) -> dict[str, Any]:
    """JSON-ready snapshot of a Job, matching the JobSummary API shape.

    Used internally by ``_emit_*`` to publish ``job_event(...)`` into the
    global event broker. The /events stream forwards this payload to the
    Library so rows reflect title/status changes without a round-trip.
    """
    return {
        "id": job.id,
        "url": job.url,
        "kind": job.kind,
        "status": job.status,
        "title": job.title,
        "duration_seconds": job.duration_seconds,
        "progress_stage": job.progress_stage,
        "transcript_source": job.transcript_source,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


# ---------------------------------------------------------------------------
# Event publishing — every write function below runs one of these as a
# side effect so callers don't have to remember. Late import keeps the
# storage layer free of any compile-time dependency on workers/.
# Failures are swallowed: a transient broker hiccup must not roll back the
# DB write the user just made.
# ---------------------------------------------------------------------------


def _publish_job_event(action: str, payload: dict[str, Any]) -> None:
    try:
        from src.workers.broker import get_event_broker, job_event
    except Exception:
        return
    with contextlib.suppress(Exception):
        get_event_broker().publish(job_event(action, payload))


def _emit_created(job_id: str) -> None:
    job = get_job(job_id)
    if job is not None:
        _publish_job_event("created", job_summary_dict(job))


def _emit_updated(job_id: str) -> None:
    job = get_job(job_id)
    if job is not None:
        _publish_job_event("updated", job_summary_dict(job))


def _emit_deleted(job_id: str) -> None:
    _publish_job_event("deleted", {"id": job_id})


def list_jobs(
    *,
    status: str | Iterable[str] | None = None,
    kind: str | None = None,
    since: datetime | None = None,
    url: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Job], int]:
    """List jobs filtered by status / kind / since / url, with pagination.

    ``status`` accepts either a single string ("queued") or an iterable
    (["queued", "running"]) — the comma-split form from the API maps cleanly
    to the latter.

    ``url`` does an exact match. Used by the extension to look up whether the
    current tab has already been summarized.

    Returns a (rows, total_count) tuple where ``total_count`` is the number of
    rows matching the filters *before* pagination (so the UI can render proper
    pagination controls).
    """
    with session_scope() as session:
        base_stmt = select(Job)
        count_stmt: Any = select(func.count()).select_from(Job)

        if status is not None:
            statuses = [status] if isinstance(status, str) else list(status)
            base_stmt = base_stmt.where(Job.status.in_(statuses))  # type: ignore[attr-defined]
            count_stmt = count_stmt.where(Job.status.in_(statuses))  # type: ignore[attr-defined]
        if kind is not None:
            base_stmt = base_stmt.where(Job.kind == kind)
            count_stmt = count_stmt.where(Job.kind == kind)
        if since is not None:
            base_stmt = base_stmt.where(Job.created_at >= since)
            count_stmt = count_stmt.where(Job.created_at >= since)
        if url is not None:
            base_stmt = base_stmt.where(Job.url == url)
            count_stmt = count_stmt.where(Job.url == url)

        total = int(session.exec(count_stmt).one())

        # Skip the heavyweight text columns — list view never reads them, and
        # raw_text can be megabytes per row. Cuts the SQLite read + Python
        # decode time on long videos by orders of magnitude. The deferred
        # columns become inaccessible after expunge() — that's intentional;
        # callers who need them must use get_job(id).
        ordered = (
            base_stmt.options(
                defer(Job.raw_text),  # type: ignore[arg-type]
                defer(Job.summary_md),  # type: ignore[arg-type]
                defer(Job.error),  # type: ignore[arg-type]
            )
            .order_by(Job.created_at.desc())  # type: ignore[attr-defined]
            .offset(offset)
            .limit(limit)
        )
        rows = list(session.exec(ordered).all())
        for row in rows:
            session.expunge(row)
        return rows, total


def find_pending_for_restart() -> list[Job]:
    """Return all jobs left in ``queued`` or ``running`` state.

    Used by the worker on startup to re-enqueue work that was in flight when
    the daemon was previously stopped.
    """
    with session_scope() as session:
        stmt = select(Job).where(Job.status.in_(["queued", "running"]))  # type: ignore[attr-defined]
        rows = list(session.exec(stmt).all())
        for row in rows:
            session.expunge(row)
        return rows


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def delete_job(job_id: str) -> bool:
    """Delete the Job row. FTS triggers and Message FK cascade handle cleanup.

    Also unlinks any cached audio file (``job.audio_path``) so we don't leave
    orphaned multi-MB files on disk after the row is gone.

    Emits ``job_event("deleted", {"id": …})``. Returns True if a row was
    deleted, False if the id was not found.
    """
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return False
        cached_audio = job.audio_path
        session.delete(job)
    if cached_audio:
        _safe_unlink(Path(cached_audio))
    _emit_deleted(job_id)
    return True


def _safe_unlink(path: Path) -> None:
    """Best-effort file removal; warns and continues on OSError."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.warning("repo: failed to unlink cached file %s", path)


# ---------------------------------------------------------------------------
# Messages (chat history)
# ---------------------------------------------------------------------------


def add_message(job_id: str, *, role: str, content: str) -> Message:
    """Insert a chat message for ``job_id`` and return it (detached)."""
    if role not in ("user", "assistant"):
        raise ValueError(f"invalid role: {role!r}")
    msg = Message(
        job_id=job_id,
        role=role,
        content=content,
        created_at=datetime.utcnow(),
    )
    with session_scope() as session:
        if session.get(Job, job_id) is None:
            raise KeyError(f"Job {job_id} not found")
        session.add(msg)
        session.flush()
        session.refresh(msg)
        session.expunge(msg)
    return msg


def list_messages(job_id: str) -> list[Message]:
    """Return all messages for ``job_id`` ordered by created_at ascending."""
    with session_scope() as session:
        stmt = (
            select(Message)
            .where(Message.job_id == job_id)
            .order_by(Message.created_at.asc(), Message.id.asc())  # type: ignore[union-attr,attr-defined]
        )
        rows = list(session.exec(stmt).all())
        for row in rows:
            session.expunge(row)
        return rows


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


def delete_jobs_older_than(cutoff: datetime) -> int:
    """Delete jobs whose ``created_at`` is strictly before ``cutoff``.

    Returns the number of jobs deleted. Message rows are removed by FK
    cascade (foreign_keys pragma is ON). Emits one ``job_event("deleted", …)``
    per row so any open Library updates immediately.
    """
    with session_scope() as session:
        stmt = select(Job.id).where(Job.created_at < cutoff)
        ids = list(session.exec(stmt).all())
        if not ids:
            return 0
        # Delete via ORM so cascades fire reliably.
        for job_id in ids:
            job = session.get(Job, job_id)
            if job is not None:
                # Clear linked rows explicitly — keep behaviour stable across
                # SQLite pragma states (matches delete_job).
                session.exec(
                    Message.__table__.delete().where(Message.job_id == job_id)  # type: ignore[attr-defined]
                )
                session.delete(job)
    for job_id in ids:
        _emit_deleted(job_id)
    return len(ids)


__all__ = [
    "add_message",
    "create_job",
    "delete_job",
    "delete_jobs_older_than",
    "find_pending_for_restart",
    "get_job",
    "job_summary_dict",
    "list_jobs",
    "list_messages",
    "mark_done",
    "mark_failed",
    "reset_for_retry",
    "set_audio",
    "set_extracted",
    "update_status",
]
