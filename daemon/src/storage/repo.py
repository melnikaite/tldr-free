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
import json
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


def generate_job_id() -> str:
    """Public wrapper around ``_new_id`` for callers outside this module
    that need to mint a job id BEFORE the row exists — currently only
    ``storage.bundle``'s import path, which must know the new id up front
    to rewrite frame_url references and place frame files under it before
    the row itself is inserted (see ``insert_imported_job``)."""
    return _new_id()


# ---------------------------------------------------------------------------
# Create / update
# ---------------------------------------------------------------------------


def create_job(
    *,
    url: str,
    kind: str,
    title: str | None = None,
    progress_stage: str | None = None,
    alt_media_candidates_json: str | None = None,
) -> Job:
    """Insert a fresh Job row in ``status='running'`` and return it.

    ``alt_media_candidates_json`` is the pre-serialised JSON payload
    discovered by the extension's page scanner (see the column comment
    on ``Job.alt_media_candidates_json``). Persisted only at create time
    — we don't re-scan a page after the job exists.

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
        alt_media_candidates_json=alt_media_candidates_json,
        created_at=now,
        added_at=now,
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


def insert_imported_job(
    *,
    job_id: str,
    url: str,
    kind: str,
    title: str | None,
    duration_seconds: int | None,
    created_at: datetime,
    completed_at: datetime | None,
    raw_text: str | None,
    summary_md: str | None,
    transcript_source: str | None,
    video_id: str | None,
    transcript_language: str | None,
    raw_segments_json: str | None,
    alt_media_candidates_json: str | None,
    messages: list[dict[str, Any]],
    translations: list[dict[str, Any]],
) -> Job:
    """Insert a fully-formed Job (plus its Messages + TranscriptTranslations)
    as one atomic transaction — used by ``storage.bundle`` when importing an
    exported bundle onto a new machine.

    ``job_id`` is caller-minted (see ``generate_job_id``) rather than
    generated here: the caller (``bundle.import_bundle``) needs the new id
    up front to rewrite frame_url references and place frame files on disk
    BEFORE this call, so the id has to exist before the row does.

    Always ``status="done"`` / ``progress_stage=None`` / ``error=None`` —
    the contract guarantees only ``status=="done"`` jobs are ever exported
    (see ``bundle.export_jobs``), so anything reaching here is, by
    definition, a finished job being replayed onto a new machine, not one
    resuming in-flight work it never actually did here.

    ``created_at`` is taken from the bundle (preserves "when the material
    was processed" on the exporting machine); ``added_at`` is always set to
    ``now`` here — this row is appearing on THIS machine right now,
    regardless of how old the material is. That's what keeps the retention
    sweep (which now runs on ``added_at``, see ``delete_jobs_older_than``)
    from deleting a freshly imported archive of old jobs on its very next
    pass.

    ``messages`` is a list of ``{"role", "content", "created_at",
    "frame_refs_json"}`` dicts, already in the order they should get
    (ascending) ids in — this function inserts them in list order inside
    the same transaction as the Job row, so autoincrement ids come out
    monotonic. ``translations`` is a list of ``{"language_code", "text",
    "created_at", "updated_at"}`` dicts, always inserted with
    ``status="done"``/``progress_percent=100`` (only ``done`` translations
    are ever exported — see ``list_done_translations``).

    All-or-nothing: if anything inside the transaction raises, the Job and
    every Message/TranscriptTranslation for it roll back together — no
    half-written job survives. The caller is expected to call this once
    per job and catch failures per-job so one bad job in a batch doesn't
    abort the rest (see ``bundle.import_bundle``).

    Emits ``job_event("created", …)`` same as ``create_job`` so an open
    Library page renders the imported row live.
    """
    from src.storage.db import TranscriptTranslation

    now = datetime.utcnow()
    job = Job(
        id=job_id,
        url=url,
        kind=kind,
        status="done",
        title=title,
        duration_seconds=duration_seconds,
        created_at=created_at,
        added_at=now,
        updated_at=now,
        completed_at=completed_at,
        error=None,
        progress_stage=None,
        raw_text=raw_text,
        summary_md=summary_md,
        transcript_source=transcript_source,
        video_id=video_id,
        transcript_language=transcript_language,
        raw_segments_json=raw_segments_json,
        alt_media_candidates_json=alt_media_candidates_json,
    )
    with session_scope() as session:
        session.add(job)
        session.flush()  # job.id becomes a valid FK target for what follows

        for m in messages:
            session.add(
                Message(
                    job_id=job.id,
                    role=m["role"],
                    content=m["content"],
                    created_at=m.get("created_at") or now,
                    frame_refs_json=m.get("frame_refs_json"),
                )
            )
        for t in translations:
            session.add(
                TranscriptTranslation(
                    job_id=job.id,
                    language_code=t["language_code"],
                    status="done",
                    text=t.get("text"),
                    progress_percent=100,
                    created_at=t.get("created_at") or now,
                    updated_at=t.get("updated_at") or now,
                )
            )
        session.flush()
        session.refresh(job)
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
    transcript_language: str | None = None,
    raw_segments_json: str | None = None,
) -> None:
    """Finalise a job with status=done, persisting all extracted fields.

    ``transcript_language`` is the ISO-639-1 code of the source transcript.
    Setting it here AND in ``set_extracted`` is redundant on the happy path,
    but ``set_extracted`` is what catches mid-pipeline failures (summary
    errors after transcription succeeded), so both paths persist it.

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
        if transcript_language is not None:
            job.transcript_language = transcript_language
        if raw_segments_json is not None:
            job.raw_segments_json = raw_segments_json
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
    transcript_language: str | None = None,
    raw_segments_json: str | None = None,
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
        if transcript_language is not None:
            job.transcript_language = transcript_language
        if raw_segments_json is not None:
            job.raw_segments_json = raw_segments_json
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
        "transcript_language": getattr(job, "transcript_language", None),
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "added_at": job.added_at.isoformat() if getattr(job, "added_at", None) else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


def list_translations(job_id: str) -> list[Any]:
    """Return cached translation summaries for ``job_id``.

    Empty list when no translation has been requested. Returned as plain
    dicts (Pydantic in the API layer coerces them into the
    ``TranscriptTranslationSummary`` schema).
    """
    from src.storage.db import TranscriptTranslation

    with session_scope() as session:
        rows = session.exec(
            select(TranscriptTranslation).where(TranscriptTranslation.job_id == job_id)
        ).all()
        return [
            {
                "language_code": r.language_code,
                "status": r.status,
                "progress_percent": int(r.progress_percent),
                "error": r.error,
            }
            for r in rows
        ]


def get_translation(job_id: str, language_code: str) -> dict[str, Any] | None:
    """Return a single cached translation row (or ``None`` if absent).

    Returns the full text in ``"text"`` — used by ``GET /transcript`` to
    serve the body. Callers should check ``status == "done"`` before
    relying on the text (in-flight rows are present too with text=None).
    """
    from src.storage.db import TranscriptTranslation

    with session_scope() as session:
        row = session.get(TranscriptTranslation, (job_id, language_code))
        if row is None:
            return None
        return {
            "language_code": row.language_code,
            "status": row.status,
            "progress_percent": int(row.progress_percent),
            "text": row.text,
            "error": row.error,
        }


def list_done_translations(job_id: str) -> list[dict[str, Any]]:
    """Return this job's cached translations that are actually ``done`` —
    together with the full ``text`` and both timestamps.

    Used only by ``storage.bundle`` when packing an export bundle: the
    sidepanel-facing ``list_translations``/``get_translation`` above
    intentionally omit ``text`` (list) or timestamps (both) because
    nothing in the live UI needs them; the export format needs both, and
    only ever wants translations that carry actual text (queued/running/
    failed rows have none worth exporting).
    """
    from src.storage.db import TranscriptTranslation

    with session_scope() as session:
        rows = session.exec(
            select(TranscriptTranslation).where(
                TranscriptTranslation.job_id == job_id,
                TranscriptTranslation.status == "done",
            )
        ).all()
        return [
            {
                "language_code": r.language_code,
                "text": r.text,
                "created_at": r.created_at,
                "updated_at": r.updated_at,
            }
            for r in rows
        ]


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

    Also unlinks any cached audio file (``job.audio_path``) and any video
    frames fetched for this job, so we don't leave orphaned multi-MB files on
    disk after the row is gone.

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
    _delete_job_frames(job_id)
    _emit_deleted(job_id)
    return True


def _delete_job_frames(job_id: str) -> None:
    """Best-effort removal of a job's fetched video frames.

    Imported locally: ``workers`` is a higher layer than ``storage``, and a
    module-level import here would invert that. Frames are optional and their
    absence is the common case, so a failure must never block deleting a row.
    """
    try:
        from src.workers.frames import delete_job_frames

        delete_job_frames(job_id)
    except Exception:
        log.warning("repo: failed to delete frames for job %s", job_id)


def _safe_unlink(path: Path) -> None:
    """Best-effort file removal; warns and continues on OSError."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.warning("repo: failed to unlink cached file %s", path)


# ---------------------------------------------------------------------------
# Messages (chat history)
# ---------------------------------------------------------------------------


def add_message(
    job_id: str,
    *,
    role: str,
    content: str,
    frame_refs: list[dict[str, Any]] | None = None,
) -> Message:
    """Insert a chat message for ``job_id`` and return it (detached).

    ``frame_refs`` is the LOOK step's list of ``FrameRef``-shaped dicts
    (``api.schemas.FrameRef.model_dump()``-compatible) — pre-serialised
    here to JSON on ``Message.frame_refs_json`` so the storage layer stays
    Pydantic-free, matching how ``alt_media_candidates_json`` is handled on
    ``Job``. ``None`` or empty is stored as NULL; the API layer treats a
    missing/malformed value as "no frames" (see ``api/jobs.py._to_message``).
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"invalid role: {role!r}")
    frame_refs_json: str | None = None
    if frame_refs:
        frame_refs_json = json.dumps(frame_refs, ensure_ascii=False, separators=(",", ":"))
    msg = Message(
        job_id=job_id,
        role=role,
        content=content,
        created_at=datetime.utcnow(),
        frame_refs_json=frame_refs_json,
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
    """Delete jobs whose ``added_at`` — when the row appeared ON THIS
    MACHINE, not when the material was processed — is strictly before
    ``cutoff``.

    Deliberately NOT ``created_at``: an imported bundle (see
    ``storage.bundle``) preserves the exporting machine's ``created_at``,
    so sweeping on that column could delete a freshly imported archive of
    old material on its very next pass. ``added_at`` answers "how long has
    this been sitting on THIS machine", which is what retention is actually
    supposed to measure. ``created_at`` keeps its own meaning untouched —
    it's still what the Library shows and sorts by.

    Returns the number of jobs deleted. Message rows are removed by FK
    cascade (foreign_keys pragma is ON). Emits one ``job_event("deleted", …)``
    per row so any open Library updates immediately.

    Unlinks each job's cached audio and fetched frames, same as
    ``delete_job``. This path used to drop the rows only, which left orphaned
    multi-MB audio files behind on every retention sweep.
    """
    cached_audio: list[str] = []
    with session_scope() as session:
        stmt = select(Job.id).where(Job.added_at < cutoff)
        ids = list(session.exec(stmt).all())
        if not ids:
            return 0
        # Delete via ORM so cascades fire reliably.
        for job_id in ids:
            job = session.get(Job, job_id)
            if job is not None:
                if job.audio_path:
                    cached_audio.append(job.audio_path)
                # Clear linked rows explicitly — keep behaviour stable across
                # SQLite pragma states (matches delete_job).
                session.exec(
                    Message.__table__.delete().where(Message.job_id == job_id)  # type: ignore[attr-defined]
                )
                session.delete(job)
    for path in cached_audio:
        _safe_unlink(Path(path))
    for job_id in ids:
        _delete_job_frames(job_id)
        _emit_deleted(job_id)
    return len(ids)


__all__ = [
    "add_message",
    "create_job",
    "delete_job",
    "delete_jobs_older_than",
    "find_pending_for_restart",
    "generate_job_id",
    "get_job",
    "insert_imported_job",
    "list_done_translations",
    "list_jobs",
    "list_messages",
    "mark_done",
    "mark_failed",
    "reset_for_retry",
    "set_audio",
    "set_extracted",
    "update_status",
]
