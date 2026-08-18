"""SQLite engine + connection lifecycle + ORM models.

Responsibilities:
- Define the SQLModel ORM models (Job, Message).
- Build a SQLModel/SQLAlchemy engine with the path from ``config.storage.db_path``.
- Apply per-connection pragmas (WAL, NORMAL, mmap, foreign_keys, ...) — SQLite
  pragmas are connection-scoped, so they must be installed on every checkout.
- Expose a session factory and a FastAPI-friendly dependency.
- Run migrations on startup (orchestrated from ``main.lifespan``).

Synchronous SQLModel/SQLAlchemy. FastAPI calls into this module from
coroutine handlers but each call is short and CPU-bound, so no async-driver
gain in v1. The actual CREATE TABLE / FTS / triggers live in migrations.py
(raw SQL), not in ``SQLModel.metadata.create_all`` — classes here are the
read/write surface used by the repo.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Field, Session, SQLModel, create_engine

from src.config import get_config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------


class Job(SQLModel, table=True):
    """One processed page or video. Status flows: queued → running → done|failed."""

    __tablename__ = "job"

    id: str = Field(primary_key=True)               # nanoid(12)
    url: str = Field(index=True)
    kind: str                                       # "page" | "youtube"
    status: str = Field(index=True)                 # queued | running | done | failed
    title: str | None = None
    duration_seconds: int | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    # When this row appeared ON THIS MACHINE — distinct from created_at (when
    # the material was actually processed). Equal to created_at for every
    # normally-created job; diverges only for a job brought in via
    # ``POST /jobs/import`` (see storage/bundle.py), which preserves the
    # exporting machine's created_at but sets added_at=now. Retention sweeps
    # by added_at (repo.delete_jobs_older_than) so a freshly imported archive
    # can't be deleted on its very next pass just because the material itself
    # is old. created_at stays what the Library shows and sorts by. Backfilled
    # to created_at for every pre-existing row by migration v7.
    added_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    error: str | None = None
    progress_stage: str | None = None
    raw_text: str | None = None
    summary_md: str | None = None
    transcript_source: str | None = None
    video_id: str | None = None
    # ISO-639-1 code of the transcript's source language. Filled mid-pipeline
    # from Whisper's auto-detect (verbose_json) for media/youtube-via-whisper
    # jobs, or from the YouTube caption track's language for the fast path.
    # ``None`` for PDF / HTML pages (we don't run language detection on
    # extracted text — falls back to "Original" in the language switcher).
    transcript_language: str | None = None
    # Fine-grained Whisper / YouTube segments as JSON — used by the
    # Transcript tab UI to render one line per segment (typically 1-5 s
    # each) with binary-search highlight on playback. ``raw_text`` stays
    # at the 30-second-bucketed shape for summary / Q&A; this is its
    # sibling for line-by-line display.
    #
    # Shape: ``[{"start": float, "end": float, "text": str}, ...]``,
    # JSON-encoded as a text column for SQLite simplicity. Null for
    # legacy jobs created before this column existed and for non-timed
    # jobs (PAGE / PDF) where there's no segment structure.
    raw_segments_json: str | None = None
    # Other playable media sources the extension's scanner discovered on
    # the page at job-creation time, JSON-encoded as
    # ``[{"media_url": "...", "kind": "video"|"audio"|"iframe",
    #    "label": "..."}, ...]``. Surfaced by the sidepanel as a "wrong
    # source?" picker so the user can switch without re-running the
    # whole extraction. The daemon never reads these itself — round-trip
    # storage only. Null for legacy jobs and for pages with one (or
    # zero) candidates.
    alt_media_candidates_json: str | None = None
    # Set by the Whisper worker after a successful audio download. Persisted
    # so that a retry of a failed-mid-pipeline job can skip re-downloading.
    # Cleared (with the file unlinked) on mark_done and on delete_job.
    audio_path: str | None = None
    audio_duration_seconds: float | None = None
    # Set by workers/transcribe.py's coverage check (Whisper jobs only) when
    # the transcript's last segment still falls short of the audio's known
    # duration after bounded retries — see transcribe.TranscribeResult and
    # transcribe._ensure_coverage for the full mechanism, and
    # timecodes.collapse_repeated_segments for why a repetition-loop
    # collapse can create exactly this gap. ``None`` means either "not a
    # Whisper job" (PAGE/PDF/YouTube-caption jobs never touch this column)
    # or "checked, coverage was complete" — a job can look identical to the
    # user in both cases, which is fine: only a non-null, positive value is
    # actionable. A job's status stays "done" either way (soft-pause /
    # restart-safety are unaffected); this is purely an honesty flag on top
    # of an otherwise normal completion, cleared back to None if a later
    # retry of the same job achieves full coverage (repo.set_extracted /
    # repo.mark_done write it unconditionally when the Whisper runner
    # passes it, unlike the other optional fields on this model).
    transcript_missing_seconds: float | None = None


class Message(SQLModel, table=True):
    """One chat message attached to a job (Q&A history)."""

    __tablename__ = "message"

    id: int | None = Field(default=None, primary_key=True)
    job_id: str = Field(foreign_key="job.id", index=True)
    role: str                                       # "user" | "assistant"
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    # JSON-encoded list of FrameRef dicts (api.schemas.FrameRef) — the video
    # frame(s) that actually backed a visual claim in this message, one per
    # relevant LOOK-step moment (see llm/qa.py). Null for user messages and
    # for assistant messages that never looked at a frame, or looked but
    # found nothing relevant. Round-trip storage only; the daemon never
    # reads this back into anything other than the API response.
    frame_refs_json: str | None = None


class TranscriptTranslation(SQLModel, table=True):
    """One cached translation of a job's transcript into a target language.

    The original transcript stays in ``Job.raw_text`` (in
    ``Job.transcript_language``). Translations live here so they're cheap
    to look up by ``(job_id, language_code)`` and CASCADE-delete with the
    parent Job.

    Status flow: ``queued`` → ``running`` → ``done`` | ``partial`` | ``failed``.
    ``text`` is populated on ``done`` AND ``partial``; ``error`` on
    ``partial`` AND ``failed``. ``progress_percent`` is updated mid-group
    so the UI shows movement without partial text leaking out (we render
    a spinner + percent, not the streaming tokens — per the agreed UX).

    ``partial`` exists because the translator (``workers/translator.py``)
    never trusts the model's line alignment — it verifies each group's
    output against the source line-for-line and, when that keeps failing
    even after bisecting the group down to single lines, gives up on just
    those lines rather than the whole job. ``text`` is then the full
    transcript with the untranslated lines left in the source language,
    and ``error`` is a plain-English count of how many. The UI treats
    ``partial`` like ``done`` for selection/export purposes (it has real
    text) but flags it visually and offers it to "Retry all" alongside
    ``failed``.

    Restart-safety: rows left in ``running`` at daemon startup are
    re-enqueued by ``re_enqueue_pending`` because we have the source text
    and the target language code — everything needed to continue from
    scratch (we don't checkpoint partial output mid-group).
    """

    __tablename__ = "transcript_translation"

    job_id: str = Field(foreign_key="job.id", primary_key=True)
    language_code: str = Field(primary_key=True)
    status: str = Field(index=True)
    text: str | None = None
    error: str | None = None
    progress_percent: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Engine + connection lifecycle
# ---------------------------------------------------------------------------

# Pragmas applied to every connection. SQLite pragmas live on a connection,
# so they must be re-applied on each checkout from the pool.
_PRAGMAS: tuple[tuple[str, str], ...] = (
    ("journal_mode", "WAL"),
    ("synchronous", "NORMAL"),
    ("cache_size", "-64000"),         # KiB → ~64 MiB
    ("mmap_size", "268435456"),       # 256 MiB
    ("temp_store", "MEMORY"),
    ("foreign_keys", "ON"),
)


_engine: Engine | None = None


def _install_pragmas(engine: Engine) -> None:
    """Register a connect listener that applies all pragmas on every connection."""

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn: Any, _connection_record: Any) -> None:  # noqa: ANN401
        cursor = dbapi_conn.cursor()
        try:
            for name, value in _PRAGMAS:
                cursor.execute(f"PRAGMA {name} = {value};")
        finally:
            cursor.close()


def _build_engine(db_path: Path | str) -> Engine:
    """Create a fresh engine for ``db_path`` (or `:memory:` SQLite URL).

    Used both by the production lifespan and by tests that want an isolated DB.
    """
    if isinstance(db_path, Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{db_path}"
    else:
        # Pre-formed sqlite:// URL or ":memory:" sentinel
        url = db_path if db_path.startswith("sqlite:") else f"sqlite:///{db_path}"

    engine = create_engine(
        url,
        echo=False,
        connect_args={"check_same_thread": False, "timeout": 30.0},
    )
    _install_pragmas(engine)
    return engine


def init_engine(db_path: Path | str | None = None) -> Engine:
    """Create the global engine if needed, return it.

    If ``db_path`` is provided, the previous engine (if any) is disposed first
    and a new one is built. Without an argument, the configured DB path is used.
    """
    global _engine
    if db_path is not None:
        if _engine is not None:
            _engine.dispose()
        _engine = _build_engine(db_path)
        log.info("storage: engine initialised at %s", db_path)
        return _engine

    if _engine is None:
        configured = get_config().storage.db_path
        _engine = _build_engine(configured)
        log.info("storage: engine initialised at %s", configured)
    return _engine


def get_engine() -> Engine:
    """Return the active engine, raising if it hasn't been initialised yet."""
    if _engine is None:
        raise RuntimeError(
            "Storage engine not initialised — call init_engine() from main.lifespan"
        )
    return _engine


def dispose_engine() -> None:
    """Close all pooled connections; called on shutdown."""
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None
        log.info("storage: engine disposed")


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager around a SQLModel Session with commit/rollback handling."""
    session = Session(get_engine())
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a session bound to the active engine."""
    with session_scope() as session:
        yield session


__all__ = [
    "Job",
    "Message",
    "SQLModel",
    "dispose_engine",
    "get_engine",
    "get_session",
    "init_engine",
    "session_scope",
]
