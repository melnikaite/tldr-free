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
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    error: str | None = None
    progress_stage: str | None = None
    raw_text: str | None = None
    summary_md: str | None = None
    transcript_source: str | None = None
    video_id: str | None = None
    # Set by the Whisper worker after a successful audio download. Persisted
    # so that a retry of a failed-mid-pipeline job can skip re-downloading.
    # Cleared (with the file unlinked) on mark_done and on delete_job.
    audio_path: str | None = None
    audio_duration_seconds: float | None = None


class Message(SQLModel, table=True):
    """One chat message attached to a job (Q&A history)."""

    __tablename__ = "message"

    id: int | None = Field(default=None, primary_key=True)
    job_id: str = Field(foreign_key="job.id", index=True)
    role: str                                       # "user" | "assistant"
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


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
