"""Idempotent schema migrations.

A ``_migrations`` table tracks which migrations have been applied. Each
migration is a callable taking a sqlite3-style connection (``Connection``-
compatible — SQLAlchemy gives us ``raw_connection`` which exposes
``cursor()`` / ``execute()`` of the underlying DB-API).

Migration v1 creates:
- The Job and Message tables (matching the SQLModel classes in db.py).
- An FTS5 virtual table ``job_fts`` indexing title/raw_text/summary_md
  with the unicode61 tokenizer (remove_diacritics=2).
- Triggers (AI/AD/AU) that mirror Job rows into job_fts so FTS stays in sync.

Pragmas live in db.py (per-connection); we don't repeat them here, but the
runner verifies WAL mode is active after migrations as a sanity check.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)


Migration = Callable[[Any], None]


_MIGRATIONS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS _migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
)
"""


# ---------------------------------------------------------------------------
# v1 — initial schema
# ---------------------------------------------------------------------------


_V1_STATEMENTS: tuple[str, ...] = (
    # Core tables. We create them via raw SQL (rather than
    # SQLModel.metadata.create_all) because we need the FTS table and triggers
    # to live alongside, and we want one consistent migration story.
    """
    CREATE TABLE IF NOT EXISTS job (
        id TEXT PRIMARY KEY,
        url TEXT NOT NULL,
        kind TEXT NOT NULL,
        status TEXT NOT NULL,
        title TEXT,
        duration_seconds INTEGER,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT,
        error TEXT,
        progress_stage TEXT,
        raw_text TEXT,
        summary_md TEXT,
        transcript_source TEXT,
        video_id TEXT,
        audio_path TEXT,
        audio_duration_seconds REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_job_url ON job (url)",
    "CREATE INDEX IF NOT EXISTS ix_job_status ON job (status)",
    "CREATE INDEX IF NOT EXISTS ix_job_created_at ON job (created_at)",
    """
    CREATE TABLE IF NOT EXISTS message (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (job_id) REFERENCES job(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_message_job_id ON message (job_id)",
    "CREATE INDEX IF NOT EXISTS ix_message_created_at ON message (created_at)",
    # FTS5 virtual table over Job rows. content='job' / content_rowid='rowid'
    # makes this an "external content" table — the actual text lives in the
    # base table, FTS only stores the index. Triggers below mirror writes.
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS job_fts USING fts5(
        title,
        raw_text,
        summary_md,
        content='job',
        content_rowid='rowid',
        tokenize='unicode61 remove_diacritics 2'
    )
    """,
    # AI: after an INSERT into job, write the same row into job_fts.
    """
    CREATE TRIGGER IF NOT EXISTS job_ai
    AFTER INSERT ON job
    BEGIN
        INSERT INTO job_fts(rowid, title, raw_text, summary_md)
        VALUES (new.rowid, new.title, new.raw_text, new.summary_md);
    END
    """,
    # AD: before a DELETE on job, send a "delete" command to FTS for that rowid.
    # The FTS5 special-syntax "delete" needs the old values to keep its index
    # consistent.
    """
    CREATE TRIGGER IF NOT EXISTS job_ad
    AFTER DELETE ON job
    BEGIN
        INSERT INTO job_fts(job_fts, rowid, title, raw_text, summary_md)
        VALUES ('delete', old.rowid, old.title, old.raw_text, old.summary_md);
    END
    """,
    # AU: on UPDATE, both delete the old FTS row and insert the new one.
    """
    CREATE TRIGGER IF NOT EXISTS job_au
    AFTER UPDATE ON job
    BEGIN
        INSERT INTO job_fts(job_fts, rowid, title, raw_text, summary_md)
        VALUES ('delete', old.rowid, old.title, old.raw_text, old.summary_md);
        INSERT INTO job_fts(rowid, title, raw_text, summary_md)
        VALUES (new.rowid, new.title, new.raw_text, new.summary_md);
    END
    """,
)


def _migration_v1(conn: Any) -> None:  # noqa: ANN401
    cursor = conn.cursor()
    try:
        for stmt in _V1_STATEMENTS:
            cursor.execute(stmt)
    finally:
        cursor.close()


# ---------------------------------------------------------------------------
# v2 — drop unused FTS5 infrastructure
# ---------------------------------------------------------------------------
# job_fts and its sync triggers were added "for future search" but no search
# endpoint was ever implemented. Drop them to remove dead maintenance overhead.

_V2_STATEMENTS: tuple[str, ...] = (
    "DROP TRIGGER IF EXISTS job_ai",
    "DROP TRIGGER IF EXISTS job_ad",
    "DROP TRIGGER IF EXISTS job_au",
    "DROP TABLE IF EXISTS job_fts",
)


def _migration_v2(conn: Any) -> None:  # noqa: ANN401
    cursor = conn.cursor()
    try:
        for stmt in _V2_STATEMENTS:
            cursor.execute(stmt)
    finally:
        cursor.close()


# ---------------------------------------------------------------------------
# v3 — transcript_language + transcript_translation table
# ---------------------------------------------------------------------------
# Adds the source-language column on Job (populated by Whisper's auto-detect
# via the patched mlx-server response, or by the YouTube caption track's
# language for the fast path), and a separate ``transcript_translation``
# table to cache user-requested translations of a job's transcript.
#
# Legacy rows have no language captured (this column didn't exist yet). We
# backfill them as ``'en'`` per the agreed policy — most existing local
# transcripts are English by accident of the dev environment, and an
# explicit code is better than ``None`` (the UI would otherwise show
# "Original" with no chip-clickable language).

_V3_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE job ADD COLUMN transcript_language TEXT",
    "UPDATE job SET transcript_language = 'en' WHERE transcript_language IS NULL",
    # Translation cache. PK is (job_id, language_code) so a re-POST for the
    # same language idempotently updates the same row instead of inserting.
    # ``status`` flows: queued → running → done | failed. ``text`` is only
    # populated on ``done``; ``error`` only on ``failed``.
    """
    CREATE TABLE IF NOT EXISTS transcript_translation (
        job_id TEXT NOT NULL,
        language_code TEXT NOT NULL,
        status TEXT NOT NULL,
        text TEXT,
        error TEXT,
        progress_percent INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (job_id, language_code),
        FOREIGN KEY (job_id) REFERENCES job(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_transcript_translation_status "
    "ON transcript_translation (status)",
)


def _migration_v3(conn: Any) -> None:  # noqa: ANN401
    cursor = conn.cursor()
    try:
        for stmt in _V3_STATEMENTS:
            cursor.execute(stmt)
    finally:
        cursor.close()


# ---------------------------------------------------------------------------
# v4 — fine-grained transcript segments
# ---------------------------------------------------------------------------
# Adds Job.raw_segments_json so the Transcript-tab UI can render one line
# per actual Whisper segment (typically 1-5 s) instead of the 30 s buckets
# raw_text uses for summary / Q&A. Legacy jobs and PAGE / PDF jobs leave
# this null — the Transcript tab falls back to raw_text in that case.

_V4_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE job ADD COLUMN raw_segments_json TEXT",
)


def _migration_v4(conn: Any) -> None:  # noqa: ANN401
    cursor = conn.cursor()
    try:
        for stmt in _V4_STATEMENTS:
            cursor.execute(stmt)
    finally:
        cursor.close()


# ---------------------------------------------------------------------------
# v5 — alt_media_candidates_json
# ---------------------------------------------------------------------------
# Round-trip storage for the "other playable sources we saw on the page"
# list that the extension's media scanner produces. The sidepanel uses
# this to render a "wrong source?" picker chip when a page has more than
# one candidate (lecture page with 3 talks, news article with embedded
# video + promo, …). Daemon never reads it itself — pure UI state.

_V5_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE job ADD COLUMN alt_media_candidates_json TEXT",
)


def _migration_v5(conn: Any) -> None:  # noqa: ANN401
    cursor = conn.cursor()
    try:
        for stmt in _V5_STATEMENTS:
            cursor.execute(stmt)
    finally:
        cursor.close()


# ---------------------------------------------------------------------------
# Registry + runner
# ---------------------------------------------------------------------------


MIGRATIONS: list[tuple[int, Migration]] = [
    (1, _migration_v1),
    (2, _migration_v2),
    (3, _migration_v3),
    (4, _migration_v4),
    (5, _migration_v5),
]


def _applied_versions(conn: Any) -> set[int]:  # noqa: ANN401
    cursor = conn.cursor()
    try:
        cursor.execute(_MIGRATIONS_TABLE_DDL)
        cursor.execute("SELECT version FROM _migrations")
        return {row[0] for row in cursor.fetchall()}
    finally:
        cursor.close()


def _record_applied(conn: Any, version: int) -> None:  # noqa: ANN401
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT OR IGNORE INTO _migrations (version, applied_at) "
            "VALUES (?, datetime('now'))",
            (version,),
        )
    finally:
        cursor.close()


def run_migrations(engine: Engine) -> list[int]:
    """Apply every registered migration that has not yet been recorded.

    Returns the list of versions that were freshly applied during this call
    (empty if the DB is already up to date).

    Idempotent: running twice on the same engine is a no-op for the second call.
    """
    raw = engine.raw_connection()
    try:
        applied = _applied_versions(raw)
        newly_applied: list[int] = []
        for version, migration in sorted(MIGRATIONS, key=lambda m: m[0]):
            if version in applied:
                continue
            log.info("storage: applying migration v%d", version)
            migration(raw)
            _record_applied(raw, version)
            newly_applied.append(version)
        raw.commit()
        if newly_applied:
            log.info("storage: applied migrations %s", newly_applied)
        else:
            log.debug("storage: no new migrations to apply")
        return newly_applied
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()


__all__ = ["MIGRATIONS", "run_migrations"]
