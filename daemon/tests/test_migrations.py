"""Migration runner tests — verifies v1 schema lands and is idempotent."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.storage.db import dispose_engine, init_engine
from src.storage.migrations import run_migrations


@pytest.fixture
def fresh_engine(tmp_path: Path):
    """Build an isolated SQLite engine in a temp file. Disposes on teardown."""
    db_path = tmp_path / "test.db"
    engine = init_engine(db_path)
    try:
        yield engine
    finally:
        dispose_engine()


def _table_names(engine) -> set[str]:
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
        return {row[0] for row in cur.fetchall()}
    finally:
        raw.close()


def _trigger_names(engine) -> set[str]:
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
        return {row[0] for row in cur.fetchall()}
    finally:
        raw.close()


def test_v1_creates_tables_fts_and_triggers(fresh_engine) -> None:
    applied = run_migrations(fresh_engine)
    assert applied == [1]

    tables = _table_names(fresh_engine)
    # Core tables
    for required in ("job", "message", "_migrations", "job_fts"):
        assert required in tables, f"missing table {required!r}; got {tables}"

    triggers = _trigger_names(fresh_engine)
    assert {"job_ai", "job_ad", "job_au"} <= triggers


def test_migration_runner_is_idempotent(fresh_engine) -> None:
    first = run_migrations(fresh_engine)
    second = run_migrations(fresh_engine)
    assert first == [1]
    assert second == []  # nothing new to apply

    # And the schema should still be intact
    tables = _table_names(fresh_engine)
    assert "job" in tables and "job_fts" in tables


def test_pragmas_are_applied(fresh_engine) -> None:
    """journal_mode=WAL etc. must hold after migrations run."""
    run_migrations(fresh_engine)
    raw = fresh_engine.raw_connection()
    try:
        cur = raw.cursor()
        cur.execute("PRAGMA journal_mode")
        mode = cur.fetchone()[0]
        # WAL is the goal, but in-memory DBs collapse to "memory". For file-
        # backed test DBs we expect WAL.
        assert str(mode).lower() == "wal"

        cur.execute("PRAGMA foreign_keys")
        assert int(cur.fetchone()[0]) == 1
    finally:
        raw.close()


def test_fts_triggers_mirror_job_writes(fresh_engine) -> None:
    """Insert/update/delete on job propagate into job_fts via the AI/AU/AD triggers."""
    run_migrations(fresh_engine)
    raw = fresh_engine.raw_connection()
    try:
        cur = raw.cursor()
        # Insert a job row directly via raw SQL to keep the test focused on triggers.
        cur.execute(
            """
            INSERT INTO job (id, url, kind, status, title, created_at, updated_at,
                             raw_text, summary_md)
            VALUES ('abc123def456', 'https://x', 'page', 'done', 'Hello world',
                    datetime('now'), datetime('now'), 'a quick brown fox', 'tldr summary')
            """
        )
        raw.commit()

        cur.execute("SELECT count(*) FROM job_fts WHERE job_fts MATCH 'fox'")
        assert cur.fetchone()[0] == 1

        # Update title — should re-index
        cur.execute("UPDATE job SET title = 'Different title' WHERE id = 'abc123def456'")
        raw.commit()
        cur.execute("SELECT count(*) FROM job_fts WHERE job_fts MATCH 'Different'")
        assert cur.fetchone()[0] == 1

        # Delete — should remove from FTS
        cur.execute("DELETE FROM job WHERE id = 'abc123def456'")
        raw.commit()
        cur.execute("SELECT count(*) FROM job_fts WHERE job_fts MATCH 'fox'")
        assert cur.fetchone()[0] == 0
    finally:
        raw.close()
