"""Migration runner tests — verifies schema migrations land and are idempotent."""

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


def test_migrations_create_core_tables(fresh_engine) -> None:
    """All migrations applied on a fresh DB produce the expected schema."""
    applied = run_migrations(fresh_engine)
    assert applied == [1, 2]

    tables = _table_names(fresh_engine)
    for required in ("job", "message", "_migrations"):
        assert required in tables, f"missing table {required!r}; got {tables}"

    # v2 dropped FTS5 infrastructure
    assert "job_fts" not in tables
    triggers = _trigger_names(fresh_engine)
    assert not {"job_ai", "job_ad", "job_au"} & triggers


def test_migration_runner_is_idempotent(fresh_engine) -> None:
    first = run_migrations(fresh_engine)
    second = run_migrations(fresh_engine)
    assert first == [1, 2]
    assert second == []  # nothing new to apply

    tables = _table_names(fresh_engine)
    assert "job" in tables
    assert "job_fts" not in tables


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
