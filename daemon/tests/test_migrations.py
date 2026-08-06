"""Migration runner tests — verifies schema migrations land and are idempotent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.storage import repo
from src.storage.db import dispose_engine, init_engine
from src.storage.migrations import (
    _MIGRATIONS_TABLE_DDL,
    MIGRATIONS,
    _record_applied,
    run_migrations,
)


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
    assert applied == [1, 2, 3, 4, 5, 6, 7]

    tables = _table_names(fresh_engine)
    for required in ("job", "message", "transcript_translation", "_migrations"):
        assert required in tables, f"missing table {required!r}; got {tables}"

    # v2 dropped FTS5 infrastructure
    assert "job_fts" not in tables
    triggers = _trigger_names(fresh_engine)
    assert not {"job_ai", "job_ad", "job_au"} & triggers

    # v3 added the language column; v4 added raw_segments_json;
    # v5 added alt_media_candidates_json
    raw = fresh_engine.raw_connection()
    try:
        cur = raw.cursor()
        cur.execute("PRAGMA table_info(job)")
        job_cols = {row[1] for row in cur.fetchall()}
        cur.execute("PRAGMA table_info(message)")
        message_cols = {row[1] for row in cur.fetchall()}
    finally:
        raw.close()
    assert "transcript_language" in job_cols
    assert "raw_segments_json" in job_cols
    assert "alt_media_candidates_json" in job_cols
    # v6 added message.frame_refs_json
    assert "frame_refs_json" in message_cols
    # v7 added job.added_at
    assert "added_at" in job_cols


def test_migration_runner_is_idempotent(fresh_engine) -> None:
    first = run_migrations(fresh_engine)
    second = run_migrations(fresh_engine)
    assert first == [1, 2, 3, 4, 5, 6, 7]
    assert second == []  # nothing new to apply

    tables = _table_names(fresh_engine)
    assert "job" in tables
    assert "transcript_translation" in tables
    assert "job_fts" not in tables


# ---------------------------------------------------------------------------
# v6 regression — a DB that already ran v1-v5 (i.e. every real pre-existing
# install) must pick up ONLY v6 on the next start, and the resulting schema
# must actually support the frame_refs write path end to end. A test that
# only ever sees a freshly-created schema (run_migrations once, from empty)
# can't catch a bug where a new column was added to a FROZEN v1 instead of
# its own ALTER-TABLE version — v1 never re-runs on a DB that already
# recorded it, so the column would simply never appear. This is exactly the
# class of bug that slipped through before this test existed: `message` got
# `frame_refs_json` added straight into `_V1_STATEMENTS` instead of a new
# migration, which worked on every fresh test DB (v1 always runs there) and
# broke on every pre-existing one (v1 never runs there again).
# ---------------------------------------------------------------------------


def test_v6_applies_alone_on_a_db_already_at_v5(fresh_engine) -> None:
    """Simulates every real pre-existing daemon install: stop the runner at
    v5 (as if this code predated v6), then upgrade — v6 (and, since this
    fixture stops at v5, v7 right behind it — there's nothing between them
    to stop at) should apply, and frame_refs_json must land without
    touching anything else."""
    v1_through_v5 = [m for m in MIGRATIONS if m[0] <= 5]
    assert [v for v, _ in v1_through_v5] == [1, 2, 3, 4, 5]

    raw = fresh_engine.raw_connection()
    try:
        cur = raw.cursor()
        cur.execute(_MIGRATIONS_TABLE_DDL)
        cur.close()
        for version, migration in v1_through_v5:
            migration(raw)
            _record_applied(raw, version)
        raw.commit()
    finally:
        raw.close()

    # Sanity check: message has no frame_refs_json yet, matching the owner's
    # real (pre-v6) database exactly.
    raw = fresh_engine.raw_connection()
    try:
        cur = raw.cursor()
        cur.execute("PRAGMA table_info(message)")
        cols_before = {row[1] for row in cur.fetchall()}
    finally:
        raw.close()
    assert "frame_refs_json" not in cols_before

    # The upgrade: v6 and v7 are both new from this v5 baseline.
    applied = run_migrations(fresh_engine)
    assert applied == [6, 7]

    raw = fresh_engine.raw_connection()
    try:
        cur = raw.cursor()
        cur.execute("PRAGMA table_info(message)")
        cols_after = {row[1] for row in cur.fetchall()}
    finally:
        raw.close()
    assert "frame_refs_json" in cols_after

    # End-to-end: the actual write path this bug broke (a real daemon dies
    # here with "table message has no column named frame_refs_json" before
    # this fix) must work against the freshly-upgraded schema.
    job = repo.create_job(url="https://x", kind="page")
    frame_refs = [
        {
            "seconds": 12.0,
            "timecode": "00:12",
            "phrase": "this cream",
            "frame_url": f"/jobs/{job.id}/frames/t12/frame_02.jpg",
        }
    ]
    msg = repo.add_message(
        job.id, role="assistant", content="It's ACME cream.", frame_refs=frame_refs
    )
    assert msg.frame_refs_json is not None

    [stored] = repo.list_messages(job.id)
    assert stored.id == msg.id
    assert json.loads(stored.frame_refs_json) == frame_refs


# ---------------------------------------------------------------------------
# v7 regression — same class of bug the v6 test above guards against: a DB
# that already ran v1-v6 (every real pre-existing install as of the
# added_at feature) must pick up ONLY v7 on the next start, AND the backfill
# (added_at = created_at) must land per-row using EACH row's own created_at,
# not one shared value — a naive backfill using a single "now" timestamp
# for every row would silently make every pre-existing job look like it was
# just imported, defeating the entire point of the column.
# ---------------------------------------------------------------------------


def test_v7_backfills_added_at_to_each_rows_own_created_at_on_a_v6_db(
    fresh_engine,
) -> None:
    """Simulates every real pre-existing daemon install: stop the runner at
    v6 (as if this code predated v7), insert several rows with DIFFERENT
    created_at values using only the v1-v6 schema (no added_at column exists
    yet), then upgrade — only v7 should apply, and every row's added_at
    must equal ITS OWN created_at, not a single backfill-time value."""
    v1_through_v6 = [m for m in MIGRATIONS if m[0] <= 6]
    assert [v for v, _ in v1_through_v6] == [1, 2, 3, 4, 5, 6]

    raw = fresh_engine.raw_connection()
    try:
        cur = raw.cursor()
        cur.execute(_MIGRATIONS_TABLE_DDL)
        cur.close()
        for version, migration in v1_through_v6:
            migration(raw)
            _record_applied(raw, version)
        raw.commit()

        # Insert rows with distinct created_at values directly — the v1-v6
        # schema has no added_at column, so this must go through raw SQL
        # rather than the Job ORM model (which already declares added_at).
        cur = raw.cursor()
        rows = [
            ("job-old", "https://old.example", "2010-03-14T00:00:00"),
            ("job-mid", "https://mid.example", "2018-07-04T12:00:00"),
            ("job-new", "https://new.example", "2023-11-30T23:59:59"),
        ]
        for job_id, url, created_at in rows:
            cur.execute(
                "INSERT INTO job (id, url, kind, status, created_at, updated_at) "
                "VALUES (?, ?, 'page', 'done', ?, ?)",
                (job_id, url, created_at, created_at),
            )
        raw.commit()
        cur.close()
    finally:
        raw.close()

    # Sanity check: no added_at column yet, matching a real pre-v7 database.
    raw = fresh_engine.raw_connection()
    try:
        cur = raw.cursor()
        cur.execute("PRAGMA table_info(job)")
        cols_before = {row[1] for row in cur.fetchall()}
    finally:
        raw.close()
    assert "added_at" not in cols_before

    applied = run_migrations(fresh_engine)
    assert applied == [7]

    raw = fresh_engine.raw_connection()
    try:
        cur = raw.cursor()
        cur.execute("PRAGMA table_info(job)")
        cols_after = {row[1] for row in cur.fetchall()}
        cur.execute("SELECT id, created_at, added_at FROM job ORDER BY id")
        by_id = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
    finally:
        raw.close()

    assert "added_at" in cols_after
    for job_id, _url, created_at in rows:
        stored_created_at, stored_added_at = by_id[job_id]
        assert stored_created_at == created_at
        # Backfilled to ITS OWN created_at, not a single shared timestamp.
        assert stored_added_at == created_at


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
