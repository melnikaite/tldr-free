"""Repo-level CRUD tests.

Each test uses a fresh on-disk SQLite database and runs the v1 migration first
so triggers and FTS exist. The default global engine is rebuilt per test
through the ``isolated_db`` fixture.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.storage import repo
from src.storage.db import dispose_engine, init_engine
from src.storage.migrations import run_migrations


@pytest.fixture
def isolated_db(tmp_path: Path):
    db_path = tmp_path / "repo.db"
    engine = init_engine(db_path)
    run_migrations(engine)
    try:
        yield engine
    finally:
        dispose_engine()


# ---------------------------------------------------------------------------
# create / get
# ---------------------------------------------------------------------------


def test_create_job_assigns_id_and_default_status(isolated_db) -> None:
    job = repo.create_job(url="https://example.com", kind="page", title="Example")
    assert isinstance(job.id, str)
    assert len(job.id) == 12
    assert job.status == "running"
    assert job.url == "https://example.com"
    assert job.kind == "page"
    assert job.title == "Example"


def test_get_job_returns_none_for_missing(isolated_db) -> None:
    assert repo.get_job("nope") is None


def test_get_job_round_trip(isolated_db) -> None:
    j = repo.create_job(url="https://x", kind="page")
    found = repo.get_job(j.id)
    assert found is not None
    assert found.id == j.id


# ---------------------------------------------------------------------------
# update_status / mark_done / mark_failed
# ---------------------------------------------------------------------------


def test_update_status_changes_progress_and_error(isolated_db) -> None:
    j = repo.create_job(url="https://x", kind="youtube")
    repo.update_status(j.id, status="queued", progress_stage="downloading")
    out = repo.get_job(j.id)
    assert out is not None
    assert out.status == "queued"
    assert out.progress_stage == "downloading"
    assert out.error is None

    repo.update_status(j.id, status="failed", error="boom")
    out2 = repo.get_job(j.id)
    assert out2 is not None
    assert out2.status == "failed"
    assert out2.error == "boom"


def test_update_status_unknown_id_raises(isolated_db) -> None:
    with pytest.raises(KeyError):
        repo.update_status("missing", status="failed", error="x")


def test_mark_done_persists_summary_and_clears_error(isolated_db) -> None:
    j = repo.create_job(url="https://x", kind="page")
    repo.update_status(j.id, status="running", error="transient")

    repo.mark_done(
        j.id,
        raw_text="hello world",
        summary_md="**summary**",
        transcript_source="page_extract",
        title="The Title",
        duration_seconds=42,
        video_id="abc",
    )

    out = repo.get_job(j.id)
    assert out is not None
    assert out.status == "done"
    assert out.raw_text == "hello world"
    assert out.summary_md == "**summary**"
    assert out.transcript_source == "page_extract"
    assert out.title == "The Title"
    assert out.duration_seconds == 42
    assert out.video_id == "abc"
    assert out.completed_at is not None
    assert out.error is None
    assert out.progress_stage is None


def test_mark_failed_sets_error_and_completed(isolated_db) -> None:
    j = repo.create_job(url="https://x", kind="page")
    repo.mark_failed(j.id, error="kaboom")
    out = repo.get_job(j.id)
    assert out is not None
    assert out.status == "failed"
    assert out.error == "kaboom"
    assert out.completed_at is not None


# ---------------------------------------------------------------------------
# list_jobs filters
# ---------------------------------------------------------------------------


def test_list_jobs_filters_by_status_and_kind(isolated_db) -> None:
    a = repo.create_job(url="https://a", kind="page")
    b = repo.create_job(url="https://b", kind="youtube")
    c = repo.create_job(url="https://c", kind="page")

    repo.mark_done(a.id, raw_text="x", summary_md="y", transcript_source="trafilatura")
    repo.mark_failed(c.id, error="e")
    # `b` stays in 'running'

    rows, total = repo.list_jobs(status="done", limit=50, offset=0)
    assert total == 1
    assert {r.id for r in rows} == {a.id}

    rows, total = repo.list_jobs(status=["failed", "running"], limit=50, offset=0)
    assert {r.id for r in rows} == {b.id, c.id}
    assert total == 2

    rows, total = repo.list_jobs(kind="youtube", limit=50, offset=0)
    assert {r.id for r in rows} == {b.id}
    assert total == 1


def test_list_jobs_since_filter(isolated_db) -> None:
    a = repo.create_job(url="https://a", kind="page")
    b = repo.create_job(url="https://b", kind="page")

    # Move a back in time by patching its created_at directly via the engine.
    raw = isolated_db.raw_connection()
    try:
        cur = raw.cursor()
        old = (datetime.utcnow() - timedelta(days=7)).isoformat()
        cur.execute("UPDATE job SET created_at = ? WHERE id = ?", (old, a.id))
        raw.commit()
    finally:
        raw.close()

    cutoff = datetime.utcnow() - timedelta(days=1)
    rows, total = repo.list_jobs(since=cutoff, limit=50, offset=0)
    assert total == 1
    assert rows[0].id == b.id


def test_list_jobs_pagination_and_total(isolated_db) -> None:
    ids = [repo.create_job(url=f"https://e{i}", kind="page").id for i in range(5)]

    page1, total = repo.list_jobs(limit=2, offset=0)
    page2, _ = repo.list_jobs(limit=2, offset=2)
    page3, _ = repo.list_jobs(limit=2, offset=4)
    assert total == 5
    assert len(page1) == 2
    assert len(page2) == 2
    assert len(page3) == 1
    seen = {r.id for r in page1 + page2 + page3}
    assert seen == set(ids)


# ---------------------------------------------------------------------------
# delete_job
# ---------------------------------------------------------------------------


def test_delete_job_removes_row(isolated_db) -> None:
    j = repo.create_job(url="https://x", kind="page")
    assert repo.delete_job(j.id) is True
    assert repo.get_job(j.id) is None


def test_delete_unknown_returns_false(isolated_db) -> None:
    assert repo.delete_job("not-here") is False


# ---------------------------------------------------------------------------
# find_pending_for_restart
# ---------------------------------------------------------------------------


def test_find_pending_for_restart_picks_queued_and_running(isolated_db) -> None:
    a = repo.create_job(url="https://a", kind="page")            # running
    b = repo.create_job(url="https://b", kind="page")
    c = repo.create_job(url="https://c", kind="page")
    d = repo.create_job(url="https://d", kind="page")

    repo.update_status(b.id, status="queued")
    repo.mark_done(c.id, raw_text="x", summary_md="y", transcript_source="trafilatura")
    repo.mark_failed(d.id, error="e")

    pending = {j.id for j in repo.find_pending_for_restart()}
    assert pending == {a.id, b.id}
