"""Every state-changing repo function publishes a ``job_event`` to the global
event broker — that's the contract the Library and Side panel rely on for
real-time updates without polling.

These tests existed in spirit before but lived implicitly inside
pipeline/runner. After centralising emit inside ``repo`` we test it once,
here, against the real broker.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from src.storage import repo
from src.storage.db import dispose_engine, init_engine
from src.storage.migrations import run_migrations
from src.workers import broker as broker_mod
from src.workers.broker import get_event_broker


@pytest.fixture
def isolated_db(tmp_path: Path) -> Any:
    db_path = tmp_path / "repo_emit.db"
    engine = init_engine(db_path)
    run_migrations(engine)
    broker_mod.reset_broker()
    try:
        yield engine
    finally:
        dispose_engine()
        broker_mod.reset_broker()


def _drain(queue: asyncio.Queue[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pull every event currently in the queue without blocking."""
    out: list[dict[str, Any]] = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


# ---------------------------------------------------------------------------
# create_job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_job_emits_created_event(isolated_db: Any) -> None:
    queue = get_event_broker().subscribe()

    job = repo.create_job(
        url="https://example.com",
        kind="page",
        title="Hello",
        progress_stage="extracting",
    )

    event = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert event["type"] == "job"
    assert event["action"] == "created"
    payload = event["job"]
    assert payload["id"] == job.id
    assert payload["url"] == "https://example.com"
    assert payload["kind"] == "page"
    assert payload["title"] == "Hello"
    assert payload["status"] == "running"
    assert payload["progress_stage"] == "extracting"


# ---------------------------------------------------------------------------
# update_status / mark_done / mark_failed / set_extracted / reset_for_retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_status_emits_updated_event(isolated_db: Any) -> None:
    job = repo.create_job(url="https://x", kind="page")
    queue = get_event_broker().subscribe()

    repo.update_status(job.id, status="queued", progress_stage="downloading")

    event = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert event["type"] == "job"
    assert event["action"] == "updated"
    assert event["job"]["status"] == "queued"
    assert event["job"]["progress_stage"] == "downloading"


@pytest.mark.asyncio
async def test_mark_done_emits_updated_with_final_title(isolated_db: Any) -> None:
    """Real bug we hit: title written by mark_done didn't reach the Library
    until refresh. With emit inside repo this is now table-stakes."""
    job = repo.create_job(url="https://x", kind="youtube", title="placeholder-id")
    queue = get_event_broker().subscribe()

    repo.mark_done(
        job.id,
        raw_text="text",
        summary_md="**summary**",
        transcript_source="whisper",
        title="The Real YouTube Title",
        video_id="abcdef12345",
    )

    event = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert event["action"] == "updated"
    assert event["job"]["status"] == "done"
    assert event["job"]["title"] == "The Real YouTube Title"


@pytest.mark.asyncio
async def test_set_extracted_emits_updated_with_canonical_title(isolated_db: Any) -> None:
    """The path that surfaces yt-dlp's canonical title to the Library
    mid-pipeline (before mark_done)."""
    job = repo.create_job(url="https://yt", kind="youtube", title="placeholder")
    queue = get_event_broker().subscribe()

    repo.set_extracted(
        job.id,
        raw_text="...",
        transcript_source="youtube_api",
        title="Canonical YT Title",
        video_id="vid12345",
    )

    event = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert event["action"] == "updated"
    assert event["job"]["title"] == "Canonical YT Title"


@pytest.mark.asyncio
async def test_mark_failed_emits_updated_event(isolated_db: Any) -> None:
    job = repo.create_job(url="https://x", kind="page")
    queue = get_event_broker().subscribe()

    repo.mark_failed(job.id, error="kaboom")

    event = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert event["action"] == "updated"
    assert event["job"]["status"] == "failed"


@pytest.mark.asyncio
async def test_reset_for_retry_emits_updated_event(isolated_db: Any) -> None:
    job = repo.create_job(url="https://x", kind="page")
    repo.mark_failed(job.id, error="x")
    queue = get_event_broker().subscribe()  # subscribe AFTER mark_failed to skip its event

    repo.reset_for_retry(job.id)

    event = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert event["action"] == "updated"
    assert event["job"]["status"] == "running"
    assert event["job"]["progress_stage"] == "extracting"


# ---------------------------------------------------------------------------
# delete_job / delete_jobs_older_than
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_job_emits_deleted_event(isolated_db: Any) -> None:
    job = repo.create_job(url="https://x", kind="page")
    queue = get_event_broker().subscribe()

    deleted = repo.delete_job(job.id)
    assert deleted is True

    event = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert event["type"] == "job"
    assert event["action"] == "deleted"
    assert event["job"]["id"] == job.id


@pytest.mark.asyncio
async def test_delete_jobs_older_than_emits_one_event_per_row(isolated_db: Any) -> None:
    a = repo.create_job(url="https://a", kind="page")
    b = repo.create_job(url="https://b", kind="page")
    repo.create_job(url="https://c", kind="page")  # stays

    # Backdate a + b so they're caught by the cutoff.
    raw = isolated_db.raw_connection()
    try:
        cur = raw.cursor()
        old = (datetime.utcnow() - timedelta(days=10)).isoformat()
        cur.execute("UPDATE job SET created_at=? WHERE id IN (?, ?)", (old, a.id, b.id))
        raw.commit()
    finally:
        raw.close()

    queue = get_event_broker().subscribe()

    n = repo.delete_jobs_older_than(datetime.utcnow() - timedelta(days=1))
    assert n == 2

    # Two deleted events, in the order they were processed.
    await asyncio.sleep(0)  # let publishes settle
    events = _drain(queue)
    assert {e["job"]["id"] for e in events} == {a.id, b.id}
    assert all(e["action"] == "deleted" for e in events)


# ---------------------------------------------------------------------------
# set_audio is intentionally silent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_audio_does_not_emit(isolated_db: Any) -> None:
    """``audio_path`` is internal plumbing the UI doesn't render — the global
    stream stays quiet to avoid waking every Library/sidebar tab on every
    download/cleanup."""
    job = repo.create_job(url="https://x", kind="youtube")
    queue = get_event_broker().subscribe()

    repo.set_audio(job.id, audio_path="/tmp/foo.opus", audio_duration_seconds=42.0)
    repo.set_audio(job.id, audio_path=None)

    await asyncio.sleep(0)
    assert queue.empty()


# ---------------------------------------------------------------------------
# Broker resilience: a write must not roll back if publish fails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_succeeds_even_if_broker_publish_raises(
    isolated_db: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A misbehaving subscriber must not block ``mark_done`` from completing."""
    job = repo.create_job(url="https://x", kind="page")

    def boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("subscriber went south")

    monkeypatch.setattr(get_event_broker(), "publish", boom)

    repo.mark_done(
        job.id, raw_text="t", summary_md="s", transcript_source="page_extract"
    )

    out = repo.get_job(job.id)
    assert out is not None
    assert out.status == "done"
