"""Concurrency tests for ``POST /jobs``.

The real bug we hit: two near-simultaneous clicks on the same URL both saw
"no existing job" and both inserted a row, leaving the Library with
duplicates. The fix was an ``asyncio.Lock`` around dedup-check + create.

These tests fire concurrent posts via ``httpx.AsyncClient`` + ``ASGITransport``
so everything runs on a single event loop — that's the configuration the
lock actually defends.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.main import app
from src.storage.db import dispose_engine, init_engine
from src.storage.migrations import run_migrations
from src.workers import broker as broker_mod
from src.workers import control as control_mod
from src.workers import queue as queue_mod


@pytest.fixture
def stub_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Tear up an isolated DB and replace the background pipeline with a no-op
    so the test focuses on dedup / locking, not on the full extract → summary
    chain."""
    db_path = tmp_path / "race.db"
    engine = init_engine(db_path)
    run_migrations(engine)

    queue_mod.reset_queue()
    broker_mod.reset_broker()
    control_mod.reset_control()

    from src.workers import pipeline as pipeline_mod

    async def _noop_pipeline(*_a: Any, **_k: Any) -> None:
        return

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _noop_pipeline)

    yield

    queue_mod.reset_queue()
    broker_mod.reset_broker()
    control_mod.reset_control()
    dispose_engine()


@pytest.mark.asyncio
async def test_post_jobs_dedupes_concurrent_requests_for_same_url(
    stub_pipeline: Any,
) -> None:
    """Five parallel POSTs for the same URL must collapse to one row."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        url = "https://example.com/race"
        responses = await asyncio.gather(
            *[
                client.post(
                    "/jobs",
                    json={"url": url, "kind": "page", "page_text": "hello"},
                )
                for _ in range(5)
            ]
        )

        for r in responses:
            assert r.status_code == 202, r.text

        ids = {r.json()["id"] for r in responses}
        assert len(ids) == 1, f"expected one dedup'd id, got {ids}"

        # And /jobs?url= confirms the DB has exactly one row.
        listing = await client.get("/jobs", params={"url": url})
        assert listing.json()["total"] == 1


@pytest.mark.asyncio
async def test_post_jobs_concurrent_distinct_urls_create_distinct_rows(
    stub_pipeline: Any,
) -> None:
    """Sanity: the lock must not serialise distinct URLs into one job."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        urls = [f"https://example.com/distinct/{i}" for i in range(4)]
        responses = await asyncio.gather(
            *[
                client.post(
                    "/jobs",
                    json={"url": u, "kind": "page", "page_text": "x"},
                )
                for u in urls
            ]
        )

        for r in responses:
            assert r.status_code == 202

        ids = {r.json()["id"] for r in responses}
        assert len(ids) == 4

        listing = await client.get("/jobs", params={"limit": 50})
        assert listing.json()["total"] == 4
