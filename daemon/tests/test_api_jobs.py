"""Integration tests for /jobs and /ai endpoints against a fresh in-memory DB.

Uses ``TestClient`` and seeds an isolated SQLite per test. External services
(LLM, youtube_transcript_api, trafilatura, the Whisper worker) are mocked so
tests stay hermetic.

Note: POST /jobs is now ASYNC — it returns 202 with the new id and runs the
extraction + summary in a background task. Tests that need the final state
either:
  - poll GET /jobs/{id} until status transitions, or
  - subscribe to POST /ai/stream {job_id} and read events.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.storage.db import dispose_engine, init_engine
from src.storage.migrations import run_migrations


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test client with mocked external services.

    - LLM stream_summarize is patched to yield a deterministic markdown.
    - LLM qa.stream_answer is patched to yield a deterministic answer.
    - youtube fetch_transcript_with_retry is patched to fail with a permanent
      error so YouTube paths take the deferred branch (no real network calls).
    - trafilatura is patched to return deterministic text.
    - The whisper queue worker AND the retention worker are replaced with
      no-op coroutines so the lifespan starts cleanly.
    """
    db_path = tmp_path / "api.db"
    engine = init_engine(db_path)
    run_migrations(engine)

    # 1) llm.summary.stream_summarize → fake stream
    from src.llm import summary as llm_summary

    async def _fake_stream_summarize(
        text: str, *, title: Any, output_language: str
    ) -> AsyncIterator[str]:
        yield f"## summary\n\nfor {len(text)} chars in {output_language}"

    monkeypatch.setattr(llm_summary, "stream_summarize", _fake_stream_summarize)

    # 2) llm.qa.stream_answer → fake stream
    from src.llm import qa as llm_qa

    async def _fake_qa_stream(
        *, job: Any, question: str, output_language: str
    ) -> AsyncIterator[str]:
        yield f"answer to {question!r} in {output_language}"

    monkeypatch.setattr(llm_qa, "stream_answer", _fake_qa_stream)

    # 3) Permanent transcript error → YouTube paths defer to whisper queue.
    from src.workers import youtube as yt_worker
    from src.workers.errors import PermanentTranscriptError

    async def _fake_fetch(*, video_id, cookies, max_attempts, backoff_seconds):  # noqa: ANN001
        raise PermanentTranscriptError("test mode: no transcript")

    monkeypatch.setattr(yt_worker, "fetch_transcript_with_retry", _fake_fetch)

    # 3b) yt-dlp subtitle fallback returns nothing in tests, so the pipeline
    # falls all the way through to the whisper queue branch.
    async def _fake_subs(*, url, cookies, dir, lang_preferences):  # noqa: ANN001
        return None

    monkeypatch.setattr(yt_worker, "download_subtitles", _fake_subs)

    # 4) Trafilatura.
    from src.workers import page as page_worker

    async def _fake_extract(url: str) -> tuple[str | None, str]:
        return (None, "extracted page text")

    monkeypatch.setattr(page_worker, "extract_with_trafilatura", _fake_extract)

    # 5) Whisper worker → no-op so lifespan can spin up cleanly.
    from src.workers import runner as runner_mod

    async def _noop_worker(queue, repo_module):  # noqa: ANN001
        try:
            while True:
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            return

    monkeypatch.setattr(runner_mod, "whisper_worker", _noop_worker)
    import src.main as main_mod

    monkeypatch.setattr(main_mod, "whisper_worker", _noop_worker)

    # 6) Retention worker → no-op so it doesn't run during tests.
    from src.workers import retention as retention_mod

    async def _noop_retention() -> None:
        try:
            while True:
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            return

    monkeypatch.setattr(retention_mod, "retention_worker", _noop_retention)
    monkeypatch.setattr(main_mod, "retention_worker", _noop_retention)

    # 7) Reset queue + broker + workers control singletons between tests.
    from src.workers import broker as broker_mod
    from src.workers import control as control_mod
    from src.workers import queue as queue_mod

    queue_mod.reset_queue()
    broker_mod.reset_broker()
    control_mod.reset_control()

    with TestClient(app) as c:
        yield c
    dispose_engine()
    queue_mod.reset_queue()
    broker_mod.reset_broker()
    control_mod.reset_control()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_until_done(client: TestClient, job_id: str, *, timeout: float = 5.0) -> dict:
    """Poll GET /jobs/{id} until status is done|failed (or timeout)."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/jobs/{job_id}")
        body = r.json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(
        f"job {job_id} did not reach done/failed within {timeout}s; last={body}"
    )


# ---------------------------------------------------------------------------
# POST /jobs is async — returns 202 immediately with running/queued status
# ---------------------------------------------------------------------------


def test_post_jobs_returns_202_with_running_status(client: TestClient) -> None:
    r = client.post(
        "/jobs",
        json={"url": "https://example.com", "kind": "page", "page_text": "hello"},
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["kind"] == "page"
    assert body["status"] == "running"
    assert isinstance(body["id"], str) and len(body["id"]) == 12

    # Background pipeline finishes the job; we can poll for the final state.
    final = _wait_until_done(client, body["id"])
    assert final["status"] == "done"
    assert final["transcript_source"] == "page_extract"
    assert final["summary_md"] is not None
    assert "summary" in final["summary_md"].lower()


def test_post_jobs_page_without_text_uses_trafilatura(client: TestClient) -> None:
    r = client.post(
        "/jobs",
        json={"url": "https://example.com", "kind": "page"},
    )
    assert r.status_code == 202, r.text
    body = r.json()

    final = _wait_until_done(client, body["id"])
    assert final["status"] == "done"
    assert final["transcript_source"] == "trafilatura"


def test_post_jobs_youtube_without_transcript_defers(client: TestClient) -> None:
    """YouTube path: fake fetch raises PermanentTranscriptError →
    pipeline puts the job in the whisper queue (status=queued)."""
    r = client.post(
        "/jobs",
        json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "kind": "auto"},
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["kind"] == "youtube"

    # Wait briefly for the pipeline to push to the queue.
    import time

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        detail = client.get(f"/jobs/{body['id']}").json()
        if detail["status"] == "queued":
            break
        time.sleep(0.05)
    assert detail["status"] == "queued", detail


def test_get_jobs_filters_and_total(client: TestClient) -> None:
    a = client.post(
        "/jobs",
        json={"url": "https://example.com/a", "kind": "page"},
    ).json()
    client.post("/jobs", json={"url": "https://example.com/b", "kind": "page"})
    client.post(
        "/jobs",
        json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "kind": "youtube"},
    )

    # Wait for jobs to settle (page jobs go done; YouTube goes queued).
    for jid in [a["id"]]:
        _wait_until_done(client, jid)

    r = client.get("/jobs?kind=page")
    body = r.json()
    assert body["total"] == 2
    assert all(item["kind"] == "page" for item in body["items"])


def test_get_job_detail_includes_raw_text_length(client: TestClient) -> None:
    create = client.post(
        "/jobs",
        json={"url": "https://x", "kind": "page", "page_text": "abcdef"},
    ).json()

    final = _wait_until_done(client, create["id"])
    assert final["raw_text_length"] == 6
    assert final["video_id"] is None


def test_delete_job_204_then_404(client: TestClient) -> None:
    create = client.post(
        "/jobs", json={"url": "https://x", "kind": "page", "page_text": "x"}
    ).json()
    _wait_until_done(client, create["id"])
    r = client.delete(f"/jobs/{create['id']}")
    assert r.status_code == 204

    r = client.delete(f"/jobs/{create['id']}")
    assert r.status_code == 404


def test_get_job_404(client: TestClient) -> None:
    r = client.get("/jobs/does-not-exist")
    assert r.status_code == 404


def test_list_filters_by_exact_url(client: TestClient) -> None:
    """Extension uses ?url= to find the cached job for the current tab."""
    target = "https://example.com/article-x"
    other = "https://example.com/article-y"
    j = client.post(
        "/jobs", json={"url": target, "kind": "page", "page_text": "first"}
    ).json()
    client.post("/jobs", json={"url": other, "kind": "page", "page_text": "z"})
    _wait_until_done(client, j["id"])

    r = client.get("/jobs", params={"url": target})
    body = r.json()
    assert body["total"] == 1
    assert body["items"][0]["id"] == j["id"]

    r = client.get("/jobs", params={"url": "https://nowhere"})
    body = r.json()
    assert body["total"] == 0


def test_post_jobs_returns_existing_for_same_url(client: TestClient) -> None:
    """POST /jobs is dedup'd: re-submitting the same URL returns the existing job."""
    payload = {"url": "https://example.com/dup", "kind": "page", "page_text": "first"}
    a = client.post("/jobs", json=payload).json()
    _wait_until_done(client, a["id"])

    # Same URL again — should return the existing id, not create a new row.
    b = client.post(
        "/jobs",
        json={**payload, "page_text": "second (would be ignored)"},
    ).json()
    assert b["id"] == a["id"]

    r = client.get("/jobs", params={"url": payload["url"]})
    assert r.json()["total"] == 1


# ---------------------------------------------------------------------------
# /ai/stream — summary + QA modes
# ---------------------------------------------------------------------------


def _read_sse_events(response) -> list[dict]:
    """Parse a TestClient streaming response body into a list of SSE event dicts."""
    out: list[dict] = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: "):].strip()
            if payload:
                out.append(json.loads(payload))
    return out


def test_ai_stream_summary_replays_cached_done(client: TestClient) -> None:
    """When the job is already done, /ai/stream replays summary_md as one
    delta + done — no broker subscription needed."""
    create = client.post(
        "/jobs", json={"url": "https://x", "kind": "page", "page_text": "hello"}
    ).json()
    _wait_until_done(client, create["id"])

    r = client.post("/ai/stream", json={"job_id": create["id"]})
    assert r.status_code == 200
    events = _read_sse_events(r)
    types = [e["type"] for e in events]
    assert types == ["delta", "done"]
    assert "summary" in events[0]["delta"].lower()
    assert events[1]["content"] == events[0]["delta"]


def test_ai_stream_qa_persists_messages(client: TestClient) -> None:
    """QA mode persists user + assistant messages, emits done with message_id."""
    create = client.post(
        "/jobs", json={"url": "https://x", "kind": "page", "page_text": "hello"}
    ).json()
    _wait_until_done(client, create["id"])

    r = client.post(
        "/ai/stream",
        json={"job_id": create["id"], "question": "What is this about?"},
    )
    assert r.status_code == 200
    events = _read_sse_events(r)
    types = [e["type"] for e in events]
    assert "stage" in types and "delta" in types and "done" in types
    done = events[-1]
    assert done["type"] == "done"
    assert done["message_id"] is not None
    assert "What is this about?" in done["content"] or "answer" in done["content"]

    # GET /jobs/{id}/messages returns both user + assistant rows.
    msgs = client.get(f"/jobs/{create['id']}/messages").json()
    assert len(msgs["items"]) == 2
    assert msgs["items"][0]["role"] == "user"
    assert msgs["items"][0]["content"] == "What is this about?"
    assert msgs["items"][1]["role"] == "assistant"
    assert msgs["items"][1]["id"] == done["message_id"]


def test_ai_stream_404_for_unknown_job(client: TestClient) -> None:
    r = client.post("/ai/stream", json={"job_id": "nope"})
    assert r.status_code == 404


