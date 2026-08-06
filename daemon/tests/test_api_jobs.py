"""Integration tests for /jobs and /ai endpoints against a fresh in-memory DB.

Uses ``TestClient`` and seeds an isolated SQLite per test. External services
(LLM, youtube_transcript_api, trafilatura, the Whisper worker) are mocked so
tests stay hermetic.

Note: POST /jobs is now ASYNC — it returns 202 with the new id and runs the
extraction + summary in a background task. Tests that need the final state
either:
  - poll GET /jobs/{id} until status transitions, or
  - subscribe to POST /ai/qa {job_id, question} and read events.
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
        text: str, *, title: Any, output_language: str, from_audio_transcript: bool = False
    ) -> AsyncIterator[str]:
        yield f"## summary\n\nfor {len(text)} chars in {output_language}"

    monkeypatch.setattr(llm_summary, "stream_summarize", _fake_stream_summarize)

    # 2) llm.qa.stream_answer → fake stream
    from src.llm import qa as llm_qa

    async def _fake_qa_stream(
        *, job: Any, question: str, output_language: str, from_audio: bool
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

    # 4) DuckDuckGo search — return a deterministic result so tests never hit the network.
    from src.workers import search as search_mod

    async def _fake_ddg_search(query: str, max_results: int = 5) -> list[dict]:  # noqa: ANN001
        return [{"title": "Fake result", "href": "https://example.com/result", "body": query}]

    monkeypatch.setattr(search_mod, "ddg_search", _fake_ddg_search)

    # 5) Trafilatura.
    from src.workers import page as page_worker

    async def _fake_extract(url: str) -> tuple[str | None, str]:
        return (None, "extracted page text")

    monkeypatch.setattr(page_worker, "extract_with_trafilatura", _fake_extract)

    # 6) Whisper worker → no-op so lifespan can spin up cleanly.
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

    # 7) Retention worker → no-op so it doesn't run during tests.
    from src.workers import retention as retention_mod

    async def _noop_retention() -> None:
        try:
            while True:
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            return

    monkeypatch.setattr(retention_mod, "retention_worker", _noop_retention)
    monkeypatch.setattr(main_mod, "retention_worker", _noop_retention)

    # 8) Reset queue + broker + workers control singletons between tests.
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


def test_get_job_detail_has_no_raw_text_length(client: TestClient) -> None:
    create = client.post(
        "/jobs",
        json={"url": "https://x", "kind": "page", "page_text": "abcdef"},
    ).json()

    final = _wait_until_done(client, create["id"])
    assert "raw_text_length" not in final
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


def test_get_transcript_serves_original_text(client: TestClient) -> None:
    """Without ``?lang=`` the endpoint returns Job.raw_text directly."""
    create = client.post(
        "/jobs",
        json={"url": "https://x/transcript", "kind": "page", "page_text": "hello world"},
    ).json()
    _wait_until_done(client, create["id"])

    r = client.get(f"/jobs/{create['id']}/transcript")
    assert r.status_code == 200
    body = r.json()
    assert body["is_original"] is True
    assert "hello" in body["text"].lower()


def test_get_transcript_404_for_unknown_job(client: TestClient) -> None:
    r = client.get("/jobs/does-not-exist/transcript")
    assert r.status_code == 404


def test_get_transcript_unknown_language_404(client: TestClient) -> None:
    """No cached translation row at all → 404; the caller is expected to POST
    the translate endpoint to start one. In-flight translations come back as
    ``is_pending: true`` (see ``test_get_transcript_pending_when_translating``
    over in the translator tests)."""
    create = client.post(
        "/jobs",
        json={"url": "https://x/tx-ru", "kind": "page", "page_text": "english body"},
    ).json()
    _wait_until_done(client, create["id"])

    r = client.get(f"/jobs/{create['id']}/transcript", params={"lang": "ru"})
    assert r.status_code == 404


def test_get_transcript_serves_segments_when_available(client: TestClient) -> None:
    """When ``Job.raw_segments_json`` is populated (Whisper / YouTube fast
    path with our v4 schema), the endpoint serves fine-grained text — one
    line per segment — instead of the 30 s buckets in ``raw_text``. This is
    the key payoff of Fix C / Fix I; without it the Transcript tab would
    show coarse buckets even when the data is finer."""
    from src.storage import repo as repo_module

    create = client.post(
        "/jobs",
        json={"url": "https://x/finesegs", "kind": "page", "page_text": "x"},
    ).json()
    _wait_until_done(client, create["id"])

    # Force a fine-grained segments payload onto the row + set source lang.
    import json as _json
    segments = [
        {"start": 0.0, "end": 1.5, "text": "first"},
        {"start": 1.5, "end": 3.0, "text": "second"},
        {"start": 3.0, "end": 4.5, "text": "third"},
    ]
    repo_module.set_extracted(
        create["id"],
        raw_text="[00:00] coarse bucket\n",
        transcript_source="whisper",
        transcript_language="en",
        raw_segments_json=_json.dumps(segments),
    )

    r = client.get(f"/jobs/{create['id']}/transcript")
    assert r.status_code == 200
    body = r.json()
    # The fine-grained text won the source preference.
    assert "first" in body["text"]
    assert "second" in body["text"]
    assert "third" in body["text"]
    # Three lines from three segments — not one coarse line.
    lines = [ln for ln in body["text"].splitlines() if ln.strip()]
    assert len(lines) == 3


def test_get_transcript_falls_back_to_raw_text_without_segments(client: TestClient) -> None:
    """Legacy jobs and PAGE/PDF jobs have no raw_segments_json — the
    endpoint must still serve raw_text rather than 404."""
    create = client.post(
        "/jobs",
        json={"url": "https://x/legacy", "kind": "page", "page_text": "body content"},
    ).json()
    _wait_until_done(client, create["id"])

    r = client.get(f"/jobs/{create['id']}/transcript")
    assert r.status_code == 200
    body = r.json()
    assert body["is_original"] is True
    assert body["is_pending"] is False
    # PAGE jobs have raw_text equal to the input text — verify we got it.
    assert "body content" in body["text"].lower() or body["text"]


def test_get_transcript_pending_when_raw_text_missing(client: TestClient) -> None:
    """Job exists but raw_text not yet saved → 200 with is_pending=true so
    the Transcript tab shows a placeholder instead of an error toast."""
    # Submit a YouTube job that defers to Whisper (no page_text). The
    # background pipeline will go into "queued" / "running" without
    # raw_text immediately. We don't wait for it to finish — we hit the
    # endpoint while raw_text is still empty.
    from src.storage import repo as repo_module

    job = repo_module.create_job(
        url="https://example.com/pending",
        kind="page",
        title="Pending",
    )
    # ``create_job`` leaves raw_text=None — perfect for this test.

    r = client.get(f"/jobs/{job.id}/transcript")
    assert r.status_code == 200
    body = r.json()
    assert body["is_pending"] is True
    assert body["text"] is None
    assert body["is_original"] is True


def test_post_jobs_persists_alt_media_candidates(client: TestClient) -> None:
    """When the extension finds multiple media sources on a page (lecture
    page with 3 talks, news article with promo + main video, etc.) the
    primary one is used as ``media_url`` and the rest ride along on the
    job. GET /jobs/{id} surfaces them under ``alt_media_candidates`` so
    the sidepanel can render the "wrong source?" picker."""
    from unittest.mock import patch

    # Don't actually run yt-dlp — we only care that POST/GET round-trips
    # the alt-candidates list. The pipeline can fail; the field is
    # written before the pipeline ever starts.
    with patch(
        "src.workers.pipeline.run_pipeline",
        new=lambda *a, **kw: __import__("asyncio").sleep(0),
    ):
        r = client.post(
            "/jobs",
            json={
                "url": "https://lecture.example/talks",
                "kind": "media",
                "media_url": "https://lecture.example/talk-1.mp4",
                "alt_media_candidates": [
                    {
                        "media_url": "https://lecture.example/talk-2.mp4",
                        "kind": "video",
                        "label": "Talk 2",
                    },
                    {
                        "media_url": "https://lecture.example/talk-3.mp4",
                        "kind": "video",
                        "label": "Talk 3",
                    },
                ],
            },
        )
        assert r.status_code == 202, r.text
        job_id = r.json()["id"]

        detail = client.get(f"/jobs/{job_id}").json()
        assert "alt_media_candidates" in detail
        alts = detail["alt_media_candidates"]
        assert len(alts) == 2
        assert {a["media_url"] for a in alts} == {
            "https://lecture.example/talk-2.mp4",
            "https://lecture.example/talk-3.mp4",
        }
        assert all(a["kind"] == "video" for a in alts)
        assert {a["label"] for a in alts} == {"Talk 2", "Talk 3"}


def test_get_job_returns_empty_alt_media_for_legacy_rows(client: TestClient) -> None:
    """Jobs created before v5 (or jobs from pages with a single source)
    must return an empty list — not null — so the sidepanel can rely on
    ``.length`` without an extra null check."""
    r = client.post(
        "/jobs",
        json={"url": "https://example.com/single", "kind": "page", "page_text": "hi"},
    )
    job_id = r.json()["id"]
    _wait_until_done(client, job_id)

    detail = client.get(f"/jobs/{job_id}").json()
    assert detail["alt_media_candidates"] == []


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
# /ai/qa — Q&A mode
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


def test_ai_qa_persists_messages(client: TestClient) -> None:
    """QA mode persists user + assistant messages, emits done with message_id."""
    create = client.post(
        "/jobs", json={"url": "https://x", "kind": "page", "page_text": "hello"}
    ).json()
    _wait_until_done(client, create["id"])

    r = client.post(
        "/ai/qa",
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


def test_ai_qa_strips_degeneration_tail(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model that finishes then loops on <br> filler has the tail cut and the
    stored message cleaned — only the real answer is persisted."""
    from src.llm import qa as llm_qa

    async def _degen_stream(*, job: Any, question: str, output_language: str, from_audio: bool):
        yield "Real answer."
        for _ in range(8):
            yield "\n<br>"

    monkeypatch.setattr(llm_qa, "stream_answer", _degen_stream)

    create = client.post(
        "/jobs", json={"url": "https://degen", "kind": "page", "page_text": "hello"}
    ).json()
    _wait_until_done(client, create["id"])

    r = client.post("/ai/qa", json={"job_id": create["id"], "question": "q"})
    assert r.status_code == 200

    msgs = client.get(f"/jobs/{create['id']}/messages").json()
    assistant = msgs["items"][-1]
    assert assistant["role"] == "assistant"
    assert "<br>" not in assistant["content"]
    assert assistant["content"] == "Real answer."


def test_ai_qa_empty_answer_errors(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A whitespace-only answer yields an error and persists no assistant row."""
    from src.llm import qa as llm_qa

    async def _empty_stream(*, job: Any, question: str, output_language: str, from_audio: bool):
        yield "   "

    monkeypatch.setattr(llm_qa, "stream_answer", _empty_stream)

    create = client.post(
        "/jobs", json={"url": "https://empty", "kind": "page", "page_text": "hello"}
    ).json()
    _wait_until_done(client, create["id"])

    r = client.post("/ai/qa", json={"job_id": create["id"], "question": "q"})
    assert r.status_code == 200
    events = _read_sse_events(r)
    assert any(e["type"] == "error" for e in events)

    # The user question is persisted; no assistant row for an empty answer.
    msgs = client.get(f"/jobs/{create['id']}/messages").json()
    assert len(msgs["items"]) == 1
    assert msgs["items"][0]["role"] == "user"


def test_ai_qa_strips_timecodes_for_document_source(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page/PDF job has no real timecodes — any [MM:SS] the model hallucinates
    must be stripped deterministically before the answer is persisted."""
    from src.llm import qa as llm_qa

    async def _fake_stream(*, job: Any, question: str, output_language: str, from_audio: bool):
        assert from_audio is False
        yield "The answer [01:30] is here."

    monkeypatch.setattr(llm_qa, "stream_answer", _fake_stream)

    create = client.post(
        "/jobs", json={"url": "https://doc", "kind": "page", "page_text": "hello"}
    ).json()
    _wait_until_done(client, create["id"])

    r = client.post("/ai/qa", json={"job_id": create["id"], "question": "q"})
    assert r.status_code == 200

    msgs = client.get(f"/jobs/{create['id']}/messages").json()
    assistant = msgs["items"][-1]
    assert assistant["role"] == "assistant"
    assert "01:30" not in assistant["content"]
    assert "[" not in assistant["content"]


def test_ai_qa_keeps_timecodes_for_audio_source(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transcript-backed job (YouTube/Whisper) legitimately has [MM:SS]
    markers in the material — they must survive the QA answer untouched."""
    from src.llm import qa as llm_qa
    from src.storage.db import Job as JobRow
    from src.storage.db import session_scope

    async def _fake_stream(*, job: Any, question: str, output_language: str, from_audio: bool):
        assert from_audio is True
        yield "The answer [01:30] is here."

    monkeypatch.setattr(llm_qa, "stream_answer", _fake_stream)

    create = client.post(
        "/jobs", json={"url": "https://audio", "kind": "page", "page_text": "hello"}
    ).json()
    _wait_until_done(client, create["id"])

    # Flip the persisted transcript_source to an audio one to simulate a
    # YouTube/Whisper-derived job without re-running the whole pipeline.
    with session_scope() as session:
        row = session.get(JobRow, create["id"])
        row.transcript_source = "whisper"
        session.add(row)

    r = client.post("/ai/qa", json={"job_id": create["id"], "question": "q"})
    assert r.status_code == 200

    msgs = client.get(f"/jobs/{create['id']}/messages").json()
    assistant = msgs["items"][-1]
    assert assistant["role"] == "assistant"
    assert "[01:30]" in assistant["content"]


def test_ai_qa_404_for_unknown_job(client: TestClient) -> None:
    r = client.post("/ai/qa", json={"job_id": "nope", "question": "hi"})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# frame_refs — LOOK-step thumbnails persisted with the assistant message
# ---------------------------------------------------------------------------


def test_ai_qa_forwards_and_persists_frame_refs(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `{"type": "frames", ...}` event from stream_answer is forwarded over
    SSE AND persisted on the assistant Message row — so reopening the chat
    later renders the identical thumbnail (see FrameRef / api/jobs.py's
    `_to_message`)."""
    from src.llm import qa as llm_qa

    frame_item = {
        "seconds": 12.0,
        "timecode": "00:12",
        "phrase": "this cream",
        "frame_url": "/jobs/whatever/frames/t12/frame_02.jpg",
    }

    async def _fake_stream(*, job: Any, question: str, output_language: str, from_audio: bool):
        yield {"type": "frames", "items": [frame_item]}
        yield "the cream is ACME brand"

    monkeypatch.setattr(llm_qa, "stream_answer", _fake_stream)

    create = client.post(
        "/jobs", json={"url": "https://frames-test", "kind": "page", "page_text": "hello"}
    ).json()
    _wait_until_done(client, create["id"])

    r = client.post("/ai/qa", json={"job_id": create["id"], "question": "what cream?"})
    assert r.status_code == 200
    events = _read_sse_events(r)
    frames_events = [e for e in events if e.get("type") == "frames"]
    assert len(frames_events) == 1
    assert frames_events[0]["items"] == [frame_item]

    msgs = client.get(f"/jobs/{create['id']}/messages").json()
    assistant = msgs["items"][-1]
    assert assistant["role"] == "assistant"
    assert assistant["frame_refs"] == [frame_item]
    # The user turn (and any job with no LOOK step at all) has no frame_refs.
    assert msgs["items"][0]["frame_refs"] == []


def test_ai_qa_no_frame_refs_when_no_frames_event(client: TestClient) -> None:
    """The common case — no LOOK step ran (page/PDF job, or nothing deemed
    worth a look) — leaves frame_refs empty, not merely absent."""
    create = client.post(
        "/jobs", json={"url": "https://no-frames", "kind": "page", "page_text": "hello"}
    ).json()
    _wait_until_done(client, create["id"])

    client.post("/ai/qa", json={"job_id": create["id"], "question": "q"})

    msgs = client.get(f"/jobs/{create['id']}/messages").json()
    assistant = msgs["items"][-1]
    assert assistant["frame_refs"] == []


# ---------------------------------------------------------------------------
# GET /jobs/{id}/frames/{rel_path} — serves a LOOK-step JPEG, hard path checks
# ---------------------------------------------------------------------------


def test_get_frame_happy_path(client: TestClient) -> None:
    from src.config import get_config

    create = client.post(
        "/jobs", json={"url": "https://frame-file", "kind": "page", "page_text": "hello"}
    ).json()
    _wait_until_done(client, create["id"])

    frame_dir = Path(get_config().storage.data_dir) / "frames" / create["id"] / "t12"
    frame_dir.mkdir(parents=True)
    jpeg_bytes = b"\xff\xd8\xff\xe0fake-jpeg-bytes"
    (frame_dir / "frame_02.jpg").write_bytes(jpeg_bytes)

    r = client.get(f"/jobs/{create['id']}/frames/t12/frame_02.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content == jpeg_bytes


def test_get_frame_404_for_unknown_job(client: TestClient) -> None:
    r = client.get("/jobs/no-such-job/frames/t12/frame_02.jpg")
    assert r.status_code == 404


def test_get_frame_404_for_missing_file(client: TestClient) -> None:
    create = client.post(
        "/jobs", json={"url": "https://frame-missing", "kind": "page", "page_text": "hello"}
    ).json()
    _wait_until_done(client, create["id"])

    r = client.get(f"/jobs/{create['id']}/frames/t12/frame_99.jpg")
    assert r.status_code == 404


def test_get_frame_404_for_traversal_attempt(client: TestClient) -> None:
    """A crafted rel_path trying to escape the job's own frame directory
    must 404, not leak an arbitrary file off disk."""
    from src.config import get_config

    create = client.post(
        "/jobs", json={"url": "https://frame-traversal", "kind": "page", "page_text": "hello"}
    ).json()
    _wait_until_done(client, create["id"])

    # A file that exists on disk but OUTSIDE this job's own frame directory —
    # must never be reachable via a `..`-crafted rel_path.
    secret_dir = Path(get_config().storage.data_dir) / "frames" / "some-other-job" / "t1"
    secret_dir.mkdir(parents=True, exist_ok=True)
    (secret_dir / "frame_00.jpg").write_bytes(b"secret")

    r = client.get(f"/jobs/{create['id']}/frames/../some-other-job/t1/frame_00.jpg")
    assert r.status_code == 404


def test_ai_qa_409_when_job_not_done(client: TestClient) -> None:
    """Q&A on a job that hasn't finished summarizing returns 409."""
    # YouTube without transcript stays in queued state with the no-op worker.
    r = client.post(
        "/jobs",
        json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "kind": "youtube"},
    )
    assert r.status_code == 202
    job_id = r.json()["id"]

    # Wait until the pipeline parks it in queued (Whisper deferred).
    import time
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        detail = client.get(f"/jobs/{job_id}").json()
        if detail["status"] == "queued":
            break
        time.sleep(0.05)

    r = client.post("/ai/qa", json={"job_id": job_id, "question": "What is this?"})
    assert r.status_code == 409
    assert "not done" in r.json()["detail"]


# ---------------------------------------------------------------------------
# POST /jobs/{id}/retry
# ---------------------------------------------------------------------------


def test_retry_endpoint_returns_202_and_reruns(client: TestClient) -> None:
    """Retry a failed job: endpoint returns 202, job transitions back to running."""
    create = client.post(
        "/jobs", json={"url": "https://x", "kind": "page", "page_text": "hello"}
    ).json()
    _wait_until_done(client, create["id"])

    # Force the job into failed state via internal API (no route to fail directly).
    from src.storage import repo

    repo.update_status(create["id"], status="failed", error="injected failure")

    r = client.post(f"/jobs/{create['id']}/retry")
    assert r.status_code == 202
    body = r.json()
    assert body["id"] == create["id"]
    assert body["status"] == "running"

    # Pipeline reruns and reaches done again.
    final = _wait_until_done(client, create["id"])
    assert final["status"] == "done"
    assert final["error"] is None


def test_retry_endpoint_404_for_unknown_job(client: TestClient) -> None:
    r = client.post("/jobs/nope/retry")
    assert r.status_code == 404


def test_retry_endpoint_409_when_not_failed(client: TestClient) -> None:
    """Only failed jobs can be retried."""
    create = client.post(
        "/jobs", json={"url": "https://x", "kind": "page", "page_text": "hello"}
    ).json()
    _wait_until_done(client, create["id"])

    r = client.post(f"/jobs/{create['id']}/retry")
    assert r.status_code == 409
    assert "failed" in r.json()["detail"]


# ---------------------------------------------------------------------------
# GET /jobs/{id}/moments, POST /jobs/{id}/frames — on-demand "look"
# affordance (see workers/deixis.py, workers/frames.py). Deliberately
# hermetic like the rest of this file: `workers.frames.fetch_frames` is
# monkeypatched, never yt-dlp/ffmpeg.
# ---------------------------------------------------------------------------

# Real deixis phrases (matched against src/workers/deixis.py's English
# marker table, verified directly against find_deixis_candidates before
# being written here) spaced well beyond COLLAPSE_WINDOW_SECONDS apart so
# each stays its own candidate.
_ACTION_SECONDS = 10.0
_OBJECT_SECONDS = 50.0
_EXTERNAL_SECONDS = 90.0
_DEIXIS_SEGMENTS = [
    {"start": _ACTION_SECONDS, "text": "Now watch this, follow along."},
    {"start": _OBJECT_SECONDS, "text": "You want to apply this cream twice a day."},
    {"start": _EXTERNAL_SECONDS, "text": "The link is in the description below."},
]


def _make_audio_job(
    client: TestClient,
    *,
    segments: list[dict[str, Any]] | None = None,
    transcript_source: str = "whisper",
    language: str = "en",
    url: str = "https://example.com/video",
) -> str:
    """Create a job the normal (hermetic) way, then promote it via
    ``repo.mark_done`` into a "done" job carrying a real audio
    ``transcript_source`` + ``raw_segments_json`` — this test module's
    pipeline mocks never produce a genuine audio transcript (YouTube always
    defers to the whisper queue, which is a no-op here), so this is the
    direct route to a job GET /moments / POST /frames can actually see
    candidates for. ``workers.deixis.candidates_for_job`` only looks at
    ``transcript_source`` / ``raw_segments_json`` / ``transcript_language``
    — the job's ``kind`` is irrelevant to it.
    """
    from src.storage import repo

    create = client.post(
        "/jobs", json={"url": url, "kind": "page", "page_text": "placeholder"}
    ).json()
    job_id = create["id"]
    _wait_until_done(client, job_id)
    repo.mark_done(
        job_id,
        raw_text="placeholder raw text",
        summary_md="placeholder summary",
        transcript_source=transcript_source,
        transcript_language=language,
        raw_segments_json=json.dumps(segments if segments is not None else _DEIXIS_SEGMENTS),
    )
    return job_id


def test_list_moments_excludes_external_and_shapes_response(client: TestClient) -> None:
    job_id = _make_audio_job(client)

    r = client.get(f"/jobs/{job_id}/moments")
    assert r.status_code == 200, r.text
    items = r.json()["items"]

    # EXTERNAL is dropped; only ACTION + OBJECT survive.
    assert len(items) == 2
    categories = {item["category"] for item in items}
    assert categories == {"action", "object"}
    assert all(item["seconds"] != _EXTERNAL_SECONDS for item in items)

    action = next(i for i in items if i["category"] == "action")
    assert action["seconds"] == _ACTION_SECONDS
    assert action["timecode"] == "00:10"
    assert action["phrase"] == "watch this"


def test_list_moments_page_job_returns_empty(client: TestClient) -> None:
    """A job whose transcript_source isn't an audio source (page/PDF) must
    return an empty list, never an error — same rule the QA LOOK step
    enforces (AUDIO_TRANSCRIPT_SOURCES)."""
    create = client.post(
        "/jobs", json={"url": "https://a-page.example", "kind": "page", "page_text": "hello"}
    ).json()
    _wait_until_done(client, create["id"])

    r = client.get(f"/jobs/{create['id']}/moments")
    assert r.status_code == 200
    assert r.json()["items"] == []


def test_list_moments_audio_job_with_empty_segments_list_returns_empty(
    client: TestClient,
) -> None:
    """transcript_source qualifies, but raw_segments_json parses to an
    empty list — still an empty result, never an error."""
    job_id = _make_audio_job(client, segments=[])
    r = client.get(f"/jobs/{job_id}/moments")
    assert r.status_code == 200
    assert r.json()["items"] == []


def test_list_moments_audio_job_with_no_segments_column_returns_empty(
    client: TestClient,
) -> None:
    """transcript_source qualifies, but raw_segments_json was never set at
    all (None) — still an empty result, never an error."""
    from src.storage import repo

    create = client.post(
        "/jobs", json={"url": "https://example.com/video", "kind": "page",
                       "page_text": "placeholder"},
    ).json()
    job_id = create["id"]
    _wait_until_done(client, job_id)
    repo.mark_done(
        job_id,
        raw_text="placeholder",
        summary_md="placeholder",
        transcript_source="whisper",
        transcript_language="en",
        # raw_segments_json intentionally omitted — stays None.
    )

    r = client.get(f"/jobs/{job_id}/moments")
    assert r.status_code == 200
    assert r.json()["items"] == []


def test_list_moments_unknown_job_404(client: TestClient) -> None:
    r = client.get("/jobs/does-not-exist/moments")
    assert r.status_code == 404


def test_fetch_moment_frames_happy_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job_id = _make_audio_job(client)

    frame_paths = [tmp_path / "frame_01.jpg", tmp_path / "frame_02.jpg"]
    for p in frame_paths:
        p.write_bytes(b"jpeg")

    captured: dict[str, Any] = {}

    async def fake_fetch_frames(**kwargs: Any) -> list[Path]:
        captured.update(kwargs)
        return frame_paths

    from src.workers import frames as frames_mod

    monkeypatch.setattr(frames_mod, "fetch_frames", fake_fetch_frames)

    r = client.post(f"/jobs/{job_id}/frames", json={"seconds": _ACTION_SECONDS})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["items"]) == 2
    for item, path in zip(body["items"], frame_paths, strict=True):
        assert item["seconds"] == _ACTION_SECONDS
        assert item["timecode"] == "00:10"
        assert item["phrase"] == "watch this"
        assert item["frame_url"] == f"/jobs/{job_id}/frames/{path.parent.name}/{path.name}"

    # ACTION -> SECTION_MAX_HEIGHT_PX (not the "readable" resolution).
    assert captured["max_height_px"] == frames_mod.SECTION_MAX_HEIGHT_PX
    assert captured["reuse_existing"] is True
    assert captured["timestamp_seconds"] == _ACTION_SECONDS


def test_fetch_moment_frames_uses_readable_height_for_object_category(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job_id = _make_audio_job(client)
    frame_path = tmp_path / "frame_01.jpg"
    frame_path.write_bytes(b"jpeg")

    captured: dict[str, Any] = {}

    async def fake_fetch_frames(**kwargs: Any) -> list[Path]:
        captured.update(kwargs)
        return [frame_path]

    from src.workers import frames as frames_mod

    monkeypatch.setattr(frames_mod, "fetch_frames", fake_fetch_frames)

    r = client.post(f"/jobs/{job_id}/frames", json={"seconds": _OBJECT_SECONDS})
    assert r.status_code == 200, r.text
    assert captured["max_height_px"] == frames_mod.SECTION_MAX_HEIGHT_READABLE_PX


def test_fetch_moment_frames_rejects_external_moment(client: TestClient) -> None:
    job_id = _make_audio_job(client)
    r = client.post(f"/jobs/{job_id}/frames", json={"seconds": _EXTERNAL_SECONDS})
    assert r.status_code == 400
    assert "outside the video" in r.json()["detail"]


def test_fetch_moment_frames_unknown_seconds_404(client: TestClient) -> None:
    job_id = _make_audio_job(client)
    r = client.post(f"/jobs/{job_id}/frames", json={"seconds": 12345.0})
    assert r.status_code == 404


def test_fetch_moment_frames_unknown_job_404(client: TestClient) -> None:
    r = client.post("/jobs/does-not-exist/frames", json={"seconds": 10.0})
    assert r.status_code == 404


def test_fetch_moment_frames_budget_exhausted_returns_409(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = _make_audio_job(client)

    async def fake_fetch_frames(**kwargs: Any) -> list[Path]:
        return []  # workers.frames.fetch_frames's own "budget spent" signal

    from src.workers import frames as frames_mod

    monkeypatch.setattr(frames_mod, "fetch_frames", fake_fetch_frames)

    r = client.post(f"/jobs/{job_id}/frames", json={"seconds": _ACTION_SECONDS})
    assert r.status_code == 409
    assert "budget" in r.json()["detail"]


def test_fetch_moment_frames_download_failure_returns_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = _make_audio_job(client)

    from src.workers import frames as frames_mod
    from src.workers.errors import FrameExtractionError

    async def fake_fetch_frames(**kwargs: Any) -> list[Path]:
        raise FrameExtractionError(
            f"section download failed for x after {frames_mod.SECTION_DOWNLOAD_MAX_ATTEMPTS} "
            "attempts: ffmpeg exited with code 8"
        )

    monkeypatch.setattr(frames_mod, "fetch_frames", fake_fetch_frames)

    r = client.post(f"/jobs/{job_id}/frames", json={"seconds": _ACTION_SECONDS})
    assert r.status_code == 502
    assert "ffmpeg exited with code 8" in r.json()["detail"]


