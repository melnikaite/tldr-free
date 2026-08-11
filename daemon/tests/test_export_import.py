"""Integration tests for POST /jobs/export and POST /jobs/import.

Uses the same ``TestClient`` + mocked-external-services fixture style as
``test_api_jobs.py`` (external LLM/whisper/trafilatura calls are faked so a
"done" job can be produced hermetically). Frame files and translations that
the pipeline itself never produces in test mode are seeded directly via
``repo``/the DB, same as other tests reach into internals for setup.
"""

from __future__ import annotations

import asyncio
import io
import json
import zipfile
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.config import get_config
from src.main import app
from src.storage import repo
from src.storage.db import TranscriptTranslation, dispose_engine, init_engine, session_scope
from src.storage.migrations import run_migrations
from src.workers import frames


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Same mocked-external-services setup as ``test_api_jobs.py``'s fixture
    — see there for why each mock exists. Duplicated rather than imported
    because pytest fixtures aren't meant to be shared by cross-module
    import, and this file's needs are a strict subset."""
    db_path = tmp_path / "api.db"
    engine = init_engine(db_path)
    run_migrations(engine)

    from src.llm import summary as llm_summary

    async def _fake_stream_summarize(
        text: str, *, title: Any, output_language: str, from_audio_transcript: bool = False
    ) -> AsyncIterator[str]:
        yield f"## summary\n\nfor {len(text)} chars in {output_language}"

    monkeypatch.setattr(llm_summary, "stream_summarize", _fake_stream_summarize)

    from src.llm import qa as llm_qa

    async def _fake_qa_stream(
        *, job: Any, question: str, output_language: str, from_audio: bool
    ) -> AsyncIterator[str]:
        yield f"answer to {question!r} in {output_language}"

    monkeypatch.setattr(llm_qa, "stream_answer", _fake_qa_stream)

    from src.workers import youtube as yt_worker
    from src.workers.errors import PermanentTranscriptError

    async def _fake_fetch(*, video_id, cookies, max_attempts, backoff_seconds):  # noqa: ANN001
        raise PermanentTranscriptError("test mode: no transcript")

    monkeypatch.setattr(yt_worker, "fetch_transcript_with_retry", _fake_fetch)

    async def _fake_subs(*, url, cookies, dir, lang_preferences):  # noqa: ANN001
        return None

    monkeypatch.setattr(yt_worker, "download_subtitles", _fake_subs)

    from src.workers import search as search_mod

    async def _fake_ddg_search(query: str, max_results: int = 5) -> list[dict]:  # noqa: ANN001
        return [{"title": "Fake result", "href": "https://example.com/result", "body": query}]

    monkeypatch.setattr(search_mod, "ddg_search", _fake_ddg_search)

    from src.workers import page as page_worker

    async def _fake_extract(url: str) -> tuple[str | None, str]:
        return (None, "extracted page text")

    monkeypatch.setattr(page_worker, "extract_with_trafilatura", _fake_extract)

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

    from src.workers import retention as retention_mod

    async def _noop_retention() -> None:
        try:
            while True:
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            return

    monkeypatch.setattr(retention_mod, "retention_worker", _noop_retention)
    monkeypatch.setattr(main_mod, "retention_worker", _noop_retention)

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


def _wait_until_done(client: TestClient, job_id: str, *, timeout: float = 5.0) -> dict:
    import time

    deadline = time.monotonic() + timeout
    body: dict[str, Any] = {}
    while time.monotonic() < deadline:
        r = client.get(f"/jobs/{job_id}")
        body = r.json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach done/failed within {timeout}s; last={body}")


def _make_done_job(client: TestClient, url: str) -> dict:
    r = client.post("/jobs", json={"url": url, "kind": "page", "page_text": "hello world"})
    body = r.json()
    done = _wait_until_done(client, body["id"])
    assert done["status"] == "done"
    return done


def _write_frame(job_id: str, sub: str, name: str, content: bytes) -> Path:
    frame_dir = Path(get_config().storage.data_dir) / "frames" / job_id / sub
    frame_dir.mkdir(parents=True, exist_ok=True)
    path = frame_dir / name
    path.write_bytes(content)
    return path


def _add_done_translation(job_id: str, *, language_code: str, text: str) -> None:
    with session_scope() as session:
        session.add(
            TranscriptTranslation(
                job_id=job_id,
                language_code=language_code,
                status="done",
                text=text,
                progress_percent=100,
            )
        )


# ---------------------------------------------------------------------------
# 1. Round trip
# ---------------------------------------------------------------------------


def test_export_import_round_trip(client: TestClient) -> None:
    done = _make_done_job(client, "https://export-roundtrip.example")
    job_id = done["id"]

    # Three messages: a question, and two "looked at a moment" answers —
    # frames in TWO DIFFERENT t<seconds> directories, so the round trip
    # proves nesting is preserved per-moment, not just within one directory.
    repo.add_message(job_id, role="user", content="what happens at 12s?")
    frame_url_12 = f"/jobs/{job_id}/frames/t12/frame_01.jpg"
    repo.add_message(
        job_id,
        role="assistant",
        content="here's what's shown at 12s",
        frame_refs=[
            {
                "seconds": 12.0, "timecode": "00:12", "phrase": "look here",
                "frame_url": frame_url_12,
            }
        ],
    )
    frame_url_45 = f"/jobs/{job_id}/frames/t45/frame_01.jpg"
    repo.add_message(
        job_id,
        role="assistant",
        content="and here's what's shown at 45s",
        frame_refs=[
            {
                "seconds": 45.0, "timecode": "00:45", "phrase": "look there",
                "frame_url": frame_url_45,
            }
        ],
    )

    # One done translation.
    _add_done_translation(job_id, language_code="en", text="Hello world (en)")

    # Frame files on disk in TWO different t<seconds> moment directories —
    # frame_02.jpg alongside frame_01.jpg in t12 is never referenced by any
    # message, proving a whole directory's contents export, not just the
    # ones a frame_ref points at.
    frame1_bytes = b"\xff\xd8\xff\xe0-frame-one"
    frame2_bytes = b"\xff\xd8\xff\xe0-frame-two"
    frame3_bytes = b"\xff\xd8\xff\xe0-frame-three"
    _write_frame(job_id, "t12", "frame_01.jpg", frame1_bytes)
    _write_frame(job_id, "t12", "frame_02.jpg", frame2_bytes)
    _write_frame(job_id, "t45", "frame_01.jpg", frame3_bytes)

    # --- export ---
    r = client.post("/jobs/export", json={"ids": [job_id]})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    disposition = r.headers["content-disposition"]
    assert "attachment" in disposition
    assert 'filename="tldr-export-1-jobs.zip"' in disposition
    zip_bytes = r.content

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["format"] == "tldr-export"
        assert manifest["version"] == 1
        assert manifest["jobs"] == [job_id]
        job_json = json.loads(zf.read(f"jobs/{job_id}/job.json"))
        assert job_json["url"] == "https://export-roundtrip.example"
        assert len(job_json["messages"]) == 3
        assert job_json["messages"][0]["role"] == "user"
        assert job_json["messages"][1]["frame_refs"][0]["frame_url"] == frame_url_12
        assert job_json["messages"][2]["frame_refs"][0]["frame_url"] == frame_url_45
        assert len(job_json["translations"]) == 1
        assert job_json["translations"][0]["text"] == "Hello world (en)"
        # The zip member names use the REAL nested relative path — no
        # flattening — matching the on-disk t<seconds>/frame_NN.jpg layout.
        frame_members = sorted(
            n for n in zf.namelist() if n.startswith(f"jobs/{job_id}/frames/")
        )
        assert frame_members == [
            f"jobs/{job_id}/frames/t12/frame_01.jpg",
            f"jobs/{job_id}/frames/t12/frame_02.jpg",
            f"jobs/{job_id}/frames/t45/frame_01.jpg",
        ]

    # --- delete original, then import ---
    assert client.delete(f"/jobs/{job_id}").status_code == 204

    r2 = client.post("/jobs/import", content=zip_bytes)
    assert r2.status_code == 200
    body = r2.json()
    assert body["skipped"] == []
    assert body["failed"] == []
    assert len(body["imported"]) == 1
    new_id = body["imported"][0]["job_id"]
    assert new_id != job_id
    assert body["imported"][0]["url"] == "https://export-roundtrip.example"

    # Job detail matches.
    new_detail = client.get(f"/jobs/{new_id}").json()
    assert new_detail["status"] == "done"
    assert new_detail["title"] == done["title"]
    assert new_detail["summary_md"] == done["summary_md"]

    # Messages preserved, in order, frame_url rewritten to the new id —
    # including the SECOND moment's nested t45/ path.
    messages = client.get(f"/jobs/{new_id}/messages").json()["items"]
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["frame_refs"][0]["frame_url"] == f"/jobs/{new_id}/frames/t12/frame_01.jpg"
    assert messages[2]["frame_refs"][0]["frame_url"] == f"/jobs/{new_id}/frames/t45/frame_01.jpg"

    # Frame files exist on disk under the new id, nested exactly as before
    # (two different t<seconds> directories recreated, not flattened).
    new_frames_root = Path(get_config().storage.data_dir) / "frames" / new_id
    assert (new_frames_root / "t12" / "frame_01.jpg").read_bytes() == frame1_bytes
    assert (new_frames_root / "t12" / "frame_02.jpg").read_bytes() == frame2_bytes
    assert (new_frames_root / "t45" / "frame_01.jpg").read_bytes() == frame3_bytes

    # The rewritten frame_urls resolve through the same path-safety check
    # GET /jobs/{id}/frames/{rel_path} uses — proves the nested layout is
    # not just "some file happens to exist" but resolves the exact way the
    # live serving path expects.
    assert frames.resolve_frame_path(new_id, "t12/frame_01.jpg") is not None
    assert frames.resolve_frame_path(new_id, "t45/frame_01.jpg") is not None
    for msg in messages[1:]:
        for ref in msg["frame_refs"]:
            rel = ref["frame_url"].split(f"/jobs/{new_id}/frames/", 1)[1]
            assert frames.resolve_frame_path(new_id, rel) is not None

    # Translation preserved.
    translations = new_detail["transcript_translations"]
    assert len(translations) == 1
    assert translations[0]["language_code"] == "en"
    assert translations[0]["status"] == "done"
    transcript = client.get(f"/jobs/{new_id}/transcript?lang=en").json()
    assert transcript["text"] == "Hello world (en)"


# ---------------------------------------------------------------------------
# 1b. added_at — machine-local, must never leave the exporting machine, and
# the importing machine must set its own rather than inherit the bundle's
# created_at for retention purposes.
# ---------------------------------------------------------------------------


def test_export_bundle_omits_added_at(client: TestClient) -> None:
    """added_at is machine-local (Job.added_at docstring / bundle.py module
    docstring's "deliberately NOT included" list) — it must never appear in
    job.json, unlike every other Job field this export/import path carries."""
    done = _make_done_job(client, "https://export-no-added-at.example")
    job_id = done["id"]

    r = client.post("/jobs/export", json={"ids": [job_id]})
    assert r.status_code == 200

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        job_json = json.loads(zf.read(f"jobs/{job_id}/job.json"))

    assert "added_at" not in job_json


def test_imported_job_gets_added_at_now_and_survives_retention_that_would_catch_created_at(
    client: TestClient,
) -> None:
    """An imported job keeps the bundle's (old) created_at but gets
    added_at=now on the importing machine — so a retention cutoff that
    would have swept the old created_at must NOT sweep it, since the sweep
    reads added_at (repo.delete_jobs_older_than)."""
    done = _make_done_job(client, "https://export-retention-immune.example")
    job_id = done["id"]

    # Back-date created_at to simulate material that was actually processed
    # years ago on the exporting machine.
    old_created_at = "2015-01-01T00:00:00"
    with session_scope() as session:
        from src.storage.db import Job

        job_row = session.get(Job, job_id)
        assert job_row is not None
        job_row.created_at = datetime.fromisoformat(old_created_at)
        session.add(job_row)

    r = client.post("/jobs/export", json={"ids": [job_id]})
    assert r.status_code == 200
    zip_bytes = r.content

    assert client.delete(f"/jobs/{job_id}").status_code == 204

    r2 = client.post("/jobs/import", content=zip_bytes)
    assert r2.status_code == 200
    body = r2.json()
    assert body["failed"] == []
    assert body["skipped"] == []
    new_id = body["imported"][0]["job_id"]

    new_detail = client.get(f"/jobs/{new_id}").json()
    assert new_detail["created_at"].startswith("2015-01-01")
    assert not new_detail["added_at"].startswith("2015-01-01")

    # A cutoff that would have caught the old created_at many times over —
    # the imported job must survive because retention reads added_at.
    cutoff = datetime(2020, 1, 1)
    deleted = repo.delete_jobs_older_than(cutoff)

    assert deleted == 0
    assert repo.get_job(new_id) is not None


# ---------------------------------------------------------------------------
# 2. Non-done jobs are excluded; all-non-exportable -> 400
# ---------------------------------------------------------------------------


def test_export_excludes_non_done_and_unknown_ids(client: TestClient) -> None:
    done = _make_done_job(client, "https://export-mixed.example")
    non_done = repo.create_job(url="https://export-not-done.example", kind="page")
    assert non_done.status == "running"

    r = client.post("/jobs/export", json={"ids": [done["id"], non_done.id, "no-such-job"]})
    assert r.status_code == 200
    assert 'filename="tldr-export-1-jobs.zip"' in r.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["jobs"] == [done["id"]]


def test_export_400_when_nothing_exportable(client: TestClient) -> None:
    non_done = repo.create_job(url="https://export-all-bad.example", kind="page")

    r = client.post("/jobs/export", json={"ids": [non_done.id, "totally-unknown"]})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# 3. Re-importing a bundle whose job already exists (done) -> skipped/duplicate
# ---------------------------------------------------------------------------


def test_import_skips_duplicate_when_job_already_done(client: TestClient) -> None:
    done = _make_done_job(client, "https://export-dup.example")
    job_id = done["id"]

    r = client.post("/jobs/export", json={"ids": [job_id]})
    zip_bytes = r.content

    before = client.get("/jobs").json()["total"]

    # Do NOT delete the original this time — its URL already exists as done.
    r2 = client.post("/jobs/import", content=zip_bytes)
    assert r2.status_code == 200
    body = r2.json()
    assert body["imported"] == []
    assert body["failed"] == []
    assert len(body["skipped"]) == 1
    assert body["skipped"][0]["url"] == "https://export-dup.example"
    assert body["skipped"][0]["reason"] == "duplicate"

    after = client.get("/jobs").json()["total"]
    assert after == before


def test_import_duplicate_check_ignores_newer_failed_sibling(client: TestClient) -> None:
    """The duplicate check must ask "does a done row for this URL exist",
    not "is the newest row for this URL done" — a machine can have both a
    failed and a done job for the same URL (e.g. a retry that left the old
    failed row behind). Here the FAILED row is created (and thus ordered
    newest by created_at) AFTER the done one, so a buggy "check the newest
    row's status" implementation would miss the duplicate entirely.
    """
    url = "https://export-dup-with-failed-sibling.example"
    done = _make_done_job(client, url)

    # A second, NEWER row for the same URL that ends up failed — simulates
    # a retry attempt left behind. repo.create_job sets created_at=utcnow()
    # at call time, so this row is guaranteed newer than `done`'s.
    newer_failed = repo.create_job(url=url, kind="page")
    repo.mark_failed(newer_failed.id, error="simulated retry failure")

    r = client.post("/jobs/export", json={"ids": [done["id"]]})
    zip_bytes = r.content

    before = client.get("/jobs").json()["total"]
    r2 = client.post("/jobs/import", content=zip_bytes)
    assert r2.status_code == 200
    body = r2.json()
    assert body["imported"] == []
    assert body["failed"] == []
    assert len(body["skipped"]) == 1
    assert body["skipped"][0]["reason"] == "duplicate"

    after = client.get("/jobs").json()["total"]
    assert after == before


# ---------------------------------------------------------------------------
# 4. Zip-slip guards
# ---------------------------------------------------------------------------


def _manifest_bytes(job_ids: list[str]) -> bytes:
    return json.dumps(
        {
            "format": "tldr-export",
            "version": 1,
            "exported_at": "2024-01-01T00:00:00",
            "daemon_api_version": 4,
            "jobs": job_ids,
        }
    ).encode("utf-8")


def _build_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_import_rejects_relative_traversal_member(client: TestClient, tmp_path: Path) -> None:
    evil_zip = _build_zip(
        {
            "manifest.json": _manifest_bytes(["x"]),
            "jobs/x/job.json": json.dumps({"url": "https://evil.example"}).encode(),
            "jobs/x/frames/../../../evil.jpg": b"evil-bytes",
        }
    )
    r = client.post("/jobs/import", content=evil_zip)
    assert r.status_code == 400

    data_dir = Path(get_config().storage.data_dir)
    assert not (data_dir.parent / "evil.jpg").exists()
    assert not (data_dir / "evil.jpg").exists()
    frames_root = data_dir / "frames"
    if frames_root.is_dir():
        assert not any(p.name == "evil.jpg" for p in frames_root.rglob("evil.jpg"))


def test_import_rejects_absolute_path_member(client: TestClient) -> None:
    evil_zip = _build_zip(
        {
            "manifest.json": _manifest_bytes(["x"]),
            "jobs/x/job.json": json.dumps({"url": "https://evil2.example"}).encode(),
            "/etc/evil2.jpg": b"evil-bytes",
        }
    )
    r = client.post("/jobs/import", content=evil_zip)
    assert r.status_code == 400
    assert not Path("/etc/evil2.jpg").exists()


# ---------------------------------------------------------------------------
# 5. Not a zip / missing manifest / wrong format
# ---------------------------------------------------------------------------


def test_import_rejects_non_zip_payload(client: TestClient) -> None:
    r = client.post("/jobs/import", content=b"this is not a zip file at all")
    assert r.status_code == 400


def test_import_rejects_missing_manifest(client: TestClient) -> None:
    bad_zip = _build_zip({"jobs/x/job.json": b"{}"})
    r = client.post("/jobs/import", content=bad_zip)
    assert r.status_code == 400


def test_import_rejects_wrong_format(client: TestClient) -> None:
    bad_manifest = json.dumps(
        {"format": "something-else", "version": 1, "jobs": []}
    ).encode("utf-8")
    bad_zip = _build_zip({"manifest.json": bad_manifest})
    r = client.post("/jobs/import", content=bad_zip)
    assert r.status_code == 400


def test_import_rejects_future_version(client: TestClient) -> None:
    bad_manifest = json.dumps(
        {"format": "tldr-export", "version": 99, "jobs": []}
    ).encode("utf-8")
    bad_zip = _build_zip({"manifest.json": bad_manifest})
    r = client.post("/jobs/import", content=bad_zip)
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# 6. Invalid 'kind' in a job entry fails just that job, not the whole bundle
# ---------------------------------------------------------------------------


def test_import_rejects_job_with_invalid_kind(client: TestClient) -> None:
    """``kind`` gets written straight into a column the UI switches on
    (page/youtube/media/pdf) — a bundle claiming some other string must
    fail that job entry into ``failed`` rather than being written to the
    DB, and must not block the rest of the bundle."""
    bad_job = json.dumps(
        {"url": "https://bad-kind.example", "title": "Bad kind job", "kind": "not-a-real-kind"}
    ).encode("utf-8")
    bundle_zip = _build_zip(
        {
            "manifest.json": _manifest_bytes(["x"]),
            "jobs/x/job.json": bad_job,
        }
    )

    before = client.get("/jobs").json()["total"]
    r = client.post("/jobs/import", content=bundle_zip)
    assert r.status_code == 200
    body = r.json()
    assert body["imported"] == []
    assert body["skipped"] == []
    assert len(body["failed"]) == 1
    assert body["failed"][0]["url"] == "https://bad-kind.example"
    assert "kind" in body["failed"][0]["reason"]

    after = client.get("/jobs").json()["total"]
    assert after == before


def test_import_rejects_job_with_missing_kind(client: TestClient) -> None:
    bad_job = json.dumps({"url": "https://missing-kind.example"}).encode("utf-8")
    bundle_zip = _build_zip(
        {
            "manifest.json": _manifest_bytes(["x"]),
            "jobs/x/job.json": bad_job,
        }
    )

    r = client.post("/jobs/import", content=bundle_zip)
    assert r.status_code == 200
    body = r.json()
    assert body["imported"] == []
    assert len(body["failed"]) == 1
    assert body["failed"][0]["url"] == "https://missing-kind.example"


# ---------------------------------------------------------------------------
# 7. A translation's "status" is a whitelist, not trusted verbatim
# ---------------------------------------------------------------------------


def test_import_normalizes_bogus_translation_status_to_done(client: TestClient) -> None:
    """``TranscriptTranslationSummary.status`` is a pydantic ``Literal`` and
    ``GET /jobs/{id}`` declares ``response_model=JobDetails`` — an
    out-of-set value stored in the DB makes that endpoint 500 for this job
    PERMANENTLY, with no way to fix it from the UI. A bundle is untrusted
    input (same class of concern as the zip-slip guards above), so a
    translation entry claiming an unrecognised status (or a LIVE one like
    "running", which ``re_enqueue_running_on_startup`` would pick up and
    start re-translating on the next daemon restart) must be normalised to
    "done" on import, never written verbatim."""
    job_payload = {
        "url": "https://bogus-translation-status.example",
        "title": "Bogus translation status job",
        "kind": "page",
        "raw_text": "hello world",
        "summary_md": "**hi**",
        "translations": [
            {
                "language_code": "ru",
                "text": "привет мир",
                "status": "bogus-status-not-in-the-literal",
                "error": "some attacker-controlled string",
            },
        ],
    }
    bundle_zip = _build_zip(
        {
            "manifest.json": _manifest_bytes(["x"]),
            "jobs/x/job.json": json.dumps(job_payload).encode("utf-8"),
        }
    )

    r = client.post("/jobs/import", content=bundle_zip)
    assert r.status_code == 200
    body = r.json()
    assert body["failed"] == []
    assert len(body["imported"]) == 1
    new_id = body["imported"][0]["job_id"]

    # The endpoint that would 500 forever on an un-whitelisted status must
    # serve this job fine.
    detail_resp = client.get(f"/jobs/{new_id}")
    assert detail_resp.status_code == 200
    translations = detail_resp.json()["transcript_translations"]
    assert len(translations) == 1
    assert translations[0]["language_code"] == "ru"
    assert translations[0]["status"] == "done"
    # error is dropped along with the bogus status — "error" only ever
    # legitimately accompanies "partial".
    assert translations[0]["error"] is None

    transcript = client.get(f"/jobs/{new_id}/transcript?lang=ru")
    assert transcript.status_code == 200
    assert transcript.json()["text"] == "привет мир"


def test_import_normalizes_live_translation_status_to_done(client: TestClient) -> None:
    """A translation claiming "running" (a live, in-progress status) must
    not be imported verbatim — ``re_enqueue_running_on_startup`` would
    otherwise pick up the imported row on the next daemon restart and
    start re-translating a job that has no worker for it."""
    job_payload = {
        "url": "https://live-translation-status.example",
        "title": "Live translation status job",
        "kind": "page",
        "raw_text": "hello world",
        "summary_md": "**hi**",
        "translations": [
            {"language_code": "de", "text": "hallo welt", "status": "running"},
        ],
    }
    bundle_zip = _build_zip(
        {
            "manifest.json": _manifest_bytes(["x"]),
            "jobs/x/job.json": json.dumps(job_payload).encode("utf-8"),
        }
    )

    r = client.post("/jobs/import", content=bundle_zip)
    assert r.status_code == 200
    body = r.json()
    assert body["failed"] == []
    new_id = body["imported"][0]["job_id"]

    translations = client.get(f"/jobs/{new_id}").json()["transcript_translations"]
    assert translations[0]["status"] == "done"
