"""Deterministic concurrency / isolation tests for the daemon API.

Goal: prove that concurrent requests for DIFFERENT jobs (and concurrent
requests on the SAME job) never mix their state — SSE payloads, persisted
messages, the per-job stream replay buffer, the global ``/events`` fan-out,
and on-disk frame directories.

How the LLM is mocked: at ``llm.client.call_with_dialect_adaptation`` — i.e.
BELOW ``_acquire_llm_slot`` / ``_llm_lock()``. Every real code path above it
(``qa.stream_answer`` plan → synthesis, ``summary.stream_summarize``,
``stream_with_messages``' chunk loop, the semaphore) runs for real. The mock
derives its output from markers it finds in the prompt it was actually
given (``MAT-<x>`` in the job's material, ``Q-<x>`` in the question), so a
response routed to the wrong job/request — or a prompt built from the wrong
job's context — shows up as a marker mismatch.

Two lock regimes are exercised:

- ``max_concurrent_calls=1`` (the shipped default): the real semaphore
  serialises the streams. We assert correctness AND that the mock saw no
  overlap (proves the lock is in the path, not bypassed).
- ``max_concurrent_calls=4``: the same real semaphore admits both, and the
  mock's streams are gated with asyncio events so stream A is provably
  mid-flight while stream B starts AND finishes first. This is the regime in
  which shared state would actually corrupt output, so it's the one that
  would catch a regression.

Driven via ``httpx.AsyncClient`` + ``ASGITransport`` on one event loop
(same setup as ``test_api_jobs_race.py``). ASGITransport buffers each
response body, so the infinite ``GET /events`` stream is observed by
subscribing to ``broker.get_event_broker()`` directly — the exact queue
``/events`` reads from.
"""

from __future__ import annotations

import asyncio
import json
import re
import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.config import get_config
from src.llm import client as llm_client
from src.main import app
from src.storage import repo
from src.storage.db import dispose_engine, init_engine
from src.storage.migrations import run_migrations
from src.workers import broker as broker_mod
from src.workers import control as control_mod
from src.workers import queue as queue_mod

_MAT_RE = re.compile(r"MAT-([A-Z0-9]+)")
_Q_RE = re.compile(r"Q-([A-Z0-9]+)")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    engine = init_engine(tmp_path / "iso.db")
    run_migrations(engine)
    cfg = get_config()
    monkeypatch.setattr(cfg.storage, "data_dir", str(tmp_path / "data"))
    # Never hit the network from QA even if a plan says "search".
    monkeypatch.setattr(cfg.qa, "web_search", False)
    queue_mod.reset_queue()
    broker_mod.reset_broker()
    control_mod.reset_control()
    llm_client._llm_lock.cache_clear()
    yield cfg
    llm_client._llm_lock.cache_clear()
    queue_mod.reset_queue()
    broker_mod.reset_broker()
    control_mod.reset_control()
    dispose_engine()


def _set_llm_concurrency(cfg: Any, monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    monkeypatch.setattr(cfg.llm, "max_concurrent_calls", n)
    llm_client._llm_lock.cache_clear()


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _make_done_job(key: str, *, url: str | None = None) -> str:
    job = repo.create_job(url=url or f"https://example.com/{key}", kind="page")
    repo.mark_done(
        job.id,
        raw_text=f"Material for MAT-{key}. " * 5,
        summary_md=f"Summary of MAT-{key}.",
        transcript_source="page",
        title=f"Title {key}",
    )
    return job.id


# ---------------------------------------------------------------------------
# Fake LLM backend (below the semaphore)
# ---------------------------------------------------------------------------


def _chunk(text: str) -> Any:
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=text))]
    )


def _plan_completion() -> Any:
    args = json.dumps({"material_sufficient": True, "search_query": ""})
    func = types.SimpleNamespace(name="plan", arguments=args)
    tc = types.SimpleNamespace(id="call_plan", type="function", function=func)
    msg = types.SimpleNamespace(content="en", tool_calls=[tc])
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class FakeLLM:
    """Scripted backend. ``gates[key]`` (optional) is awaited after the first
    chunk of the stream identified by ``key``; ``finished[key]`` is set when
    that stream is exhausted. ``log`` records (event, key) in real order."""

    def __init__(self, chunks_for: Any) -> None:
        self.chunks_for = chunks_for
        self.gates: dict[str, asyncio.Event] = {}
        self.started: dict[str, asyncio.Event] = {}
        self.finished: dict[str, asyncio.Event] = {}
        self.log: list[tuple[str, str]] = []
        self.active = 0
        self.max_active = 0

    def ev(self, table: dict[str, asyncio.Event], key: str) -> asyncio.Event:
        return table.setdefault(key, asyncio.Event())

    async def __call__(self, *, messages: list[dict[str, Any]], stream: bool, **_: Any) -> Any:
        text = "\n".join(str(m.get("content") or "") for m in messages)
        if not stream:
            return _plan_completion()
        key = self.chunks_for.key(text)
        chunks = self.chunks_for.chunks(text)
        fake = self

        async def gen() -> AsyncIterator[Any]:
            fake.active += 1
            fake.max_active = max(fake.max_active, fake.active)
            fake.log.append(("start", key))
            fake.ev(fake.started, key).set()
            try:
                for i, c in enumerate(chunks):
                    yield _chunk(c)
                    await asyncio.sleep(0)
                    if i == 0 and key in fake.gates:
                        await asyncio.wait_for(fake.gates[key].wait(), timeout=5)
            finally:
                fake.active -= 1
                fake.log.append(("end", key))
                fake.ev(fake.finished, key).set()

        return gen()


class QAScript:
    @staticmethod
    def key(text: str) -> str:
        return _Q_RE.findall(text)[-1]

    @staticmethod
    def chunks(text: str) -> list[str]:
        q = _Q_RE.findall(text)[-1]
        mats = sorted(set(_MAT_RE.findall(text)))
        mat = "+".join(mats)
        return [f"ans<{q}>", f"mat<{mat}>", f"end<{q}>"]


def _expected_answer(q: str, mat: str) -> str:
    return f"ans<{q}>mat<{mat}>end<{q}>"


def _sse_events(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(line[len("data: "):])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def _assert_qa_response(body: str, q: str, mat: str) -> None:
    events = _sse_events(body)
    deltas = "".join(e["delta"] for e in events if e["type"] == "delta")
    dones = [e for e in events if e["type"] == "done"]
    assert not [e for e in events if e["type"] == "error"], events
    assert deltas == _expected_answer(q, mat), events
    assert len(dones) == 1
    assert dones[0]["content"] == _expected_answer(q, mat)


# ---------------------------------------------------------------------------
# 1. POST /ai/qa — different jobs, and the same job, concurrently
# ---------------------------------------------------------------------------


async def _run_two_qas(
    fake: FakeLLM, reqs: list[tuple[str, str]]
) -> list[httpx.Response]:
    async with _client() as c:
        return list(
            await asyncio.gather(
                *[
                    c.post("/ai/qa", json={"job_id": jid, "question": f"What about Q-{q}?"})
                    for jid, q in reqs
                ]
            )
        )


async def _assert_messages(job_id: str, expected: list[tuple[str, str]]) -> None:
    """``expected`` = [(question_key, material_key)] in any order."""
    async with _client() as c:
        r = await c.get(f"/jobs/{job_id}/messages")
    assert r.status_code == 200, r.text
    items = r.json()["items"]
    users = sorted(m["content"] for m in items if m["role"] == "user")
    assistants = sorted(m["content"] for m in items if m["role"] == "assistant")
    assert users == sorted(f"What about Q-{q}?" for q, _ in expected)
    assert assistants == sorted(_expected_answer(q, m) for q, m in expected)
    # DB level: every row carries this job_id.
    assert {m.job_id for m in repo.list_messages(job_id)} == {job_id}


@pytest.mark.parametrize("concurrency", [1, 4])
async def test_qa_two_jobs_concurrently_never_mix(
    env: Any, monkeypatch: pytest.MonkeyPatch, concurrency: int
) -> None:
    _set_llm_concurrency(env, monkeypatch, concurrency)
    fake = FakeLLM(QAScript)
    monkeypatch.setattr(llm_client, "call_with_dialect_adaptation", fake)
    job_a, job_b = _make_done_job("A"), _make_done_job("B")

    if concurrency > 1:
        # A streams its first chunk, then parks until B's stream has fully
        # started AND finished. B must not start before A is mid-flight.
        fake.gates["QA"] = fake.ev(fake.finished, "QB")

    ra, rb = await _run_two_qas(fake, [(job_a, "QA"), (job_b, "QB")])
    assert ra.status_code == rb.status_code == 200
    _assert_qa_response(ra.text, "QA", "A")
    _assert_qa_response(rb.text, "QB", "B")

    if concurrency > 1:
        assert fake.log == [("start", "QA"), ("start", "QB"), ("end", "QB"), ("end", "QA")]
        assert fake.max_active == 2
    else:
        assert fake.max_active == 1, fake.log  # real semaphore serialised them

    await _assert_messages(job_a, [("QA", "A")])
    await _assert_messages(job_b, [("QB", "B")])


@pytest.mark.parametrize("concurrency", [1, 4])
async def test_qa_same_job_two_questions_concurrently(
    env: Any, monkeypatch: pytest.MonkeyPatch, concurrency: int
) -> None:
    _set_llm_concurrency(env, monkeypatch, concurrency)
    fake = FakeLLM(QAScript)
    monkeypatch.setattr(llm_client, "call_with_dialect_adaptation", fake)
    job = _make_done_job("S")
    other = _make_done_job("O")  # bystander: must stay empty

    if concurrency > 1:
        fake.gates["Q1"] = fake.ev(fake.finished, "Q2")

    r1, r2 = await _run_two_qas(fake, [(job, "Q1"), (job, "Q2")])
    _assert_qa_response(r1.text, "Q1", "S")
    _assert_qa_response(r2.text, "Q2", "S")
    if concurrency > 1:
        assert fake.log == [("start", "Q1"), ("start", "Q2"), ("end", "Q2"), ("end", "Q1")]

    await _assert_messages(job, [("Q1", "S"), ("Q2", "S")])
    # Each assistant row follows its OWN user row's question (pairing is by
    # content; ids interleave under concurrency, so order is not asserted).
    rows = repo.list_messages(job)
    assert len(rows) == 4
    assert repo.list_messages(other) == []


# ---------------------------------------------------------------------------
# 2. Summary streaming: per-job replay buffer + global /events job_id tagging
# ---------------------------------------------------------------------------


class SummaryScript:
    @staticmethod
    def key(text: str) -> str:
        return _MAT_RE.findall(text)[0]

    @staticmethod
    def chunks(text: str) -> list[str]:
        k = _MAT_RE.findall(text)[0]
        # >= FLUSH_CHARS (64) each so the pipeline publishes every chunk
        # immediately instead of batching on a timer. No brackets (the
        # timecode capper would hold those back).
        return [f"SUM-{k}-part{i} " + ("x" * 70) + "\n" for i in range(3)]


async def _wait_for(cond: Any, timeout: float = 5.0) -> None:
    async def loop() -> None:
        while not cond():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(loop(), timeout)


async def test_summary_streams_isolated_per_job(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_llm_concurrency(env, monkeypatch, 4)
    fake = FakeLLM(SummaryScript)
    monkeypatch.setattr(llm_client, "call_with_dialect_adaptation", fake)
    # A: first chunk, then park until B's stream is fully done.
    # B: first chunk, then park until the test has checked mid-flight state.
    checked = asyncio.Event()
    fake.gates["A"] = fake.ev(fake.finished, "B")
    fake.gates["B"] = checked

    events_q = broker_mod.get_event_broker().subscribe()
    chunks_a = SummaryScript.chunks("MAT-A")
    chunks_b = SummaryScript.chunks("MAT-B")

    async with _client() as c:
        ra = await c.post(
            "/jobs", json={"url": "https://e.com/a", "kind": "page", "page_text": "MAT-A " * 20}
        )
        await asyncio.wait_for(fake.ev(fake.started, "A").wait(), 5)
        rb = await c.post(
            "/jobs", json={"url": "https://e.com/b", "kind": "page", "page_text": "MAT-B " * 20}
        )
        ida, idb = ra.json()["id"], rb.json()["id"]
        assert ida != idb

        # Both are mid-flight: each has published exactly its first chunk.
        await _wait_for(
            lambda: broker_mod.get_stream_buffer(ida) == chunks_a[0]
            and broker_mod.get_stream_buffer(idb) == chunks_b[0]
        )
        # Late subscriber (GET /jobs/{id} partial_summary is the replay path)
        # sees only its own job's buffered stream.
        ja = (await c.get(f"/jobs/{ida}")).json()
        jb = (await c.get(f"/jobs/{idb}")).json()
        assert ja["partial_summary"] == chunks_a[0]
        assert jb["partial_summary"] == chunks_b[0]
        checked.set()

        # B finishes first (A resumes only on B's stream end — order is
        # asserted via fake.log below); B's terminal event clears only B's
        # buffer.
        await _wait_for(lambda: repo.get_job(idb).status == "done")  # type: ignore[union-attr]
        assert broker_mod.get_stream_buffer(idb) == ""

        await _wait_for(lambda: repo.get_job(ida).status == "done")  # type: ignore[union-attr]

    assert fake.log == [("start", "A"), ("start", "B"), ("end", "B"), ("end", "A")]
    assert repo.get_job(ida).summary_md == "".join(chunks_a).strip()  # type: ignore[union-attr]
    assert repo.get_job(idb).summary_md == "".join(chunks_b).strip()  # type: ignore[union-attr]

    # Global /events queue: every per-job event carries the right job_id, and
    # every delta's content belongs to that job.
    seen: list[dict[str, Any]] = []
    while not events_q.empty():
        seen.append(events_q.get_nowait())
    by_job = {ida: "A", idb: "B"}
    deltas: dict[str, str] = {ida: "", idb: ""}
    for e in seen:
        if e["type"] in ("stage", "delta", "done", "error"):
            assert e.get("job_id") in by_job, e
        if e["type"] == "delta":
            assert f"SUM-{by_job[e['job_id']]}-" in e["delta"], e
            other = "B" if by_job[e["job_id"]] == "A" else "A"
            assert f"SUM-{other}-" not in e["delta"], e
            deltas[e["job_id"]] += e["delta"]
        if e["type"] == "job":
            assert e["job"]["id"] in by_job, e
    assert deltas[ida] == "".join(chunks_a)
    assert deltas[idb] == "".join(chunks_b)


def test_job_event_broker_buffers_are_per_job(env: Any) -> None:
    """Unit-level: interleaved publishes for two jobs; a terminal event for
    one job clears only its own replay buffer."""
    b = broker_mod.get_broker()
    q = broker_mod.get_event_broker().subscribe()
    for i in range(3):
        b.publish("A", broker_mod.delta_event(f"a{i}"))
        b.publish("B", broker_mod.delta_event(f"b{i}"))
    assert broker_mod.get_stream_buffer("A") == "a0a1a2"
    assert broker_mod.get_stream_buffer("B") == "b0b1b2"
    b.publish("B", broker_mod.done_event("b"))
    assert broker_mod.get_stream_buffer("B") == ""
    assert broker_mod.get_stream_buffer("A") == "a0a1a2"
    got = [q.get_nowait() for _ in range(7)]
    for e in got:
        if e["type"] == "delta":
            assert e["delta"][0] == e["job_id"].lower()


# ---------------------------------------------------------------------------
# 3. POST /jobs/{id}/frames — two jobs concurrently
# ---------------------------------------------------------------------------

_SEGMENTS = [{"start": 10.0, "text": "Now watch this, follow along."}]


def _make_audio_job(key: str) -> str:
    job = repo.create_job(url=f"https://video.example/{key}", kind="page")
    repo.mark_done(
        job.id,
        raw_text="raw",
        summary_md="sum",
        transcript_source="whisper",
        transcript_language="en",
        raw_segments_json=json.dumps(_SEGMENTS),
    )
    return job.id


async def test_frames_two_jobs_concurrently_write_own_dirs(
    env: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mocks only yt-dlp (`_download_video_section`) and ffmpeg
    (`_extract_frames`) — the real `fetch_frames` decides out_dir. Job A's
    download parks until job B's whole request has completed."""
    from src.workers import frames as frames_mod

    job_a, job_b = _make_audio_job("A"), _make_audio_job("B")
    url_to_key = {"https://video.example/A": "A", "https://video.example/B": "B"}
    b_done = asyncio.Event()
    a_started = asyncio.Event()
    order: list[str] = []

    async def fake_download(*, url: str, dir: Path, **_: Any) -> Path:
        key = url_to_key[url]
        order.append(f"dl-{key}")
        if key == "A":
            a_started.set()
            await asyncio.wait_for(b_done.wait(), 5)
        else:
            await asyncio.wait_for(a_started.wait(), 5)
        p = dir / f"src-{key}.mp4"
        p.write_bytes(key.encode())
        return p

    async def fake_extract(*, video_path: Path, out_dir: Path, num_frames: int, **_: Any) -> list[Path]:
        await asyncio.sleep(0)
        out_dir.mkdir(parents=True, exist_ok=True)
        key = video_path.read_bytes().decode()
        paths = []
        for i in range(num_frames):
            p = out_dir / f"frame_{i:02d}.jpg"
            p.write_bytes(f"jpeg-{key}".encode())
            paths.append(p)
        return paths

    monkeypatch.setattr(frames_mod, "_download_video_section", fake_download)
    monkeypatch.setattr(frames_mod, "_extract_frames", fake_extract)

    async with _client() as c:

        async def post_b() -> httpx.Response:
            await asyncio.wait_for(a_started.wait(), 5)
            r = await c.post(f"/jobs/{job_b}/frames", json={"seconds": 10.0})
            b_done.set()
            return r

        ra, rb = await asyncio.gather(
            c.post(f"/jobs/{job_a}/frames", json={"seconds": 10.0}), post_b()
        )
        assert ra.status_code == 200, ra.text
        assert rb.status_code == 200, rb.text
        assert order == ["dl-A", "dl-B"]

        frames_root = Path(env.storage.data_dir) / "frames"
        for jid, key, resp in ((job_a, "A", ra), (job_b, "B", rb)):
            items = resp.json()["items"]
            assert items
            for it in items:
                assert it["frame_url"].startswith(f"/jobs/{jid}/frames/t10/"), it
                rel = it["frame_url"].split(f"/jobs/{jid}/frames/", 1)[1]
                on_disk = frames_root / jid / rel
                assert on_disk.read_bytes() == f"jpeg-{key}".encode()
                served = await c.get(it["frame_url"])
                assert served.status_code == 200
                assert served.content == f"jpeg-{key}".encode()
            files = list((frames_root / jid).rglob("*.jpg"))
            assert {f.read_bytes() for f in files} == {f"jpeg-{key}".encode()}
        assert sorted(p.name for p in frames_root.iterdir()) == sorted([job_a, job_b])


# ---------------------------------------------------------------------------
# 4. POST /jobs dedupe under concurrency, by (url, kind)
# ---------------------------------------------------------------------------


@pytest.fixture
def noop_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.workers import pipeline as pipeline_mod

    async def _noop(*_a: Any, **_k: Any) -> None:
        return

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _noop)


async def test_concurrent_same_url_different_kinds_make_two_jobs(
    env: Any, noop_pipeline: None
) -> None:
    url = "https://example.com/article-with-video"
    async with _client() as c:
        rs = await asyncio.gather(
            *[
                c.post("/jobs", json={"url": url, "kind": "page", "page_text": "t"})
                for _ in range(3)
            ],
            *[
                c.post(
                    "/jobs",
                    json={"url": url, "kind": "media", "media_url": "https://cdn.example/v.m3u8"},
                )
                for _ in range(3)
            ],
        )
    for r in rs:
        assert r.status_code == 202, r.text
    page_ids = {r.json()["id"] for r in rs[:3]}
    media_ids = {r.json()["id"] for r in rs[3:]}
    assert len(page_ids) == 1
    assert len(media_ids) == 1
    assert page_ids != media_ids
    rows, total = repo.list_jobs(url=url, limit=50)
    assert total == 2
    assert sorted(r.kind for r in rows) == ["media", "page"]
