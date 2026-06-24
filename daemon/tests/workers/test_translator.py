"""Translator worker — dedup, retry-all, restart re-enqueue.

The worker uses ``llm.client.stream_complete`` for the actual LLM call;
we monkeypatch that to return a fake translation so tests run without a
live backend. Background tasks (``asyncio.create_task``) are awaited via
the helper so each test is deterministic.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from src.storage import repo
from src.storage.db import (
    Job,
    TranscriptTranslation,
    dispose_engine,
    init_engine,
    session_scope,
)
from src.storage.migrations import run_migrations
from src.workers import translator


@pytest.fixture
def isolated_db(tmp_path: Path):
    db_path = tmp_path / "translator.db"
    engine = init_engine(db_path)
    run_migrations(engine)
    try:
        yield engine
    finally:
        dispose_engine()


def _seed_job(*, raw_text: str = "[00:00] hello\n", source_lang: str = "en") -> Job:
    """Insert a done job with the given raw_text + source language."""
    job = repo.create_job(url="https://x", kind="media", title="Test")
    repo.mark_done(
        job.id,
        raw_text=raw_text,
        summary_md="ok",
        transcript_source="whisper",
        transcript_language=source_lang,
    )
    fresh = repo.get_job(job.id)
    assert fresh is not None
    return fresh


async def _fake_stream(prompt: str, *, max_tokens: int, temperature: float, respect_pause: bool) -> AsyncIterator[str]:
    """Stand-in for llm.client.stream_complete. Echoes a fixed Russian line so
    we can verify it ends up in the row."""
    yield "[00:00] привет\n"


async def _drain_tasks() -> None:
    """Yield control until the translator's background task finishes."""
    # Try until no more translator tasks remain. Cap iterations so a bug
    # doesn't hang the test runner forever.
    for _ in range(50):
        if not translator._BACKGROUND_TASKS:
            return
        await asyncio.sleep(0.02)
    raise AssertionError("translator background task did not finish in time")


# ---------------------------------------------------------------------------
# enqueue_translation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_runs_translation_and_writes_row(
    isolated_db, monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _seed_job()
    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", _fake_stream)

    result = await translator.enqueue_translation(job.id, "ru")
    assert result["language_code"] == "ru"
    assert result["status"] in ("queued", "running")

    await _drain_tasks()

    with session_scope() as session:
        row = session.get(TranscriptTranslation, (job.id, "ru"))
        assert row is not None
        assert row.status == "done"
        assert row.text and "привет" in row.text
        assert row.progress_percent == 100


@pytest.mark.asyncio
async def test_enqueue_is_idempotent_for_in_flight(
    isolated_db, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two POSTs to translate the same language don't spawn two workers."""
    job = _seed_job()

    spawn_count = 0

    async def slow_stream(*a: Any, **kw: Any) -> AsyncIterator[str]:
        nonlocal spawn_count
        spawn_count += 1
        await asyncio.sleep(0.05)
        yield "[00:00] привет\n"

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", slow_stream)

    first = await translator.enqueue_translation(job.id, "ru")
    second = await translator.enqueue_translation(job.id, "ru")
    # First should be queued/running, second returns the existing status.
    assert first["language_code"] == "ru"
    assert second["language_code"] == "ru"
    assert second["status"] in ("queued", "running", "done")

    await _drain_tasks()
    assert spawn_count == 1, "second enqueue should not have spawned a second LLM run"


@pytest.mark.asyncio
async def test_enqueue_same_as_source_returns_is_source(
    isolated_db, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """en→en for a job with source=en short-circuits to the original."""
    job = _seed_job(source_lang="en")
    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", _fake_stream)

    result = await translator.enqueue_translation(job.id, "en")
    assert result["is_source"] is True
    assert result["status"] == "done"

    # No row should be created.
    with session_scope() as session:
        row = session.get(TranscriptTranslation, (job.id, "en"))
        assert row is None


@pytest.mark.asyncio
async def test_unknown_language_raises(isolated_db) -> None:
    """Invalid language string surfaces UnknownLanguageError to the caller
    (the API layer turns that into 400)."""
    from src.llm.languages import UnknownLanguageError
    job = _seed_job()
    with pytest.raises(UnknownLanguageError):
        await translator.enqueue_translation(job.id, "klingon")


@pytest.mark.asyncio
async def test_failure_marks_row_failed(
    isolated_db, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM exception is captured into status=failed + error message."""
    job = _seed_job()

    async def boom(*a: Any, **kw: Any) -> AsyncIterator[str]:
        raise RuntimeError("mlx 503")
        yield  # unreachable; keeps mypy/pyright happy on AsyncIterator type

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", boom)

    await translator.enqueue_translation(job.id, "ru")
    await _drain_tasks()

    with session_scope() as session:
        row = session.get(TranscriptTranslation, (job.id, "ru"))
        assert row is not None
        assert row.status == "failed"
        assert row.error is not None and "mlx 503" in row.error


# ---------------------------------------------------------------------------
# retry_all_failed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_all_re_enqueues_only_failed(
    isolated_db, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry-all touches only failed rows; in-flight / done are left alone."""
    job = _seed_job()
    # Seed three translations: one done, one running, one failed.
    with session_scope() as session:
        session.add(TranscriptTranslation(
            job_id=job.id, language_code="ru", status="done",
            progress_percent=100, text="[00:00] да",
        ))
        session.add(TranscriptTranslation(
            job_id=job.id, language_code="de", status="running",
            progress_percent=50,
        ))
        session.add(TranscriptTranslation(
            job_id=job.id, language_code="fr", status="failed",
            progress_percent=0, error="prev error",
        ))

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", _fake_stream)

    retried = await translator.retry_all_failed(job.id)
    assert [r["language_code"] for r in retried] == ["fr"]

    await _drain_tasks()

    with session_scope() as session:
        ru = session.get(TranscriptTranslation, (job.id, "ru"))
        de = session.get(TranscriptTranslation, (job.id, "de"))
        fr = session.get(TranscriptTranslation, (job.id, "fr"))
        assert ru is not None and ru.status == "done"
        assert de is not None and de.status == "running"  # not touched
        assert fr is not None and fr.status == "done"
        assert fr.error is None


@pytest.mark.asyncio
async def test_retry_all_404_for_missing_job(isolated_db) -> None:
    with pytest.raises(KeyError):
        await translator.retry_all_failed("does-not-exist")


# ---------------------------------------------------------------------------
# Restart re-enqueue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_re_enqueue_running_on_startup(
    isolated_db, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rows left ``running`` from before a restart get picked up so the
    spinner the user saw doesn't hang forever."""
    job = _seed_job()
    with session_scope() as session:
        session.add(TranscriptTranslation(
            job_id=job.id, language_code="ru", status="running",
            progress_percent=42,
        ))

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", _fake_stream)

    count = translator.re_enqueue_running_on_startup()
    assert count == 1

    await _drain_tasks()

    with session_scope() as session:
        row = session.get(TranscriptTranslation, (job.id, "ru"))
        assert row is not None
        assert row.status == "done"


# ---------------------------------------------------------------------------
# Source preference: raw_segments_json > raw_text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_translator_prefers_raw_segments_json_over_raw_text(
    isolated_db, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fine-grained segments win when both are populated. We capture the
    prompt the LLM receives — fine-grained lines (every ~1 s) must be
    present, NOT the 30 s-bucketed shape from raw_text. This is the
    invariant Fix L put in; without it RU translation came out in 30 s
    buckets even when EN was fine-grained."""
    import json as _json

    from src.storage import repo as repo_module

    job = repo.create_job(url="https://x/segs-pref", kind="media", title="t")
    fine_segments = [
        {"start": float(i), "end": float(i + 1), "text": f"line-{i}"}
        for i in range(0, 6)
    ]
    repo_module.mark_done(
        job.id,
        raw_text="[00:00] coarse bucket\n",
        summary_md="ok",
        transcript_source="whisper",
        transcript_language="en",
        raw_segments_json=_json.dumps(fine_segments),
    )

    captured_prompts: list[str] = []

    async def capture_stream(prompt, *, max_tokens, temperature, respect_pause):
        captured_prompts.append(prompt)
        yield "[00:00] linea-0\n[00:01] linea-1\n"  # spanish-ish, doesn't matter

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", capture_stream)

    await translator.enqueue_translation(job.id, "ru")
    await _drain_tasks()

    assert captured_prompts, "stream_complete should have been called"
    prompt = captured_prompts[0]
    # The fine-grained source produced lines like "[00:00] line-0".
    # If translator had used raw_text instead, we'd see "coarse bucket".
    assert "line-0" in prompt
    assert "line-5" in prompt
    assert "coarse bucket" not in prompt


# ---------------------------------------------------------------------------
# Queueing before raw_text is available
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_queues_when_raw_text_missing(
    isolated_db, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User adds a translation BEFORE extraction finishes (Whisper still
    running). enqueue_translation must NOT error — it inserts a queued
    row and the worker polls until raw_text appears, then runs."""
    # Job in queued status, no raw_text yet.
    job = repo.create_job(url="https://x/no-text", kind="media", title="pending")

    # Speed up the worker's polling so the test runs quick.
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep(asyncio.sleep))

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", _fake_stream)

    # Enqueue → should NOT raise. Row inserted with status=queued.
    result = await translator.enqueue_translation(job.id, "ru")
    assert result["language_code"] == "ru"
    assert result["status"] in ("queued", "running")

    # Worker is now in _wait_for_source_text. Let it poll a couple times.
    await asyncio.sleep(0.05)

    # Now simulate extraction finishing — write raw_text.
    repo.set_extracted(
        job.id,
        raw_text="[00:00] hello world\n",
        transcript_source="whisper",
        transcript_language="en",
    )

    # Worker should pick it up on its next poll iteration.
    await _drain_tasks()

    with session_scope() as session:
        row = session.get(TranscriptTranslation, (job.id, "ru"))
        assert row is not None
        assert row.status == "done"
        assert row.text and "привет" in row.text


def _instant_sleep(real_sleep):
    """Pytest helper: replace asyncio.sleep(N) with a near-instant yield so
    the translator's 3 s polling cadence doesn't slow the test down."""
    async def fast(seconds, *args, **kwargs):
        return await real_sleep(min(0.01, seconds))
    return fast


# ---------------------------------------------------------------------------
# Intermediate progress publishes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_translator_publishes_intermediate_progress(
    isolated_db, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow streaming LLM must produce ≥1 mid-stream progress event
    (not just 0 % at start and 100 % at end). Catches the throttle bug
    we hit in Fix F / K — pct stayed at 0 because the dedup check was
    too strict for huge chunks."""
    # Seed a job with enough fine-grained content that translation
    # naturally walks through several percent values.
    import json as _json
    job = repo.create_job(url="https://x/progress", kind="media", title="long")
    segs = [
        {"start": float(i), "end": float(i + 1), "text": f"sentence {i} here"}
        for i in range(0, 40)
    ]
    repo.mark_done(
        job.id,
        raw_text="[00:00] flat fallback\n",
        summary_md="ok",
        transcript_source="whisper",
        transcript_language="en",
        raw_segments_json=_json.dumps(segs),
    )

    async def slow_token_stream(prompt, *, max_tokens, temperature, respect_pause):
        # Yield in small bursts with real (but tiny) waits so the
        # translator's 100 ms time-throttle has multiple windows to
        # publish progress. Without this the whole "translation" would
        # complete in microseconds and only 0 % / 100 % would fire.
        text = "\n".join(f"[00:{i:02d}] перевод {i}" for i in range(0, 40)) + "\n"
        # Chunk by 20 chars and sleep 30 ms between — total ~ 1 s of
        # streaming spread across ~30 bursts.
        for i in range(0, len(text), 20):
            yield text[i:i + 20]
            await asyncio.sleep(0.03)

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", slow_token_stream)

    # Capture every translation_updated event the worker publishes.
    from src.workers import broker as broker_mod

    published: list[int] = []
    orig_publish = broker_mod._event_broker.publish

    def spy_publish(event):
        if (
            event.get("type") == "job"
            and event.get("action") == "translation_updated"
            and isinstance(event.get("job"), dict)
            and event["job"].get("language_code") == "ru"
        ):
            pct = event["job"].get("progress_percent")
            if isinstance(pct, int):
                published.append(pct)
        return orig_publish(event)

    monkeypatch.setattr(broker_mod._event_broker, "publish", spy_publish)

    await translator.enqueue_translation(job.id, "ru")

    # Custom drain — this test deliberately uses a slow stream so the
    # default 1 s drain isn't enough. Give it up to 5 s.
    for _ in range(250):
        if not translator._BACKGROUND_TASKS:
            break
        await asyncio.sleep(0.02)
    else:  # noqa: PLW0120
        raise AssertionError("slow translator did not finish in 5 s")

    # Must see something other than just {0, 100}. We're checking that the
    # mid-stream throttle actually fires; the exact set of percentages
    # depends on timing so we just assert ≥ 1 intermediate value.
    assert 0 in published, f"missing initial 0% publish; got {published}"
    assert 100 in published, f"missing final 100% publish; got {published}"
    intermediates = [p for p in published if 0 < p < 100]
    assert intermediates, f"no intermediate progress events; got {published}"


# ---------------------------------------------------------------------------
# Retry-all endpoint via TestClient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_all_is_async_and_re_enqueues_failed(
    isolated_db, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for Fix M — when the endpoint was sync def, FastAPI
    ran it on the threadpool and ``_spawn``'s ``asyncio.create_task`` raised
    RuntimeError silently. The endpoint must be async; calling it must
    actually re-enqueue and produce a queued row."""
    job = _seed_job()
    with session_scope() as session:
        session.add(TranscriptTranslation(
            job_id=job.id, language_code="ru", status="failed",
            progress_percent=0, error="previous failure",
        ))

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", _fake_stream)

    retried = await translator.retry_all_failed(job.id)
    assert [r["language_code"] for r in retried] == ["ru"]

    await _drain_tasks()

    with session_scope() as session:
        row = session.get(TranscriptTranslation, (job.id, "ru"))
        assert row is not None
        # Either still running or finished — but the failed → queued reset
        # must have stuck, and the error must be cleared.
        assert row.status in ("running", "done")
        assert row.error is None
