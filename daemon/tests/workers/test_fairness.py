"""Fairness/concurrency tests for the Whisper worker pool + HTTP gating.

Covers the point of this whole feature: a short job's chunk should be able
to complete while a long job (many chunks) is still mid-flight, once
``whisper.max_concurrent_jobs >= 2`` lets a pool of workers pull from the
queue concurrently. Two independent knobs are exercised separately:

- ``runner.whisper_worker`` pool size (queue-level fairness — which JOB
  gets picked up next).
- ``transcribe._global_whisper_lock`` / the per-job ``asyncio.Semaphore(1)``
  (HTTP-level gating — how many Whisper REQUESTS may be in flight at once,
  globally and per job).

See ``.claude/workers.md`` and ``WhisperConfig.max_concurrent_requests``'s
comment for the measured fact that motivates all of this: the real backend
serialises Whisper requests FIFO, so concurrency here is a fairness knob,
never a throughput one.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from src.workers import queue as queue_mod
from src.workers import runner as runner_mod
from src.workers import transcribe
from src.workers.control import reset_control
from src.workers.queue import WhisperQueue, WhisperTask


@pytest.fixture(autouse=True)
def _reset() -> Any:
    # transcribe._global_whisper_lock is @lru_cache'd (mirrors
    # llm.client._llm_lock) so it binds to one event loop for the process
    # lifetime in production; tests must clear it so each test gets a fresh
    # semaphore bound to its own loop and reflecting whatever config it set.
    transcribe._global_whisper_lock.cache_clear()
    reset_control()
    queue_mod.reset_queue()
    yield
    transcribe._global_whisper_lock.cache_clear()
    reset_control()
    queue_mod.reset_queue()


class _FakeRepo:
    def mark_failed(self, job_id: str, *, error: str) -> None:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# 1. Queue-level fairness: pool of 2 workers interleaves a short job's
#    single chunk with a long job's multi-chunk sequence.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_short_job_completes_before_long_job_with_pool_of_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completions: dict[str, float] = {}
    long_chunk_finish_times: list[float] = []

    async def fake_process_one(
        task_url: str,
        task_cookies: list[Any],
        task_job_id: str,
        repo_module: object,
        task_page_text: str | None = None,
    ) -> None:
        if task_job_id == "long":
            for _ in range(5):
                await asyncio.sleep(0.03)
                long_chunk_finish_times.append(time.monotonic())
        else:
            await asyncio.sleep(0.03)
        completions[task_job_id] = time.monotonic()

    monkeypatch.setattr(runner_mod, "_process_one", fake_process_one)

    q = WhisperQueue()
    # Long job enqueued FIRST — with a single serial worker (the old
    # behaviour) the short job would have to wait for all 5 of the long
    # job's chunks. With a pool of 2, a second worker can pick up the
    # short job immediately instead of queueing behind it.
    await q.put(WhisperTask(job_id="long", url="https://x/long"))
    await q.put(WhisperTask(job_id="short", url="https://x/short"))

    workers = [
        asyncio.create_task(runner_mod.whisper_worker(q, _FakeRepo()))
        for _ in range(2)
    ]
    try:
        deadline = time.monotonic() + 5.0
        while len(completions) < 2 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
    finally:
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    assert set(completions) == {"long", "short"}
    assert long_chunk_finish_times, "long job never produced any chunk"
    # The whole point: short finishes before the long job's LAST chunk,
    # i.e. it did not queue behind the entire long sequence.
    assert completions["short"] < long_chunk_finish_times[-1]


@pytest.mark.asyncio
async def test_pool_of_one_still_processes_both_jobs_without_deadlock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Degenerate case: whisper.max_concurrent_jobs=1 (a single worker, the
    old shape) must still process every job correctly — no fairness
    benefit expected here, just no regression/deadlock."""
    completions: list[str] = []

    async def fake_process_one(
        task_url: str,
        task_cookies: list[Any],
        task_job_id: str,
        repo_module: object,
        task_page_text: str | None = None,
    ) -> None:
        await asyncio.sleep(0.01)
        completions.append(task_job_id)

    monkeypatch.setattr(runner_mod, "_process_one", fake_process_one)

    q = WhisperQueue()
    await q.put(WhisperTask(job_id="a", url="https://x/a"))
    await q.put(WhisperTask(job_id="b", url="https://x/b"))

    worker = asyncio.create_task(runner_mod.whisper_worker(q, _FakeRepo()))
    try:
        await asyncio.wait_for(_wait_until(lambda: len(completions) == 2), timeout=5.0)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

    # Single worker: strictly FIFO, no interleaving possible or expected.
    assert completions == ["a", "b"]


async def _wait_until(predicate: Any, *, interval: float = 0.01) -> None:
    while not predicate():
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# 2. HTTP-level global bound: max_concurrent_requests=N is actually
#    respected across concurrent "jobs".
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_global_bound_respected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = transcribe.get_config()
    monkeypatch.setattr(cfg.whisper, "max_concurrent_requests", 2)
    monkeypatch.setattr(transcribe, "get_config", lambda: cfg)
    transcribe._global_whisper_lock.cache_clear()

    lock = asyncio.Lock()
    concurrent = 0
    max_concurrent = 0

    async def fake_post_audio(path: Path, **_kwargs: object) -> dict:
        nonlocal concurrent, max_concurrent
        async with lock:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.05)
        async with lock:
            concurrent -= 1
        # A segment spanning far past any window_duration used below avoids
        # tripping the coverage-recheck machinery — we're testing the HTTP
        # gate here, not coverage logic (covered separately).
        return {"segments": [{"start": 0.0, "end": 999999.0, "text": "hi"}], "language": "en"}

    monkeypatch.setattr(transcribe, "_post_audio", fake_post_audio)

    paths = []
    for i in range(6):
        p = tmp_path / f"job{i}.opus"
        p.write_bytes(b"x")
        paths.append(p)

    await asyncio.gather(
        *(transcribe.transcribe_audio(p, total_duration=None) for p in paths)
    )

    assert max_concurrent <= 2
    # Not just "never exceeds" — confirm it actually reaches the limit, so
    # this isn't accidentally passing because everything ran serially.
    assert max_concurrent == 2


# ---------------------------------------------------------------------------
# 3. HTTP-level per-job cap of 1: two concurrent jobs, each with several
#    chunks, never have 2+ of their OWN chunks in flight at once — while
#    the cross-job total may exceed 1.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_job_cap_of_one_while_global_allows_more(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = transcribe.get_config()
    monkeypatch.setattr(cfg.whisper, "max_upload_mb", 1)
    monkeypatch.setattr(cfg.whisper, "max_concurrent_requests", 4)  # not the limiting factor
    monkeypatch.setattr(transcribe, "get_config", lambda: cfg)
    transcribe._global_whisper_lock.cache_clear()

    # Distinct, non-overlapping substrings (not "job_a"/"job_b" — "a" is a
    # substring of "job_b.opus" too via other letters, so a naive substring
    # check on chunk filenames derived from these would misclassify).
    job_a = tmp_path / "jobAAA.opus"
    job_a.write_bytes(b"x" * (2 * 1024 * 1024))
    job_b = tmp_path / "jobBBB.opus"
    job_b.write_bytes(b"x" * (2 * 1024 * 1024))

    def job_key(path: Path) -> str:
        return "a" if "AAA" in path.name else "b"

    def fake_split_audio(
        audio_path: Path, num_chunks: int, chunk_seconds: float
    ) -> list[tuple[Path, float]]:
        key = job_key(audio_path)
        tag = "AAA" if key == "a" else "BBB"
        chunks = []
        for i in range(num_chunks):
            p = tmp_path / f"{tag}_chunk{i}.opus"
            p.write_bytes(b"x")
            chunks.append((p, i * chunk_seconds))
        return chunks

    lock = asyncio.Lock()
    per_job_concurrent = {"a": 0, "b": 0}
    per_job_max = {"a": 0, "b": 0}
    global_concurrent = 0
    global_max = 0

    async def fake_post_audio(path: Path, **_kwargs: object) -> dict:
        nonlocal global_concurrent, global_max
        key = job_key(path)
        async with lock:
            per_job_concurrent[key] += 1
            per_job_max[key] = max(per_job_max[key], per_job_concurrent[key])
            global_concurrent += 1
            global_max = max(global_max, global_concurrent)
        await asyncio.sleep(0.03)
        async with lock:
            per_job_concurrent[key] -= 1
            global_concurrent -= 1
        # Span the whole window so _ensure_coverage sees full coverage and
        # doesn't issue extra recheck calls of its own.
        return {"segments": [{"start": 0.0, "end": 999999.0, "text": key}], "language": "en"}

    monkeypatch.setattr(transcribe, "_split_audio", fake_split_audio)
    monkeypatch.setattr(transcribe, "_post_audio", fake_post_audio)

    await asyncio.gather(
        transcribe.transcribe_audio(job_a, total_duration=40.0),
        transcribe.transcribe_audio(job_b, total_duration=40.0),
    )

    assert per_job_max["a"] == 1
    assert per_job_max["b"] == 1
    # Cross-job concurrency DID happen — otherwise this test would pass
    # vacuously even with a bug forcing everything serial.
    assert global_max >= 2


# ---------------------------------------------------------------------------
# 4. Ordering/monotonicity survives concurrency: gating never reorders.
# ---------------------------------------------------------------------------


def _payload(segments: list[dict]) -> dict:
    return {"segments": segments, "language": "en", "duration": 100.0}


@pytest.mark.asyncio
async def test_ensure_coverage_ordering_survives_concurrent_jobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(transcribe, "_PREFIX_DISTRUST_SECONDS", 0.0)

    cfg = transcribe.get_config()
    monkeypatch.setattr(cfg.whisper, "max_concurrent_requests", 2)
    monkeypatch.setattr(transcribe, "get_config", lambda: cfg)
    transcribe._global_whisper_lock.cache_clear()

    audio_a = tmp_path / "job_a.opus"
    audio_a.write_bytes(b"x")
    audio_b = tmp_path / "job_b.opus"
    audio_b.write_bytes(b"x")

    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        return src_path.parent / f"{src_path.stem}_retry.opus"

    concurrent = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    async def fake_post(path: Path, **_kwargs: object) -> dict:
        nonlocal concurrent, max_concurrent
        async with lock:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.02)
        async with lock:
            concurrent -= 1
        if "job_a" in path.name:
            return _payload([{"start": 0.0, "end": 30.0, "text": "AAA recovered"}])
        return _payload([{"start": 0.0, "end": 30.0, "text": "BBB recovered"}])

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)
    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    segments_a = [{"start": 0.0, "end": 10.0, "text": "intro A"}]
    segments_b = [{"start": 0.0, "end": 10.0, "text": "intro B"}]

    async def run(segments: list[dict], audio_path: Path) -> tuple[list[dict], float]:
        return await transcribe._ensure_coverage(
            list(segments),
            source_path=audio_path,
            window_duration=40.0,
            per_job_lock=asyncio.Semaphore(1),
        )

    (result_a, missing_a), (result_b, missing_b) = await asyncio.gather(
        run(segments_a, audio_a), run(segments_b, audio_b),
    )

    # Interleaving actually happened — otherwise this test would prove
    # nothing about concurrent-safety.
    assert max_concurrent >= 2

    def is_monotonic(segs: list[dict]) -> bool:
        starts = [s["start"] for s in segs]
        return all(a <= b for a, b in zip(starts, starts[1:], strict=False))

    assert is_monotonic(result_a)
    assert is_monotonic(result_b)

    texts_a = [s["text"] for s in result_a]
    texts_b = [s["text"] for s in result_b]
    assert "AAA recovered" in texts_a and "BBB recovered" not in texts_a
    assert "BBB recovered" in texts_b and "AAA recovered" not in texts_b

    # Baseline: run the exact same two calls again, this time sequentially
    # (no gather) — the result for each job must be byte-identical to the
    # concurrent run. Gating only wraps the network call and never reorders
    # anything, so contention must not change the outcome.
    baseline_a, _ = await run(segments_a, audio_a)
    baseline_b, _ = await run(segments_b, audio_b)
    assert result_a == baseline_a
    assert result_b == baseline_b


# ---------------------------------------------------------------------------
# 5. Degenerate case: max_concurrent_requests=1 still works end-to-end for
#    a single multi-chunk job, no deadlock.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_concurrent_requests_one_no_deadlock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = transcribe.get_config()
    monkeypatch.setattr(cfg.whisper, "max_upload_mb", 1)
    monkeypatch.setattr(cfg.whisper, "max_concurrent_requests", 1)
    monkeypatch.setattr(transcribe, "get_config", lambda: cfg)
    transcribe._global_whisper_lock.cache_clear()

    audio = tmp_path / "big.opus"
    audio.write_bytes(b"x" * (2 * 1024 * 1024))

    def fake_split_audio(
        audio_path: Path, num_chunks: int, chunk_seconds: float
    ) -> list[tuple[Path, float]]:
        chunks = []
        for i in range(num_chunks):
            p = tmp_path / f"chunk{i}.opus"
            p.write_bytes(b"x")
            chunks.append((p, i * chunk_seconds))
        return chunks

    call_count = 0

    async def fake_post_audio(path: Path, **_kwargs: object) -> dict:
        nonlocal call_count
        call_count += 1
        return {"segments": [{"start": 0.0, "end": 999999.0, "text": "ok"}], "language": "en"}

    monkeypatch.setattr(transcribe, "_split_audio", fake_split_audio)
    monkeypatch.setattr(transcribe, "_post_audio", fake_post_audio)

    result = await asyncio.wait_for(
        transcribe.transcribe_audio(audio, total_duration=40.0), timeout=5.0
    )

    assert call_count >= 1
    assert result.segments
    assert result.missing_seconds == 0.0
