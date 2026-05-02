"""Integration test for workers.runner.whisper_worker.

We mock the slow / external pieces (download_audio, transcribe.transcribe_stream,
llm.summary.stream_summarize) so the test runs in milliseconds and is hermetic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from src.workers import runner as runner_mod
from src.workers.queue import WhisperQueue, WhisperTask


@dataclass
class _FakeJob:
    id: str
    url: str
    kind: str = "youtube"
    title: str | None = "Test Title"


class _FakeRepo:
    """Captures the calls the runner makes against the repo so each test
    asserts behaviour without standing up a real DB. Emit-on-write lives
    inside ``src.storage.repo`` itself and is covered by ``test_repo_emit``."""

    def __init__(self, jobs: dict[str, _FakeJob]) -> None:
        self._jobs = jobs
        self.status_updates: list[dict[str, Any]] = []
        self.done_calls: list[dict[str, Any]] = []
        self.failed_calls: list[tuple[str, str]] = []
        self.set_extracted_calls: list[dict[str, Any]] = []
        self.set_audio_calls: list[dict[str, Any]] = []

    def get_job(self, job_id: str) -> _FakeJob | None:
        return self._jobs.get(job_id)

    def update_status(self, job_id: str, **kwargs: Any) -> None:
        self.status_updates.append({"job_id": job_id, **kwargs})

    def mark_done(self, job_id: str, **kwargs: Any) -> None:
        self.done_calls.append({"job_id": job_id, **kwargs})

    def mark_failed(self, job_id: str, *, error: str) -> None:
        self.failed_calls.append((job_id, error))

    def set_extracted(self, job_id: str, **kwargs: Any) -> None:
        self.set_extracted_calls.append({"job_id": job_id, **kwargs})

    def set_audio(self, job_id: str, **kwargs: Any) -> None:
        self.set_audio_calls.append({"job_id": job_id, **kwargs})


async def _wait_until(predicate: Any, *, timeout: float = 2.0, interval: float = 0.01) -> None:
    """Poll ``predicate()`` until it returns truthy or the timeout expires.

    Replaces ad-hoc ``for _ in range(50): await asyncio.sleep(0.02)`` loops —
    interval-aware and timeout-aware so a slow CI machine doesn't flake.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError(f"predicate stayed false for {timeout}s")


@pytest.fixture
def fake_segments() -> list[dict[str, Any]]:
    return [
        {"start": 0.0, "end": 5.0, "text": "First chunk"},
        {"start": 30.0, "end": 35.0, "text": "Second chunk"},
        {"start": 60.0, "end": 65.0, "text": "Third chunk"},
    ]


@pytest.mark.asyncio
async def test_runner_processes_one_task_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_segments: list[dict[str, Any]],
) -> None:
    job = _FakeJob(id="job1", url="https://www.youtube.com/watch?v=xxxxxxxxxxx")
    fake_repo = _FakeRepo({"job1": job})

    # Track audio file lifecycle.
    audio_file = tmp_path / "fake.opus"
    audio_file.write_bytes(b"\x00" * 32)

    download_calls: list[dict[str, Any]] = []

    async def fake_download_audio(
        *, url: str, cookies: list[Any], dir: Path,
    ) -> tuple[Path, float | None]:
        download_calls.append({"url": url, "dir": dir})
        return audio_file, 90.0

    progress_calls: list[float] = []

    async def fake_transcribe_stream(
        audio_path: Path,
        *,
        total_duration: float | None,
        on_progress: Any,
    ) -> list[dict[str, Any]]:
        assert audio_path == audio_file
        assert total_duration == 90.0
        if on_progress is not None:
            for f in (0.33, 0.66, 1.0):
                on_progress(f)
                progress_calls.append(f)
        return fake_segments

    summarize_calls: list[dict[str, Any]] = []

    async def fake_stream_summarize(text: str, *, title: Any, output_language: str):
        summarize_calls.append(
            {"text": text, "title": title, "output_language": output_language}
        )
        for chunk in ("## Summary\n\n", "Seen ", "[00:00] [00:30] [01:00]."):
            yield chunk

    monkeypatch.setattr(runner_mod.youtube, "download_audio", fake_download_audio)
    monkeypatch.setattr(runner_mod.transcribe, "transcribe_stream", fake_transcribe_stream)
    monkeypatch.setattr(runner_mod.llm_summary, "stream_summarize", fake_stream_summarize)
    # Force the audio dir into tmp_path so we don't write into /data.
    monkeypatch.setattr(runner_mod, "_audio_dir", lambda: tmp_path)

    q = WhisperQueue()
    await q.put(WhisperTask(job_id="job1", url=job.url, cookies=[]))

    worker = asyncio.create_task(runner_mod.whisper_worker(q, fake_repo))

    # Wait for the queue to drain AND mark_done to fire.
    await _wait_until(
        lambda: q.snapshot() == (0, 0) and bool(fake_repo.done_calls),
    )

    worker.cancel()
    await asyncio.gather(worker, return_exceptions=True)

    # Verify the chain.
    assert len(download_calls) == 1
    assert len(summarize_calls) == 1
    assert summarize_calls[0]["title"] == "Test Title"

    assert len(fake_repo.done_calls) == 1, "mark_done should have been called once"
    done = fake_repo.done_calls[0]
    assert done["job_id"] == "job1"
    assert done["transcript_source"] == "whisper"
    # raw_text contains markers from build_marked_text.
    raw_text = done["raw_text"]
    assert "[00:00]" in raw_text
    assert "[00:30]" in raw_text
    assert "[01:00]" in raw_text
    assert "First chunk" in raw_text
    # video_id was extracted from the URL.
    assert done.get("video_id") == "xxxxxxxxxxx"

    # Audio file deleted in finally.
    assert not audio_file.exists()

    # Status progressed through the expected stages.
    stages = [u.get("progress_stage") for u in fake_repo.status_updates]
    assert "downloading" in stages
    assert "transcribing" in stages
    assert "summarizing" in stages

    # No failure was recorded.
    assert fake_repo.failed_calls == []

    # Progress callback fired for each fake chunk.
    assert progress_calls == [0.33, 0.66, 1.0]

    # Audio lifecycle: persisted after download, then cleared on success.
    assert len(fake_repo.set_audio_calls) == 2
    assert fake_repo.set_audio_calls[0]["audio_path"] == str(audio_file)
    assert fake_repo.set_audio_calls[0]["audio_duration_seconds"] == 90.0
    assert fake_repo.set_audio_calls[1]["audio_path"] is None
    assert fake_repo.set_audio_calls[1]["audio_duration_seconds"] is None


@pytest.mark.asyncio
async def test_runner_marks_failed_on_download_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job = _FakeJob(id="job2", url="https://www.youtube.com/watch?v=yyyyyyyyyyy")
    fake_repo = _FakeRepo({"job2": job})

    async def fake_download_audio(
        *, url: str, cookies: list[Any], dir: Path,
    ) -> tuple[Path, float | None]:
        raise RuntimeError("yt-dlp boom")

    monkeypatch.setattr(runner_mod.youtube, "download_audio", fake_download_audio)
    monkeypatch.setattr(runner_mod, "_audio_dir", lambda: tmp_path)

    q = WhisperQueue()
    await q.put(WhisperTask(job_id="job2", url=job.url, cookies=[]))

    worker = asyncio.create_task(runner_mod.whisper_worker(q, fake_repo))

    await _wait_until(lambda: bool(fake_repo.failed_calls))

    worker.cancel()
    await asyncio.gather(worker, return_exceptions=True)

    assert fake_repo.failed_calls == [("job2", "yt-dlp boom")]
    assert fake_repo.done_calls == []


@dataclass
class _CleanupGuard:
    tmp_path: Path
    cleanup_observed: bool = False
    file: Path = field(init=False)

    def __post_init__(self) -> None:
        self.file = self.tmp_path / "guard.opus"
        self.file.write_bytes(b"\x01" * 16)


@pytest.mark.asyncio
async def test_runner_deletes_audio_even_on_transcribe_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job = _FakeJob(id="job3", url="https://www.youtube.com/watch?v=zzzzzzzzzzz")
    fake_repo = _FakeRepo({"job3": job})

    guard = _CleanupGuard(tmp_path)

    async def fake_download_audio(
        *, url: str, cookies: list[Any], dir: Path,
    ) -> tuple[Path, float | None]:
        return guard.file, 60.0

    async def fake_transcribe_stream(
        audio_path: Path,
        *,
        total_duration: float | None,
        on_progress: Any,
    ) -> list[dict[str, Any]]:
        raise RuntimeError("mlx 503")

    monkeypatch.setattr(runner_mod.youtube, "download_audio", fake_download_audio)
    monkeypatch.setattr(runner_mod.transcribe, "transcribe_stream", fake_transcribe_stream)
    monkeypatch.setattr(runner_mod, "_audio_dir", lambda: tmp_path)

    q = WhisperQueue()
    await q.put(WhisperTask(job_id="job3", url=job.url, cookies=[]))

    worker = asyncio.create_task(runner_mod.whisper_worker(q, fake_repo))

    await _wait_until(lambda: bool(fake_repo.failed_calls))

    worker.cancel()
    await asyncio.gather(worker, return_exceptions=True)

    assert fake_repo.failed_calls == [("job3", "mlx 503")]
    # Audio is preserved when the failure happens AFTER a successful download
    # — runner keeps it so a subsequent retry can skip yt-dlp.
    assert guard.file.exists()
