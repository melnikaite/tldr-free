"""Per-job audio isolation: generic media URLs share yt-dlp ids (``master``
for every ``.../master.m3u8``), so each job downloads into its own
``audio/job-<id>/`` directory. A retry of the same job still reuses it."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.storage import repo
from src.storage.db import dispose_engine, init_engine
from src.storage.migrations import run_migrations
from src.workers import runner as runner_mod
from src.workers.queue import WhisperQueue, WhisperTask
from src.workers.transcribe import TranscribeResult

from .test_runner import _FakeRepo, _wait_until


@dataclass
class _Job:
    id: str
    url: str
    kind: str = "youtube"
    title: str | None = "T"
    audio_path: str | None = None
    audio_duration_seconds: float | None = None


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, fail_for: set[str],
) -> tuple[list[Path], dict[str, bytes]]:
    download_dirs: list[Path] = []
    transcribed: dict[str, bytes] = {}

    async def fake_download_audio(
        *, url: str, cookies: list[Any], dir: Path, http_headers: Any = None,
    ) -> tuple[Path, float | None]:
        # Mimic yt-dlp: id = URL basename, and an existing file is NOT
        # re-downloaded (no overwrites).
        download_dirs.append(dir)
        target = dir / "master.opus"
        if not target.exists():
            target.write_bytes(url.encode())
        return target, 60.0

    async def fake_transcribe_audio(audio_path: Path, **_kw: object) -> TranscribeResult:
        job_id = audio_path.parent.name
        transcribed[job_id] = audio_path.read_bytes()
        if any(job_id.endswith(f) for f in fail_for):
            raise RuntimeError("whisper down")
        return TranscribeResult(
            segments=[{"start": 0.0, "end": 5.0, "text": "hi"}],
            language="en", duration_seconds=60.0,
        )

    async def fake_stream_summarize(text: str, **_kw: Any):
        yield "## Summary\n\nok"

    async def fake_metadata(**_kw: Any) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(runner_mod.youtube, "download_audio", fake_download_audio)
    monkeypatch.setattr(runner_mod.transcribe, "transcribe_audio", fake_transcribe_audio)
    monkeypatch.setattr(runner_mod.youtube, "fetch_video_metadata", fake_metadata)
    monkeypatch.setattr(runner_mod.llm_summary, "stream_summarize", fake_stream_summarize)
    monkeypatch.setattr(runner_mod, "_audio_dir", lambda: tmp_path)
    return download_dirs, transcribed


async def _run(fake_repo: _FakeRepo, tasks: list[WhisperTask], until: Any) -> None:
    q = WhisperQueue()
    for t in tasks:
        await q.put(t)
    worker = asyncio.create_task(runner_mod.whisper_worker(q, fake_repo))
    try:
        await _wait_until(until)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_jobs_with_identical_ytdlp_ids_get_different_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    a = _Job(id="A", url="https://cdn-a.example/hls/master.m3u8")
    b = _Job(id="B", url="https://cdn-b.example/hls/master.m3u8")
    fake_repo = _FakeRepo({"A": a, "B": b})  # type: ignore[dict-item]
    dirs, transcribed = _install_fakes(monkeypatch, tmp_path, fail_for={"A"})

    await _run(
        fake_repo,
        [WhisperTask(job_id="A", url=a.url, cookies=[]),
         WhisperTask(job_id="B", url=b.url, cookies=[])],
        lambda: bool(fake_repo.failed_calls) and bool(fake_repo.done_calls),
    )

    assert dirs == [tmp_path / "job-A", tmp_path / "job-B"]
    # B was transcribed from its own download, not A's leftover.
    assert transcribed["job-A"] == a.url.encode()
    assert transcribed["job-B"] == b.url.encode()
    # A failed after download: its file is kept for retry; B's dir is gone.
    a_path = tmp_path / "job-A" / "master.opus"
    assert a_path.exists()
    assert not (tmp_path / "job-B").exists()
    stored = [c["audio_path"] for c in fake_repo.set_audio_calls if c["audio_path"]]
    assert stored == [str(a_path), str(tmp_path / "job-B" / "master.opus")]


@pytest.mark.asyncio
async def test_same_job_retry_reuses_its_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job-A"
    job_dir.mkdir()
    cached = job_dir / "master.opus"
    cached.write_bytes(b"first download")
    a = _Job(id="A", url="https://cdn/master.m3u8", audio_path=str(cached),
             audio_duration_seconds=60.0)
    fake_repo = _FakeRepo({"A": a})  # type: ignore[dict-item]
    dirs, transcribed = _install_fakes(monkeypatch, tmp_path, fail_for=set())

    await _run(fake_repo, [WhisperTask(job_id="A", url=a.url, cookies=[])],
               lambda: bool(fake_repo.done_calls))

    assert dirs == [], "retry must reuse the cached file, not re-download"
    assert transcribed["job-A"] == b"first download"
    assert not job_dir.exists(), "per-job dir removed after success"


@pytest.mark.asyncio
async def test_legacy_flat_audio_path_still_reused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    legacy = tmp_path / "abc.opus"
    legacy.write_bytes(b"legacy")
    a = _Job(id="L", url="https://www.youtube.com/watch?v=abcabcabcab",
             audio_path=str(legacy), audio_duration_seconds=60.0)
    fake_repo = _FakeRepo({"L": a})  # type: ignore[dict-item]
    dirs, _ = _install_fakes(monkeypatch, tmp_path, fail_for=set())

    await _run(fake_repo, [WhisperTask(job_id="L", url=a.url, cookies=[])],
               lambda: bool(fake_repo.done_calls))

    assert dirs == []
    assert not legacy.exists()
    assert tmp_path.exists()


@pytest.fixture
def isolated_db(tmp_path: Path):
    engine = init_engine(tmp_path / "repo.db")
    run_migrations(engine)
    try:
        yield engine
    finally:
        dispose_engine()


def test_delete_job_removes_per_job_audio_dir(isolated_db, tmp_path: Path) -> None:
    j = repo.create_job(url="https://x/master.m3u8", kind="media")
    d = tmp_path / "audio" / f"job-{j.id}"
    d.mkdir(parents=True)
    (d / "master.opus").write_bytes(b"x")
    (d / "master.mp4.part").write_bytes(b"x")
    repo.set_audio(j.id, audio_path=str(d / "master.opus"), audio_duration_seconds=1.0)
    assert repo.delete_job(j.id)
    assert not d.exists()


def test_delete_job_keeps_shared_legacy_audio(isolated_db, tmp_path: Path) -> None:
    shared = tmp_path / "audio" / "master.opus"
    shared.parent.mkdir(parents=True)
    shared.write_bytes(b"x")
    j1 = repo.create_job(url="https://a/master.m3u8", kind="media")
    j2 = repo.create_job(url="https://b/master.m3u8", kind="media")
    repo.set_audio(j1.id, audio_path=str(shared), audio_duration_seconds=1.0)
    repo.set_audio(j2.id, audio_path=str(shared), audio_duration_seconds=1.0)
    repo.delete_job(j1.id)
    assert shared.exists(), "still referenced by j2"
    repo.delete_job(j2.id)
    assert not shared.exists()
