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
        {"start": 0.0, "end": 5.0, "text": "First chunk."},
        {"start": 30.0, "end": 35.0, "text": "Second chunk."},
        {"start": 60.0, "end": 65.0, "text": "Third chunk."},
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

    async def fake_transcribe_audio(
        audio_path: Path,
        *,
        total_duration: float | None,
    ):
        assert audio_path == audio_file
        assert total_duration == 90.0
        # Real return shape: TranscribeResult with segments + language.
        from src.workers.transcribe import TranscribeResult
        return TranscribeResult(
            segments=fake_segments,
            language="en",
            duration_seconds=total_duration,
        )

    summarize_calls: list[dict[str, Any]] = []

    async def fake_stream_summarize(
        text: str, *, title: Any, output_language: str, from_audio_transcript: bool = False
    ):
        summarize_calls.append(
            {"text": text, "title": title, "output_language": output_language}
        )
        for chunk in ("## Summary\n\n", "Seen ", "[00:00] [00:30] [01:00]."):
            yield chunk

    async def fake_metadata(*, url: str, cookies: list[Any], scratch_dir: Path):
        return {"title": "Canonical Title", "language": "en"}

    monkeypatch.setattr(runner_mod.youtube, "download_audio", fake_download_audio)
    monkeypatch.setattr(runner_mod.transcribe, "transcribe_audio", fake_transcribe_audio)
    monkeypatch.setattr(runner_mod.youtube, "fetch_video_metadata", fake_metadata)
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
    # Authoritative title from yt-dlp metadata overrides the DB row's scrape.
    assert summarize_calls[0]["title"] == "Canonical Title"

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

    # Detected source language persisted on mark_done.
    assert done.get("transcript_language") == "en"

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

    async def fake_transcribe_audio(
        audio_path: Path,
        *,
        total_duration: float | None,
    ):
        raise RuntimeError("mlx 503")

    monkeypatch.setattr(runner_mod.youtube, "download_audio", fake_download_audio)
    monkeypatch.setattr(runner_mod.transcribe, "transcribe_audio", fake_transcribe_audio)
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


# ---------------------------------------------------------------------------
# kind=media: duration probe + page-text fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_media_short_probed_duration_skips_download_uses_page_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Probe returns 1.5s duration -> download never attempted, page_text summarized."""
    job = _FakeJob(
        id="job_media_short", url="https://example.com/ding.mp3",
        kind="media", title="Helpdesk Page",
    )
    fake_repo = _FakeRepo({"job_media_short": job})

    download_calls: list[dict[str, Any]] = []

    async def fake_download_audio(
        *, url: str, cookies: list[Any], dir: Path,
    ) -> tuple[Path, float | None]:
        download_calls.append({"url": url})
        raise AssertionError("download_audio must not be called for a too-short probe")

    async def fake_metadata(*, url: str, cookies: list[Any], scratch_dir: Path):
        return {"title": None, "language": None, "duration": 1.5}

    summarize_calls: list[dict[str, Any]] = []

    async def fake_stream_summarize(
        text: str, *, title: Any, output_language: str, from_audio_transcript: bool = False
    ):
        summarize_calls.append(
            {"text": text, "title": title, "from_audio_transcript": from_audio_transcript}
        )
        yield "Page summary."

    monkeypatch.setattr(runner_mod.youtube, "download_audio", fake_download_audio)
    monkeypatch.setattr(runner_mod.youtube, "fetch_video_metadata", fake_metadata)
    monkeypatch.setattr(runner_mod.llm_summary, "stream_summarize", fake_stream_summarize)
    monkeypatch.setattr(runner_mod, "_audio_dir", lambda: tmp_path)

    q = WhisperQueue()
    await q.put(
        WhisperTask(
            job_id="job_media_short", url=job.url, cookies=[],
            page_text="This is the helpdesk page content.",
        )
    )

    worker = asyncio.create_task(runner_mod.whisper_worker(q, fake_repo))
    await _wait_until(
        lambda: q.snapshot() == (0, 0) and bool(fake_repo.done_calls or fake_repo.failed_calls),
    )
    worker.cancel()
    await asyncio.gather(worker, return_exceptions=True)

    assert download_calls == [], "download_audio must never be called"
    assert fake_repo.failed_calls == []
    assert len(fake_repo.done_calls) == 1
    done = fake_repo.done_calls[0]
    assert done["transcript_source"] == "page_extract"
    assert "Page summary" in done["summary_md"]
    assert summarize_calls[0]["from_audio_transcript"] is False
    assert summarize_calls[0]["text"] == "This is the helpdesk page content."


@pytest.mark.asyncio
async def test_runner_media_long_probed_duration_normal_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_segments: list[dict[str, Any]],
) -> None:
    """Probe returns 900s duration -> normal download + transcribe path."""
    job = _FakeJob(
        id="job_media_long", url="https://example.com/podcast.mp3",
        kind="media", title="Podcast Page",
    )
    fake_repo = _FakeRepo({"job_media_long": job})

    audio_file = tmp_path / "podcast.opus"
    audio_file.write_bytes(b"\x00" * 8)

    download_calls: list[dict[str, Any]] = []

    async def fake_download_audio(
        *, url: str, cookies: list[Any], dir: Path,
    ) -> tuple[Path, float | None]:
        download_calls.append({"url": url})
        return audio_file, 900.0

    async def fake_transcribe_audio(audio_path: Path, *, total_duration: float | None):
        from src.workers.transcribe import TranscribeResult
        return TranscribeResult(segments=fake_segments, language="en", duration_seconds=total_duration)

    async def fake_metadata(*, url: str, cookies: list[Any], scratch_dir: Path):
        return {"title": "Podcast Episode", "language": "en", "duration": 900.0}

    async def fake_stream_summarize(
        text: str, *, title: Any, output_language: str, from_audio_transcript: bool = False
    ):
        yield "Podcast summary."

    monkeypatch.setattr(runner_mod.youtube, "download_audio", fake_download_audio)
    monkeypatch.setattr(runner_mod.transcribe, "transcribe_audio", fake_transcribe_audio)
    monkeypatch.setattr(runner_mod.youtube, "fetch_video_metadata", fake_metadata)
    monkeypatch.setattr(runner_mod.llm_summary, "stream_summarize", fake_stream_summarize)
    monkeypatch.setattr(runner_mod, "_audio_dir", lambda: tmp_path)

    q = WhisperQueue()
    await q.put(
        WhisperTask(job_id="job_media_long", url=job.url, cookies=[], page_text="unused"),
    )

    worker = asyncio.create_task(runner_mod.whisper_worker(q, fake_repo))
    await _wait_until(lambda: q.snapshot() == (0, 0) and bool(fake_repo.done_calls))
    worker.cancel()
    await asyncio.gather(worker, return_exceptions=True)

    assert len(download_calls) == 1
    assert fake_repo.failed_calls == []
    done = fake_repo.done_calls[0]
    assert done["transcript_source"] == "whisper"
    assert done["title"] == "Podcast Episode"


@pytest.mark.asyncio
async def test_runner_media_probe_failure_falls_through_to_normal_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_segments: list[dict[str, Any]],
) -> None:
    """Probe fails / returns no duration -> normal path unaffected (regression guard)."""
    job = _FakeJob(
        id="job_media_probefail", url="https://example.com/clip2.mp3",
        kind="media", title="Fallback Title",
    )
    fake_repo = _FakeRepo({"job_media_probefail": job})

    audio_file = tmp_path / "clip2.opus"
    audio_file.write_bytes(b"\x00" * 8)

    download_calls: list[dict[str, Any]] = []

    async def fake_download_audio(
        *, url: str, cookies: list[Any], dir: Path,
    ) -> tuple[Path, float | None]:
        download_calls.append({"url": url})
        return audio_file, 120.0

    async def fake_transcribe_audio(audio_path: Path, *, total_duration: float | None):
        from src.workers.transcribe import TranscribeResult
        return TranscribeResult(segments=fake_segments, language="en", duration_seconds=total_duration)

    async def fake_metadata(*, url: str, cookies: list[Any], scratch_dir: Path):
        return {}  # probe failed entirely — no title, no language, no duration

    async def fake_stream_summarize(
        text: str, *, title: Any, output_language: str, from_audio_transcript: bool = False
    ):
        yield "Normal summary."

    async def fake_probe_url_duration(url: str, *, timeout: float = 5.0):
        # Bonus pre-download URL probe also fails/unknown — keep this test
        # hermetic (no real ffprobe/network call) and isolated to exercising
        # the "everything unknown" fallthrough.
        return None

    monkeypatch.setattr(runner_mod.youtube, "download_audio", fake_download_audio)
    monkeypatch.setattr(runner_mod.transcribe, "transcribe_audio", fake_transcribe_audio)
    monkeypatch.setattr(runner_mod.youtube, "fetch_video_metadata", fake_metadata)
    monkeypatch.setattr(runner_mod.transcribe, "probe_url_duration", fake_probe_url_duration)
    monkeypatch.setattr(runner_mod.llm_summary, "stream_summarize", fake_stream_summarize)
    monkeypatch.setattr(runner_mod, "_audio_dir", lambda: tmp_path)

    q = WhisperQueue()
    await q.put(
        WhisperTask(job_id="job_media_probefail", url=job.url, cookies=[], page_text="unused"),
    )

    worker = asyncio.create_task(runner_mod.whisper_worker(q, fake_repo))
    await _wait_until(lambda: q.snapshot() == (0, 0) and bool(fake_repo.done_calls))
    worker.cancel()
    await asyncio.gather(worker, return_exceptions=True)

    assert len(download_calls) == 1
    assert fake_repo.failed_calls == []
    assert fake_repo.done_calls[0]["transcript_source"] == "whisper"


@pytest.mark.asyncio
async def test_runner_media_empty_transcript_falls_back_to_page_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Whisper produces an empty transcript for a media job -> page_text fallback,
    not "LLM returned empty summary"."""
    job = _FakeJob(
        id="job_media_empty", url="https://example.com/clip.mp3",
        kind="media", title="Some Page",
    )
    fake_repo = _FakeRepo({"job_media_empty": job})

    audio_file = tmp_path / "clip.opus"
    audio_file.write_bytes(b"\x00" * 8)

    async def fake_download_audio(
        *, url: str, cookies: list[Any], dir: Path,
    ) -> tuple[Path, float | None]:
        return audio_file, 30.0

    async def fake_transcribe_audio(audio_path: Path, *, total_duration: float | None):
        from src.workers.transcribe import TranscribeResult
        # No segments -> build_marked_text produces "" -> empty transcript.
        return TranscribeResult(segments=[], language=None, duration_seconds=total_duration)

    async def fake_metadata(*, url: str, cookies: list[Any], scratch_dir: Path):
        # Duration is above threshold, so the EARLY probe doesn't reject —
        # this exercises the separate empty-transcript trigger.
        return {"title": None, "language": None, "duration": 30.0}

    summarize_calls: list[dict[str, Any]] = []

    async def fake_stream_summarize(
        text: str, *, title: Any, output_language: str, from_audio_transcript: bool = False
    ):
        summarize_calls.append({"text": text, "from_audio_transcript": from_audio_transcript})
        yield "Fallback summary."

    monkeypatch.setattr(runner_mod.youtube, "download_audio", fake_download_audio)
    monkeypatch.setattr(runner_mod.transcribe, "transcribe_audio", fake_transcribe_audio)
    monkeypatch.setattr(runner_mod.youtube, "fetch_video_metadata", fake_metadata)
    monkeypatch.setattr(runner_mod.llm_summary, "stream_summarize", fake_stream_summarize)
    monkeypatch.setattr(runner_mod, "_audio_dir", lambda: tmp_path)

    q = WhisperQueue()
    await q.put(
        WhisperTask(
            job_id="job_media_empty", url=job.url, cookies=[],
            page_text="Fallback page content.",
        )
    )

    worker = asyncio.create_task(runner_mod.whisper_worker(q, fake_repo))
    await _wait_until(
        lambda: q.snapshot() == (0, 0) and bool(fake_repo.done_calls or fake_repo.failed_calls),
    )
    worker.cancel()
    await asyncio.gather(worker, return_exceptions=True)

    assert fake_repo.failed_calls == []
    assert len(fake_repo.done_calls) == 1
    done = fake_repo.done_calls[0]
    assert done["transcript_source"] == "page_extract"
    assert "Fallback summary" in done["summary_md"]
    assert summarize_calls[0]["from_audio_transcript"] is False
    assert summarize_calls[0]["text"] == "Fallback page content."
    # Audio was downloaded and transcription "succeeded" (just produced
    # nothing) -> cleaned up like any other completed job.
    assert not audio_file.exists()


@pytest.mark.asyncio
async def test_runner_media_short_duration_no_page_text_marks_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No page_text at all (old extension build) -> explicit mark_failed,
    not a silent/empty success."""
    job = _FakeJob(id="job_media_nopage", url="https://example.com/ding2.mp3", kind="media")
    fake_repo = _FakeRepo({"job_media_nopage": job})

    download_calls: list[dict[str, Any]] = []

    async def fake_download_audio(
        *, url: str, cookies: list[Any], dir: Path,
    ) -> tuple[Path, float | None]:
        download_calls.append({"url": url})
        raise AssertionError("download_audio must not be called for a too-short probe")

    async def fake_metadata(*, url: str, cookies: list[Any], scratch_dir: Path):
        return {"title": None, "language": None, "duration": 0.5}

    monkeypatch.setattr(runner_mod.youtube, "download_audio", fake_download_audio)
    monkeypatch.setattr(runner_mod.youtube, "fetch_video_metadata", fake_metadata)
    monkeypatch.setattr(runner_mod, "_audio_dir", lambda: tmp_path)

    q = WhisperQueue()
    await q.put(
        WhisperTask(job_id="job_media_nopage", url=job.url, cookies=[], page_text=None),
    )

    worker = asyncio.create_task(runner_mod.whisper_worker(q, fake_repo))
    await _wait_until(lambda: bool(fake_repo.failed_calls))
    worker.cancel()
    await asyncio.gather(worker, return_exceptions=True)

    assert download_calls == []
    assert fake_repo.done_calls == []
    assert len(fake_repo.failed_calls) == 1
    failed_job_id, error = fake_repo.failed_calls[0]
    assert failed_job_id == "job_media_nopage"
    assert "too short to contain speech" in error
    assert "no page text to fall back to" in error


@pytest.mark.asyncio
async def test_runner_youtube_job_ignores_media_only_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_segments: list[dict[str, Any]],
) -> None:
    """kind=youtube is unaffected: even a tiny probed duration must not
    trigger the media-only duration-reject fallback."""
    job = _FakeJob(id="job_yt_tiny", url="https://www.youtube.com/watch?v=shortshort1")
    fake_repo = _FakeRepo({"job_yt_tiny": job})

    audio_file = tmp_path / "yt.opus"
    audio_file.write_bytes(b"\x00" * 8)

    download_calls: list[dict[str, Any]] = []

    async def fake_download_audio(
        *, url: str, cookies: list[Any], dir: Path,
    ) -> tuple[Path, float | None]:
        download_calls.append({"url": url})
        return audio_file, 90.0

    async def fake_transcribe_audio(audio_path: Path, *, total_duration: float | None):
        from src.workers.transcribe import TranscribeResult
        return TranscribeResult(segments=fake_segments, language="en", duration_seconds=total_duration)

    async def fake_metadata(*, url: str, cookies: list[Any], scratch_dir: Path):
        # Tiny "duration" — must be ignored entirely for kind=youtube.
        return {"title": "Real Title", "language": "en", "duration": 1.0}

    async def fake_stream_summarize(
        text: str, *, title: Any, output_language: str, from_audio_transcript: bool = False
    ):
        yield "Video summary."

    monkeypatch.setattr(runner_mod.youtube, "download_audio", fake_download_audio)
    monkeypatch.setattr(runner_mod.transcribe, "transcribe_audio", fake_transcribe_audio)
    monkeypatch.setattr(runner_mod.youtube, "fetch_video_metadata", fake_metadata)
    monkeypatch.setattr(runner_mod.llm_summary, "stream_summarize", fake_stream_summarize)
    monkeypatch.setattr(runner_mod, "_audio_dir", lambda: tmp_path)

    q = WhisperQueue()
    await q.put(WhisperTask(job_id="job_yt_tiny", url=job.url, cookies=[]))

    worker = asyncio.create_task(runner_mod.whisper_worker(q, fake_repo))
    await _wait_until(lambda: q.snapshot() == (0, 0) and bool(fake_repo.done_calls))
    worker.cancel()
    await asyncio.gather(worker, return_exceptions=True)

    # Download WAS attempted — the media-only duration-reject never applies
    # to kind=youtube.
    assert len(download_calls) == 1
    assert fake_repo.failed_calls == []
    assert fake_repo.done_calls[0]["transcript_source"] == "whisper"


# ---------------------------------------------------------------------------
# kind=media: post-download local-ffprobe gate (Gap 1) + segments-based
# transcript-usability gate (Gap 2). These cover the real bug found live:
# a plain static-asset URL (e.g. /assets/notification.mp3) never reports
# `duration` via yt-dlp's generic extractor, so the pre-download probes
# (metadata + optional URL ffprobe) come back unknown, and Whisper — fed a
# few seconds of a "ding" — can hallucinate short plausible-looking text
# that isn't literally empty.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_media_static_asset_post_download_probe_skips_whisper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Static asset: yt-dlp metadata probe AND the pre-download URL probe
    both report no duration (Gap 1's exact scenario). Download proceeds
    (cheap), but the post-download local ffprobe reveals 3s -> Whisper must
    NEVER be called, and the job completes via the page-text fallback."""
    job = _FakeJob(
        id="job_media_static_short", url="https://example.com/assets/notification.mp3",
        kind="media", title="App Page",
    )
    fake_repo = _FakeRepo({"job_media_static_short": job})

    audio_file = tmp_path / "notification.mp3"
    audio_file.write_bytes(b"\x00" * 48)

    async def fake_metadata(*, url: str, cookies: list[Any], scratch_dir: Path):
        # Plain static-asset URL: yt-dlp's generic extractor reports no
        # duration at all, confirmed live.
        return {"duration": None}

    async def fake_probe_url_duration(url: str, *, timeout: float = 5.0):
        return None  # bonus pre-download URL probe also inconclusive

    download_calls: list[dict[str, Any]] = []

    async def fake_download_audio(
        *, url: str, cookies: list[Any], dir: Path,
    ) -> tuple[Path, float | None]:
        download_calls.append({"url": url})
        return audio_file, None  # yt-dlp's own probe doesn't know duration either

    async def fake_probe_duration(path: Path):
        assert path == audio_file
        return 3.0  # post-download local ffprobe: the real, authoritative value

    async def fake_transcribe_audio(audio_path: Path, *, total_duration: float | None):
        raise AssertionError("Whisper must not be called for a 3s post-download probe")

    summarize_calls: list[dict[str, Any]] = []

    async def fake_stream_summarize(
        text: str, *, title: Any, output_language: str, from_audio_transcript: bool = False
    ):
        summarize_calls.append({"text": text, "from_audio_transcript": from_audio_transcript})
        yield "Notification page summary."

    monkeypatch.setattr(runner_mod.youtube, "fetch_video_metadata", fake_metadata)
    monkeypatch.setattr(runner_mod.transcribe, "probe_url_duration", fake_probe_url_duration)
    monkeypatch.setattr(runner_mod.youtube, "download_audio", fake_download_audio)
    monkeypatch.setattr(runner_mod.transcribe, "probe_duration", fake_probe_duration)
    monkeypatch.setattr(runner_mod.transcribe, "transcribe_audio", fake_transcribe_audio)
    monkeypatch.setattr(runner_mod.llm_summary, "stream_summarize", fake_stream_summarize)
    monkeypatch.setattr(runner_mod, "_audio_dir", lambda: tmp_path)

    q = WhisperQueue()
    await q.put(
        WhisperTask(
            job_id="job_media_static_short", url=job.url, cookies=[],
            page_text="This app notifies you when a task finishes.",
        )
    )

    worker = asyncio.create_task(runner_mod.whisper_worker(q, fake_repo))
    await _wait_until(
        lambda: q.snapshot() == (0, 0) and bool(fake_repo.done_calls or fake_repo.failed_calls),
    )
    worker.cancel()
    await asyncio.gather(worker, return_exceptions=True)

    assert len(download_calls) == 1, "download should still happen — it's cheap"
    assert fake_repo.failed_calls == []
    assert len(fake_repo.done_calls) == 1
    done = fake_repo.done_calls[0]
    assert done["transcript_source"] == "page_extract"
    assert "Notification page summary" in done["summary_md"]
    assert summarize_calls[0]["from_audio_transcript"] is False
    assert summarize_calls[0]["text"] == "This app notifies you when a task finishes."

    # The downloaded chime file is a genuine terminal success (mark_done via
    # page-text fallback), not a failure a retry could improve on — must be
    # cleaned up like any other completed job, not leaked.
    assert not audio_file.exists()
    assert fake_repo.set_audio_calls[-1]["audio_path"] is None


@pytest.mark.asyncio
async def test_runner_media_static_asset_post_download_probe_confirms_long_duration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_segments: list[dict[str, Any]],
) -> None:
    """Same unknown-duration-until-download setup, but the post-download
    local ffprobe reveals 40s (above threshold) -> normal Whisper path."""
    job = _FakeJob(
        id="job_media_static_long", url="https://example.com/assets/podcast.mp3",
        kind="media", title="Podcast Asset Page",
    )
    fake_repo = _FakeRepo({"job_media_static_long": job})

    audio_file = tmp_path / "podcast.mp3"
    audio_file.write_bytes(b"\x00" * 626)

    async def fake_metadata(*, url: str, cookies: list[Any], scratch_dir: Path):
        return {"duration": None}

    async def fake_probe_url_duration(url: str, *, timeout: float = 5.0):
        return None

    download_calls: list[dict[str, Any]] = []

    async def fake_download_audio(
        *, url: str, cookies: list[Any], dir: Path,
    ) -> tuple[Path, float | None]:
        download_calls.append({"url": url})
        return audio_file, None

    async def fake_probe_duration(path: Path):
        assert path == audio_file
        return 40.0

    transcribe_calls: list[dict[str, Any]] = []

    async def fake_transcribe_audio(audio_path: Path, *, total_duration: float | None):
        from src.workers.transcribe import TranscribeResult
        transcribe_calls.append({"audio_path": audio_path, "total_duration": total_duration})
        return TranscribeResult(segments=fake_segments, language="en", duration_seconds=total_duration)

    async def fake_stream_summarize(
        text: str, *, title: Any, output_language: str, from_audio_transcript: bool = False
    ):
        yield "Real podcast summary."

    monkeypatch.setattr(runner_mod.youtube, "fetch_video_metadata", fake_metadata)
    monkeypatch.setattr(runner_mod.transcribe, "probe_url_duration", fake_probe_url_duration)
    monkeypatch.setattr(runner_mod.youtube, "download_audio", fake_download_audio)
    monkeypatch.setattr(runner_mod.transcribe, "probe_duration", fake_probe_duration)
    monkeypatch.setattr(runner_mod.transcribe, "transcribe_audio", fake_transcribe_audio)
    monkeypatch.setattr(runner_mod.llm_summary, "stream_summarize", fake_stream_summarize)
    monkeypatch.setattr(runner_mod, "_audio_dir", lambda: tmp_path)

    q = WhisperQueue()
    await q.put(
        WhisperTask(
            job_id="job_media_static_long", url=job.url, cookies=[],
            page_text="unused",
        )
    )

    worker = asyncio.create_task(runner_mod.whisper_worker(q, fake_repo))
    await _wait_until(lambda: q.snapshot() == (0, 0) and bool(fake_repo.done_calls))
    worker.cancel()
    await asyncio.gather(worker, return_exceptions=True)

    assert len(download_calls) == 1
    assert len(transcribe_calls) == 1, "Whisper must run once duration is confirmed above threshold"
    # The local-probe-confirmed duration is threaded through to Whisper.
    assert transcribe_calls[0]["total_duration"] == 40.0
    assert fake_repo.failed_calls == []
    done = fake_repo.done_calls[0]
    assert done["transcript_source"] == "whisper"


@pytest.mark.asyncio
async def test_runner_media_annotation_only_transcript_falls_back_to_page_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Whisper produces a short NON-empty but annotation-only transcript
    (e.g. "[chime]") for a media job whose duration is known to be above
    threshold. The old naive `not raw_text.strip()` check would have let
    this through to a fabricated summary; transcript_is_unusable must
    catch it via the segments-based gate."""
    job = _FakeJob(
        id="job_media_garbage", url="https://example.com/clip3.mp3",
        kind="media", title="Some Page",
    )
    fake_repo = _FakeRepo({"job_media_garbage": job})

    audio_file = tmp_path / "clip3.opus"
    audio_file.write_bytes(b"\x00" * 8)

    async def fake_metadata(*, url: str, cookies: list[Any], scratch_dir: Path):
        # Duration known and comfortably above threshold — this test is
        # specifically about the post-Whisper transcript-quality gate, not
        # the duration gate.
        return {"title": None, "language": None, "duration": 15.0}

    async def fake_download_audio(
        *, url: str, cookies: list[Any], dir: Path,
    ) -> tuple[Path, float | None]:
        return audio_file, 15.0

    async def fake_transcribe_audio(audio_path: Path, *, total_duration: float | None):
        from src.workers.transcribe import TranscribeResult
        # 7 non-empty characters, annotation-only — NOT caught by
        # `not raw_text.strip()`.
        return TranscribeResult(
            segments=[{"start": 0.0, "end": 3.0, "text": "[chime]"}],
            language=None,
            duration_seconds=total_duration,
        )

    summarize_calls: list[dict[str, Any]] = []

    async def fake_stream_summarize(
        text: str, *, title: Any, output_language: str, from_audio_transcript: bool = False
    ):
        summarize_calls.append({"text": text, "from_audio_transcript": from_audio_transcript})
        yield "Garbage-transcript fallback summary."

    monkeypatch.setattr(runner_mod.youtube, "fetch_video_metadata", fake_metadata)
    monkeypatch.setattr(runner_mod.youtube, "download_audio", fake_download_audio)
    monkeypatch.setattr(runner_mod.transcribe, "transcribe_audio", fake_transcribe_audio)
    monkeypatch.setattr(runner_mod.llm_summary, "stream_summarize", fake_stream_summarize)
    monkeypatch.setattr(runner_mod, "_audio_dir", lambda: tmp_path)

    q = WhisperQueue()
    await q.put(
        WhisperTask(
            job_id="job_media_garbage", url=job.url, cookies=[],
            page_text="Real page content describing the clip.",
        )
    )

    worker = asyncio.create_task(runner_mod.whisper_worker(q, fake_repo))
    await _wait_until(
        lambda: q.snapshot() == (0, 0) and bool(fake_repo.done_calls or fake_repo.failed_calls),
    )
    worker.cancel()
    await asyncio.gather(worker, return_exceptions=True)

    assert fake_repo.failed_calls == []
    assert len(fake_repo.done_calls) == 1
    done = fake_repo.done_calls[0]
    assert done["transcript_source"] == "page_extract"
    assert "Garbage-transcript fallback summary" in done["summary_md"]
    assert summarize_calls[0]["from_audio_transcript"] is False
    assert summarize_calls[0]["text"] == "Real page content describing the clip."
    # Audio downloaded and "transcribed" (just produced garbage) -> cleaned
    # up like any other completed job.
    assert not audio_file.exists()
