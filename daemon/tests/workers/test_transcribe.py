"""Tests for workers.transcribe — single-shot vs chunked transcription."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.workers import transcribe


def _payload(segments: list[dict], language: str = "en") -> dict:
    return {"segments": segments, "language": language, "duration": 100.0}


@pytest.mark.asyncio
async def test_small_file_takes_single_post(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x" * 1024)  # 1 KB, well under the cap

    calls: list[Path] = []

    async def fake_post(path: Path) -> dict:
        calls.append(path)
        return _payload([{"start": 1.0, "end": 2.0, "text": "hi"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)
    monkeypatch.setattr(
        transcribe, "_transcribe_chunked", lambda *a, **k: pytest.fail("should not chunk")
    )

    result = await transcribe.transcribe_audio(audio, total_duration=100.0)
    assert calls == [audio]
    assert result.segments == [{"start": 1.0, "end": 2.0, "text": "hi"}]
    assert result.language == "en"


@pytest.mark.asyncio
async def test_chunked_merges_with_offsets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = tmp_path / "big.opus"
    audio.write_bytes(b"x" * 2048)

    # Two chunks at offsets 0 and 600s.
    c0, c1 = tmp_path / "c0.opus", tmp_path / "c1.opus"
    c0.write_bytes(b"0")
    c1.write_bytes(b"1")
    monkeypatch.setattr(
        transcribe, "_split_audio", lambda *a, **k: [(c0, 0.0), (c1, 600.0)]
    )

    async def fake_post(path: Path) -> dict:
        if path == c0:
            return _payload([{"start": 5.0, "end": 10.0, "text": "first"}], "ru")
        return _payload([{"start": 3.0, "end": 8.0, "text": "second"}], "en")

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result = await transcribe._transcribe_chunked(
        audio, total_duration=1200.0, max_bytes=1024
    )
    # Second chunk's timestamps are shifted by its 600 s offset.
    assert result.segments == [
        {"start": 5.0, "end": 10.0, "text": "first"},
        {"start": 603.0, "end": 608.0, "text": "second"},
    ]
    # Language comes from the first chunk that reports one.
    assert result.language == "ru"
    assert result.duration_seconds == 1200.0


@pytest.mark.asyncio
async def test_chunked_falls_back_when_no_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = tmp_path / "big.opus"
    audio.write_bytes(b"x" * 2048)
    monkeypatch.setattr(transcribe, "_probe_duration", lambda _p: None)

    async def fake_post(path: Path) -> dict:
        return _payload([{"start": 0.0, "end": 1.0, "text": "whole"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)
    monkeypatch.setattr(
        transcribe, "_split_audio", lambda *a, **k: pytest.fail("should not split")
    )

    result = await transcribe._transcribe_chunked(
        audio, total_duration=None, max_bytes=1024
    )
    assert result.segments == [{"start": 0.0, "end": 1.0, "text": "whole"}]


@pytest.mark.asyncio
async def test_large_file_routes_to_chunked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = tmp_path / "big.opus"
    audio.write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MB

    # Force a tiny cap so the 2 MB file is "oversized".
    cfg = transcribe.get_config()
    monkeypatch.setattr(cfg.whisper, "max_upload_mb", 1)
    monkeypatch.setattr(transcribe, "get_config", lambda: cfg)

    async def fake_chunked(*_a: object, **_k: object) -> transcribe.TranscribeResult:
        return transcribe.TranscribeResult(segments=[], language=None, duration_seconds=1.0)

    monkeypatch.setattr(transcribe, "_transcribe_chunked", fake_chunked)
    monkeypatch.setattr(
        transcribe, "_post_audio", lambda *a, **k: pytest.fail("should chunk")
    )

    result = await transcribe.transcribe_audio(audio, total_duration=10.0)
    assert result.duration_seconds == 1.0
