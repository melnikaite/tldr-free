"""Tests for workers.vad — LocalAI's optional POST /vad extension.

Mocks at the same boundaries workers/transcribe.py's own tests already
use for the analogous Whisper calls: HTTP (``httpx.AsyncClient``, same
fake-client pattern as ``test_post_audio_sends_language_form_field_only_
when_given`` in test_transcribe.py) and the ffmpeg cut
(``vad._extract_pcm_f32le``, the direct equivalent of mocking
``transcribe._cut_audio_segment`` rather than ``subprocess.run`` itself).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.workers import vad


def _enable_vad(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> Any:
    """Point ``vad.get_config()`` at the real cached Config singleton with
    ``whisper.vad_model`` (and any other whisper.* overrides) mutated in
    place — same pattern test_transcribe.py already uses (e.g.
    test_large_file_routes_to_chunked) to tweak config for one test without
    touching the on-disk file."""
    cfg = vad.get_config()
    monkeypatch.setattr(cfg.whisper, "vad_model", "silero-vad-ggml")
    for key, value in overrides.items():
        monkeypatch.setattr(cfg.whisper, key, value)
    monkeypatch.setattr(vad, "get_config", lambda: cfg)
    return cfg


def _fake_pcm(num_samples: int = 4) -> bytes:
    """Arbitrary-but-valid float32 PCM bytes — content doesn't matter since
    ``_extract_pcm_f32le`` is mocked out in every test that needs it; this
    just has to be a value multiple-of-4-bytes length."""
    return b"\x00\x00\x00\x00" * num_samples


class _FakeResponse:
    def __init__(self, payload: object, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://localhost/vad")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self) -> object:
        return self._payload


class _FakeClient:
    """Minimal async-context-manager stand-in for httpx.AsyncClient, same
    shape as test_transcribe.py's own ``_FakeClient``."""

    def __init__(self, post_fn: Any) -> None:
        self._post_fn = post_fn

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def post(self, url: str, *, content: bytes, headers: dict) -> _FakeResponse:
        return await self._post_fn(url, content, headers)


def _install_fake_client(monkeypatch: pytest.MonkeyPatch, post_fn: Any) -> None:
    monkeypatch.setattr(vad.httpx, "AsyncClient", lambda **_k: _FakeClient(post_fn))


@pytest.mark.asyncio
async def test_feature_off_returns_none_without_any_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = vad.get_config()
    monkeypatch.setattr(cfg.whisper, "vad_model", "")
    monkeypatch.setattr(vad, "get_config", lambda: cfg)

    def fail_extract(*_a: object, **_k: object) -> bytes:
        pytest.fail("ffmpeg should never be invoked when the feature is off")

    monkeypatch.setattr(vad, "_extract_pcm_f32le", fail_extract)

    async def fail_post(*_a: object, **_k: object) -> _FakeResponse:
        pytest.fail("HTTP should never be invoked when the feature is off")

    _install_fake_client(monkeypatch, fail_post)

    result = await vad.speech_seconds(tmp_path / "a.opus", [(0.0, 10.0)])
    assert result is None


@pytest.mark.asyncio
async def test_empty_windows_returns_empty_dict_not_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Feature ON but nothing to examine — a real (empty) answer, not "no
    VAD data": the caller must be able to tell "VAD ran, found nothing to
    do" apart from "VAD unavailable"."""
    _enable_vad(monkeypatch)
    result = await vad.speech_seconds(tmp_path / "a.opus", [])
    assert result == {}


@pytest.mark.asyncio
async def test_http_404_on_first_window_returns_none_logged_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _enable_vad(monkeypatch)
    monkeypatch.setattr(vad, "_extract_pcm_f32le", lambda *a, **k: _fake_pcm())

    call_count = 0

    async def post_404(_url: str, _content: bytes, _headers: dict) -> _FakeResponse:
        nonlocal call_count
        call_count += 1
        return _FakeResponse({}, status=404)

    _install_fake_client(monkeypatch, post_404)

    with caplog.at_level(logging.WARNING, logger="src.workers.vad"):
        result = await vad.speech_seconds(
            tmp_path / "a.opus", [(0.0, 10.0), (20.0, 30.0)]
        )

    assert result is None
    # Only the FIRST window's request should ever fire — a total failure on
    # the very first request stops the whole call, never continues to a
    # second window.
    assert call_count == 1
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_connection_error_on_first_window_returns_none_logged_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _enable_vad(monkeypatch)
    monkeypatch.setattr(vad, "_extract_pcm_f32le", lambda *a, **k: _fake_pcm())

    async def post_conn_error(_url: str, _content: bytes, _headers: dict) -> _FakeResponse:
        raise httpx.ConnectError("connection refused")

    _install_fake_client(monkeypatch, post_conn_error)

    with caplog.at_level(logging.WARNING, logger="src.workers.vad"):
        result = await vad.speech_seconds(tmp_path / "a.opus", [(0.0, 10.0)])

    assert result is None
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1


@pytest.mark.asyncio
async def test_missing_ffmpeg_on_first_window_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ffmpeg unavailable is one of the enumerated "first request failed"
    causes — same None-and-log-once contract as an HTTP-level failure."""
    _enable_vad(monkeypatch)

    def fail_extract(*_a: object, **_k: object) -> bytes:
        raise RuntimeError("vad: ffmpeg unavailable")

    monkeypatch.setattr(vad, "_extract_pcm_f32le", fail_extract)

    async def fail_post(*_a: object, **_k: object) -> _FakeResponse:
        pytest.fail("HTTP should never be reached when the ffmpeg cut itself fails")

    _install_fake_client(monkeypatch, fail_post)

    result = await vad.speech_seconds(tmp_path / "a.opus", [(0.0, 10.0)])
    assert result is None


@pytest.mark.asyncio
async def test_malformed_response_missing_segments_key_on_first_window_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _enable_vad(monkeypatch)
    monkeypatch.setattr(vad, "_extract_pcm_f32le", lambda *a, **k: _fake_pcm())

    async def post_malformed(_url: str, _content: bytes, _headers: dict) -> _FakeResponse:
        return _FakeResponse({"no_segments_here": True})

    _install_fake_client(monkeypatch, post_malformed)

    result = await vad.speech_seconds(tmp_path / "a.opus", [(0.0, 10.0)])
    assert result is None


@pytest.mark.asyncio
async def test_later_window_failure_returns_partial_data_not_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A window that fails AFTER at least one earlier window already
    succeeded degrades to a partial dict, not a total None — some real
    data was already gathered and is worth keeping."""
    _enable_vad(monkeypatch)
    monkeypatch.setattr(vad, "_extract_pcm_f32le", lambda *a, **k: _fake_pcm())

    calls = 0

    async def post_second_fails(_url: str, _content: bytes, _headers: dict) -> _FakeResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _FakeResponse({"segments": [{"start": 1.0, "end": 3.0}]})
        return _FakeResponse({}, status=500)

    _install_fake_client(monkeypatch, post_second_fails)

    with caplog.at_level(logging.WARNING, logger="src.workers.vad"):
        result = await vad.speech_seconds(
            tmp_path / "a.opus", [(0.0, 10.0), (20.0, 30.0)]
        )

    assert result == {(0.0, 10.0): 2.0}
    assert calls == 2
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1


@pytest.mark.asyncio
async def test_budget_cap_stops_partway_and_returns_partial_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _enable_vad(monkeypatch, vad_max_seconds=25.0)
    monkeypatch.setattr(vad, "_extract_pcm_f32le", lambda *a, **k: _fake_pcm())

    calls: list[tuple[float, float]] = []

    async def post_ok(_url: str, _content: bytes, _headers: dict) -> _FakeResponse:
        return _FakeResponse({"segments": []})

    _install_fake_client(monkeypatch, post_ok)

    # Windows of 10s, 10s, 10s — budget 25s: first two (20s total) fit, the
    # third would push to 30s > 25s and must be skipped entirely (never
    # even attempted).
    def tracking_extract(_path: Path, start: float, duration: float) -> bytes:
        calls.append((start, start + duration))
        return _fake_pcm()

    monkeypatch.setattr(vad, "_extract_pcm_f32le", tracking_extract)

    result = await vad.speech_seconds(
        tmp_path / "a.opus", [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
    )

    assert set(result) == {(0.0, 10.0), (10.0, 20.0)}
    assert (20.0, 30.0) not in result
    assert len(calls) == 2
    assert cfg.whisper.vad_max_seconds == 25.0


@pytest.mark.asyncio
async def test_segment_partially_outside_window_is_clipped_not_overcounted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _enable_vad(monkeypatch)
    monkeypatch.setattr(vad, "_extract_pcm_f32le", lambda *a, **k: _fake_pcm())

    async def post_straddling(_url: str, _content: bytes, _headers: dict) -> _FakeResponse:
        # Window is 10s long (local timeline [0, 10)); one segment starts
        # before 0, one ends after 10 — both must clip to the window's own
        # bounds rather than counting the overhang.
        return _FakeResponse(
            {
                "segments": [
                    {"start": -5.0, "end": 2.0},  # clips to [0, 2) -> 2.0s
                    {"start": 8.0, "end": 15.0},   # clips to [8, 10) -> 2.0s
                ]
            }
        )

    _install_fake_client(monkeypatch, post_straddling)

    result = await vad.speech_seconds(tmp_path / "a.opus", [(100.0, 110.0)])
    assert result == {(100.0, 110.0): 4.0}


def test_build_request_body_uses_byte_join_not_json_dumps() -> None:
    """The request body is hand-built (byte-join over the unpacked float32
    samples), not json.dumps over a Python list — see the module
    docstring's measured cost comparison. Sanity-check the actual bytes
    produced are valid JSON with the expected shape."""
    import json
    import struct

    samples = [0.5, -0.25, 1.0]
    pcm = struct.pack("<3f", *samples)
    body = vad._build_request_body("silero-vad-ggml", pcm)
    parsed = json.loads(body)
    assert parsed["model"] == "silero-vad-ggml"
    assert parsed["audio"] == pytest.approx(samples, abs=1e-4)


@pytest.mark.asyncio
async def test_requests_are_sequential_never_concurrent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No two VAD requests are ever in flight at the same time — a plain
    sequential await loop, never gather/create_task (see the module
    docstring on why concurrency would only compete with the job's own ASR
    calls for no benefit)."""
    _enable_vad(monkeypatch)
    monkeypatch.setattr(vad, "_extract_pcm_f32le", lambda *a, **k: _fake_pcm())

    active = 0
    max_active = 0

    async def post_tracking(_url: str, _content: bytes, _headers: dict) -> _FakeResponse:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        active -= 1
        return _FakeResponse({"segments": []})

    _install_fake_client(monkeypatch, post_tracking)

    await vad.speech_seconds(
        tmp_path / "a.opus", [(0.0, 5.0), (10.0, 15.0), (20.0, 25.0)]
    )

    assert max_active == 1
