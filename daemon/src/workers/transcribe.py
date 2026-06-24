"""Whisper transcription via mlx-server's ``/v1/audio/transcriptions``.

    async def transcribe_audio(audio_path, *, total_duration) -> TranscribeResult

Non-streaming ``verbose_json``. The mlx-server install is patched (see
``scripts/mlx-patches/``) so the response actually carries the
per-segment timing + auto-detected language that ``mlx_whisper.transcribe``
produces internally — upstream's handler used to drop both.

Why non-streaming + verbose_json (not streaming + plain json)
-------------------------------------------------------------

The streaming endpoint only emits text deltas; no segment boundaries, no
language. With it we'd be back to "one giant bucket" — exactly the
problem the transcript-tab UI needs to solve. ``verbose_json`` returns
segments and language in one shot, so we make a single request and get
exactly what the downstream code needs.

Trade-off: we lose mid-transcription UI progress (the previous stream
form gave a chunk every 30 s of audio). Whisper-turbo on mlx is ~1×
realtime, so the daemon publishes a single ``transcribing`` stage and
the Library row sits on it until the call returns. The user explicitly
chose accuracy of timestamps over real-time progress; if that flips, we
synthesise an elapsed-vs-expected timer here without changing callers.

The mlx-server side timeout (``queue_timeout`` in
``~/.mlx-server/config.yaml``) must be ≥ expected transcription wall
time — for hour-long audio we leave it at the install default of an
hour. ``httpx`` here uses ``timeout=None`` to match.

Fallback if the server isn't patched
------------------------------------

Older / unpatched mlx-server responses lack ``segments`` and
``language``. We don't fail — we fabricate one segment spanning the
whole audio so downstream ``build_marked_text`` and summary still work
(same shape as the pre-patch behaviour). ``language`` ends up ``None``;
callers persist it as ``None`` and the UI falls back to the "Original"
label.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from src.config import get_config

log = logging.getLogger(__name__)


@dataclass
class TranscribeResult:
    """Per-segment timing + detected language from a Whisper transcription.

    ``segments`` is a list of dicts with ``start`` / ``end`` / ``text`` —
    the canonical shape ``timecodes.build_marked_text`` consumes. When
    the server didn't return real segments (unpatched mlx-server) we
    construct a single all-encompassing segment so the rest of the
    pipeline behaves normally.

    ``language`` is an ISO-639-1 code (e.g. ``"en"``, ``"ru"``) or
    ``None`` if the server didn't surface it.

    ``duration_seconds`` mirrors the upstream ``duration`` field — handy
    for synthesised progress and as a sanity check vs yt-dlp's metadata.
    """

    segments: list[dict[str, Any]]
    language: str | None
    duration_seconds: float | None


async def transcribe_audio(
    audio_path: Path,
    *,
    total_duration: float | None,
) -> TranscribeResult:
    """Transcribe the audio, splitting it first if it exceeds the upload cap.

    Most Whisper backends reject large bodies (LocalAI ~15 MB, OpenAI 25 MB).
    For audio over ``whisper.max_upload_mb`` we split it into time-based chunks
    with ffmpeg, transcribe each, and stitch the segments back together with
    their original timestamps. Small audio takes the single-request path.

    Raises ``httpx.HTTPStatusError`` on server-side failure (caller turns that
    into a friendly error). Returns an empty-segments result when the server
    reports success but didn't transcribe anything (e.g. silent audio).
    """
    cfg = get_config().whisper
    max_bytes = max(1, cfg.max_upload_mb) * 1024 * 1024
    size = audio_path.stat().st_size

    if size <= max_bytes:
        payload = await _post_audio(audio_path)
        return _parse_payload(payload, total_duration=total_duration)

    return await _transcribe_chunked(
        audio_path, total_duration=total_duration, max_bytes=max_bytes
    )


async def _transcribe_chunked(
    audio_path: Path,
    *,
    total_duration: float | None,
    max_bytes: int,
) -> TranscribeResult:
    """Split oversized audio with ffmpeg, transcribe parts, merge segments."""
    duration = total_duration if total_duration and total_duration > 0 else None
    if duration is None:
        duration = await asyncio.to_thread(_probe_duration, audio_path)
    if duration is None or duration <= 0:
        # Can't time-slice without a duration — fall back to one shot and let
        # the backend's own error surface if it really is too big.
        log.warning("transcribe: unknown duration, cannot chunk; trying single upload")
        payload = await _post_audio(audio_path)
        return _parse_payload(payload, total_duration=total_duration)

    size = audio_path.stat().st_size
    # Target 90% of the cap for VBR headroom; at least 2 chunks since we're here.
    target = max(1, int(max_bytes * 0.9))
    num_chunks = max(2, math.ceil(size / target))
    chunk_seconds = duration / num_chunks
    log.info(
        "transcribe: audio %.1f MB > cap → %d chunks of ~%.0f s",
        size / 1024 / 1024,
        num_chunks,
        chunk_seconds,
    )

    chunks = await asyncio.to_thread(
        _split_audio, audio_path, num_chunks, chunk_seconds
    )
    if not chunks:
        log.warning("transcribe: ffmpeg split produced nothing; trying single upload")
        payload = await _post_audio(audio_path)
        return _parse_payload(payload, total_duration=total_duration)

    all_segments: list[dict[str, Any]] = []
    language: str | None = None
    try:
        for idx, (chunk_path, offset) in enumerate(chunks):
            payload = await _post_audio(chunk_path)
            part = _parse_payload(payload, total_duration=chunk_seconds)
            for seg in part.segments:
                all_segments.append(
                    {
                        "start": seg["start"] + offset,
                        "end": seg["end"] + offset,
                        "text": seg["text"],
                    }
                )
            if language is None:
                language = part.language
            log.info("transcribe: chunk %d/%d done", idx + 1, len(chunks))
    finally:
        for chunk_path, _ in chunks:
            chunk_path.unlink(missing_ok=True)
        # chunks share one mkdtemp dir; remove it once emptied.
        with contextlib.suppress(OSError):
            chunks[0][0].parent.rmdir()

    return TranscribeResult(
        segments=all_segments,
        language=language,
        duration_seconds=duration,
    )


async def _post_audio(audio_path: Path) -> dict[str, Any]:
    """POST one audio file to the transcription endpoint, return parsed JSON."""
    cfg = get_config().whisper
    endpoint = f"{cfg.base_url.rstrip('/')}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {cfg.api_key}"}

    with audio_path.open("rb") as fh:
        files = {"file": (audio_path.name, fh, "application/octet-stream")}
        data = {"model": cfg.model, "response_format": "verbose_json"}
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(endpoint, headers=headers, data=data, files=files)
            r.raise_for_status()
            result: dict[str, Any] = r.json()
            return result


def _parse_payload(
    payload: dict[str, Any], *, total_duration: float | None
) -> TranscribeResult:
    """Turn one transcription response into a TranscribeResult."""
    segments = _normalise_segments(payload.get("segments"))
    if not segments:
        # Unpatched server, or model returned text only. Construct a
        # single segment so build_marked_text still produces something
        # usable. raw_text loses fine-grained markers but summary works.
        full_text = str(payload.get("text") or "").strip()
        if full_text:
            end = float(total_duration) if total_duration and total_duration > 0 else 0.0
            segments = [{"start": 0.0, "end": end, "text": full_text}]
            log.warning(
                "transcribe: server returned no segments — using one-bucket "
                "fallback (no per-segment timing from this backend).",
            )

    raw_lang = payload.get("language")
    language: str | None = None
    if isinstance(raw_lang, str) and raw_lang.strip():
        # Some Whisper backends use full names ("english"); we normalise
        # the casing but leave the value as-is — the LLM language helper
        # later canonicalises to ISO-639-1.
        language = raw_lang.strip().lower()

    raw_duration = payload.get("duration")
    duration_seconds: float | None = None
    if isinstance(raw_duration, (int, float)) and raw_duration > 0:
        duration_seconds = float(raw_duration)
    elif total_duration and total_duration > 0:
        duration_seconds = float(total_duration)

    return TranscribeResult(
        segments=segments,
        language=language,
        duration_seconds=duration_seconds,
    )


def _ffmpeg_bin(name: str) -> str | None:
    """Path to ffmpeg/ffprobe from the resolver, or None if unavailable."""
    from src.workers.ffmpeg import resolve_ffmpeg_dir

    directory = resolve_ffmpeg_dir()
    if not directory:
        return None
    exe = f"{name}.exe" if os.name == "nt" else name
    candidate = Path(directory) / exe
    return str(candidate) if candidate.is_file() else None


def _probe_duration(audio_path: Path) -> float | None:
    """Audio duration in seconds via ffprobe, or None if it can't be read."""
    ffprobe = _ffmpeg_bin("ffprobe")
    if not ffprobe:
        return None
    try:
        out = subprocess.run(
            [
                ffprobe, "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(audio_path),
            ],
            check=True, capture_output=True, text=True,
        )
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, OSError, ValueError) as exc:
        log.warning("transcribe: ffprobe duration failed (%s)", exc)
        return None


def _split_audio(
    audio_path: Path, num_chunks: int, chunk_seconds: float
) -> list[tuple[Path, float]]:
    """Cut ``audio_path`` into ``num_chunks`` time slices, codec-copied.

    Returns ``[(chunk_path, start_offset_seconds), ...]``. Chunks land in a
    temp dir next to the source; the caller unlinks them. Returns ``[]`` when
    ffmpeg is unavailable or every cut fails.
    """
    ffmpeg = _ffmpeg_bin("ffmpeg")
    if not ffmpeg:
        log.warning("transcribe: no ffmpeg to split audio")
        return []

    tmp_dir = Path(tempfile.mkdtemp(prefix="tldr-chunks-", dir=audio_path.parent))
    suffix = audio_path.suffix or ".opus"
    chunks: list[tuple[Path, float]] = []
    for i in range(num_chunks):
        offset = i * chunk_seconds
        out = tmp_dir / f"chunk{i:03d}{suffix}"
        try:
            subprocess.run(
                [
                    ffmpeg, "-y", "-loglevel", "error",
                    "-ss", f"{offset:.3f}", "-t", f"{chunk_seconds:.3f}",
                    "-i", str(audio_path), "-c", "copy", str(out),
                ],
                check=True, capture_output=True,
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            log.warning("transcribe: ffmpeg chunk %d failed (%s)", i, exc)
            continue
        if out.is_file() and out.stat().st_size > 0:
            chunks.append((out, offset))
    return chunks


def _normalise_segments(raw: Any) -> list[dict[str, Any]]:
    """Coerce server's segment list into the ``build_marked_text`` shape.

    Drops malformed entries quietly rather than failing the whole
    transcription — one corrupt segment shouldn't kill an hour of work.
    """
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for seg in raw:
        if not isinstance(seg, dict):
            continue
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        out.append({"start": start, "end": end, "text": text})
    return out


__all__ = ["transcribe_audio", "TranscribeResult"]
