"""Whisper transcription via mlx-server's ``/v1/audio/transcriptions``.

    async def transcribe_stream(audio_path, *, total_duration, on_progress)
            -> list[dict]
        Multipart-uploads ``audio_path`` with ``stream=true`` and
        ``response_format=json`` and consumes the SSE chunk stream. mlx-server
        emits one chunk per 30-second slice of audio (``CHUNK_SIZE`` in
        mlx-server's ``MLX_Whisper._transcribe_generator``), so each chunk we
        receive ≈ 30 s of progress. ``on_progress(fraction)`` is called after
        each chunk with the share of audio processed (0.0–1.0). Returns the
        canonical one-segment list shaped for ``timecodes.build_marked_text``:
            [{"start": 0.0, "end": total_duration, "text": full_transcript}]

mlx-server v1.8 dropped per-segment timestamps from its non-streaming JSON
response. The streaming path doesn't expose them either, so a real [MM:SS]
breakdown is unavailable for Whisper-pathway videos. The YouTube
transcript-API fast path retains real timestamps.

We pass ``timeout=None`` to httpx because transcription of a long file can
take many minutes. The matching server-side timeout (``queue_timeout``)
is bumped in ``~/.mlx-server/config.yaml`` (the live mlx-server config,
outside the tldr repo so it can be shared across tools).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx

from src.config import get_config

log = logging.getLogger(__name__)


# mlx-server's MLX_Whisper streams in 30-second slices.
_CHUNK_SIZE_SECONDS = 30.0


ProgressCallback = Callable[[float], Awaitable[None]] | Callable[[float], None]


async def transcribe_stream(
    audio_path: Path,
    *,
    total_duration: float | None,
    on_progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Stream-transcribe ``audio_path`` and report progress per chunk.

    ``total_duration`` is the length of the audio in seconds (from yt-dlp's
    info dict). When supplied, ``on_progress(fraction)`` is invoked after
    every received chunk with ``min(1.0, processed / total)``. When unknown
    we still call ``on_progress`` but always pass ``0.0`` — the caller can
    decide whether to fall back to a static label.
    """
    cfg = get_config().whisper
    base_url = cfg.base_url.rstrip("/")
    endpoint = f"{base_url}/audio/transcriptions"

    headers = {"Authorization": f"Bearer {cfg.api_key}", "Accept": "text/event-stream"}

    chunks_received = 0
    parts: list[str] = []

    with audio_path.open("rb") as fh:
        files = {"file": (audio_path.name, fh, "application/octet-stream")}
        data = {
            "model": cfg.model,
            "response_format": "json",
            "stream": "true",
        }
        async with (
            httpx.AsyncClient(timeout=None) as client,
            client.stream(
                "POST", endpoint, headers=headers, data=data, files=files,
            ) as response,
        ):
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload_raw = line[6:].strip()
                    if not payload_raw or payload_raw == "[DONE]":
                        continue
                    try:
                        payload = json.loads(payload_raw)
                    except ValueError:
                        log.warning("transcribe: malformed SSE frame: %r", payload_raw)
                        continue
                    delta = ""
                    for choice in payload.get("choices") or []:
                        delta_obj = choice.get("delta") or {}
                        delta += str(delta_obj.get("content") or "")
                    if not delta:
                        continue
                    parts.append(delta)
                    chunks_received += 1
                    if on_progress is not None:
                        fraction = 0.0
                        if total_duration and total_duration > 0:
                            processed = chunks_received * _CHUNK_SIZE_SECONDS
                            fraction = min(1.0, processed / total_duration)
                        result = on_progress(fraction)
                        if hasattr(result, "__await__"):
                            await result  # type: ignore[misc]

    text = "".join(parts).strip()
    if not text:
        return []
    end = float(total_duration) if total_duration and total_duration > 0 else 0.0
    return [{"start": 0.0, "end": end, "text": text}]


__all__ = ["transcribe_stream"]
