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

import logging
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
    """POST the audio file, return parsed segments + language.

    Raises ``httpx.HTTPStatusError`` on server-side failure (caller turns
    that into a friendly error). Returns an empty-segments result when the
    server reports success but didn't transcribe anything (e.g. silent
    audio) — caller can treat that as a soft failure.
    """
    cfg = get_config().whisper
    base_url = cfg.base_url.rstrip("/")
    endpoint = f"{base_url}/audio/transcriptions"

    headers = {"Authorization": f"Bearer {cfg.api_key}"}

    with audio_path.open("rb") as fh:
        files = {"file": (audio_path.name, fh, "application/octet-stream")}
        data = {
            "model": cfg.model,
            "response_format": "verbose_json",
        }
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(endpoint, headers=headers, data=data, files=files)
            r.raise_for_status()
            payload = r.json()

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
                "fallback. Whisper segments require the mlx-server patch "
                "(see scripts/mlx-patches/).",
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
