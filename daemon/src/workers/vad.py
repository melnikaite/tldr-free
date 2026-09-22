"""Optional voice-activity detection via LocalAI's ``POST /vad`` extension.

    async def speech_seconds(audio_path, windows) -> dict[(start,end), float] | None

Bonus on top of ``workers.transcribe``'s coverage-recheck loop, never a
requirement — CLAUDE.md is explicit that the daemon targets ANY OpenAI-
compatible backend, and ``/vad`` is a LocalAI-only extension (it lives at
the server root, NOT under ``/v1`` like every other route this project
calls). Off by default (``config.whisper.vad_model == ""``); every failure
mode — feature disabled, HTTP 404 (backend doesn't have the route),
connection error, timeout, missing ffmpeg, a malformed response — degrades
to "no VAD data" (``None``, or a partial mapping) and is NEVER raised. A
job's transcription must never fail, or even behave differently in a way
that matters, because this bonus backend happens to be absent.

Why this exists — measured problem
-----------------------------------

The daemon now uses ``parakeet-cpp-tdt-0.6b-v3`` instead of Whisper.
Parakeet emits nothing at all for non-speech; Whisper used to emit
``*Musik*``-style pseudo-segments there instead.
``TranscribeResult.missing_seconds`` counts "time not covered by any
segment" — switching models silently turned that metric from "speech we
lost" into "the show has music in it".

Numbers from a real 24:57 ZDF episode transcribed by the current daemon
(job ``zuglcbAoGy9h``):

- reported ``transcript_missing_seconds``: 470 s (31% of the episode)
- actual dropped dialogue, measured against the official ZDF subtitles:
  ~27 s (17 short interjections)
- silero VAD over the same gaps: 66 s of speech in 607 s of gaps — and
  crucially the LONG gaps are almost entirely empty: gaps >=10 s hold
  242 s of audio and only 2 s of speech; gaps >=6 s hold 384 s and only
  9 s of speech.

That last line matters because ``_ensure_coverage``'s recheck loop spends
its bounded budget on the LARGEST unresolved windows first — today that
budget goes almost entirely to music, while the short windows that
actually held the dropped dialogue never get looked at. See
``workers/transcribe.py``'s use of this module for how the recheck loop's
window selection changes once VAD data is available, and
``config.WhisperConfig.vad_model``/``vad_url``/``vad_max_seconds`` for the
knobs.

The backend API (verified working against a real LocalAI instance)
--------------------------------------------------------------------

``POST {vad_url}`` (``vad_url`` derived from ``whisper.base_url`` by
stripping a trailing ``/v1`` unless overridden — see ``_resolve_vad_url``).

Request: ``{"model": "<vad model name>", "audio": [<float32 PCM samples>]}``,
audio at 16 kHz mono. Response:
``{"segments": [{"start": <seconds>, "end": <seconds>}, ...]}``, timestamps
relative to the submitted audio (i.e. local to whatever window was cut and
sent, not the source file's own timeline).

Verified by hand against ``silero-vad-ggml`` (already installed on this
machine's LocalAI, running on the already-installed ``whisper`` backend): a
20 s speech window returned 6 segments / 12.2 s of speech; a 27 s music
window returned 0 segments.

Cost, and why the request body is hand-built
----------------------------------------------

Measured for one 27 s window: ffmpeg extraction 0.06 s + JSON body build
0.10 s (using ``b','.join(b'%.4f' % x ...)`` over the unpacked PCM floats —
about 2x faster and ~25% smaller than ``json.dumps`` over an equivalent
Python list of rounded floats) + HTTP round trip including inference
0.78 s. Roughly 0.035 s of wall time per second of audio examined — see
``_build_request_body`` for the body-construction approach and
``WhisperConfig.vad_max_seconds`` for the resulting per-call budget.

Sequential, never concurrent, with the job's own ASR calls
-------------------------------------------------------------

Every request this module makes goes out one at a time, in the SAME
sequential manner as ``transcribe.py``'s own Whisper calls (never
``asyncio.gather``/``create_task``) — the backend serialises requests FIFO
internally anyway (see ``WhisperConfig.max_concurrent_requests``'s
docstring for the measured evidence backing that), so concurrency here
would buy nothing and would compete with the job's own transcription
requests for the same FIFO queue.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import subprocess
from pathlib import Path
from typing import Any

import httpx

from src.config import WhisperConfig, get_config

log = logging.getLogger(__name__)

_SAMPLE_RATE = 16000

# Per-request HTTP timeout: a floor plus a per-audio-second allowance well
# above the measured ~0.035s/s total cost (ffmpeg + body-build + HTTP now
# scoped out — this timeout covers the HTTP leg alone) — generous enough
# for a slower/loaded backend without risking an unbounded hang eating into
# the job's own transcription time.
_MIN_TIMEOUT_SECONDS = 30.0
_TIMEOUT_SECONDS_PER_AUDIO_SECOND = 0.5


def _resolve_vad_url(cfg: WhisperConfig) -> str:
    """``cfg.vad_url`` if set; otherwise ``cfg.base_url`` with a trailing
    ``/v1`` stripped, plus ``/vad`` — LocalAI serves this endpoint at the
    server ROOT, not under ``/v1`` like ``/audio/transcriptions`` and every
    other route this project calls."""
    if cfg.vad_url:
        return cfg.vad_url.rstrip("/")
    base = cfg.base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return f"{base}/vad"


def _ffmpeg_bin() -> str | None:
    """Path to ffmpeg from the shared resolver, or None if unavailable.
    Mirrors ``workers/transcribe.py``'s ``_ffmpeg_bin`` (duplicated rather
    than imported — this module must not import ``workers.transcribe``,
    which imports this module)."""
    from src.workers.ffmpeg import resolve_ffmpeg_dir

    directory = resolve_ffmpeg_dir()
    if not directory:
        return None
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    candidate = Path(directory) / exe
    return str(candidate) if candidate.is_file() else None


def _extract_pcm_f32le(audio_path: Path, start: float, duration: float) -> bytes:
    """Cut ``[start, start+duration)`` out of ``audio_path`` with ffmpeg,
    resampled to 16 kHz mono float32 PCM, entirely IN MEMORY — ffmpeg's
    stdout is captured directly, no temp files written or read.

    Raises ``RuntimeError``/``subprocess.CalledProcessError``/``OSError`` on
    any failure (ffmpeg missing, the cut itself failing); the caller
    (``speech_seconds``) treats every exception from this function as "no
    VAD data for this window", never as a reason to fail the job.
    """
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        raise RuntimeError("vad: ffmpeg unavailable")
    proc = subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
            "-i", str(audio_path),
            "-f", "f32le", "-ac", "1", "-ar", str(_SAMPLE_RATE),
            "-",
        ],
        check=True, capture_output=True,
    )
    return proc.stdout


def _build_request_body(model: str, pcm: bytes) -> bytes:
    """Hand-rolled JSON body for ``POST /vad`` — see the module docstring's
    "Cost, and why the request body is hand-built" section for the
    measurement behind this. ``pcm`` is raw little-endian float32 samples
    (as produced by ``_extract_pcm_f32le``); each is formatted to 4 decimal
    places (ample precision for a speech/non-speech decision) and joined
    with commas as BYTES directly — never materialised as a Python list of
    floats and handed to ``json.dumps``, which measurably costs about 2x
    more wall time and ~25% more bytes for the same data.

    ``model`` is JSON-escaped properly (via ``json.dumps`` on just the
    string, a negligible cost) since it's user-configured and must not be
    trusted to be safe to splice in raw.
    """
    n = len(pcm) // 4
    if n * 4 != len(pcm):
        pcm = pcm[: n * 4]  # drop a trailing partial sample from an odd-sized cut
    samples = struct.unpack(f"<{n}f", pcm) if n else ()
    floats = b",".join(b"%.4f" % s for s in samples)
    model_json = json.dumps(model).encode("utf-8")
    return b'{"model":%s,"audio":[%s]}' % (model_json, floats)


def _sum_speech_seconds(payload: Any, window_length: float) -> float:
    """Seconds of speech inside ``[0, window_length)`` per the VAD
    response's ``segments`` list (timestamps are LOCAL to the submitted
    window — see the module docstring). Raises ``ValueError`` for a
    malformed payload (no ``segments`` list at all) so the caller's
    generic exception handling covers this failure mode the same as every
    other one; an individual malformed segment entry is skipped rather
    than failing the whole window, matching ``transcribe._normalise_segments``'s
    "drop the bad entry, not the whole result" posture.
    """
    if not isinstance(payload, dict):
        raise ValueError("vad: response is not a JSON object")
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError("vad: response missing a 'segments' list")
    total = 0.0
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        start = max(0.0, start)
        end = min(window_length, end)
        if end > start:
            total += end - start
    return total


async def _speech_seconds_one_window(
    cfg: WhisperConfig, url: str, audio_path: Path, start: float, length: float
) -> float:
    """One full round trip for one window: cut -> build body -> POST ->
    parse. Any failure propagates to the caller as an ordinary exception —
    this function has no fallback behaviour of its own; ``speech_seconds``
    owns the "never fail the job" contract."""
    pcm = await asyncio.to_thread(_extract_pcm_f32le, audio_path, start, length)
    body = await asyncio.to_thread(_build_request_body, cfg.vad_model, pcm)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.effective_api_key}",
    }
    timeout = max(_MIN_TIMEOUT_SECONDS, length * _TIMEOUT_SECONDS_PER_AUDIO_SECOND)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, content=body, headers=headers)
        r.raise_for_status()
        payload = r.json()
    return _sum_speech_seconds(payload, length)


async def speech_seconds(
    audio_path: Path, windows: list[tuple[float, float]]
) -> dict[tuple[float, float], float] | None:
    """For each ``(start, end)`` in ``windows``, how many seconds of speech
    VAD found inside it (local to that window, i.e. already the true
    seconds-of-speech regardless of window length). Returns ``None`` —
    meaning "no VAD data at all, caller must behave exactly as if this
    module didn't exist" — when the feature is off (``whisper.vad_model``
    unset) or the very FIRST request this call attempts fails for any
    reason.

    A failure on a LATER window (not the first) instead stops the loop and
    returns whatever was gathered so far as a partial mapping — the
    remaining windows are simply absent from the result, which callers
    treat the same as "budget ran out" (see below): still counted the old
    way, nothing about the call raised.

    Every failure mode — the feature being off, HTTP 404 (a non-LocalAI
    backend simply lacking this route), a connection error, a timeout,
    ffmpeg being unavailable, a malformed response — is caught here and
    logged at most ONCE per call to this function (not once per window),
    and NEVER raised. This must be a silent, cheap no-op against any
    backend that doesn't speak this LocalAI extension — see the module
    docstring.

    Requests go out strictly SEQUENTIALLY (see the module docstring), and
    stop once the summed length of examined windows would exceed
    ``whisper.vad_max_seconds`` — the remaining windows are left out of
    the returned mapping and the cutoff is logged once.
    """
    cfg = get_config().whisper
    if not cfg.vad_model:
        return None
    if not windows:
        return {}

    url = _resolve_vad_url(cfg)
    results: dict[tuple[float, float], float] = {}
    used_seconds = 0.0
    logged_failure = False
    first_attempt = True

    for start, end in windows:
        length = end - start
        if length <= 0:
            continue
        if used_seconds + length > cfg.vad_max_seconds:
            log.info(
                "vad: budget (%.0fs) reached after examining %.0fs across "
                "%d window(s) — remaining candidate windows fall back to "
                "full-length weighting",
                cfg.vad_max_seconds, used_seconds, len(results),
            )
            break
        is_first_request = first_attempt
        first_attempt = False
        try:
            speech = await _speech_seconds_one_window(cfg, url, audio_path, start, length)
        except Exception:
            if not logged_failure:
                log.warning(
                    "vad: request failed (backend may not implement /vad, "
                    "or is unreachable) — treating as no VAD data for %s",
                    "this call" if is_first_request else "the remaining windows",
                    exc_info=True,
                )
                logged_failure = True
            if is_first_request:
                return None
            break
        results[(start, end)] = speech
        used_seconds += length

    return results


__all__ = ["speech_seconds"]
