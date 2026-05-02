"""YouTube transcript fast path + yt-dlp audio download.

Public surface:
    extract_video_id(url: str) -> str
        Parse common YouTube URL forms; raise ValueError for anything else.

    fetch_transcript_with_retry(*, video_id, cookies, max_attempts, backoff_seconds)
        -> list[dict]
        Calls youtube-transcript-api once per attempt with a Retry decorator.
        Permanent errors → PermanentTranscriptError immediately.
        Transient errors → retry with backoff, then ExhaustedRetriesError.
        Returns a list of {"start": float, "duration": float, "text": str}.

    download_audio(*, url, cookies, dir) -> Path
        Downloads the worst-quality audio with yt-dlp inside asyncio.to_thread.
        Returns the path to the downloaded file. Caller must delete after use.

The transcript-API behaves synchronously and uses a ``requests.Session``;
we keep the call site sync internally and wrap with ``asyncio.to_thread``
so the FastAPI event loop isn't blocked.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests  # type: ignore[import-untyped]
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_chain,
    wait_fixed,
)
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    AgeRestricted,
    CouldNotRetrieveTranscript,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)

from src.api.schemas import Cookie
from src.config import get_config
from src.storage.cookies import build_requests_session, write_netscape_cookie_file
from src.workers.errors import (
    ExhaustedRetriesError,
    NetworkTranscriptError,
    PermanentTranscriptError,
    TransientTranscriptError,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URL → video id
# ---------------------------------------------------------------------------


# Strict YouTube video-id pattern: 11 chars from [A-Za-z0-9_-].
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url: str) -> str:
    """Parse a YouTube URL and return the 11-char video id.

    Handles:
        https://www.youtube.com/watch?v=ID
        https://youtu.be/ID
        https://www.youtube.com/shorts/ID
        https://www.youtube.com/embed/ID
        https://m.youtube.com/watch?v=ID
        https://music.youtube.com/watch?v=ID

    Raises ``ValueError`` if the URL does not contain a recognisable video id.
    """
    if not url:
        raise ValueError("empty url")

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""

    # 1) youtu.be/<id>
    if host.endswith("youtu.be"):
        candidate = path.lstrip("/").split("/")[0]
        if _VIDEO_ID_RE.match(candidate):
            return candidate
        raise ValueError(f"could not extract video id from {url!r}")

    # 2) youtube.com/* (any subdomain)
    if host == "youtube.com" or host.endswith(".youtube.com"):
        # /watch?v=<id>
        if path == "/watch" or path.startswith("/watch"):
            qs = parse_qs(parsed.query)
            vs = qs.get("v") or []
            if vs and _VIDEO_ID_RE.match(vs[0]):
                return vs[0]
        # /shorts/<id>, /embed/<id>, /v/<id>, /live/<id>
        for prefix in ("/shorts/", "/embed/", "/v/", "/live/"):
            if path.startswith(prefix):
                candidate = path[len(prefix):].split("/")[0]
                if _VIDEO_ID_RE.match(candidate):
                    return candidate
        raise ValueError(f"could not extract video id from {url!r}")

    raise ValueError(f"not a youtube url: {url!r}")


# ---------------------------------------------------------------------------
# Transcript classification
# ---------------------------------------------------------------------------

# These exceptions are domain-permanent: retrying won't help.
_PERMANENT_EXC: tuple[type[Exception], ...] = (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    AgeRestricted,
)

# These exceptions are transient: retry with backoff.
_TRANSIENT_EXC: tuple[type[Exception], ...] = (
    IpBlocked,
    RequestBlocked,
)


def _classify_transcript_exception(exc: BaseException) -> Exception:
    """Translate a youtube-transcript-api exception into our domain hierarchy."""
    if isinstance(exc, _PERMANENT_EXC):
        return PermanentTranscriptError(str(exc))
    if isinstance(exc, _TRANSIENT_EXC):
        return TransientTranscriptError(str(exc))
    if isinstance(exc, requests.exceptions.RequestException):
        # Connection refused / timeout / DNS / SSL → network class.
        return NetworkTranscriptError(str(exc))
    if isinstance(exc, CouldNotRetrieveTranscript):
        # Anything else from the library is treated as transient — it usually
        # boils down to YouTube returning an unexpected payload (rate-limit page).
        return TransientTranscriptError(str(exc))
    # Unknown error — wrap as transient so the caller still defers to whisper.
    return TransientTranscriptError(str(exc))


# ---------------------------------------------------------------------------
# Transcript fetch
# ---------------------------------------------------------------------------


def _fetch_transcript_sync(
    *,
    video_id: str,
    http_session: requests.Session | None,
) -> list[dict[str, Any]]:
    """One synchronous fetch via youtube-transcript-api 1.x."""
    api = YouTubeTranscriptApi(http_client=http_session)
    fetched = api.fetch(video_id)
    # ``fetched`` iterates as FetchedTranscriptSnippet(text, start, duration).
    out: list[dict[str, Any]] = []
    for snippet in fetched:
        out.append(
            {
                "start": float(getattr(snippet, "start", 0.0)),
                "duration": float(getattr(snippet, "duration", 0.0)),
                "text": str(getattr(snippet, "text", "") or ""),
            }
        )
    return out


async def fetch_transcript_with_retry(
    *,
    video_id: str,
    cookies: list[Cookie],
    max_attempts: int,
    backoff_seconds: list[int],
) -> list[dict[str, Any]]:
    """Fetch the transcript with classification + tenacity retry on transient errors.

    Permanent errors raise immediately. Transient errors are retried up to
    ``max_attempts`` times, with the inter-attempt waits taken from
    ``backoff_seconds`` (extended with the last value if needed). After the
    final attempt fails, ``ExhaustedRetriesError`` is raised carrying the
    last error's ``code``.
    """
    if max_attempts < 1:
        max_attempts = 1
    waits = [wait_fixed(s) for s in backoff_seconds] or [wait_fixed(1)]
    # If callers supplied fewer backoff entries than max_attempts, extend with
    # the last entry so tenacity has something to wait on between attempts.
    while len(waits) < max_attempts:
        waits.append(waits[-1])

    http_session = build_requests_session(cookies) if cookies else None

    def _attempt() -> list[dict[str, Any]]:
        try:
            return _fetch_transcript_sync(
                video_id=video_id, http_session=http_session
            )
        except _PERMANENT_EXC as exc:
            # Permanent — translate and bubble up so tenacity does NOT retry.
            raise _classify_transcript_exception(exc) from exc
        except Exception as exc:
            translated = _classify_transcript_exception(exc)
            if isinstance(translated, PermanentTranscriptError):
                # Unexpected case — but keep the contract.
                raise translated from exc
            raise translated from exc

    last_translated_error: TransientTranscriptError | NetworkTranscriptError | None = (
        None
    )

    def _run_blocking() -> list[dict[str, Any]]:
        nonlocal last_translated_error
        try:
            for attempt in Retrying(
                wait=wait_chain(*waits),
                stop=stop_after_attempt(max_attempts),
                retry=retry_if_exception_type(
                    (TransientTranscriptError, NetworkTranscriptError)
                ),
                reraise=True,
            ):
                with attempt:
                    return _attempt()
            # Unreachable: Retrying either returns a value through the with-block
            # or raises. Keep mypy happy.
            raise RuntimeError("unreachable")
        except PermanentTranscriptError:
            raise
        except (TransientTranscriptError, NetworkTranscriptError) as final:
            last_translated_error = final
            raise ExhaustedRetriesError(final) from final
        except RetryError as retry_err:  # pragma: no cover — reraise=True so unlikely
            wrapped = retry_err.last_attempt.exception()
            if isinstance(wrapped, (TransientTranscriptError, NetworkTranscriptError)):
                last_translated_error = wrapped
                raise ExhaustedRetriesError(wrapped) from retry_err
            raise

    return await asyncio.to_thread(_run_blocking)


# ---------------------------------------------------------------------------
# Audio download (yt-dlp)
# ---------------------------------------------------------------------------


def _download_audio_sync(
    *,
    url: str,
    cookies: list[Cookie],
    dir: Path,
) -> tuple[Path, float | None]:
    """Run yt-dlp synchronously inside this thread.

    Returns ``(audio_path, duration_seconds | None)``. Duration is taken from
    yt-dlp's info dict and is used by the Whisper streaming progress reporter
    to compute a real percentage of audio processed.

    Imports ``yt_dlp`` lazily so module-load doesn't pay the cost in the hot
    fast-path."""
    from yt_dlp import YoutubeDL

    cfg = get_config().youtube
    dir.mkdir(parents=True, exist_ok=True)

    cookie_path: Path | None = None
    if cookies:
        cookie_path = write_netscape_cookie_file(cookies, dir)

    sleep_min, sleep_max = (
        (cfg.ytdlp_sleep_interval[0], cfg.ytdlp_sleep_interval[1])
        if len(cfg.ytdlp_sleep_interval) >= 2
        else (3, 8)
    )

    output_template = str(dir / "%(id)s.%(ext)s")

    # Build the post-processor: re-encode to opus at the configured cap so we
    # don't need to depend on what stream YouTube is currently serving.
    postprocessors: list[dict[str, Any]] = [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": cfg.audio_format,
            "preferredquality": str(cfg.audio_bitrate_max),
        }
    ]

    bitrate_cap = cfg.audio_bitrate_max
    # Preference order:
    #   1. cheapest audio-only stream within our bitrate cap
    #   2. any audio-only stream
    #   3. cheapest muxed stream (FFmpegExtractAudio postprocessor pulls audio
    #      out via ffmpeg) — this is the fallback when YouTube returns no pure
    #      audio formats for the cookied session, which happens more often
    #      since YouTube started requiring a JS runtime for full extraction.
    fmt_filter = (
        f"worstaudio[abr<={bitrate_cap}]/worstaudio/"
        f"bestaudio[abr<={bitrate_cap}]/bestaudio/"
        f"worst[height<=480]/worst/best"
    )

    ydl_opts: dict[str, Any] = {
        "format": fmt_filter,
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "sleep_interval": sleep_min,
        "max_sleep_interval": sleep_max,
        "postprocessors": postprocessors,
        # Allow yt-dlp to auto-fetch the EJS challenge solver from GitHub on
        # first need (paired with the deno runtime baked into the image).
        # Without this YouTube's "n" challenge cannot be solved and some
        # formats are silently dropped from the available set.
        "remote_components": ["ejs:github"],
    }
    if cookie_path is not None:
        ydl_opts["cookiefile"] = str(cookie_path)

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        if info is None:
            raise RuntimeError(f"yt-dlp returned no info for {url}")

        # Prefer the post-processed file path if available; fall back to
        # the originally requested download.
        path: str | None = None
        requested = info.get("requested_downloads") or []
        if requested:
            entry = requested[0] or {}
            path = entry.get("filepath") or entry.get("_filename")
        if not path:
            # ydl.prepare_filename gives the *pre*-postprocessor name. After
            # extracting audio, the file actually on disk has the audio_format
            # extension (e.g. .opus).
            from yt_dlp import YoutubeDL as _Y  # noqa: F811

            with _Y(ydl_opts) as ydl2:
                base = ydl2.prepare_filename(info)
            stem = Path(base).with_suffix("")
            ext_path = stem.with_suffix(f".{cfg.audio_format}")
            path = str(ext_path) if ext_path.exists() else base

        result = Path(path)
        if not result.exists():
            raise RuntimeError(f"yt-dlp reported success but file missing: {result}")

        raw_duration = info.get("duration")
        duration: float | None = None
        if raw_duration is not None:
            try:
                duration = float(raw_duration)
            except (TypeError, ValueError):
                duration = None
        return result, duration
    finally:
        if cookie_path is not None:
            try:
                cookie_path.unlink(missing_ok=True)
            except OSError:
                log.warning("failed to unlink cookie file %s", cookie_path)


async def download_audio(
    *,
    url: str,
    cookies: list[Cookie],
    dir: Path,
) -> tuple[Path, float | None]:
    """Async wrapper around the blocking yt-dlp call.

    Returns the path to the downloaded audio file plus its duration in
    seconds (``None`` if yt-dlp didn't report it).
    """
    return await asyncio.to_thread(
        _download_audio_sync, url=url, cookies=cookies, dir=dir
    )


# ---------------------------------------------------------------------------
# Subtitles via yt-dlp (fallback before Whisper)
# ---------------------------------------------------------------------------


def _ydl_base_opts(cookie_path: Path | None) -> dict[str, Any]:
    """Common yt-dlp opts for our YouTube callers (deno + EJS solver, quiet)."""
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # Auto-fetch the EJS challenge solver so YouTube's "n" challenge can be
        # solved by the bundled deno runtime.
        "remote_components": ["ejs:github"],
    }
    if cookie_path is not None:
        opts["cookiefile"] = str(cookie_path)
    return opts


def _pick_subtitle_lang(
    available: dict[str, Any], original_lang: str | None, preferences: list[str]
) -> str | None:
    """Choose the best caption language from what's available.

    Priority order:
      1. original-language track (whatever language the video is in)
      2. user-configured preferences in order
      3. first language alphabetically (deterministic last resort)
    """
    if not available:
        return None
    for lang in [original_lang, *preferences]:
        if lang and lang in available:
            return lang
    return sorted(available.keys())[0] if available else None


def _parse_subtitle_json3(path: Path) -> list[dict[str, Any]]:
    """Parse YouTube's json3 subtitle format into our segment shape.

    Schema:
        {"events": [{"tStartMs": int, "dDurationMs": int, "segs": [{"utf8": str}]}, ...]}

    Returns ``[{"start": float, "duration": float, "text": str}, ...]`` with
    blank/cue-only events skipped.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    events = raw.get("events") or []
    out: list[dict[str, Any]] = []
    for ev in events:
        start_ms = ev.get("tStartMs")
        if start_ms is None:
            continue
        dur_ms = ev.get("dDurationMs") or 0
        text = "".join(s.get("utf8", "") for s in (ev.get("segs") or []))
        text = text.replace("\n", " ").strip()
        if not text:
            continue
        out.append(
            {
                "start": float(start_ms) / 1000.0,
                "duration": float(dur_ms) / 1000.0,
                "text": text,
            }
        )
    return out


def _download_subtitles_sync(
    *,
    url: str,
    cookies: list[Cookie],
    dir: Path,
    lang_preferences: list[str],
) -> list[dict[str, Any]] | None:
    """Probe available caption tracks, pick a language, download json3, parse.

    Returns segments or None if no usable caption track exists. Raises on
    yt-dlp / network errors so the caller can decide whether to fall back
    further (i.e. to Whisper).
    """
    from yt_dlp import YoutubeDL

    dir.mkdir(parents=True, exist_ok=True)
    cookie_path = write_netscape_cookie_file(cookies, dir) if cookies else None

    try:
        # Pass 1: probe — what languages does yt-dlp see?
        probe_opts = {
            **_ydl_base_opts(cookie_path),
            "skip_download": True,
            "writesubtitles": False,
            "writeautomaticsub": False,
        }
        with YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(url, download=False) or {}

        manual = info.get("subtitles") or {}
        auto = info.get("automatic_captions") or {}
        # Prefer manually-uploaded captions over auto-generated ones when both exist.
        available: dict[str, Any] = {**auto, **manual}
        original = info.get("language") or info.get("original_language")

        chosen = _pick_subtitle_lang(available, original, lang_preferences)
        if chosen is None:
            log.info("yt-dlp subtitles: no caption track for %s", url)
            return None

        log.info(
            "yt-dlp subtitles: chosen lang=%r (manual=%s, auto=%s, original=%r) for %s",
            chosen, chosen in manual, chosen in auto, original, url,
        )

        # Pass 2: download in json3 (compact + easy to parse).
        out_template = str(dir / "%(id)s.%(ext)s")
        dl_opts = {
            **_ydl_base_opts(cookie_path),
            "skip_download": True,
            "writesubtitles": chosen in manual,
            "writeautomaticsub": chosen not in manual,
            "subtitlesformat": "json3",
            "subtitleslangs": [chosen],
            "outtmpl": out_template,
        }
        with YoutubeDL(dl_opts) as ydl:
            info2 = ydl.extract_info(url, download=True) or {}

        # Locate the produced file. yt-dlp stores it as <id>.<lang>.json3.
        requested = info2.get("requested_subtitles") or {}
        sub_info = requested.get(chosen) or {}
        sub_path_str = sub_info.get("filepath")
        sub_path: Path | None = Path(sub_path_str) if sub_path_str else None
        if sub_path is None or not sub_path.exists():
            # Fallback: try standard naming convention.
            video_id = info2.get("id")
            if video_id:
                guess = dir / f"{video_id}.{chosen}.json3"
                if guess.exists():
                    sub_path = guess
        if sub_path is None or not sub_path.exists():
            log.warning("yt-dlp subtitles: file not found after download for %s", url)
            return None

        try:
            return _parse_subtitle_json3(sub_path)
        finally:
            with contextlib.suppress(OSError):
                sub_path.unlink()
    finally:
        if cookie_path is not None:
            with contextlib.suppress(OSError):
                cookie_path.unlink(missing_ok=True)


async def download_subtitles(
    *,
    url: str,
    cookies: list[Cookie],
    dir: Path,
    lang_preferences: list[str],
) -> list[dict[str, Any]] | None:
    """Async wrapper around the blocking yt-dlp subtitles fetch.

    Returns parsed segments (with timestamps preserved) or None if no caption
    track is available. Raises on transport / yt-dlp errors.
    """
    return await asyncio.to_thread(
        _download_subtitles_sync,
        url=url,
        cookies=cookies,
        dir=dir,
        lang_preferences=lang_preferences,
    )


# ---------------------------------------------------------------------------
# Metadata probe (canonical title)
# ---------------------------------------------------------------------------


def _fetch_video_metadata_sync(
    *, url: str, cookies: list[Cookie], scratch_dir: Path
) -> dict[str, Any]:
    from yt_dlp import YoutubeDL

    scratch_dir.mkdir(parents=True, exist_ok=True)
    cookie_path = write_netscape_cookie_file(cookies, scratch_dir) if cookies else None
    try:
        with YoutubeDL({**_ydl_base_opts(cookie_path), "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False) or {}
        return {
            "title": info.get("title"),
            "language": info.get("language") or info.get("original_language"),
            "duration": info.get("duration"),
        }
    except Exception as exc:
        log.warning("yt-dlp metadata probe failed for %s: %s", url, exc)
        return {}
    finally:
        if cookie_path is not None:
            with contextlib.suppress(OSError):
                cookie_path.unlink(missing_ok=True)


async def fetch_video_metadata(
    *, url: str, cookies: list[Cookie], scratch_dir: Path
) -> dict[str, Any]:
    """Lightweight yt-dlp probe for the canonical video title (and a few
    incidental fields). Returns ``{}`` on any failure — we never want a
    metadata hiccup to break the actual transcript/summary path.

    Used by the pipeline so the persisted title comes from YouTube's own
    metadata rather than whatever the extension scraped from a possibly
    stale or partially-loaded DOM (YouTube is an SPA).
    """
    return await asyncio.to_thread(
        _fetch_video_metadata_sync,
        url=url,
        cookies=cookies,
        scratch_dir=scratch_dir,
    )


__all__ = [
    "download_audio",
    "download_subtitles",
    "extract_video_id",
    "fetch_transcript_with_retry",
    "fetch_video_metadata",
]
