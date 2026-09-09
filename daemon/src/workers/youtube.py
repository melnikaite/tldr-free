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
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_chain,
    wait_fixed,
)
from youtube_transcript_api import (
    Transcript,
    TranscriptList,
    YouTubeTranscriptApi,
)
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


def _guess_original_language(transcript_list: TranscriptList) -> str | None:
    """Best-effort guess at the video's own spoken language.

    Unlike yt-dlp's ``automatic_captions`` (which expands one ASR track into
    ~150 on-demand machine translations), youtube-transcript-api's
    ``TranscriptList`` never does that — it only exposes tracks that
    genuinely exist. YouTube generates exactly one auto-caption (ASR) track
    per video, in whatever language the audio actually is. So when there is
    exactly one generated transcript, its language IS the video's own
    language. With zero or more than one (e.g. multi-dub uploads) there's no
    such signal, so callers should fall through to the rest of the priority
    chain instead of guessing.
    """
    generated = [t for t in transcript_list if t.is_generated]
    if len(generated) == 1:
        return generated[0].language_code
    return None


def _select_youtube_api_transcript(
    transcript_list: TranscriptList,
    *,
    preferences: list[str],
    output_language: str | None,
) -> Transcript:
    """Pick a transcript deliberately instead of hard-coding a language.

    Priority order:
      1. the video's own/original language (``_guess_original_language``)
      2. ``youtube.subtitle_lang_preferences``, in order
      3. ``output.language``
      4. any manually-created track
      5. any auto-generated track

    Steps 1-3 use ``TranscriptList.find_transcript``, which itself checks
    manually-created before generated for each candidate language. Steps 4-5
    fall through to plain iteration, which yields manual tracks before
    generated ones (see ``TranscriptList.__iter__``). Raises
    ``NoTranscriptFound`` — already classified as permanent — only when the
    video has no transcripts at all.
    """
    original_lang = _guess_original_language(transcript_list)
    candidates = [lang for lang in [original_lang, *preferences, output_language] if lang]
    if candidates:
        with contextlib.suppress(NoTranscriptFound):
            return transcript_list.find_transcript(candidates)
    for transcript in transcript_list:
        return transcript
    raise NoTranscriptFound(transcript_list.video_id, candidates or ["en"], transcript_list)


def _fetch_transcript_sync(
    *,
    video_id: str,
    http_session: requests.Session | None,
    preferences: list[str],
    output_language: str | None,
) -> list[dict[str, Any]]:
    """One synchronous fetch via youtube-transcript-api 1.x.

    Uses ``list()`` + a deliberate language choice (``_select_youtube_api_transcript``)
    rather than the library's ``fetch()`` shortcut, which defaults to
    English-only and silently fails for every other language.
    """
    api = YouTubeTranscriptApi(http_client=http_session)
    transcript_list = api.list(video_id)
    transcript = _select_youtube_api_transcript(
        transcript_list, preferences=preferences, output_language=output_language,
    )
    fetched = transcript.fetch()
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
    preferences: list[str],
    output_language: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch the transcript with classification + tenacity retry on transient errors.

    Permanent errors raise immediately. Transient errors are retried up to
    ``max_attempts`` times, with the inter-attempt waits taken from
    ``backoff_seconds`` (extended with the last value if needed). After the
    final attempt fails, ``ExhaustedRetriesError`` is raised carrying the
    last error's ``code``.

    ``preferences`` (``youtube.subtitle_lang_preferences``) and
    ``output_language`` (``output.language``) drive the deliberate language
    choice in ``_fetch_transcript_sync`` — this never hard-codes English, so
    the fast path also works for non-English videos.
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
                video_id=video_id,
                http_session=http_session,
                preferences=preferences,
                output_language=output_language,
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
        # ffmpeg/ffprobe location for FFmpegExtractAudio — system binary if on
        # PATH, else a bundled static build (see workers.ffmpeg). Omitted when
        # neither is available so yt-dlp emits its own clear error.
        **_ffmpeg_opt(),
        # Deno runtime for the YouTube "n"/sig challenge solver — system binary
        # or a bundled static download (see workers.jsruntime). Critical for
        # caption-less videos whose audio we feed to Whisper.
        **_jsruntime_opt(),
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


def _ffmpeg_opt() -> dict[str, str]:
    """``{"ffmpeg_location": dir}`` when ffmpeg is resolvable, else ``{}``.

    Spread into ydl_opts so any path that postprocesses audio (or merges
    formats) finds ffmpeg without relying on the daemon's PATH — which under
    launchd/systemd often excludes Homebrew/static builds.
    """
    from src.workers.ffmpeg import resolve_ffmpeg_dir

    location = resolve_ffmpeg_dir()
    return {"ffmpeg_location": location} if location else {}


def _jsruntime_opt() -> dict[str, dict[str, dict[str, str]]]:
    """``{"js_runtimes": {"deno": {"path": <path>}}}`` when deno is resolvable.

    Lets yt-dlp solve YouTube's "n"/sig challenge without depending on the
    daemon's PATH. Empty → yt-dlp's default (deno looked up on PATH).
    """
    from src.workers.jsruntime import deno_runtime_opt

    return deno_runtime_opt()


def _ydl_base_opts(cookie_path: Path | None) -> dict[str, Any]:
    """Common yt-dlp opts for our YouTube callers (deno + EJS solver, quiet)."""
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # Auto-fetch the EJS challenge solver so YouTube's "n" challenge can be
        # solved by the bundled deno runtime. The yt-dlp-ejs package ships the
        # solver locally too; this stays as a fallback for deno's npm libs.
        "remote_components": ["ejs:github"],
        **_ffmpeg_opt(),
        **_jsruntime_opt(),
    }
    if cookie_path is not None:
        opts["cookiefile"] = str(cookie_path)
    return opts


def _pick_subtitle_lang(
    available: dict[str, Any],
    manual: dict[str, Any],
    original_lang: str | None,
    preferences: list[str],
    output_language: str | None = None,
) -> str | None:
    """Choose the best caption language from what's available.

    Shared by both callers of ``_download_subtitles_sync``: YouTube (many
    machine-translated ``automatic_captions`` alongside real tracks) and
    generic sites reached via the media pipeline (typically exactly one
    manually-created track and no ``automatic_captions`` at all — ZDF, ARD,
    Vimeo, TED, Coursera, …). Nothing here is YouTube-specific: it reasons
    purely over the ``available``/``manual`` dicts yt-dlp hands back for any
    extractor, so the generic case is just the priority chain's tail (steps
    1-3 rarely match with a single foreign-language track; step 4 picks it).

    Priority order:
      1. original-language track (whatever language the video is in)
      2. user-configured preferences (``youtube.subtitle_lang_preferences``), in order
      3. the configured output/summary language (``output.language``)
      4. any manually-created track (a human made it, so it's real)
      5. give up (``None``) — with YouTube's automatic-caption machine
         translations often covering 100+ languages, an arbitrary pick
         (e.g. alphabetically-first) is as likely to land on a translation
         nobody asked for as on anything useful. Returning ``None`` here lets
         the caller retry or defer to Whisper instead.
    """
    if not available:
        return None
    for lang in [original_lang, *preferences, output_language]:
        if lang and lang in available:
            return lang
    if manual:
        return sorted(manual.keys())[0]
    return None


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


# Pseudo-tracks yt-dlp surfaces alongside real subtitles that are NOT
# captions and must never be chosen: ``live_chat`` is the chat-replay
# stream, served as line-delimited JSON — feeding it to
# ``_parse_subtitle_json3`` raises "Expecting value: line 1 column 1"
# (json3 expects a single JSON object, not JSONL). yt-dlp only ever puts it
# in ``subtitles`` (never ``automatic_captions``), but a single junk manual
# entry outranks every real auto-caption language once ``manual`` is merged
# on top of ``auto`` — so it must be filtered before the merge, not after.
_NON_CAPTION_SUBTITLE_KEYS = frozenset({"live_chat"})


_VTT_TIME_RE = re.compile(r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{1,3})")
_VTT_CUE_TIMING_RE = re.compile(
    r"^\s*((?:\d+:)?\d{2}:\d{2}[.,]\d{1,3})\s*-->\s*((?:\d+:)?\d{2}:\d{2}[.,]\d{1,3})"
)
_VTT_TAG_RE = re.compile(r"<[^>]*>")


def _vtt_timestamp_to_seconds(ts: str) -> float:
    """Parse a WebVTT/SRT-style timestamp (``[HH:]MM:SS.mmm`` or ``,mmm``)."""
    m = _VTT_TIME_RE.fullmatch(ts.strip())
    if not m:
        raise ValueError(f"unparseable subtitle timestamp: {ts!r}")
    hours = int(m.group(1)) if m.group(1) else 0
    minutes = int(m.group(2))
    seconds = int(m.group(3))
    millis = int(m.group(4).ljust(3, "0")[:3])
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def _parse_subtitle_vtt(path: Path) -> list[dict[str, Any]]:
    """Parse a WebVTT caption file into our segment shape.

    Non-YouTube sites (ZDF, ARD, Vimeo, TED, Coursera, …) publish subtitle
    tracks yt-dlp can list and fetch, but they're WebVTT/SRT-shaped rather
    than YouTube's json3 — this is the parser format negotiation
    (``_download_subtitles_sync``'s ``subtitlesformat``) falls back to.

    Strips cue identifiers, cue settings (``align:``/``position:``/…) and
    inline markup (``<c>``, ``<00:00:01.360>``) down to plain text, and
    returns the same shape ``_parse_subtitle_json3`` does:
    ``[{"start": float, "duration": float, "text": str}, ...]``, so
    downstream timecode handling (``timecodes.build_marked_text`` etc.)
    needs no changes to consume either source.

    Blocks that aren't a cue (the ``WEBVTT`` header, ``NOTE``/``STYLE``/
    ``REGION`` blocks) have no ``-->`` timing line and are silently
    skipped, as are cues with no text.
    """
    raw = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\n+", raw)
    out: list[dict[str, Any]] = []
    for block in blocks:
        lines = block.split("\n")
        timing_idx: int | None = None
        timing_match: re.Match[str] | None = None
        for i, line in enumerate(lines):
            m = _VTT_CUE_TIMING_RE.match(line)
            if m:
                timing_idx = i
                timing_match = m
                break
        if timing_match is None or timing_idx is None:
            continue
        try:
            start = _vtt_timestamp_to_seconds(timing_match.group(1))
            end = _vtt_timestamp_to_seconds(timing_match.group(2))
        except ValueError:
            continue
        text_lines = lines[timing_idx + 1 :]
        text = " ".join(_VTT_TAG_RE.sub("", line).strip() for line in text_lines)
        text = " ".join(text.split())
        if not text:
            continue
        out.append(
            {"start": start, "duration": max(end - start, 0.0), "text": text}
        )
    return out


# Preference order for ``--sub-format``: json3 first (YouTube's compact
# format, unchanged behaviour for the YouTube path), WebVTT second (what
# most non-YouTube sites actually serve). yt-dlp still downloads *something*
# even when neither is available for a language (it falls back to the last
# offered format — see ``YoutubeDL.process_subtitles``), so the parser
# dispatch below keys off the actual downloaded file's extension rather than
# assuming the preference was honoured.
_SUBTITLE_FORMAT_PREFERENCE = "json3/vtt"
_SUBTITLE_PARSERS: dict[str, Callable[[Path], list[dict[str, Any]]]] = {
    "json3": _parse_subtitle_json3,
    "vtt": _parse_subtitle_vtt,
}


def _download_subtitles_sync(
    *,
    url: str,
    cookies: list[Cookie],
    dir: Path,
    lang_preferences: list[str],
    output_language: str | None = None,
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

        manual = {
            k: v for k, v in (info.get("subtitles") or {}).items()
            if k not in _NON_CAPTION_SUBTITLE_KEYS
        }
        auto = {
            k: v for k, v in (info.get("automatic_captions") or {}).items()
            if k not in _NON_CAPTION_SUBTITLE_KEYS
        }
        # Prefer manually-uploaded captions over auto-generated ones when both exist.
        available: dict[str, Any] = {**auto, **manual}
        original = info.get("language") or info.get("original_language")

        chosen = _pick_subtitle_lang(available, manual, original, lang_preferences, output_language)
        if chosen is None:
            log.info("yt-dlp subtitles: no caption track for %s", url)
            return None

        log.info(
            "yt-dlp subtitles: chosen lang=%r (manual=%s, auto=%s, original=%r) for %s",
            chosen, chosen in manual, chosen in auto, original, url,
        )

        # Pass 2: download, preferring json3 (YouTube) then falling back to
        # WebVTT (most non-YouTube sites). yt-dlp still writes *some* format
        # if neither is on offer for this language (see
        # ``_SUBTITLE_FORMAT_PREFERENCE``'s docstring above), so the actual
        # downloaded file's extension — not this preference string — decides
        # which parser runs.
        out_template = str(dir / "%(id)s.%(ext)s")
        dl_opts = {
            **_ydl_base_opts(cookie_path),
            "skip_download": True,
            "writesubtitles": chosen in manual,
            "writeautomaticsub": chosen not in manual,
            "subtitlesformat": _SUBTITLE_FORMAT_PREFERENCE,
            "subtitleslangs": [chosen],
            "outtmpl": out_template,
        }
        with YoutubeDL(dl_opts) as ydl:
            info2 = ydl.extract_info(url, download=True) or {}

        # Locate the produced file. yt-dlp stores it as <id>.<lang>.<ext>.
        requested = info2.get("requested_subtitles") or {}
        sub_info = requested.get(chosen) or {}
        sub_path_str = sub_info.get("filepath")
        sub_path: Path | None = Path(sub_path_str) if sub_path_str else None
        if sub_path is None or not sub_path.exists():
            # Fallback: try the standard naming convention for each format we
            # know how to parse, in preference order.
            video_id = info2.get("id")
            if video_id:
                for ext in _SUBTITLE_PARSERS:
                    guess = dir / f"{video_id}.{chosen}.{ext}"
                    if guess.exists():
                        sub_path = guess
                        break
        if sub_path is None or not sub_path.exists():
            log.warning("yt-dlp subtitles: file not found after download for %s", url)
            return None

        ext = sub_path.suffix.lower().lstrip(".")
        parser = _SUBTITLE_PARSERS.get(ext)
        if parser is None:
            log.info(
                "yt-dlp subtitles: downloaded format %r for %s has no parser, skipping",
                ext, url,
            )
            with contextlib.suppress(OSError):
                sub_path.unlink()
            return None

        try:
            return parser(sub_path)
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
    output_language: str | None = None,
    max_attempts: int = 1,
    backoff_seconds: list[int] | None = None,
    retry_on_no_track: bool = True,
) -> list[dict[str, Any]] | None:
    """Retry-wrapped async caption fetch — probe, download json3/vtt, parse.

    An empty/missing caption track, a missing downloaded file, and a parse
    failure are all treated as retryable BY DEFAULT: YouTube intermittently
    throttles/bot-checks logged-in sessions (the same video that fails now
    routinely succeeds moments later, cookies unchanged), so a single miss
    is not proof the video lacks captions. ``retry_on_no_track`` (default
    ``True``) preserves exactly this behaviour for YouTube's callers.

    That reasoning does not transfer to a generic media page
    (``pipeline._run_media``): "this URL simply has no subtitle track"
    reproduces identically on every attempt, so retrying it just burns extra
    ``extract_info`` probes and backoff sleeps on the common captionless
    case — a straight latency regression on the majority path, and its own
    rate-limiting hazard on top. Pass ``retry_on_no_track=False`` there:
    a *clean* "no usable captions" result (``_download_subtitles_sync``
    returning ``None`` without raising) is accepted as final after the
    first attempt. Genuine transient failures — a raised exception, whether
    from a yt-dlp/network error or a parse failure — are unaffected by this
    flag and still retried up to ``max_attempts``, since those really can be
    a one-off blip regardless of site. Do not "simplify" this by unifying
    the two policies — see the docstring above for why they differ.

    The first attempt uses ``cookies`` as given; every subsequent attempt
    drops them, since a cookie-less request has been observed to succeed on
    exactly the videos a cookied one failed on. Between attempts, waits are
    taken from ``backoff_seconds`` (extended with the last value if needed).

    Returns parsed segments, or ``None`` once every attempt is exhausted (or
    immediately, per ``retry_on_no_track``).
    """
    attempts = max(max_attempts, 1)
    waits = list(backoff_seconds or [])

    last_segments: list[dict[str, Any]] | None = None
    for attempt in range(1, attempts + 1):
        attempt_cookies = cookies if attempt == 1 else []
        raised = False
        try:
            last_segments = await asyncio.to_thread(
                _download_subtitles_sync,
                url=url,
                cookies=attempt_cookies,
                dir=dir,
                lang_preferences=lang_preferences,
                output_language=output_language,
            )
        except Exception:
            log.exception(
                "yt-dlp subtitles attempt %d/%d failed for %s (cookies=%s)",
                attempt, attempts, url, bool(attempt_cookies),
            )
            last_segments = None
            raised = True

        if last_segments:
            if attempt > 1:
                log.info(
                    "yt-dlp subtitles: succeeded on attempt %d/%d for %s",
                    attempt, attempts, url,
                )
            return last_segments

        log.info(
            "yt-dlp subtitles attempt %d/%d: no usable captions for %s (cookies=%s)",
            attempt, attempts, url, bool(attempt_cookies),
        )

        if not raised and not retry_on_no_track:
            # A clean (non-exception) empty result and this caller has opted
            # out of retrying that specific outcome — see docstring above.
            log.info(
                "yt-dlp subtitles: no track found for %s, not retrying "
                "(retry_on_no_track=False)", url,
            )
            return None

        if attempt < attempts:
            wait = waits[min(attempt - 1, len(waits) - 1)] if waits else 1
            await asyncio.sleep(wait)

    return None


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
