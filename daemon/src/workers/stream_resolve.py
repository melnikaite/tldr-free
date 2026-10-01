"""Resolve network-sniffed streams into what to download and transcribe.

Public API:
    parse_playlist(url, text) -> Playlist | None
    select(streams, playlists, *, default_url) -> Resolution
    resolve(streams, *, default_url, headers, cookies) -> Resolution
    fetch_subtitle_segments(choice, resolution, *, headers, cookies)
        -> list[dict] | None
    subtitles_cover(segments, duration) -> bool

The extension records every HLS/DASH manifest and subtitle file a tab
fetched (``JobCreateRequest.sniffed_streams``) but can't tell a master
playlist from a rendition playlist without the body — and both end in
``.m3u8``. Players only fetch the playlists of the CURRENTLY selected audio
and subtitle renditions (and fetch new ones on every switch), so "what the
user chose" is "what was fetched last". This module fetches the sniffed HLS
playlists (small text files) with the player's headers, classifies them,
and picks:

- master: the most recently seen master playlist. Handles players that
  serve every dub as its own master (a dub switch fetches a new master).
- audio: among the master's ``EXT-X-MEDIA TYPE=AUDIO`` renditions, the one
  whose playlist was fetched most recently; if none was sniffed, the
  ``DEFAULT=YES`` one, else the first. When chosen, ITS playlist is what
  yt-dlp downloads — audio-only, a fraction of the muxed variant's size.
  Without separate audio renditions the master itself is downloaded, as
  before (yt-dlp's format filter picks the smallest variant).
- subtitle: the most recently fetched of: a sniffed direct ``.vtt``/
  ``.srt`` file, a sniffed ``TYPE=SUBTITLES`` rendition playlist of any
  fetched master, or a standalone media playlist of WebVTT segments.
  Direct subtitle files that are just segments of a known subtitle
  playlist are not candidates of their own.

DASH manifests are passed through untouched (yt-dlp handles them).
Everything here is best-effort: a failed fetch just means less to choose
from, never a failed job.
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from src.llm import languages
from src.workers import youtube

log = logging.getLogger(__name__)

FETCH_TIMEOUT_SECONDS = 15.0
MAX_PLAYLIST_BYTES = 2 * 1024 * 1024
MAX_PLAYLISTS_FETCHED = 30
MAX_SUBTITLE_SEGMENTS = 3000
MAX_SUBTITLE_TOTAL_BYTES = 30 * 1024 * 1024
FETCH_CONCURRENCY = 6

# Subtitles become the transcript only if they plausibly cover the media:
# at least this many cues, and the last cue ends at least this far into it
# (credits/silence at the end are normal, a half-loaded track is not).
# Below that, ASR runs instead — a wrong transcript beats a hole.
SUBTITLE_MIN_CUES = 5
SUBTITLE_MIN_COVERAGE = 0.7

_SUBTITLE_SEGMENT_RE = re.compile(r"\.(?:vtt|webvtt)(?:$|[?#])", re.I)
_ATTR_RE = re.compile(r'([A-Z0-9-]+)=("[^"]*"|[^,]*)')
_TIMESTAMP_MAP_RE = re.compile(r"X-TIMESTAMP-MAP=([^\n]*)")
_ASS_OVERRIDE_RE = re.compile(r"\{\\[^}]*\}")


@dataclass(frozen=True)
class Rendition:
    type: str                   # AUDIO | SUBTITLES | VIDEO | CLOSED-CAPTIONS
    uri: str | None             # absolute
    name: str | None
    language: str | None
    default: bool
    group_id: str | None


@dataclass
class Playlist:
    url: str
    is_master: bool
    renditions: list[Rendition] = field(default_factory=list)
    variant_uris: list[str] = field(default_factory=list)
    segment_uris: list[str] = field(default_factory=list)
    segment_durations: list[float] = field(default_factory=list)
    endlist: bool = False

    @property
    def duration(self) -> float | None:
        """Sum of EXTINFs — the media's real duration, but only for a
        finished (VOD, ``#EXT-X-ENDLIST``) playlist; a live window's sum
        says nothing about the whole. ``None`` rather than a guess."""
        total = sum(self.segment_durations)
        return total if self.endlist and total > 0 else None

    @property
    def is_subtitle_media(self) -> bool:
        return bool(self.segment_uris) and all(
            _SUBTITLE_SEGMENT_RE.search(u) for u in self.segment_uris
        )


@dataclass(frozen=True)
class SubtitleChoice:
    url: str
    via: str                    # "file" (direct .vtt/.srt) | "hls" (WebVTT playlist)
    name: str | None
    language: str | None
    last_seen: float


@dataclass
class Resolution:
    download_url: str           # what yt-dlp downloads for ASR
    master_url: str | None      # carries video — what frames should use
    audio: Rendition | None
    subtitle: SubtitleChoice | None
    duration: float | None      # from the downloaded playlist's EXTINFs
    playlists: dict[str, Playlist] = field(default_factory=dict)

    def selection(self, *, subtitles_used: bool) -> dict[str, Any]:
        """Shape of ``api.schemas.MediaSelection``."""
        return {
            "audio_name": self.audio.name if self.audio else None,
            "audio_language": self.audio.language if self.audio else None,
            "subtitle_name": self.subtitle.name if self.subtitle else None,
            "subtitle_language": self.subtitle.language if self.subtitle else None,
            "subtitles_used": subtitles_used,
        }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _attrs(s: str) -> dict[str, str]:
    return {k: v.strip('"') for k, v in _ATTR_RE.findall(s)}


def parse_playlist(url: str, text: str) -> Playlist | None:
    """Parse an M3U8 body. ``None`` if it isn't one. URIs are resolved
    against ``url``."""
    text = text.lstrip("﻿")
    if not text.lstrip().startswith("#EXTM3U"):
        return None
    lines = [ln.strip() for ln in text.splitlines()]
    pl = Playlist(url=url, is_master=False)
    pending_variant = False
    pending_duration: float | None = None
    for line in lines:
        if not line:
            continue
        if line.startswith("#EXT-X-MEDIA:"):
            pl.is_master = True
            a = _attrs(line.split(":", 1)[1])
            pl.renditions.append(Rendition(
                type=a.get("TYPE", "").upper(),
                uri=urljoin(url, a["URI"]) if a.get("URI") else None,
                name=a.get("NAME") or None,
                language=a.get("LANGUAGE") or None,
                default=a.get("DEFAULT", "").upper() == "YES",
                group_id=a.get("GROUP-ID") or None,
            ))
        elif line.startswith("#EXT-X-STREAM-INF"):
            pl.is_master = True
            pending_variant = True
        elif line.startswith("#EXTINF:"):
            try:
                pending_duration = float(line.split(":", 1)[1].split(",", 1)[0])
            except ValueError:
                pending_duration = 0.0
        elif line.startswith("#EXT-X-ENDLIST"):
            pl.endlist = True
        elif line.startswith("#"):
            continue
        elif pending_variant:
            pl.variant_uris.append(urljoin(url, line))
            pending_variant = False
        else:
            pl.segment_uris.append(urljoin(url, line))
            pl.segment_durations.append(pending_duration or 0.0)
            pending_duration = None
    return pl


# ---------------------------------------------------------------------------
# Selection (pure)
# ---------------------------------------------------------------------------


def _lang_code(language: str | None, name: str | None) -> str | None:
    """EXT-X-MEDIA LANGUAGE, else a language word in NAME ("German
    Original"), else None (the caller detects from text)."""
    # Name words shorter than 3 letters ("it", "no") are too ambiguous.
    words = [w for w in re.findall(r"[^\W\d_]+", name or "") if len(w) >= 3]
    for raw in [language, *words]:
        if not raw:
            continue
        try:
            return languages.normalize_lang(raw).code
        except Exception:  # noqa: BLE001 — unknown word, try the next
            continue
    return languages.short_lang_code(language)


def select(streams: list[Any], playlists: dict[str, Playlist], *, default_url: str) -> Resolution:
    """Pick master/audio/subtitle from sniffed ``streams`` (objects with
    ``url``/``kind``/``first_seen``/``last_seen``) and the subset of them
    that were fetched and parsed (``playlists``, keyed by URL)."""
    seen = {s.url: s for s in streams}
    # Players often append their own query tokens to the URIs a master
    # lists, so an exact-URL miss falls back to host+path (latest wins).
    seen_by_path: dict[tuple[str, str], Any] = {}
    for s in sorted(streams, key=lambda s: (s.last_seen, s.first_seen)):
        p = urlparse(s.url)
        seen_by_path[(p.netloc, p.path)] = s

    def lookup(url: str | None) -> Any:
        if not url:
            return None
        s = seen.get(url)
        if s is None:
            p = urlparse(url)
            s = seen_by_path.get((p.netloc, p.path))
        return s

    def recency(url: str) -> tuple[float, float]:
        s = lookup(url)
        return (s.last_seen, s.first_seen) if s else (float("-inf"), float("-inf"))

    masters = [p for u, p in playlists.items() if p.is_master and u in seen]
    master = max(masters, key=lambda p: recency(p.url)) if masters else None

    audio: Rendition | None = None
    download_url = default_url
    if master is not None:
        audios = [r for r in master.renditions if r.type == "AUDIO" and r.uri]
        sniffed = [r for r in audios if lookup(r.uri) is not None]
        if sniffed:
            audio = max(sniffed, key=lambda r: recency(r.uri or ""))
        elif audios:
            audio = next((r for r in audios if r.default), audios[0])
        if audio and audio.uri:
            # The URL the player actually fetched carries whatever tokens
            # the CDN wants; the bare master URI may not.
            hit = lookup(audio.uri)
            download_url = hit.url if hit is not None else audio.uri
        else:
            download_url = master.url

    duration: float | None = None
    dl_pl = playlists.get(download_url)
    if dl_pl is not None and not dl_pl.is_master:
        duration = dl_pl.duration
    elif master is not None:
        for v in master.variant_uris:
            vp = playlists.get(v)
            if vp is not None and vp.duration:
                duration = vp.duration
                break

    # Subtitle candidates.
    candidates: list[SubtitleChoice] = []
    sub_playlist_urls: set[str] = set()
    for p in playlists.values():
        if not p.is_master:
            continue
        for r in p.renditions:
            if r.type != "SUBTITLES" or not r.uri:
                continue
            sub_playlist_urls.add(r.uri)
            hit = lookup(r.uri)
            if hit is not None:
                sub_playlist_urls.add(hit.url)
                candidates.append(SubtitleChoice(
                    url=hit.url, via="hls", name=r.name,
                    language=_lang_code(r.language, r.name),
                    last_seen=hit.last_seen,
                ))
    segment_urls: set[str] = set()
    for u, p in playlists.items():
        if p.is_master or not p.is_subtitle_media:
            continue
        segment_urls.update(p.segment_uris)
        if u not in sub_playlist_urls and u in seen:
            candidates.append(SubtitleChoice(
                url=u, via="hls", name=None, language=None,
                last_seen=seen[u].last_seen,
            ))
    for s in streams:
        if s.kind == "subtitle" and s.url not in segment_urls:
            path = urlparse(s.url).path.lower()
            if path.endswith(".ass"):
                continue  # no parser for ASS
            candidates.append(SubtitleChoice(
                url=s.url, via="file", name=None, language=None,
                last_seen=s.last_seen,
            ))
    subtitle = max(candidates, key=lambda c: c.last_seen) if candidates else None

    return Resolution(
        download_url=download_url,
        master_url=master.url if master else None,
        audio=audio,
        subtitle=subtitle,
        duration=duration,
        playlists=playlists,
    )


def subtitles_cover(segments: list[dict[str, Any]] | None, duration: float | None) -> bool:
    """Whether subtitle ``segments`` are complete enough to stand in for
    ASR. Unknown duration → only the cue-count check applies."""
    if not segments or len(segments) < SUBTITLE_MIN_CUES:
        return False
    if not duration or duration <= 0:
        return True
    last_end = max(float(s["start"]) + float(s.get("duration") or 0.0) for s in segments)
    return last_end >= SUBTITLE_MIN_COVERAGE * duration


# ---------------------------------------------------------------------------
# WebVTT segment concatenation (pure)
# ---------------------------------------------------------------------------


def _timestamp_map_offset(text: str) -> float | None:
    """Seconds to add to this segment's cue times per its X-TIMESTAMP-MAP
    (``MPEGTS/90000 - LOCAL``), or None without a map."""
    m = _TIMESTAMP_MAP_RE.search(text)
    if not m:
        return None
    parts = dict(
        p.split(":", 1) for p in m.group(1).strip().split(",") if ":" in p
    )
    try:
        mpegts = int(parts.get("MPEGTS", "0").strip())
        local = youtube._vtt_timestamp_to_seconds(parts.get("LOCAL", "00:00.000"))
    except ValueError:
        return None
    return mpegts / 90000.0 - local


def clean_cue_text(text: str) -> str:
    """Strip what the shared VTT parser leaves behind: HTML entities and
    ASS-style ``{\\an8}`` overrides common in fan subtitles."""
    return " ".join(_ASS_OVERRIDE_RE.sub("", html.unescape(text)).split())


def concat_vtt_segments(
    bodies: list[str], segment_starts: list[float]
) -> list[dict[str, Any]]:
    """Merge the WebVTT segments of an HLS subtitle playlist into one cue
    list on the media timeline.

    Offsets: each segment's X-TIMESTAMP-MAP gives ``MPEGTS/90000 - LOCAL``;
    those are normalised against the smallest one, on the assumption that
    the first segment lines up with the start of the media (players align
    MPEGTS with the video's first PTS, which we don't know). The common
    case — the same map in every segment and absolute cue times — thus
    leaves times untouched. A segment without a map whose cues all lie
    well before its playlist position is treated as segment-local and
    shifted by that position (``segment_starts``, from EXTINF sums).
    Cues repeated across a segment boundary are emitted once.
    """
    offsets = [_timestamp_map_offset(b) for b in bodies]
    known = [o for o in offsets if o is not None]
    base = min(known) if known else 0.0
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for body, offset, seg_start in zip(bodies, offsets, segment_starts, strict=False):
        cues = youtube.parse_subtitle_vtt_text(body)
        if not cues:
            continue
        shift = offset - base if offset is not None else 0.0
        if offset is None and seg_start > 1.0 and all(
            c["start"] < seg_start - 1.0 for c in cues
        ):
            shift = seg_start
        for c in cues:
            text = clean_cue_text(c["text"])
            if not text:
                continue
            start = max(0.0, c["start"] + shift)
            key = (round(start * 10), text)
            if key in seen:
                continue
            seen.add(key)
            out.append({"start": start, "duration": c["duration"], "text": text})
    out.sort(key=lambda c: c["start"])
    return out


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _client(headers: dict[str, str] | None, cookies: list[Any]) -> httpx.AsyncClient:
    jar = httpx.Cookies()
    for c in cookies or []:
        try:
            jar.set(c.name, c.value, domain=c.domain or "", path=c.path or "/")
        except Exception:  # noqa: BLE001 — a malformed cookie is just skipped
            continue
    hdrs = {"User-Agent": "Mozilla/5.0"}
    hdrs.update(headers or {})
    return httpx.AsyncClient(
        headers=hdrs,
        cookies=jar,
        follow_redirects=True,
        timeout=httpx.Timeout(FETCH_TIMEOUT_SECONDS, connect=10.0),
    )


async def _get_text(client: httpx.AsyncClient, url: str, cap: int = MAX_PLAYLIST_BYTES) -> str | None:
    try:
        async with client.stream("GET", url) as r:
            if r.status_code < 200 or r.status_code >= 300:
                log.info("stream_resolve: %s -> HTTP %d", url, r.status_code)
                return None
            buf = bytearray()
            async for chunk in r.aiter_bytes():
                buf.extend(chunk)
                if len(buf) > cap:
                    log.info("stream_resolve: %s over %d bytes, skipped", url, cap)
                    return None
            return buf.decode(r.encoding or "utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 — best-effort
        log.info("stream_resolve: fetch failed for %s: %s", url, exc)
        return None


async def _fetch_playlists(
    client: httpx.AsyncClient, urls: list[str]
) -> dict[str, Playlist]:
    sem = asyncio.Semaphore(FETCH_CONCURRENCY)

    async def one(u: str) -> tuple[str, Playlist | None]:
        async with sem:
            text = await _get_text(client, u)
        return u, parse_playlist(u, text) if text else None

    results = await asyncio.gather(*(one(u) for u in urls))
    return {u: p for u, p in results if p is not None}


async def resolve(
    streams: list[Any],
    *,
    default_url: str,
    headers: dict[str, str] | None,
    cookies: list[Any],
) -> Resolution:
    """Fetch the sniffed HLS playlists (most recent first, capped) and
    :func:`select`. Then fetch the chosen download playlist too if it wasn't
    sniffed, so its duration is known for the subtitle coverage check."""
    hls = sorted(
        (s for s in streams if s.kind == "hls"),
        key=lambda s: s.last_seen, reverse=True,
    )[:MAX_PLAYLISTS_FETCHED]
    async with _client(headers, cookies) as client:
        playlists = await _fetch_playlists(client, [s.url for s in hls])
        res = select(streams, playlists, default_url=default_url)
        if res.download_url not in playlists and res.master_url:
            extra = await _fetch_playlists(client, [res.download_url])
            if extra:
                playlists.update(extra)
                res = select(streams, playlists, default_url=default_url)
        # Duration still unknown (master downloaded directly, no variant
        # sniffed): one variant playlist's EXTINF sum gives it.
        master = playlists.get(res.master_url or "")
        if res.duration is None and master is not None and master.variant_uris:
            extra = await _fetch_playlists(client, master.variant_uris[:1])
            if extra:
                playlists.update(extra)
                res = select(streams, playlists, default_url=default_url)
    return res


async def fetch_subtitle_segments(
    choice: SubtitleChoice,
    resolution: Resolution,
    *,
    headers: dict[str, str] | None,
    cookies: list[Any],
) -> list[dict[str, Any]] | None:
    """Download and parse the chosen subtitle into ``{start, duration,
    text}`` segments. ``None`` on any failure."""
    async with _client(headers, cookies) as client:
        if choice.via == "file":
            text = await _get_text(client, choice.url, MAX_SUBTITLE_TOTAL_BYTES)
            if not text:
                return None
            segs = []
            for c in youtube.parse_subtitle_vtt_text(text):
                t = clean_cue_text(c["text"])
                if t:
                    segs.append({**c, "text": t})
            return segs or None

        pl = resolution.playlists.get(choice.url)
        if pl is None:
            fetched = await _fetch_playlists(client, [choice.url])
            pl = fetched.get(choice.url)
        if pl is None or pl.is_master or not pl.segment_uris:
            return None
        uris = pl.segment_uris[:MAX_SUBTITLE_SEGMENTS]
        starts: list[float] = []
        acc = 0.0
        for d in pl.segment_durations[: len(uris)]:
            starts.append(acc)
            acc += d
        sem = asyncio.Semaphore(FETCH_CONCURRENCY)

        async def one(u: str) -> str:
            async with sem:
                return await _get_text(client, u) or ""

        bodies = await asyncio.gather(*(one(u) for u in uris))
    if sum(len(b) for b in bodies) > MAX_SUBTITLE_TOTAL_BYTES:
        return None
    return concat_vtt_segments(list(bodies), starts) or None


__all__ = [
    "Playlist",
    "Rendition",
    "Resolution",
    "SubtitleChoice",
    "concat_vtt_segments",
    "fetch_subtitle_segments",
    "parse_playlist",
    "resolve",
    "select",
    "subtitles_cover",
]
