"""Fetch a few JPEG frames from around a timestamp in a video — for a
multimodal LLM to look at, without ever downloading the whole video.

Public surface:
    fetch_frames(*, job_id, url, timestamp_seconds, num_frames=..., cookies=None,
                  allow_full_download=False, full_download_max_bytes=...,
                  reuse_existing=False)
        -> list[Path]
        Downloads a short section of ``url`` around ``timestamp_seconds``
        (yt-dlp ``--download-sections``, never the full video — see below),
        then extracts a handful of JPEG frames from it with ffmpeg. Returns
        the frame paths in chronological order.

        ``reuse_existing=True`` makes a second call for the SAME
        ``(job_id, timestamp_seconds)`` return whatever is already on disk
        for that moment WITHOUT touching the network at all — for a caller
        that only wants "the frames for this moment, fetching once is
        enough" (e.g. a user re-opening the same "look" affordance).
        Default ``False`` keeps every other caller's existing behaviour: a
        second call always clears and re-downloads/re-extracts (see the
        stale-leftover comment inside the function for why that's still
        the right default when a caller might ask for a different
        ``num_frames`` between calls).

    delete_job_frames(job_id) -> bool
        RETENTION HOOK. See its docstring — wired in from both
        ``storage/repo.py`` delete paths (explicit delete + retention sweep).

Not wired into the pipeline itself — only the QA LOOK step
(``llm/qa.py``'s ``_inspect_moment``) and the on-demand "look" affordance
(``api/jobs.py``'s ``POST /jobs/{id}/frames``) call ``fetch_frames``. This
module only produces JPEG files on disk and hands back their paths; those
callers are responsible for what happens to them next (feeding them to the
multimodal LLM as ``image_url`` content, or serving them straight to the
user via ``GET /jobs/{id}/frames/{rel_path}``).

Why sections, not the whole file
---------------------------------
Measured on a 10:35 YouTube video: a 3-second 480p section downloaded via
``yt-dlp --download-sections "*MM:SS-MM:SS"`` cost 152 KB / 4.1 s wall,
against 71 MB for the full 720p video. The existing YouTube worker
(``workers/youtube.py``) already fetches audio-only for the transcript path
and that must keep working unchanged — this module is a separate, additive
capability, not a modification of that path.

Both yt-dlp's ffmpeg dependency and its JS-challenge-solver runtime are
resolved the same way every other yt-dlp caller in this daemon resolves
them (``workers.ffmpeg.resolve_ffmpeg_dir`` / ``workers.jsruntime.
deno_runtime_opt``) rather than assumed to be on PATH — under launchd/
systemd the daemon's PATH is thin, and without a resolved deno runtime
yt-dlp cannot solve YouTube's "n" challenge and downloads fail with an
HTTP 403 (hit exactly that while proving out this approach). Like
``workers/youtube.py``, yt-dlp is driven through its Python ``YoutubeDL``
API class, not the CLI — one less subprocess + reimport per call, and
consistent with the only other yt-dlp caller in this codebase.

Window slack around the target timestamp
-----------------------------------------
``yt-dlp --download-sections`` cuts on the nearest keyframe unless you pass
``--force-keyframes-at-cuts``, which is slow because it re-encodes. We don't
use it, so the exact requested second is not guaranteed to land inside the
downloaded span. Instead we request a WINDOW around the timestamp
(``WINDOW_BEFORE_SECONDS`` before, ``WINDOW_AFTER_SECONDS`` after) and
sample several frames from the whole window, so the moment the caller cares
about is very likely covered by at least one frame even if the cut's
keyframe boundary lands a couple of seconds off target.

Frame extraction shape (ffmpeg)
--------------------------------
~1 fps, scaled to ``FRAME_WIDTH_PX`` (768) wide, JPEG at ``FRAME_JPEG_
QUALITY``. 768px is the number actually sent on to the model: smaller
starts losing small on-screen text (the motivating case is reading product
labels / burned-in captions); larger just spends more image tokens for no
legibility gain at typical multimodal vision-encoder input sizes. 1 fps
keeps a few-second window at "a handful of candidate frames" rather than
dozens.

Frame count and the two caps
------------------------------
``DEFAULT_NUM_FRAMES`` / ``MAX_FRAMES_PER_CALL`` bound how many frames a
single ``fetch_frames`` call can produce (a caller asking about one moment
doesn't need dozens of near-duplicate frames of the same few seconds).
``MAX_FRAMES_PER_JOB`` bounds the total across EVERY call made for the same
``job_id`` over its lifetime — a QA session might ask about several
different moments in the same video, and every frame sent to the model
costs image tokens, so the budget has to be shared across the whole job,
not reset per call. The running total is derived by counting JPEGs already
on disk under the job's frame directory (``_existing_frame_count``) rather
than tracked in memory or the DB, so it survives daemon restarts and needs
no schema change. When the job is already at or over the cap,
``fetch_frames`` logs a warning and returns ``[]`` WITHOUT downloading
anything — a caller that keeps asking after the budget is exhausted just
gets no new frames, not an error.

Storage and retention
-----------------------
Frames for a job live under ``<data_dir>/frames/<job_id>/t<second>/``,
alongside how ``workers/youtube.download_audio`` places a job's audio under
``<data_dir>/audio`` (see ``runner._audio_dir``) and ``Job.audio_path``
tracks it for cleanup. Frames have no DB column — like ``media_url`` for
MEDIA jobs (see ``.claude/workers.md``, "Media and PDF jobs are ephemeral on
restart"), there is nothing here that's safe or worth persisting across a
restart, so this module doesn't try. That means the ``retention.py`` sweep
(keyed off ``Job.created_at`` / row deletion in the DB) has no way to find
these directories on its own — see ``delete_job_frames`` below for the
hook ``storage/repo.py`` calls to close that gap.

The section/full clip downloaded on the way to the frames is never
persisted — it lives in a ``tempfile.TemporaryDirectory`` under
``<data_dir>/frames_scratch`` for the duration of one ``fetch_frames`` call
and is removed (directory and all) before the call returns, success or
failure. Only ``frames/`` (the extracted JPEGs) needs the retention hook;
``frames_scratch/`` self-cleans every call.

Section download retries
--------------------------
A section download can fail with a plain transient error — measured
directly rather than assumed: a live ``ffmpeg exited with code 8`` on one
moment of a YouTube video turned out to be flaky, not sticky. A follow-up
probe called ``_download_section_sync`` directly against the SAME video,
three attempts each on the range that had just failed live and a range
that had just succeeded live, and saw the failure MOVE between ranges
(roughly one failure in six attempts) rather than stay with one — a
CDN-side hiccup mid-download, not a property of any particular section.
``_download_video_section`` therefore retries the section download up to
``SECTION_DOWNLOAD_MAX_ATTEMPTS`` times (constant's docstring has the
reasoning for the count/delay) before giving up — recovering most of
these for free, well before the caller has to pay for a full-video
download. Only the section path retries; the opt-in full-download
fallback below does not (see its own note).

Fallback for sites that refuse ranged/section downloads
---------------------------------------------------------
Not every yt-dlp extractor supports ``--download-sections`` — some sites
only ever hand back the whole file. ``allow_full_download=True`` is an
explicit per-call opt-in (default False) that lets ``fetch_frames`` retry
with a full download when the section attempt raises. It is never
automatic by default specifically so nobody silently pays the "download an
entire video to answer one question about one moment" cost the section
approach exists to avoid. Even opted in, ``full_download_max_bytes`` (25 MB
default — well under the 71 MB measured full-720p example, generous enough
for a low-res full-length capture) is enforced both as a pre-download
best-effort hint to yt-dlp (``max_filesize`` — not honoured by every
extractor, since it depends on the site reporting a filesize up front) and
as a hard post-download check that deletes the file and raises
``FrameExtractionError`` if it's still over budget.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from src.api.schemas import Cookie
from src.config import get_config
from src.storage.cookies import write_netscape_cookie_file
from src.workers.errors import FrameExtractionError
from src.workers.ffmpeg import ensure_ffmpeg_on_path, resolve_ffmpeg_dir
from src.workers.jsruntime import deno_runtime_opt

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables — see module docstring for the reasoning behind each constant.
# ---------------------------------------------------------------------------

# Slack around the requested timestamp. Asymmetric on purpose: yt-dlp's
# section cut starts at or before the nearest preceding keyframe and ends at
# or after the requested end, so a little more slack after the target costs
# nothing (still ~1 frame/s) and protects against the target landing late.
WINDOW_BEFORE_SECONDS = 2.0
WINDOW_AFTER_SECONDS = 5.0

# ffmpeg frame extraction.
FRAME_FPS = 1
FRAME_WIDTH_PX = 768
# ffmpeg mjpeg -q:v scale: 2 (near-top quality) rather than the more common
# "good enough" 3-5, because the motivating use case is reading small
# on-screen text (product labels, captions) — see module docstring.
FRAME_JPEG_QUALITY = 2

# How many frames one fetch_frames() call may request/produce.
DEFAULT_NUM_FRAMES = 5
MAX_FRAMES_PER_CALL = 8
# Hard cap across the whole job's lifetime — every fetch_frames call for the
# same job_id shares this budget. See "Frame count and the two caps".
MAX_FRAMES_PER_JOB = 24

# Video height cap for the section download. 480p matches the measured proof
# (3 s @ 480p = 152 KB) and is plenty when we only need to SEE something —
# a gesture, a demonstrated action, who is on screen.
SECTION_MAX_HEIGHT_PX = 480
# When the point is to READ something off the picture — a product label, an
# article number, on-screen UI text — 480p is the binding constraint, not our
# 768px extraction width: a 480p source yields 768x432 frames whose small text
# is already at the edge of legibility. 720p roughly doubles the section cost
# (measured ~112 KB/s of video vs ~50 KB/s at 480p, so ~380 KB vs ~170 KB for a
# 7 s window) which is negligible against the 71 MB a full download would take.
# Callers pick per situation via ``fetch_frames(max_height_px=...)``; this
# module deliberately knows nothing about WHY (see workers/deixis.py categories
# — mapping those to a height is the caller's job, not ours).
SECTION_MAX_HEIGHT_READABLE_PX = 720
FULL_DOWNLOAD_MAX_HEIGHT_PX = 480
# Size guard for the opt-in full-download fallback. Well under the 71 MB
# measured full-720p example; see module docstring.
FULL_DOWNLOAD_MAX_BYTES = 25 * 1024 * 1024

# Bounded retry for the SECTION download only (never the full-download
# fallback — that path is a big, rare, deliberately-opt-in escape hatch;
# retrying it automatically would multiply an already-expensive cost).
# 3 total attempts (2 retries) — see module docstring "Section download
# retries" for the measurement behind this: a transient CDN-side failure
# was observed at roughly 1-in-6 attempts, moving between ranges, so two
# retries clears the vast majority of these without piling on unbounded
# wall-clock time for a truly broken URL.
SECTION_DOWNLOAD_MAX_ATTEMPTS = 3
# Short pause between attempts — long enough to let a transient hiccup
# clear, short enough that the worst case (all attempts fail) doesn't make
# the user wait much longer than a couple of extra download attempts.
SECTION_DOWNLOAD_RETRY_DELAY_SECONDS = 1.5


# ---------------------------------------------------------------------------
# Pure helpers — window math, section-syntax formatting
# ---------------------------------------------------------------------------


def _compute_window(
    timestamp_seconds: float,
    *,
    before: float = WINDOW_BEFORE_SECONDS,
    after: float = WINDOW_AFTER_SECONDS,
) -> tuple[float, float]:
    """(start, end) in seconds around ``timestamp_seconds``, never negative.

    Clamped at 0 rather than shifting the window forward — a timestamp near
    the very start of the video just gets a shorter pre-roll, not a window
    that no longer contains the requested moment.
    """
    safe_timestamp = max(0.0, timestamp_seconds)
    start = max(0.0, safe_timestamp - before)
    end = max(start, safe_timestamp + after)
    return start, end


def _fmt_hms(seconds: float) -> str:
    """``MM:SS``, or ``H:MM:SS`` past the hour mark. Never negative."""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_section_arg(start: float, end: float) -> str:
    """``*START-END`` — yt-dlp's ``--download-sections`` syntax, for log and
    error messages only.

    The daemon drives yt-dlp through its Python API (``YoutubeDL``), like
    every other caller in this codebase, not the CLI — so the actual range
    handed to yt-dlp is the numeric ``(start, end)`` pair via
    ``yt_dlp.utils.download_range_func`` (see ``_download_section_sync``),
    not this string. This is what the equivalent
    ``yt-dlp --download-sections "..."`` invocation would say.
    """
    return f"*{_fmt_hms(start)}-{_fmt_hms(end)}"


def _effective_num_frames(num_frames: int) -> int:
    """Clamp a caller-requested frame count into ``[1, MAX_FRAMES_PER_CALL]``."""
    return max(1, min(int(num_frames), MAX_FRAMES_PER_CALL))


# ---------------------------------------------------------------------------
# yt-dlp option builders (mirrors workers/youtube.py's _ffmpeg_opt/_jsruntime_opt)
# ---------------------------------------------------------------------------


def _ffmpeg_opt() -> dict[str, str]:
    """``{"ffmpeg_location": dir}`` when ffmpeg is resolvable, else ``{}``."""
    location = resolve_ffmpeg_dir()
    return {"ffmpeg_location": location} if location else {}


def _jsruntime_opt() -> dict[str, Any]:
    """``{"js_runtimes": {"deno": {"path": ...}}}`` when deno is resolvable."""
    return deno_runtime_opt()


def _new_ytdl(opts: dict[str, Any]) -> Any:
    """Construct a ``yt_dlp.YoutubeDL``. Imported lazily so module-load
    doesn't pay yt-dlp's import cost, and isolated behind this one-line seam
    so tests can monkeypatch it instead of touching yt_dlp itself."""
    from yt_dlp import YoutubeDL

    return YoutubeDL(opts)


# ---------------------------------------------------------------------------
# Storage locations
# ---------------------------------------------------------------------------


def _frames_root_dir() -> Path:
    """Persisted per-job frame JPEGs live under here. See ``delete_job_frames``
    for the retention hook this directory needs wired up externally."""
    p = Path(get_config().storage.data_dir) / "frames"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _scratch_root_dir() -> Path:
    """Transient downloaded section/full clips — always self-cleaned within
    one ``fetch_frames`` call via ``tempfile.TemporaryDirectory``, so this
    directory needs no retention hook of its own."""
    p = Path(get_config().storage.data_dir) / "frames_scratch"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _job_frames_dir(job_id: str) -> Path:
    p = _frames_root_dir() / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _existing_frame_count(job_id: str) -> int:
    """How many frames already exist for this job, across all prior calls —
    the running total the per-job cap is checked against."""
    job_dir = _frames_root_dir() / job_id
    if not job_dir.is_dir():
        return 0
    return sum(1 for _ in job_dir.glob("**/*.jpg"))


# ---------------------------------------------------------------------------
# yt-dlp download (section, and the opt-in full-video fallback)
# ---------------------------------------------------------------------------


def _locate_downloaded_file(ydl: Any, info: dict[str, Any]) -> Path:
    """Resolve the actual file yt-dlp wrote, mirroring
    ``workers.youtube._download_audio_sync``'s fallback chain."""
    path: str | None = None
    requested = info.get("requested_downloads") or []
    if requested:
        entry = requested[0] or {}
        path = entry.get("filepath") or entry.get("_filename")
    if not path:
        path = ydl.prepare_filename(info)
    result = Path(path)
    if not result.exists():
        raise RuntimeError(f"yt-dlp reported success but file is missing: {result}")
    return result


def _download_section_sync(
    *,
    url: str,
    start: float,
    end: float,
    cookies: list[Cookie],
    dir: Path,
    max_height: int,
) -> Path:
    """Download ONLY the ``[start, end]`` span of ``url`` (video, no audio
    track needed) via yt-dlp's ``--download-sections`` equivalent.

    Raises whatever yt-dlp raises (typically ``yt_dlp.utils.DownloadError``)
    on failure — the caller decides whether to fall back to a full download.
    """
    from yt_dlp.utils import download_range_func

    # yt-dlp's download_ranges precheck resolves ffmpeg from PATH only and
    # aborts before it ever reads ``ffmpeg_location`` — see
    # ffmpeg.ensure_ffmpeg_on_path for the measurement. Under launchd the
    # daemon's PATH is thin, so without this every section download fails with
    # "ffmpeg is not installed" on a machine where ffmpeg is right there.
    ensure_ffmpeg_on_path()

    dir.mkdir(parents=True, exist_ok=True)
    cookie_path = write_netscape_cookie_file(cookies, dir) if cookies else None

    # Video-only, no postprocessor: we only need pixels for ffmpeg to sample,
    # never an audio track, so there's nothing to mux/extract afterwards.
    fmt_filter = f"bestvideo[height<={max_height}]/worst[height<={max_height}]/worst"
    output_template = str(dir / "%(id)s.section.%(ext)s")

    ydl_opts: dict[str, Any] = {
        "format": fmt_filter,
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "download_ranges": download_range_func([], [(start, end)]),
        # Explicit even though it's yt-dlp's default: keyframe-forced cuts
        # re-encode and are slow. We compensate for the imprecision with
        # window slack instead — see module docstring.
        "force_keyframes_at_cuts": False,
        "remote_components": ["ejs:github"],
        **_ffmpeg_opt(),
        **_jsruntime_opt(),
    }
    if cookie_path is not None:
        ydl_opts["cookiefile"] = str(cookie_path)

    log.info(
        "frames: downloading section %s of %s (height<=%d)",
        _format_section_arg(start, end), url, max_height,
    )
    try:
        with _new_ytdl(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise RuntimeError(f"yt-dlp returned no info for {url}")
            return _locate_downloaded_file(ydl, info)
    finally:
        if cookie_path is not None:
            with contextlib.suppress(OSError):
                cookie_path.unlink(missing_ok=True)


def _download_full_sync(
    *,
    url: str,
    cookies: list[Cookie],
    dir: Path,
    max_height: int,
    max_filesize_bytes: int,
) -> Path:
    """Opt-in fallback: download the whole video (no ``download_ranges``) for
    sites whose extractor doesn't support sectioned downloads.

    ``max_filesize`` is passed to yt-dlp as a best-effort pre-download guard
    (only effective when the site reports a filesize before downloading);
    the caller enforces the same limit again on the actual file afterwards,
    since not every extractor honours it.
    """
    dir.mkdir(parents=True, exist_ok=True)
    cookie_path = write_netscape_cookie_file(cookies, dir) if cookies else None

    fmt_filter = f"worst[height<={max_height}]/worst"
    output_template = str(dir / "%(id)s.full.%(ext)s")

    ydl_opts: dict[str, Any] = {
        "format": fmt_filter,
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "max_filesize": max_filesize_bytes,
        "remote_components": ["ejs:github"],
        **_ffmpeg_opt(),
        **_jsruntime_opt(),
    }
    if cookie_path is not None:
        ydl_opts["cookiefile"] = str(cookie_path)

    log.info("frames: full-download fallback for %s (height<=%d)", url, max_height)
    try:
        with _new_ytdl(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise RuntimeError(f"yt-dlp returned no info for {url}")
            return _locate_downloaded_file(ydl, info)
    finally:
        if cookie_path is not None:
            with contextlib.suppress(OSError):
                cookie_path.unlink(missing_ok=True)


async def _download_video_section(
    *, url: str, start: float, end: float, cookies: list[Cookie], dir: Path, max_height: int,
) -> Path:
    """Download the ``[start, end]`` section, retrying up to
    ``SECTION_DOWNLOAD_MAX_ATTEMPTS`` times on failure — see module
    docstring "Section download retries" for the measurement behind why
    this is worth doing at all.

    Each attempt gets its OWN fresh subdirectory under ``dir`` rather than
    reusing one across attempts. Checked what a failed attempt leaves
    behind first: yt-dlp's ffmpeg-backed external downloader
    (``yt_dlp.downloader.external.ExternalFD.real_download``) downloads to
    a temp-named file and only renames it to the final filename on a ZERO
    ffmpeg exit code — on failure it reports the error and returns without
    renaming OR deleting, so the temp file can survive under the original
    name. yt-dlp re-invokes ffmpeg with ``-y`` on a retry, so in practice a
    same-directory retry would just overwrite that leftover — but a fresh
    subdirectory per attempt removes the question entirely instead of
    depending on that overwrite behaviour (which isn't necessarily true for
    every extractor/downloader combination), and guarantees
    ``_locate_downloaded_file`` can never resolve a stale partial file from
    an earlier failed attempt as if it were this attempt's result.

    Raises the LAST attempt's exception once every attempt is exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(1, SECTION_DOWNLOAD_MAX_ATTEMPTS + 1):
        attempt_dir = dir / f"attempt{attempt}"
        try:
            return await asyncio.to_thread(
                _download_section_sync,
                url=url, start=start, end=end, cookies=cookies,
                dir=attempt_dir, max_height=max_height,
            )
        except Exception as exc:
            last_exc = exc
            if attempt < SECTION_DOWNLOAD_MAX_ATTEMPTS:
                log.warning(
                    "frames: section download attempt %d/%d failed for %s %s: "
                    "%s — retrying",
                    attempt, SECTION_DOWNLOAD_MAX_ATTEMPTS, url,
                    _format_section_arg(start, end), exc,
                )
                await asyncio.sleep(SECTION_DOWNLOAD_RETRY_DELAY_SECONDS)
    assert last_exc is not None  # loop always sets it before exhausting attempts
    raise last_exc


async def _download_full_video(
    *, url: str, cookies: list[Cookie], dir: Path, max_height: int, max_filesize_bytes: int,
) -> Path:
    return await asyncio.to_thread(
        _download_full_sync,
        url=url, cookies=cookies, dir=dir,
        max_height=max_height, max_filesize_bytes=max_filesize_bytes,
    )


# ---------------------------------------------------------------------------
# ffmpeg frame extraction
# ---------------------------------------------------------------------------


def _ffmpeg_bin() -> str | None:
    directory = resolve_ffmpeg_dir()
    if not directory:
        return None
    candidate = Path(directory) / "ffmpeg"
    if candidate.is_file():
        return str(candidate)
    candidate_exe = Path(directory) / "ffmpeg.exe"
    return str(candidate_exe) if candidate_exe.is_file() else None


def _build_ffmpeg_frame_cmd(
    *,
    ffmpeg_bin: str,
    video_path: Path,
    out_pattern: Path,
    num_frames: int,
    seek_start: float | None,
    seek_duration: float | None,
) -> list[str]:
    """Build the exact ffmpeg argv for frame extraction.

    ``-ss`` before ``-i`` (fast, demuxer-level seek) is only added when the
    source file covers more than the window — the section-download path
    already downloaded exactly the window, so no seek is needed there; the
    full-download fallback downloaded the whole video, so it seeks + limits
    duration to the same window the section path would have covered.
    """
    cmd = [ffmpeg_bin, "-y", "-loglevel", "error"]
    if seek_start is not None:
        cmd += ["-ss", f"{seek_start:.3f}"]
    cmd += ["-i", str(video_path)]
    if seek_duration is not None:
        cmd += ["-t", f"{seek_duration:.3f}"]
    cmd += [
        "-an",
        "-vf", f"fps={FRAME_FPS},scale={FRAME_WIDTH_PX}:-2",
        "-vframes", str(num_frames),
        "-q:v", str(FRAME_JPEG_QUALITY),
        str(out_pattern),
    ]
    return cmd


def _extract_frames_sync(
    *,
    video_path: Path,
    out_dir: Path,
    num_frames: int,
    seek_start: float | None,
    seek_duration: float | None,
) -> list[Path]:
    ffmpeg_bin = _ffmpeg_bin()
    if not ffmpeg_bin:
        raise FrameExtractionError("ffmpeg is not available for frame extraction")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = out_dir / "frame_%02d.jpg"
    cmd = _build_ffmpeg_frame_cmd(
        ffmpeg_bin=ffmpeg_bin,
        video_path=video_path,
        out_pattern=out_pattern,
        num_frames=num_frames,
        seek_start=seek_start,
        seek_duration=seek_duration,
    )
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        raise FrameExtractionError(f"ffmpeg frame extraction failed: {exc}") from exc

    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        raise FrameExtractionError(f"ffmpeg produced no frames in {out_dir}")
    return frames


async def _extract_frames(
    *,
    video_path: Path,
    out_dir: Path,
    num_frames: int,
    seek_start: float | None,
    seek_duration: float | None,
) -> list[Path]:
    return await asyncio.to_thread(
        _extract_frames_sync,
        video_path=video_path, out_dir=out_dir, num_frames=num_frames,
        seek_start=seek_start, seek_duration=seek_duration,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_frames(
    *,
    job_id: str,
    url: str,
    timestamp_seconds: float,
    num_frames: int = DEFAULT_NUM_FRAMES,
    cookies: list[Cookie] | None = None,
    max_height_px: int = SECTION_MAX_HEIGHT_PX,
    allow_full_download: bool = False,
    full_download_max_bytes: int = FULL_DOWNLOAD_MAX_BYTES,
    reuse_existing: bool = False,
) -> list[Path]:
    """Fetch a handful of JPEG frames from around ``timestamp_seconds`` in
    ``url``, for a multimodal LLM (or the user, directly) to inspect. Never
    downloads the whole video unless ``allow_full_download=True`` is passed
    AND the section download fails — see the module docstring for the full
    reasoning.

    Returns frame paths in chronological order, or ``[]`` if the job's
    per-job frame budget (``MAX_FRAMES_PER_JOB``) is already spent — that is
    not an error, just nothing more to give this job.

    ``max_height_px`` caps the source video height for the section download.
    Default ``SECTION_MAX_HEIGHT_PX`` (480p) is right when the caller only
    needs to SEE what is happening; pass ``SECTION_MAX_HEIGHT_READABLE_PX``
    (720p) when the caller intends to READ text off the frame, such as a
    product label. Note the opt-in full-download fallback stays at
    ``FULL_DOWNLOAD_MAX_HEIGHT_PX`` regardless: there the whole file is
    fetched, so the size guard matters more than legibility.

    ``reuse_existing=True`` short-circuits the whole download/extract path
    when this exact ``(job_id, timestamp_seconds)`` already has frames on
    disk from an earlier call — returns them as-is, no network, no budget
    check (they were already paid for). Default ``False`` preserves every
    existing caller's behaviour (always clear + re-fetch fresh).

    Raises ``FrameExtractionError`` (``code="network_error"`` — see that
    class's docstring for why) if the download and/or ffmpeg step fails
    after every retry the section download gets (see module docstring
    "Section download retries").
    """
    cookies = cookies or []
    timestamp_seconds = max(0.0, float(timestamp_seconds))
    requested = _effective_num_frames(num_frames)

    out_dir = _job_frames_dir(job_id) / f"t{int(timestamp_seconds)}"

    if reuse_existing:
        cached = sorted(out_dir.glob("frame_*.jpg"))
        if cached:
            log.info(
                "frames: reusing %d cached frame(s) for job %s at %ds (no download)",
                len(cached), job_id, int(timestamp_seconds),
            )
            return cached

    # A second call for the same (job_id, timestamp) that does NOT ask to
    # reuse blows away any stale directory instead. ffmpeg's -y only
    # overwrites same-NAMED files, so if this call requests fewer frames
    # than a previous one did, stale higher-indexed leftovers would
    # otherwise survive and get returned alongside the fresh ones. Clear it
    # up front — before the budget check below — so (a) every call's result
    # reflects only itself, and (b) this timestamp's old frames don't count
    # against its own replacement's budget.
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)

    existing = _existing_frame_count(job_id)
    remaining = max(0, MAX_FRAMES_PER_JOB - existing)
    if remaining <= 0:
        log.warning(
            "frames: job %s already has %d frames (cap %d); skipping fetch for %s",
            job_id, existing, MAX_FRAMES_PER_JOB, url,
        )
        return []
    requested = min(requested, remaining)

    start, end = _compute_window(timestamp_seconds)

    with tempfile.TemporaryDirectory(
        prefix="tldr-frames-src-", dir=str(_scratch_root_dir())
    ) as scratch:
        scratch_dir = Path(scratch)
        seek_start: float | None
        seek_duration: float | None

        try:
            video_path = await _download_video_section(
                url=url, start=start, end=end, cookies=cookies,
                dir=scratch_dir, max_height=max_height_px,
            )
            seek_start, seek_duration = None, None
        except Exception as exc:
            if not allow_full_download:
                raise FrameExtractionError(
                    f"section download failed for {url} "
                    f"{_format_section_arg(start, end)} after "
                    f"{SECTION_DOWNLOAD_MAX_ATTEMPTS} attempts: {exc}"
                ) from exc
            log.warning(
                "frames: section download failed for %s (%s); "
                "falling back to full download (opt-in)", url, exc,
            )
            try:
                video_path = await _download_full_video(
                    url=url, cookies=cookies, dir=scratch_dir,
                    max_height=FULL_DOWNLOAD_MAX_HEIGHT_PX,
                    max_filesize_bytes=full_download_max_bytes,
                )
            except Exception as full_exc:
                raise FrameExtractionError(
                    f"full-download fallback failed for {url}: {full_exc}"
                ) from full_exc

            size = video_path.stat().st_size
            if size > full_download_max_bytes:
                raise FrameExtractionError(
                    f"full download of {url} was {size} bytes, over the "
                    f"{full_download_max_bytes}-byte guard; refusing to "
                    f"extract frames"
                ) from exc
            seek_start, seek_duration = start, max(0.0, end - start)

        frames = await _extract_frames(
            video_path=video_path,
            out_dir=out_dir,
            num_frames=requested,
            seek_start=seek_start,
            seek_duration=seek_duration,
        )

    return frames


def job_frames_dir_if_exists(job_id: str) -> Path | None:
    """The on-disk directory holding this job's frame JPEGs (see module
    docstring "Storage and retention"), or ``None`` if it doesn't exist —
    i.e. no frames were ever fetched for this job.

    Read-only: unlike ``_job_frames_dir`` this never creates the
    directory. A caller that only wants to know "does this job have
    frames worth copying" (``storage.bundle``'s export path) must not have
    the side effect of creating an empty one just by asking.
    """
    p = _frames_root_dir() / job_id
    return p if p.is_dir() else None


def ensure_job_frames_dir(job_id: str) -> Path:
    """Public alias for ``_job_frames_dir`` — creates (if needed) and
    returns the per-job frame directory.

    Exposed for ``storage.bundle``'s import path, which writes frame files
    under a NEWLY MINTED job id that was never real until this import (so
    nothing else could have created the directory yet).
    """
    return _job_frames_dir(job_id)


def resolve_frame_path(job_id: str, rel_path: str) -> Path | None:
    """Resolve a frame JPEG's on-disk path for ``GET /jobs/{id}/frames/...``,
    or ``None`` if it doesn't exist or ``rel_path`` tries to escape the
    job's own frame directory.

    Pure path validation — no fetching, no network, no mutation (beyond the
    harmless ``mkdir`` inside ``_frames_root_dir``/``_job_frames_dir``, the
    same idempotent side effect every other reader/writer in this module
    already has). The caller (``api/jobs.py``) is responsible for the 404
    HTTP response; this just decides yes/no.

    Hardening: resolves both the job's frame root and the candidate path
    with symlinks/``..`` collapsed, then requires the candidate to still be
    *inside* the root (``Path.relative_to`` raises otherwise) — a
    ``rel_path`` like ``"../../etc/passwd"`` or an absolute path can't
    escape the job's own directory no matter how it's spelled.
    """
    job_root = _job_frames_dir(job_id).resolve()
    candidate = (job_root / rel_path).resolve()
    try:
        candidate.relative_to(job_root)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate


def delete_job_frames(job_id: str) -> bool:
    """Remove every frame JPEG (and the per-job directory) for ``job_id``.

    RETENTION HOOK — wired in at both audio-cleanup call sites in
    ``storage/repo.py`` (via the local ``_delete_job_frames`` wrapper there)
    so frame directories share the audio file's lifecycle exactly:

      1. ``repo.delete_job(job_id)`` — called right alongside the existing
         ``_safe_unlink(Path(cached_audio))`` call for ``job.audio_path``.
      2. ``repo.delete_jobs_older_than(cutoff)`` — called for each id in the
         ``ids`` loop. This is what makes the periodic
         ``workers.retention.retention_worker`` sweep actually clean up
         frame directories instead of leaking them for
         ``storage.retention_days`` (default 365).

    Both call sites matter: (1) covers the user explicitly deleting a job
    from the Library; (2) covers jobs that simply age out.

    Returns True if a directory existed and was removed, False otherwise
    (including jobs that never had frames fetched — not an error).
    """
    job_dir = _frames_root_dir() / job_id
    if not job_dir.is_dir():
        return False
    shutil.rmtree(job_dir, ignore_errors=True)
    return True


__all__ = [
    "DEFAULT_NUM_FRAMES",
    "FULL_DOWNLOAD_MAX_BYTES",
    "MAX_FRAMES_PER_CALL",
    "MAX_FRAMES_PER_JOB",
    "delete_job_frames",
    "ensure_job_frames_dir",
    "fetch_frames",
    "job_frames_dir_if_exists",
    "resolve_frame_path",
]
