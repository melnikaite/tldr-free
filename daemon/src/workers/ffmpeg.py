"""Resolve an ffmpeg/ffprobe location for yt-dlp — cross-platform, no brew.

yt-dlp's audio postprocessing (``FFmpegExtractAudio``) needs both ``ffmpeg``
AND ``ffprobe``. In Docker these came from the image; on a native (uv) install
we can't assume the host has them — Windows has no Homebrew, and a fresh
Mac/Linux box may lack them too.

Resolution order:
  1. System ``ffmpeg`` + ``ffprobe`` on PATH — fastest, offline-friendly, and
     respects a user's own build. Used as-is when both are present.
  2. ``static-ffmpeg`` (a regular uv/pip dependency) — downloads a static build
     for this platform (win32 / darwin / darwin_arm64 / linux / linux_arm64)
     once and caches it. This is what makes "install via uv" self-contained:
     no brew/apt, works the same on Windows.

The download is cached under the daemon's data dir (not the venv) so it
survives a ``uv tool`` reinstall. ``resolve_ffmpeg_dir`` returns a directory
to hand to yt-dlp's ``ffmpeg_location`` (yt-dlp accepts a dir and finds both
binaries inside), or ``None`` if neither path works — the media job then fails
with yt-dlp's own clear "ffmpeg not found" error rather than crashing here.

Note: this does NOT cover deno, which yt-dlp shells out to for YouTube's "n"
challenge (``remote_components: ["ejs:github"]``). deno has no PyPI package and
is resolved separately (or not at all on a fresh host).
"""

from __future__ import annotations

import logging
import os
import shutil
from functools import lru_cache
from pathlib import Path

from src import paths

log = logging.getLogger(__name__)


# Well-known install dirs to probe when ffmpeg isn't on PATH. A daemon under
# launchd/systemd often runs with a minimal PATH that excludes these, so we'd
# otherwise download a static build even though a system ffmpeg exists.
_KNOWN_BIN_DIRS = (
    "/opt/homebrew/bin",  # macOS Apple Silicon (Homebrew)
    "/usr/local/bin",     # macOS Intel (Homebrew), common Linux
    "/usr/bin",           # Linux distro packages
    "/snap/bin",          # Ubuntu snap
)


def _both_in(directory: str) -> bool:
    exe = ".exe" if os.name == "nt" else ""
    d = Path(directory)
    return (d / f"ffmpeg{exe}").is_file() and (d / f"ffprobe{exe}").is_file()


def _system_ffmpeg_dir() -> str | None:
    """Dir holding both system ``ffmpeg`` and ``ffprobe``, else None.

    Checks PATH first (respects a user's chosen build), then a few well-known
    install dirs that the daemon's thin service PATH may omit.
    """
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return str(Path(ffmpeg).parent)
    for directory in _KNOWN_BIN_DIRS:
        if _both_in(directory):
            return directory
    return None


def _static_ffmpeg_dir() -> str | None:
    """Fetch (once) a static ffmpeg+ffprobe build for this platform.

    Cached under ``<data_dir>/ffmpeg/bin/<platform_key>`` so it outlives a venv
    reinstall. The ``bin/<platform_key>`` suffix is required: static-ffmpeg
    extracts the zip into the *parent* of the download dir and expects the
    binaries to land in a folder named exactly after the platform key.
    """
    try:
        from static_ffmpeg import run

        key = run.get_platform_key()  # raises on an unsupported platform
        cache_dir = paths.default_data_dir() / "ffmpeg" / "bin" / key
        cache_dir.mkdir(parents=True, exist_ok=True)
        ffmpeg, _ffprobe = run.get_or_fetch_platform_executables_else_raise(
            download_dir=str(cache_dir)
        )
        return str(Path(ffmpeg).parent)
    except Exception as exc:  # noqa: BLE001 — never fatal; caller falls back
        log.warning("ffmpeg: static-ffmpeg unavailable (%s)", exc)
        return None


@lru_cache(maxsize=1)
def resolve_ffmpeg_dir() -> str | None:
    """Directory containing ffmpeg+ffprobe for yt-dlp's ``ffmpeg_location``.

    System binaries win (fast, offline); otherwise a static build is fetched
    and cached. Result is memoised — the (possibly slow, network) resolution
    happens at most once per process. Returns None if neither is available.
    """
    system = _system_ffmpeg_dir()
    if system:
        log.info("ffmpeg: using system binaries at %s", system)
        return system

    static = _static_ffmpeg_dir()
    if static:
        log.info("ffmpeg: using bundled static build at %s", static)
        return static

    log.warning(
        "ffmpeg: no ffmpeg/ffprobe found and static-ffmpeg fetch failed; "
        "media/YouTube jobs needing audio extraction will fail"
    )
    return None


def prefetch_ffmpeg() -> None:
    """Warm the cache at startup so the first media job doesn't block on a
    ~80 MB download. Best-effort: logs and returns, never raises."""
    try:
        resolve_ffmpeg_dir()
    except Exception as exc:  # noqa: BLE001 — startup must never fail here
        log.warning("ffmpeg: prefetch failed (%s)", exc)


def ensure_ffmpeg_on_path() -> str | None:
    """Prepend the resolved ffmpeg directory to this process's ``PATH``.

    Passing yt-dlp ``ffmpeg_location`` is enough for its POSTPROCESSORS, which
    is why audio extraction has always worked. It is NOT enough for partial
    downloads: yt-dlp's precheck for ``download_ranges`` looks ffmpeg up on
    PATH only and aborts with "you have requested downloading the video
    partially, but ffmpeg is not installed" before the option is consulted.
    Under launchd/systemd the daemon's PATH is thin, so that precheck fails on
    a machine where ffmpeg is plainly present — measured directly: with a thin
    PATH and ``ffmpeg_location`` set the section download aborts, and with the
    same directory prepended to PATH it succeeds.

    Idempotent, and deliberately uses whatever ``resolve_ffmpeg_dir`` returns
    rather than a fixed location — on a machine without system ffmpeg that is
    the static build cached under the data dir, and hardcoding a system path
    would break exactly the setup the bundled fallback exists to support.

    Returns the directory that is now on PATH, or None when no ffmpeg could be
    resolved at all.
    """
    location = resolve_ffmpeg_dir()
    if not location:
        return None
    current = os.environ.get("PATH", "")
    if location in current.split(os.pathsep):
        return location
    os.environ["PATH"] = f"{location}{os.pathsep}{current}" if current else location
    log.info("ffmpeg: prepended %s to PATH for subprocess lookups", location)
    return location


__all__ = ["ensure_ffmpeg_on_path", "resolve_ffmpeg_dir", "prefetch_ffmpeg"]
