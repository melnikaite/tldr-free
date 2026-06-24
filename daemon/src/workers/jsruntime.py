"""Resolve a JavaScript runtime (Deno) for yt-dlp's YouTube challenge solver.

YouTube protects stream URLs with an "n"/signature challenge. yt-dlp solves it
by running an embedded-JS solver (the ``yt-dlp-ejs`` package, a pure-Python
dependency holding the solver *code*) inside a real JS *runtime*. The runtime
must be a native binary — deno/node/bun/quickjs; there is no pure-Python path
in yt-dlp's current ``jsc`` subsystem. ``deno`` is the only runtime yt-dlp
enables by default and the one its solver targets.

Without a runtime yt-dlp falls back to "JS-less" clients and some formats go
missing — fine for captioned videos (handled by youtube-transcript-api) but it
breaks the audio download that the Whisper path needs for caption-less videos.

Resolution order (mirrors workers.ffmpeg):
  1. A system ``deno`` on PATH / known dirs / the venv's scripts dir.
  2. A static ``deno`` binary downloaded once from GitHub releases for this
     platform and cached under ``<data_dir>/deno`` — cross-platform, no brew.

The resolved path is handed to yt-dlp via its ``js_runtimes`` option as
``deno:<path>``, so the daemon's (thin, under launchd/systemd) PATH is
irrelevant. Returns None when no runtime can be obtained — yt-dlp then degrades
to JS-less clients with its own clear warning.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import stat
import sys
import sysconfig
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

from src import paths

log = logging.getLogger(__name__)

# yt-dlp's DenoJsRuntime requires >= 2.3.0; "latest" stable is well past that.
# Override the channel with TLDR_DENO_VERSION=x.y.z to pin a specific release.
_DENO_LATEST_URL = "https://github.com/denoland/deno/releases/latest/download/{asset}"
_DENO_PINNED_URL = "https://github.com/denoland/deno/releases/download/v{ver}/{asset}"

# (sys.platform, machine) → deno release asset (a .zip holding one binary).
# deno has no stable Windows-arm64 build, so it's intentionally absent.
_DENO_ASSETS = {
    ("darwin", "arm64"): "deno-aarch64-apple-darwin.zip",
    ("darwin", "x86_64"): "deno-x86_64-apple-darwin.zip",
    ("linux", "x86_64"): "deno-x86_64-unknown-linux-gnu.zip",
    ("linux", "aarch64"): "deno-aarch64-unknown-linux-gnu.zip",
    ("win32", "amd64"): "deno-x86_64-pc-windows-msvc.zip",
}

_KNOWN_BIN_DIRS = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
    str(Path.home() / ".deno" / "bin"),  # deno's own installer default
)


def _exe_name() -> str:
    return "deno.exe" if os.name == "nt" else "deno"


def _machine() -> str:
    """Normalised CPU arch: arm64 / x86_64 / aarch64 / amd64 as deno labels."""
    m = platform.machine().lower()
    if m in ("arm64", "aarch64"):
        return "arm64" if sys.platform == "darwin" else "aarch64"
    if m in ("x86_64", "amd64", "x64"):
        return "amd64" if os.name == "nt" else "x86_64"
    return m


def _system_deno() -> str | None:
    """A usable system deno: PATH, then the venv scripts dir (yt-dlp checks it
    first too), then well-known install dirs."""
    found = shutil.which("deno")
    if found:
        return found
    candidates = [Path(sysconfig.get_path("scripts")) / _exe_name()]
    candidates += [Path(d) / _exe_name() for d in _KNOWN_BIN_DIRS]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


def _download_deno() -> str | None:
    """Fetch + cache a static deno binary for this platform. Returns its path."""
    key = (sys.platform, _machine())
    asset = _DENO_ASSETS.get(key)
    if not asset:
        log.warning("deno: no prebuilt binary for platform %s; skipping", key)
        return None

    cache_dir = paths.default_data_dir() / "deno"
    cache_dir.mkdir(parents=True, exist_ok=True)
    binary = cache_dir / _exe_name()
    if binary.is_file() and os.access(binary, os.X_OK):
        return str(binary)

    pinned = os.environ.get("TLDR_DENO_VERSION")
    url = (
        _DENO_PINNED_URL.format(ver=pinned, asset=asset)
        if pinned
        else _DENO_LATEST_URL.format(asset=asset)
    )
    zip_path = cache_dir / asset
    try:
        log.info("deno: downloading %s ...", url)
        urllib.request.urlretrieve(url, zip_path)  # noqa: S310 — fixed GitHub host
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(cache_dir)
        zip_path.unlink(missing_ok=True)
        if not binary.is_file():
            log.warning("deno: archive did not contain %s", _exe_name())
            return None
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        log.info("deno: installed static build at %s", binary)
        return str(binary)
    except Exception as exc:  # noqa: BLE001 — never fatal; caller degrades
        log.warning("deno: download failed (%s)", exc)
        zip_path.unlink(missing_ok=True)
        return None


@lru_cache(maxsize=1)
def resolve_deno() -> str | None:
    """Path to a deno binary for yt-dlp's ``js_runtimes``, or None.

    System binary wins (fast, offline); otherwise a static build is fetched and
    cached. Memoised — resolution (possibly a network download) runs at most
    once per process.
    """
    system = _system_deno()
    if system:
        log.info("deno: using system runtime at %s", system)
        return system
    downloaded = _download_deno()
    if downloaded:
        log.info("deno: using bundled runtime at %s", downloaded)
    else:
        log.warning(
            "deno: no JS runtime available; caption-less YouTube videos may "
            "fail to download audio for Whisper"
        )
    return downloaded


def deno_runtime_opt() -> dict[str, dict[str, dict[str, str]]]:
    """``{"js_runtimes": {"deno": {"path": <path>}}}`` when deno is resolvable.

    Spread into ydl_opts. The programmatic API wants the already-parsed dict
    form ``{runtime: {config}}`` (the CLI's ``RUNTIME[:PATH]`` string is parsed
    into this). Empty when deno can't be resolved — yt-dlp then keeps its
    default and looks for deno on PATH itself.
    """
    path = resolve_deno()
    return {"js_runtimes": {"deno": {"path": path}}} if path else {}


def prefetch_deno() -> None:
    """Warm the cache at startup so the first caption-less video doesn't block
    on a ~40 MB download. Best-effort: logs, never raises."""
    try:
        resolve_deno()
    except Exception as exc:  # noqa: BLE001 — startup must never fail here
        log.warning("deno: prefetch failed (%s)", exc)


__all__ = ["resolve_deno", "deno_runtime_opt", "prefetch_deno"]
