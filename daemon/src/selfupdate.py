"""Refresh YouTube-related libraries in the daemon's own environment.

Native-mode counterpart of daemon/docker-entrypoint.sh: Google plays
cat-and-mouse with yt-dlp / youtube-transcript-api, so restarting the daemon
should pull the latest fixes without a reinstall. Runs once at CLI startup,
before the server begins serving.

Skipped when:
- TLDR_SKIP_PKG_UPDATE=1 (set by docker-entrypoint.sh — it already upgrades);
- running under pytest (hermetic tests must not hit the network).

Never fatal: offline / timeout / missing pip just logs a warning and the
daemon continues with the installed versions.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys

log = logging.getLogger(__name__)

PACKAGES = ("yt-dlp", "youtube-transcript-api")
TIMEOUT_SECONDS = 60.0


def _should_skip() -> bool:
    if os.environ.get("TLDR_SKIP_PKG_UPDATE") == "1":
        return True
    return "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ


# Where uv lands when not on PATH. A daemon under launchd/systemd inherits a
# minimal PATH that usually omits ~/.local/bin, so shutil.which("uv") is None
# even though uv is installed — and a uv-tool venv has no pip to fall back on.
def _find_uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, ".local", "bin", "uv"),   # standalone installer
        os.path.join(home, ".cargo", "bin", "uv"),   # cargo install
        "/opt/homebrew/bin/uv",                        # brew (Apple Silicon)
        "/usr/local/bin/uv",                           # brew (Intel) / Linux
    ]
    return next((p for p in candidates if os.path.isfile(p)), None)


def _upgrade_command() -> list[str]:
    """uv if available (fast, and the uv-tool venv has no pip), else pip."""
    uv = _find_uv()
    if uv:
        return [uv, "pip", "install", "-U", "--python", sys.executable, *PACKAGES]
    return [sys.executable, "-m", "pip", "install", "-U", *PACKAGES]


def refresh_youtube_libs(timeout: float = TIMEOUT_SECONDS) -> bool:
    """Try to upgrade yt-dlp + youtube-transcript-api in-place.

    Returns True if the upgrade ran successfully, False if skipped or failed.
    """
    if _should_skip():
        return False
    cmd = _upgrade_command()
    log.info("selfupdate: refreshing %s ...", ", ".join(PACKAGES))
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log.warning("selfupdate: upgrade timed out after %.0fs (offline?); continuing", timeout)
        return False
    except (subprocess.CalledProcessError, OSError) as exc:
        log.warning("selfupdate: upgrade failed (%s); continuing with installed versions", exc)
        return False
    log.info("selfupdate: %s up to date", ", ".join(PACKAGES))
    return True
