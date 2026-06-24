"""User-level autostart service management for native (uv) installs.

Generates and (un)registers per-user service units:

  macOS    ~/Library/LaunchAgents/dev.tldr.daemon.plist (launchd, RunAtLoad)
  Linux    ~/.config/systemd/user/tldr-daemon.service (systemd --user, hardened)
  Windows  schtasks logon task — best-effort, experimental

Content generation is pure (string in, string out) so tests can assert on
unit files without touching launchctl/systemctl. Registration shells out via
the module-level ``_run`` so tests monkeypatch it.

Docker installs never call this — the container has its own restart policy.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import httpx

from src import paths

log = logging.getLogger(__name__)

LAUNCHD_LABEL = "dev.tldr.daemon"
SYSTEMD_UNIT = "tldr-daemon.service"
SCHTASKS_NAME = "TLDR Daemon"
HEALTH_URL = "http://127.0.0.1:8765/health"


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def daemon_executable() -> str:
    """Absolute path of the installed ``tldr-daemon`` entrypoint."""
    found = shutil.which("tldr-daemon")
    if found:
        return str(Path(found).resolve())
    # Fallback: however this process was started (e.g. `uvx tldr-daemon`).
    return str(Path(sys.argv[0]).resolve())


# --- unit file paths ---------------------------------------------------------


def launchd_plist_path(home: Path | None = None) -> Path:
    home = home or Path.home()
    return home / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def systemd_unit_path(home: Path | None = None) -> Path:
    home = home or Path.home()
    return home / ".config" / "systemd" / "user" / SYSTEMD_UNIT


def unit_path(platform: str | None = None, home: Path | None = None) -> Path | None:
    platform = platform or sys.platform
    if platform == "darwin":
        return launchd_plist_path(home)
    if platform.startswith("linux"):
        return systemd_unit_path(home)
    return None  # Windows: schtasks has no user-visible unit file


# --- unit file content -------------------------------------------------------


def launchd_plist(program: str, log_dir: Path) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LAUNCHD_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{program}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_dir}/daemon.out.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/daemon.err.log</string>
</dict>
</plist>
"""


def systemd_unit(program: str, data_dir: Path) -> str:
    return f"""[Unit]
Description=TLDR daemon (page/video summaries)
After=network-online.target

[Service]
ExecStart={program}
Restart=on-failure
RestartSec=5
# Hardening — the daemon runs as the user with no container boundary.
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths={data_dir}
PrivateTmp=true

[Install]
WantedBy=default.target
"""


def _resolve_data_dir() -> Path:
    """Data dir for ReadWritePaths / launchd logs; tolerate a broken config."""
    try:
        from src.config import get_config

        return Path(get_config().storage.data_dir)
    except Exception:
        return paths.default_data_dir()


# --- install / uninstall / status --------------------------------------------


def install_service(
    platform: str | None = None,
    home: Path | None = None,
    register: bool = True,
) -> Path | None:
    """Write the unit file and (optionally) register it with the OS.

    Returns the unit file path, or None on Windows (schtasks only).
    """
    platform = platform or sys.platform
    program = daemon_executable()
    data_dir = _resolve_data_dir()

    if platform == "darwin":
        plist = launchd_plist_path(home)
        data_dir.mkdir(parents=True, exist_ok=True)
        plist.parent.mkdir(parents=True, exist_ok=True)
        plist.write_text(launchd_plist(program, data_dir))
        if register:
            domain = f"gui/{os.getuid()}"
            _run(["launchctl", "bootout", f"{domain}/{LAUNCHD_LABEL}"], check=False)
            _run(["launchctl", "bootstrap", domain, str(plist)])
        return plist

    if platform.startswith("linux"):
        unit = systemd_unit_path(home)
        unit.parent.mkdir(parents=True, exist_ok=True)
        unit.write_text(systemd_unit(program, data_dir))
        if register:
            _run(["systemctl", "--user", "daemon-reload"])
            _run(["systemctl", "--user", "enable", "--now", SYSTEMD_UNIT])
        return unit

    if platform == "win32":
        log.warning("Windows service support is experimental (schtasks logon task)")
        if register:
            _run(
                [
                    "schtasks",
                    "/Create",
                    "/F",
                    "/SC",
                    "ONLOGON",
                    "/TN",
                    SCHTASKS_NAME,
                    "/TR",
                    program,
                ]
            )
        return None

    raise RuntimeError(f"Unsupported platform for service install: {platform}")


def uninstall_service(platform: str | None = None, home: Path | None = None) -> None:
    """Stop, deregister, and remove the unit file. Idempotent."""
    platform = platform or sys.platform

    if platform == "darwin":
        plist = launchd_plist_path(home)
        _run(["launchctl", "bootout", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"], check=False)
        plist.unlink(missing_ok=True)
        return

    if platform.startswith("linux"):
        unit = systemd_unit_path(home)
        _run(["systemctl", "--user", "disable", "--now", SYSTEMD_UNIT], check=False)
        unit.unlink(missing_ok=True)
        _run(["systemctl", "--user", "daemon-reload"], check=False)
        return

    if platform == "win32":
        _run(["schtasks", "/Delete", "/F", "/TN", SCHTASKS_NAME], check=False)
        return

    raise RuntimeError(f"Unsupported platform for service uninstall: {platform}")


def daemon_healthy(url: str = HEALTH_URL) -> bool:
    try:
        return httpx.get(url, timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


def service_status(platform: str | None = None, home: Path | None = None) -> dict[str, bool]:
    """Installed = unit file present (Windows: schtasks query); healthy = /health OK."""
    platform = platform or sys.platform
    if platform == "win32":
        installed = _run(["schtasks", "/Query", "/TN", SCHTASKS_NAME], check=False).returncode == 0
    else:
        path = unit_path(platform, home)
        installed = path is not None and path.is_file()
    return {"installed": installed, "healthy": daemon_healthy()}
