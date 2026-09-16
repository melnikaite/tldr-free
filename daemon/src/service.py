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
import time
from pathlib import Path

import httpx

from src import paths

log = logging.getLogger(__name__)

LAUNCHD_LABEL = "dev.tldr.daemon"
SYSTEMD_UNIT = "tldr-daemon.service"
SCHTASKS_NAME = "TLDR Daemon"
HEALTH_URL = "http://127.0.0.1:8765/health"

# launchctl's `bootout` returns before the target job has necessarily
# finished leaving the domain (measured directly: an immediate `bootstrap`
# right after `bootout` fails with exit status 5, and the identical command
# run a moment later by hand succeeds on the first try — the signature of a
# race, not a real conflict). These bound how long we wait for the domain to
# actually clear, and how many times we retry `bootstrap` itself, before
# giving up — never an unbounded spin.
LAUNCHD_DOMAIN_WAIT_TIMEOUT_SECONDS = 5.0
LAUNCHD_DOMAIN_WAIT_POLL_SECONDS = 0.2
LAUNCHD_BOOTSTRAP_MAX_ATTEMPTS = 5
LAUNCHD_BOOTSTRAP_RETRY_DELAY_SECONDS = 0.5


class ServiceCommandError(RuntimeError):
    """A launchctl/systemctl/schtasks invocation failed after any retries
    this module attempts on its own.

    Raised instead of letting the underlying ``subprocess.CalledProcessError``
    propagate: a bare subprocess traceback is the wrong user-facing output for
    a CLI, and this message says what was being attempted and what the tool
    reported instead.
    """


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
# Optional: cloud API keys (llm.api_key et al.) can live here instead of the
# yaml config, out of the unit file and out of `systemctl show`. The leading
# "-" means "don't fail to start if the file is missing" — most installs
# don't need it (local backends, or `llm.api_key_file`/`api_key_keychain`
# already point elsewhere).
EnvironmentFile=-%h/.config/tldr/env
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


def _launchd_domain(uid: int | None = None) -> str:
    return f"gui/{uid if uid is not None else os.getuid()}"


def _launchd_service_present(domain: str) -> bool:
    """True if launchd still knows about our service in ``domain``.

    ``launchctl print <domain>/<label>`` exits non-zero (and prints
    "Could not find service ..." on stderr) once the service has genuinely
    left the domain; while `bootout` is still tearing it down it keeps
    exiting zero. This is the authoritative check — not a fixed sleep —
    for whether it's safe to `bootstrap` again.
    """
    result = _run(["launchctl", "print", f"{domain}/{LAUNCHD_LABEL}"], check=False)
    return result.returncode == 0


def _wait_until_launchd_service_gone(
    domain: str,
    *,
    timeout: float = LAUNCHD_DOMAIN_WAIT_TIMEOUT_SECONDS,
    poll_interval: float = LAUNCHD_DOMAIN_WAIT_POLL_SECONDS,
) -> None:
    """Poll ``launchctl print`` until the service has left ``domain``, or
    give up after ``timeout`` seconds — bounded, never an unbounded spin.

    Timing out here isn't fatal by itself: it just means the caller moves on
    without having confirmed the domain is clear (``install_service`` still
    retries `bootstrap` separately below if that turns out to matter).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _launchd_service_present(domain):
            return
        time.sleep(poll_interval)


def _bootstrap_launchd_with_retry(domain: str, plist: Path) -> None:
    """`launchctl bootstrap`, tolerating the bootout/bootstrap race.

    Waits for the domain to actually be free first (rather than assuming
    the preceding `bootout` was synchronous), then retries the bootstrap
    itself up to ``LAUNCHD_BOOTSTRAP_MAX_ATTEMPTS`` times, ``LAUNCHD_
    BOOTSTRAP_RETRY_DELAY_SECONDS`` apart, re-checking the domain between
    attempts. Raises ``ServiceCommandError`` with the last attempt's exit
    code/output if every attempt fails.
    """
    _wait_until_launchd_service_gone(domain)

    last_result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, LAUNCHD_BOOTSTRAP_MAX_ATTEMPTS + 1):
        result = _run(["launchctl", "bootstrap", domain, str(plist)], check=False)
        if result.returncode == 0:
            return
        last_result = result
        if attempt < LAUNCHD_BOOTSTRAP_MAX_ATTEMPTS:
            log.warning(
                "service: launchctl bootstrap attempt %d/%d failed (exit %d): %s; retrying",
                attempt, LAUNCHD_BOOTSTRAP_MAX_ATTEMPTS, result.returncode,
                result.stderr.strip() or result.stdout.strip(),
            )
            time.sleep(LAUNCHD_BOOTSTRAP_RETRY_DELAY_SECONDS)
            _wait_until_launchd_service_gone(domain)

    assert last_result is not None  # loop always sets it before exhausting attempts
    raise ServiceCommandError(
        f"launchctl bootstrap failed after {LAUNCHD_BOOTSTRAP_MAX_ATTEMPTS} attempt(s) "
        f"for {domain}/{LAUNCHD_LABEL} (exit code {last_result.returncode}): "
        f"{last_result.stderr.strip() or last_result.stdout.strip() or 'no output'}"
    )


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
            domain = _launchd_domain()
            _run(["launchctl", "bootout", f"{domain}/{LAUNCHD_LABEL}"], check=False)
            _bootstrap_launchd_with_retry(domain, plist)
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
        domain = _launchd_domain()
        _run(["launchctl", "bootout", f"{domain}/{LAUNCHD_LABEL}"], check=False)
        # Mirrors install_service's race: bootout can return before the job
        # has actually left the domain. Waiting here (bounded — see the wait
        # helper) means a caller chaining `install` right after `uninstall`
        # (the exact reported repro) is much less likely to need the retry
        # in _bootstrap_launchd_with_retry at all.
        _wait_until_launchd_service_gone(domain)
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
