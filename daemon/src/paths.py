"""Platform-aware config/data path resolution for native (uv) installs.

Inside the Docker container nothing here matters: compose mounts
``/app/config/tldr.yaml`` + ``/data`` and sets ``TLDR_CONFIG``, and both
container probes below hit first. Outside the container we follow platform
conventions:

  macOS    ~/Library/Application Support/tldr/{tldr.yaml,data}
  Linux    $XDG_CONFIG_HOME/tldr/tldr.yaml + $XDG_DATA_HOME/tldr
  Windows  %APPDATA%/tldr/tldr.yaml + %LOCALAPPDATA%/tldr/data

All functions take explicit ``platform``/``home``/``env`` so tests stay
hermetic; production callers use the defaults.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

APP_NAME = "tldr"

# What the Docker image / compose file provides. If these exist we're (almost
# certainly) inside the container and must not change behavior.
CONTAINER_CONFIG = Path("/app/config/tldr.yaml")
CONTAINER_DATA = Path("/data")


def platform_config_dir(
    platform: str | None = None,
    home: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Directory that holds ``tldr.yaml`` on a native install."""
    platform = platform or sys.platform
    home = home or Path.home()
    env = env if env is not None else os.environ
    if platform == "darwin":
        return home / "Library" / "Application Support" / APP_NAME
    if platform == "win32":
        appdata = env.get("APPDATA")
        base = Path(appdata) if appdata else home / "AppData" / "Roaming"
        return base / APP_NAME
    xdg = env.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else home / ".config"
    return base / APP_NAME


def platform_data_dir(
    platform: str | None = None,
    home: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Directory for the SQLite DB + media scratch on a native install."""
    platform = platform or sys.platform
    home = home or Path.home()
    env = env if env is not None else os.environ
    if platform == "darwin":
        return home / "Library" / "Application Support" / APP_NAME / "data"
    if platform == "win32":
        local = env.get("LOCALAPPDATA")
        base = Path(local) if local else home / "AppData" / "Local"
        return base / APP_NAME / "data"
    xdg = env.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else home / ".local" / "share"
    return base / APP_NAME


def default_config_path() -> Path:
    """Config path when TLDR_CONFIG is unset: container mount if present,
    platform-conventional path otherwise."""
    if CONTAINER_CONFIG.is_file():
        return CONTAINER_CONFIG
    return platform_config_dir() / "tldr.yaml"


def default_data_dir() -> Path:
    """Data dir fallback: ``/data`` if it exists (container), else platform."""
    if CONTAINER_DATA.is_dir():
        return CONTAINER_DATA
    return platform_data_dir()
