"""Native (uv) mode: path resolution, config auto-create, service unit files.

Hermetic — no launchctl/systemctl/schtasks, no network. Registration calls
go through ``service._run``, which is monkeypatched to record commands.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src import paths, selfupdate, service
from src.config import StorageConfig, ensure_config_file

# --- paths --------------------------------------------------------------------


def test_config_dir_macos() -> None:
    home = Path("/Users/alice")
    assert paths.platform_config_dir("darwin", home, {}) == (
        home / "Library" / "Application Support" / "tldr"
    )


def test_config_dir_linux_default_and_xdg() -> None:
    home = Path("/home/alice")
    assert paths.platform_config_dir("linux", home, {}) == home / ".config" / "tldr"
    assert paths.platform_config_dir("linux", home, {"XDG_CONFIG_HOME": "/xdg"}) == (
        Path("/xdg") / "tldr"
    )


def test_data_dir_macos_and_linux() -> None:
    home = Path("/Users/alice")
    assert paths.platform_data_dir("darwin", home, {}) == (
        home / "Library" / "Application Support" / "tldr" / "data"
    )
    home = Path("/home/alice")
    assert paths.platform_data_dir("linux", home, {}) == home / ".local" / "share" / "tldr"
    assert paths.platform_data_dir("linux", home, {"XDG_DATA_HOME": "/xdg"}) == (
        Path("/xdg") / "tldr"
    )


def test_windows_dirs_respect_env() -> None:
    home = Path("C:/Users/alice")
    env = {"APPDATA": "C:/Roaming", "LOCALAPPDATA": "C:/Local"}
    assert paths.platform_config_dir("win32", home, env) == Path("C:/Roaming") / "tldr"
    assert paths.platform_data_dir("win32", home, env) == Path("C:/Local") / "tldr" / "data"


def test_storage_config_substitutes_platform_dir_when_no_container_volume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # /data missing → "/data" (default and template value) resolves natively.
    monkeypatch.setattr(paths, "CONTAINER_DATA", tmp_path / "no-such-dir")
    cfg = StorageConfig(data_dir="/data")
    assert cfg.data_dir == str(paths.platform_data_dir())
    # An explicit non-/data value is left alone.
    assert StorageConfig(data_dir=str(tmp_path)).data_dir == str(tmp_path)


def test_storage_config_keeps_data_dir_in_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(paths, "CONTAINER_DATA", tmp_path)  # exists → container
    assert StorageConfig(data_dir="/data").data_dir == "/data"


# --- config auto-create ---------------------------------------------------------


def test_ensure_config_file_creates_from_template(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "tldr.yaml"
    assert ensure_config_file(target) is True
    text = target.read_text()
    assert "llm:" in text
    # Native rewrite: container-only hostname must not leak into native configs.
    assert "host.docker.internal" not in text
    assert "127.0.0.1" in text
    # Idempotent: second call is a no-op.
    assert ensure_config_file(target) is False


# --- service unit generation -----------------------------------------------------


def _fake_run(calls: list[list[str]]) -> object:
    def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return run


def test_launchd_plist_content(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(service, "_run", _fake_run(calls))
    monkeypatch.setattr(service, "daemon_executable", lambda: "/opt/bin/tldr-daemon")
    monkeypatch.setattr(service, "_resolve_data_dir", lambda: tmp_path / "data")

    plist = service.install_service(platform="darwin", home=tmp_path)
    assert plist == tmp_path / "Library" / "LaunchAgents" / "dev.tldr.daemon.plist"
    text = plist.read_text()
    assert "<string>dev.tldr.daemon</string>" in text
    assert "<string>/opt/bin/tldr-daemon</string>" in text
    assert "<key>RunAtLoad</key>" in text
    assert "<key>KeepAlive</key>" in text
    # Registration went through launchctl bootstrap on the gui domain.
    assert any(c[:2] == ["launchctl", "bootstrap"] for c in calls)

    service.uninstall_service(platform="darwin", home=tmp_path)
    assert not plist.exists()
    assert any(c[:2] == ["launchctl", "bootout"] for c in calls)


def test_launchd_install_without_register_runs_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(service, "_run", _fake_run(calls))
    monkeypatch.setattr(service, "daemon_executable", lambda: "/opt/bin/tldr-daemon")
    monkeypatch.setattr(service, "_resolve_data_dir", lambda: tmp_path / "data")
    plist = service.install_service(platform="darwin", home=tmp_path, register=False)
    assert plist is not None and plist.is_file()
    assert calls == []


def test_systemd_unit_content_and_hardening(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(service, "_run", _fake_run(calls))
    monkeypatch.setattr(service, "daemon_executable", lambda: "/opt/bin/tldr-daemon")
    monkeypatch.setattr(service, "_resolve_data_dir", lambda: tmp_path / "data")

    unit = service.install_service(platform="linux", home=tmp_path)
    assert unit == tmp_path / ".config" / "systemd" / "user" / "tldr-daemon.service"
    text = unit.read_text()
    # Optional key material (cloud llm.api_key) can live outside the yaml —
    # the leading "-" means startup doesn't fail if the file is absent.
    assert "EnvironmentFile=-%h/.config/tldr/env" in text
    assert "ExecStart=/opt/bin/tldr-daemon" in text
    assert "Restart=on-failure" in text
    assert "NoNewPrivileges=true" in text
    assert "ProtectSystem=strict" in text
    assert "ProtectHome=read-only" in text
    assert f"ReadWritePaths={tmp_path / 'data'}" in text
    assert "PrivateTmp=true" in text
    assert ["systemctl", "--user", "daemon-reload"] in calls
    assert ["systemctl", "--user", "enable", "--now", "tldr-daemon.service"] in calls

    service.uninstall_service(platform="linux", home=tmp_path)
    assert not unit.exists()


def test_service_status_reflects_unit_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(service, "daemon_healthy", lambda: False)
    status = service.service_status(platform="darwin", home=tmp_path)
    assert status == {"installed": False, "healthy": False}
    plist = service.launchd_plist_path(tmp_path)
    plist.parent.mkdir(parents=True)
    plist.write_text("x")
    assert service.service_status(platform="darwin", home=tmp_path)["installed"] is True


# --- selfupdate ------------------------------------------------------------------


def test_selfupdate_skipped_under_pytest() -> None:
    # We ARE under pytest here, so this must be a guaranteed no-op.
    assert selfupdate.refresh_youtube_libs() is False


def test_selfupdate_skipped_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLDR_SKIP_PKG_UPDATE", "1")
    assert selfupdate._should_skip() is True


def test_selfupdate_prefers_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(selfupdate.shutil, "which", lambda _: "/usr/local/bin/uv")
    cmd = selfupdate._upgrade_command()
    assert cmd[0] == "/usr/local/bin/uv"
    assert "--python" in cmd and "yt-dlp" in cmd
    # uv not on PATH, none of the known dirs exist → pip fallback.
    monkeypatch.setattr(selfupdate.shutil, "which", lambda _: None)
    monkeypatch.setattr(selfupdate.os.path, "isfile", lambda _p: False)
    cmd = selfupdate._upgrade_command()
    assert cmd[:3] == [selfupdate.sys.executable, "-m", "pip"]


def test_selfupdate_finds_uv_off_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under a thin launchd PATH, uv is found in a known dir, not via pip."""
    monkeypatch.setattr(selfupdate.shutil, "which", lambda _: None)
    home = selfupdate.os.path.expanduser("~")
    local_uv = selfupdate.os.path.join(home, ".local", "bin", "uv")
    monkeypatch.setattr(selfupdate.os.path, "isfile", lambda p: p == local_uv)
    cmd = selfupdate._upgrade_command()
    assert cmd[0] == local_uv
    assert "--python" in cmd
