"""Tests for workers.ffmpeg — cross-platform ffmpeg resolution for yt-dlp."""

from __future__ import annotations

import pytest

from src.workers import ffmpeg


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """resolve_ffmpeg_dir is lru_cached — reset between tests."""
    ffmpeg.resolve_ffmpeg_dir.cache_clear()


def test_system_ffmpeg_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_which(name: str) -> str | None:
        return {"ffmpeg": "/usr/bin/ffmpeg", "ffprobe": "/usr/bin/ffprobe"}.get(name)

    monkeypatch.setattr(ffmpeg.shutil, "which", fake_which)
    # static path must NOT be consulted when system binaries are present.
    monkeypatch.setattr(
        ffmpeg, "_static_ffmpeg_dir", lambda: pytest.fail("should not fetch static")
    )
    assert ffmpeg.resolve_ffmpeg_dir() == "/usr/bin"


def test_known_dir_used_when_not_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A daemon with a thin PATH still finds a system build in a known dir,
    avoiding a needless static download."""
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda name: None)
    monkeypatch.setattr(ffmpeg, "_KNOWN_BIN_DIRS", ("/opt/homebrew/bin",))
    monkeypatch.setattr(ffmpeg, "_both_in", lambda d: d == "/opt/homebrew/bin")
    monkeypatch.setattr(
        ffmpeg, "_static_ffmpeg_dir", lambda: pytest.fail("should not fetch static")
    )
    assert ffmpeg.resolve_ffmpeg_dir() == "/opt/homebrew/bin"


def test_falls_back_to_static_when_system_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only ffmpeg present, no ffprobe → system dir is None → static is used.
    monkeypatch.setattr(
        ffmpeg.shutil,
        "which",
        lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None,
    )
    monkeypatch.setattr(ffmpeg, "_both_in", lambda d: False)
    monkeypatch.setattr(ffmpeg, "_static_ffmpeg_dir", lambda: "/cache/bin/darwin_arm64")
    assert ffmpeg.resolve_ffmpeg_dir() == "/cache/bin/darwin_arm64"


def test_returns_none_when_nothing_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda name: None)
    monkeypatch.setattr(ffmpeg, "_both_in", lambda d: False)
    monkeypatch.setattr(ffmpeg, "_static_ffmpeg_dir", lambda: None)
    assert ffmpeg.resolve_ffmpeg_dir() is None


def test_result_is_memoised(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def counting_which(name: str) -> str | None:
        calls["n"] += 1
        return {"ffmpeg": "/usr/bin/ffmpeg", "ffprobe": "/usr/bin/ffprobe"}.get(name)

    monkeypatch.setattr(ffmpeg.shutil, "which", counting_which)
    ffmpeg.resolve_ffmpeg_dir()
    ffmpeg.resolve_ffmpeg_dir()
    # Two which() calls (ffmpeg + ffprobe) on the first resolve only.
    assert calls["n"] == 2


def test_static_fetch_failure_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real _static_ffmpeg_dir must degrade to None on any error, not raise."""
    from static_ffmpeg import run

    def boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("network down")

    monkeypatch.setattr(run, "get_platform_key", boom)
    assert ffmpeg._static_ffmpeg_dir() is None


def test_prefetch_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> str | None:
        raise RuntimeError("boom")

    monkeypatch.setattr(ffmpeg, "resolve_ffmpeg_dir", boom)
    ffmpeg.prefetch_ffmpeg()  # must not propagate


# ---------------------------------------------------------------------------
# ensure_ffmpeg_on_path — yt-dlp's download_ranges precheck reads PATH only
# ---------------------------------------------------------------------------


def test_ensure_ffmpeg_on_path_prepends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ffmpeg, "resolve_ffmpeg_dir", lambda: "/opt/ff/bin")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    assert ffmpeg.ensure_ffmpeg_on_path() == "/opt/ff/bin"
    assert ffmpeg.os.environ["PATH"] == "/opt/ff/bin:/usr/bin:/bin"


def test_ensure_ffmpeg_on_path_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Called per section download — must not grow PATH without bound."""
    monkeypatch.setattr(ffmpeg, "resolve_ffmpeg_dir", lambda: "/opt/ff/bin")
    monkeypatch.setenv("PATH", "/usr/bin")

    ffmpeg.ensure_ffmpeg_on_path()
    ffmpeg.ensure_ffmpeg_on_path()
    ffmpeg.ensure_ffmpeg_on_path()

    assert ffmpeg.os.environ["PATH"] == "/opt/ff/bin:/usr/bin"


def test_ensure_ffmpeg_on_path_already_present_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ffmpeg, "resolve_ffmpeg_dir", lambda: "/usr/bin")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    assert ffmpeg.ensure_ffmpeg_on_path() == "/usr/bin"
    assert ffmpeg.os.environ["PATH"] == "/usr/bin:/bin"


def test_ensure_ffmpeg_on_path_unresolvable_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ffmpeg, "resolve_ffmpeg_dir", lambda: None)
    monkeypatch.setenv("PATH", "/usr/bin")

    assert ffmpeg.ensure_ffmpeg_on_path() is None
    assert ffmpeg.os.environ["PATH"] == "/usr/bin"


def test_ensure_ffmpeg_on_path_empty_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ffmpeg, "resolve_ffmpeg_dir", lambda: "/opt/ff/bin")
    monkeypatch.setenv("PATH", "")

    assert ffmpeg.ensure_ffmpeg_on_path() == "/opt/ff/bin"
    assert ffmpeg.os.environ["PATH"] == "/opt/ff/bin"
