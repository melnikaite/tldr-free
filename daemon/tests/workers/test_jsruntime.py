"""Tests for workers.jsruntime — Deno resolution for yt-dlp's challenge solver."""

from __future__ import annotations

import pytest

from src.workers import jsruntime


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    jsruntime.resolve_deno.cache_clear()


def test_system_deno_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jsruntime.shutil, "which", lambda name: "/usr/bin/deno")
    monkeypatch.setattr(
        jsruntime, "_download_deno", lambda: pytest.fail("should not download")
    )
    assert jsruntime.resolve_deno() == "/usr/bin/deno"


def test_falls_back_to_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jsruntime, "_system_deno", lambda: None)
    monkeypatch.setattr(jsruntime, "_download_deno", lambda: "/cache/deno/deno")
    assert jsruntime.resolve_deno() == "/cache/deno/deno"


def test_returns_none_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jsruntime, "_system_deno", lambda: None)
    monkeypatch.setattr(jsruntime, "_download_deno", lambda: None)
    assert jsruntime.resolve_deno() is None


def test_runtime_opt_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jsruntime, "resolve_deno", lambda: "/x/deno")
    assert jsruntime.deno_runtime_opt() == {
        "js_runtimes": {"deno": {"path": "/x/deno"}}
    }


def test_runtime_opt_empty_when_unresolved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jsruntime, "resolve_deno", lambda: None)
    assert jsruntime.deno_runtime_opt() == {}


def test_unsupported_platform_skips_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jsruntime.sys, "platform", "sunos5")
    monkeypatch.setattr(jsruntime, "_machine", lambda: "sparc")
    # No matching asset → returns None without touching the network.
    monkeypatch.setattr(
        jsruntime.urllib.request,
        "urlretrieve",
        lambda *a, **k: pytest.fail("should not download on unsupported platform"),
    )
    assert jsruntime._download_deno() is None


def test_pinned_version_builds_versioned_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    monkeypatch.setenv("TLDR_DENO_VERSION", "2.4.0")
    monkeypatch.setattr(jsruntime.sys, "platform", "darwin")
    monkeypatch.setattr(jsruntime, "_machine", lambda: "arm64")
    monkeypatch.setattr(jsruntime.paths, "default_data_dir", lambda: tmp_path)
    captured: dict[str, str] = {}

    def fake_retrieve(url: str, dest: object) -> None:
        captured["url"] = url
        raise RuntimeError("stop after URL capture")

    monkeypatch.setattr(jsruntime.urllib.request, "urlretrieve", fake_retrieve)
    jsruntime._download_deno()
    assert captured["url"] == (
        "https://github.com/denoland/deno/releases/download/v2.4.0/"
        "deno-aarch64-apple-darwin.zip"
    )


def test_prefetch_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> str | None:
        raise RuntimeError("boom")

    monkeypatch.setattr(jsruntime, "resolve_deno", boom)
    jsruntime.prefetch_deno()
