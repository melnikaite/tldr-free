"""`_serve()` (src/cli.py) is the ONLY call site for
`truncate_legacy_launchd_logs_once` — see that function's docstring and
`tests/test_main_lifespan.py` for the negative case (the FastAPI lifespan
must NEVER call it). This test pins down the positive case: the real
native entrypoint does, so an existing launchd install still gets its
accumulated `daemon.{out,err}.log` cleaned up.

`uvicorn.run`, the self-update, and the ffmpeg/deno prefetch threads are
all mocked out — this test is about the ORDER/presence of the truncation
call, not about actually serving anything.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from src import cli, selfupdate
from src.config import validate_full_config


def test_serve_truncates_legacy_logs_before_running_uvicorn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(selfupdate, "refresh_youtube_libs", lambda: calls.append("selfupdate"))
    # Prevent the prefetch helpers from spinning up real background threads.
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)

    import uvicorn

    def _fake_run(*args: object, **kwargs: object) -> None:
        calls.append("uvicorn.run")

    monkeypatch.setattr(uvicorn, "run", _fake_run)

    config = validate_full_config({"storage": {"data_dir": str(tmp_path)}})
    monkeypatch.setattr(cli, "get_config", lambda: config)

    def _fake_truncate(cfg: object) -> None:
        assert cfg is config
        calls.append("truncate")

    monkeypatch.setattr(cli, "truncate_legacy_launchd_logs_once", _fake_truncate)

    assert cli._serve("127.0.0.1", 8765) == 0

    assert calls[0] == "truncate"
    assert "uvicorn.run" in calls
