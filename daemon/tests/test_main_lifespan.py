"""Regression test for the incident documented in `.claude/ops.md` /
`src/logging_setup.py`: running the FastAPI app (as the test suite does,
via `TestClient(app)`, and as `docker-entrypoint.sh` does for real) must
NEVER truncate `daemon.{out,err}.log`. That's a one-time, native-install-only
migration that now lives ONLY in `cli.py`'s `_serve()` (see
test_cli_serve.py for the positive case) — this test pins down that the
lifespan itself does not do it, regardless of what `storage.data_dir`
happens to resolve to.

This is a behavioral, black-box check (real files, real lifespan) rather
than "assert some function wasn't called" — it stays meaningful even if
`src.main` is refactored to call something else with a similar effect.
Critically, the fake legacy log files below are written BEFORE
`TestClient(app)` is ever entered, so a version of this code that (still,
or again) truncated at lifespan startup would fail this test — writing
them afterwards would prove nothing, since nothing would exist yet for a
buggy lifespan to find.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src import config as config_mod
from src.llm import client as llm_client
from src.logging_setup import log_dir
from src.main import app
from src.storage.db import dispose_engine, init_engine
from src.storage.migrations import run_migrations

_YAML = """
llm:
  base_url: http://127.0.0.1:1240/v1
  api_key: dummy
  model: test-model
  context_length: 32768
  single_pass_token_limit: 24000
  max_concurrent_calls: 1
whisper:
  base_url: http://127.0.0.1:1240/v1
  api_key: dummy
  model: whisper
output:
  language: en
youtube: {{}}
storage:
  data_dir: {data_dir}
  db_filename: tldr.db
""".strip()


@pytest.fixture
def prepared_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Everything test_api_config.py's `client` fixture does EXCEPT
    entering `with TestClient(app)` — this test needs to seed fake legacy
    log files on disk in between config setup and lifespan startup, which
    a fixture that already entered the TestClient context manager can't do
    (lifespan startup would already be done by the time the test body
    runs)."""
    config_file = tmp_path / "tldr.yaml"
    config_file.write_text(_YAML.format(data_dir=tmp_path))
    overrides_file = tmp_path / "tldr.local.yaml"

    monkeypatch.setenv("TLDR_CONFIG", str(config_file))
    monkeypatch.setenv("TLDR_CONFIG_OVERRIDES", str(overrides_file))
    config_mod.get_config.cache_clear()
    config_mod.keychain_backend_available.cache_clear()
    llm_client.reset_caches()

    db_path = tmp_path / "lifespan.db"
    engine = init_engine(db_path)
    run_migrations(engine)

    from src.workers import runner as runner_mod

    async def _noop_worker(queue: Any, repo_module: Any) -> None:
        try:
            while True:
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            return

    monkeypatch.setattr(runner_mod, "whisper_worker", _noop_worker)
    import src.main as main_mod

    monkeypatch.setattr(main_mod, "whisper_worker", _noop_worker)

    from src.workers import retention as retention_mod

    async def _noop_retention() -> None:
        try:
            while True:
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            return

    monkeypatch.setattr(retention_mod, "retention_worker", _noop_retention)
    monkeypatch.setattr(main_mod, "retention_worker", _noop_retention)

    from src.workers import broker as broker_mod
    from src.workers import control as control_mod
    from src.workers import queue as queue_mod

    queue_mod.reset_queue()
    broker_mod.reset_broker()
    control_mod.reset_control()

    yield tmp_path

    dispose_engine()
    queue_mod.reset_queue()
    broker_mod.reset_broker()
    control_mod.reset_control()
    config_mod.get_config.cache_clear()
    if hasattr(config_mod.keychain_backend_available, "cache_clear"):
        config_mod.keychain_backend_available.cache_clear()
    llm_client.reset_caches()


def test_lifespan_never_truncates_legacy_launchd_logs(prepared_config: Path) -> None:
    data_dir = prepared_config
    out_log = data_dir / "daemon.out.log"
    err_log = data_dir / "daemon.err.log"
    out_content = b"pre-existing launchd stdout content\n" * 100
    err_content = b"pre-existing launchd stderr content\n" * 100
    # Written BEFORE the app (and its lifespan) ever starts — see module
    # docstring for why this ordering is the whole point of the test.
    out_log.write_bytes(out_content)
    err_log.write_bytes(err_content)

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200

    assert out_log.read_bytes() == out_content
    assert err_log.read_bytes() == err_content

    from src.config import get_config

    sentinel = log_dir(get_config()) / ".legacy_logs_truncated"
    assert not sentinel.exists()
