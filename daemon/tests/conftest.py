"""Shared pytest fixtures.

Ensures `src.config.get_config()` has a valid file to read in CI/test
environments. We point TLDR_CONFIG at the example YAML if no real config
is mounted at the default path.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import pytest

from src import paths

# Try the docker-mounted path first; if it's missing (e.g. local pytest run
# outside the container, or a stripped CI image), fall back to the repo's
# example file.
_DEFAULT_CONFIG = Path("/app/config/tldr.yaml")
_REPO_EXAMPLE = Path(__file__).resolve().parents[2] / "config" / "tldr.yaml.example"

if not _DEFAULT_CONFIG.is_file():
    if _REPO_EXAMPLE.is_file():
        os.environ.setdefault("TLDR_CONFIG", str(_REPO_EXAMPLE))
    else:
        # As a last resort, ship a tiny inline config so tests can still
        # import src.config.get_config(). Its own scratch directory (not a
        # shared literal like "/tmp") — see `_isolate_native_data_dir`
        # below for why a literal path here would be just as unsafe as the
        # real native data directory this whole fixture exists to avoid.
        _fallback_dir = Path(tempfile.mkdtemp(prefix="tldr-fallback-config-"))
        tmp = _fallback_dir / "tldr_test_config.yaml"
        tmp.write_text(
            f"""
llm:
  base_url: http://localhost:18000/v1
  api_key: dummy
  model: qwen
  context_length: 32768
  single_pass_token_limit: 24000
whisper:
  base_url: http://localhost:18000/v1
  api_key: dummy
  model: whisper
output:
  language: ru
youtube:
  fast_path_max_attempts: 4
  fast_path_backoff_seconds: [1, 4, 16, 60]
  segment_window_seconds: 30
storage:
  data_dir: {_fallback_dir / "data"}
  db_filename: tldr.db
logging:
  level: INFO
""".strip(),
            encoding="utf-8",
        )
        os.environ.setdefault("TLDR_CONFIG", str(tmp))


# ---------------------------------------------------------------------------
# Guard rail: no test may ever touch the REAL native data directory.
# ---------------------------------------------------------------------------
#
# Incident this fixture exists to prevent: `StorageConfig._resolve_data_dir`
# substitutes `paths.platform_data_dir()` (the real
# `~/Library/Application Support/tldr/data` on macOS / XDG dir on Linux)
# whenever a config's `storage.data_dir` is left at the packaged template's
# default ("/data" — a path that only exists inside the Docker container).
# Most test files never override `storage.data_dir` at all (only ~10 of the
# ~47 test modules do) — they load `_REPO_EXAMPLE` above as-is, which DOES
# have `data_dir: /data`. On a host machine that quietly resolved to this
# developer's real, live daemon data directory. Once `src/logging_setup.py`
# started writing a rotating log file and — worse — truncating pre-existing
# `daemon.{out,err}.log` there, running the test suite destroyed real
# `daemon.err.log`/`daemon.out.log` content the developer was using to
# debug a live issue. See `.claude/ops.md` for the writeup.
#
# Fix: patch the ONE function `_resolve_data_dir()` (and every other native
# code path — `paths.default_data_dir()` calls it the same way) actually
# calls when it falls through to the native default — `platform_data_dir()`
# with NO arguments — to a fresh pytest tmp directory for every test.
# Deliberately narrow: calls WITH explicit `platform`/`home`/`env` arguments
# (test_native_install.py's own unit tests of the real cross-platform
# logic) are passed straight through to the real implementation, so this
# fixture can't mask a regression in that function itself. Any test that
# already points `storage.data_dir` at its own tmp_path never hits the
# substitution branch in the first place (it only fires when
# `data_dir == "/data"`), so this is a pure safety net for the ones that
# don't — it changes no test's observable behavior.
@pytest.fixture(autouse=True)
def _isolate_native_data_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_platform_data_dir = paths.platform_data_dir
    fallback_dir = tmp_path_factory.mktemp("native-data-dir")

    def _guarded_platform_data_dir(
        platform: str | None = None,
        home: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> Path:
        if platform is None and home is None and env is None:
            return fallback_dir
        return real_platform_data_dir(platform, home, env)

    monkeypatch.setattr(paths, "platform_data_dir", _guarded_platform_data_dir)
