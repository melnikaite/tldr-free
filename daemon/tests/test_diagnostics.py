"""Tests for GET /diagnostics and its scrubbing (src/api/diagnostics.py).

The whole point of this endpoint is that it's safe to paste into a public
bug report, so the scrubbing tests are the most important ones here: a
processed-page URL must never survive, a home directory must never survive,
neither API key may survive even as a fragment — while a LOCAL backend URL
(127.0.0.1/localhost) must be kept, since that's exactly the address a
diagnosis needs to see.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src import config as config_mod
from src.api import diagnostics as diagnostics_mod
from src.llm import client as llm_client
from src.main import app
from src.storage.db import dispose_engine, init_engine
from src.storage.migrations import run_migrations

_SECRET_KEY = "sk-supersecrettoken1234567890"


def _yaml(tmp_path: Path, api_key: str = "dummy") -> str:
    return f"""
llm:
  base_url: http://127.0.0.1:1240/v1
  api_key: {api_key}
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
  data_dir: {tmp_path}
  db_filename: tldr.db
""".strip()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    config_file = tmp_path / "tldr.yaml"
    config_file.write_text(_yaml(tmp_path, api_key=_SECRET_KEY))
    overrides_file = tmp_path / "tldr.local.yaml"

    monkeypatch.setenv("TLDR_CONFIG", str(config_file))
    monkeypatch.setenv("TLDR_CONFIG_OVERRIDES", str(overrides_file))
    config_mod.get_config.cache_clear()
    config_mod.keychain_backend_available.cache_clear()
    llm_client.reset_caches()

    db_path = tmp_path / "diag.db"
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

    with TestClient(app) as c:
        yield c

    dispose_engine()
    queue_mod.reset_queue()
    broker_mod.reset_broker()
    control_mod.reset_control()
    config_mod.get_config.cache_clear()
    if hasattr(config_mod.keychain_backend_available, "cache_clear"):
        config_mod.keychain_backend_available.cache_clear()
    llm_client.reset_caches()


# ---------------------------------------------------------------------------
# Pure scrub unit tests — no HTTP involved.
# ---------------------------------------------------------------------------


def test_scrub_redacts_non_local_url() -> None:
    from src.config import validate_full_config

    cfg = validate_full_config({})
    line = "fetched https://www.youtube.com/watch?v=abc123 for the user"
    scrubbed = diagnostics_mod._scrub(line, cfg)
    assert "youtube.com" not in scrubbed
    assert diagnostics_mod._URL_PLACEHOLDER in scrubbed


def test_scrub_keeps_local_backend_url() -> None:
    from src.config import validate_full_config

    cfg = validate_full_config({})
    line = "GET http://127.0.0.1:1240/v1/models -> 200; also http://localhost:1240/health"
    scrubbed = diagnostics_mod._scrub(line, cfg)
    assert "http://127.0.0.1:1240/v1/models" in scrubbed
    assert "http://localhost:1240/health" in scrubbed


def test_scrub_replaces_home_directory() -> None:
    from src.config import validate_full_config

    cfg = validate_full_config({})
    home = str(Path.home())
    line = f'Traceback: File "{home}/projects/tldr/daemon/src/main.py", line 12'
    scrubbed = diagnostics_mod._scrub(line, cfg)
    assert home not in scrubbed
    assert "~" in scrubbed


def test_scrub_removes_api_key_fragment() -> None:
    from src.config import validate_full_config

    cfg = validate_full_config({"llm": {"api_key": _SECRET_KEY}})
    line = f"completion failed: Authorization: Bearer {_SECRET_KEY} rejected"
    scrubbed = diagnostics_mod._scrub(line, cfg)
    assert _SECRET_KEY not in scrubbed
    # Not present as a long fragment either, only the whole-string match
    # `_redact_api_keys` performs — a 10-char prefix is well past what
    # could survive by coincidence.
    assert _SECRET_KEY[:10] not in scrubbed


def test_scrub_redacts_percent_encoded_external_url() -> None:
    """The regression this test exists for: `uvicorn.access` (before the
    logging_setup.py fix) wrote the request's query string verbatim,
    percent-encoded — `GET /jobs?url=https%3A%2F%2F...`. A literal-only
    `https?://` regex lets that straight through untouched."""
    from src.config import validate_full_config

    cfg = validate_full_config({})
    line = (
        'INFO uvicorn.access | 127.0.0.1:65404 - '
        '"GET /jobs?url=https%3A%2F%2Fexample.com%2Fa%2Fpath%3Fq%3Dvalue&limit=1 '
        'HTTP/1.1" 200'
    )
    scrubbed = diagnostics_mod._scrub(line, cfg)
    assert "example.com" not in scrubbed
    assert "%2F" not in scrubbed
    assert diagnostics_mod._URL_PLACEHOLDER in scrubbed


def test_scrub_keeps_percent_encoded_loopback_url() -> None:
    from src.config import validate_full_config

    cfg = validate_full_config({})
    line = (
        'INFO uvicorn.access | 127.0.0.1:65404 - '
        '"GET /jobs?url=http%3A%2F%2F127.0.0.1%3A1240%2Fv1%2Fmodels&limit=1 '
        'HTTP/1.1" 200'
    )
    scrubbed = diagnostics_mod._scrub(line, cfg)
    # Kept verbatim (still encoded) — not rewritten, not redacted.
    assert "http%3A%2F%2F127.0.0.1%3A1240%2Fv1%2Fmodels" in scrubbed
    assert diagnostics_mod._URL_PLACEHOLDER not in scrubbed


def test_scrub_does_not_leak_long_text_embedded_in_encoded_query_string() -> None:
    """Reproduces the worse-than-a-URL case reported live: a query string
    can carry arbitrary user content (e.g. a translated message), not just
    a page address. The whole percent-encoded span must be swept away with
    the URL it's attached to, not just the host portion of it."""
    from src.config import validate_full_config

    cfg = validate_full_config({})
    private_text = "this is a private message the user typed " * 5
    line = (
        "GET /jobs?url=https%3A%2F%2Ftranslate.google.com%2F%3Ftext%3D"
        + private_text.replace(" ", "%20")
        + "&limit=1"
    )
    scrubbed = diagnostics_mod._scrub(line, cfg)
    assert private_text not in scrubbed
    assert "private message" not in scrubbed
    assert diagnostics_mod._URL_PLACEHOLDER in scrubbed


def test_scrub_still_redacts_partially_encoded_url_path() -> None:
    """A literal scheme with only the PATH percent-encoded (e.g. spaces as
    %20) already worked before this fix and must keep working."""
    from src.config import validate_full_config

    cfg = validate_full_config({})
    line = "fetched https://example.com/a%20path%20with%20spaces for the user"
    scrubbed = diagnostics_mod._scrub(line, cfg)
    assert "example.com" not in scrubbed
    assert diagnostics_mod._URL_PLACEHOLDER in scrubbed


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


def test_get_diagnostics_ok_and_scrubbed(client: TestClient) -> None:
    r = client.get("/diagnostics")
    assert r.status_code == 200
    body = r.json()
    assert body["daemon_version"]
    assert body["python_version"]
    assert body["platform"]
    assert "health" in body
    assert "config" in body
    assert _SECRET_KEY not in r.text
    assert str(Path.home()) not in r.text
    assert "job_status_summary" in body
    assert body["job"] is None


def test_get_diagnostics_backend_unreachable_is_200(client: TestClient) -> None:
    # base_url in this fixture's config already points at 127.0.0.1:1240,
    # which is almost certainly not listening during the test run — health's
    # own probe already treats that as "degraded", not an exception.
    r = client.get("/diagnostics")
    assert r.status_code == 200
    body = r.json()
    assert body["health"]["status"] in ("ok", "degraded")


def test_get_diagnostics_job_id_returns_metadata_only(client: TestClient) -> None:
    from src.storage import repo as repo_module

    job = repo_module.create_job(
        url="https://example.com/secret-page-the-user-visited",
        kind="page",
        title="A Very Private Video Title",
    )
    repo_module.mark_failed(job.id, error="boom: connection reset")

    r = client.get(f"/diagnostics?job_id={job.id}")
    assert r.status_code == 200
    body = r.json()
    assert body["job"]["job_id"] == job.id
    assert body["job"]["status"] == "failed"
    assert body["job"]["kind"] == "page"
    assert body["job"]["error"] == "boom: connection reset"

    # Never present anywhere in the response, scrubbed or not.
    assert "secret-page-the-user-visited" not in r.text
    assert "A Very Private Video Title" not in r.text
    assert "title" not in body["job"]
    assert "url" not in body["job"]
    assert "raw_text" not in body["job"]
    assert "summary_md" not in body["job"]


def test_get_diagnostics_unknown_job_id_404(client: TestClient) -> None:
    r = client.get("/diagnostics?job_id=does-not-exist")
    assert r.status_code == 404


def test_get_diagnostics_job_status_summary_counts(client: TestClient) -> None:
    from src.storage import repo as repo_module

    repo_module.create_job(url="https://a.example/1", kind="page")
    j2 = repo_module.create_job(url="https://a.example/2", kind="page")
    repo_module.mark_failed(j2.id, error="x")

    r = client.get("/diagnostics")
    body = r.json()
    summary = body["job_status_summary"]
    assert summary.get("running", 0) >= 1
    assert summary.get("failed", 0) >= 1


def test_get_diagnostics_never_includes_api_key_hint(client: TestClient) -> None:
    """The regression this test exists for: `_to_response()` (reused for
    its api-key resolution logic) computes `api_key_hint` for the
    options-page UI, but a diagnostics report has no legitimate use for it
    — `api_key_set` already answers "is a key configured" — while the hint
    is 4 real characters of the configured key. It must be absent from
    BOTH sections, not merely null."""
    r = client.get("/diagnostics")
    assert r.status_code == 200
    body = r.json()
    assert "api_key_hint" not in body["config"]["llm"]
    assert "api_key_hint" not in body["config"]["whisper"]
    # Belt and suspenders: the field name itself never appears in the raw
    # response body (covers it not sneaking back in via some other key).
    assert "api_key_hint" not in r.text


def test_get_diagnostics_scrubs_config_and_overrides_paths_under_home(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduces the exact leak reported from a live daemon: when
    `TLDR_CONFIG`/`TLDR_CONFIG_OVERRIDES` are unset, `config_path()` /
    `overrides_path()` (src/config.py) resolve under the real
    `platform_config_dir()` — the user's actual home directory — and
    `_to_response()` reports those absolute paths verbatim. This test's
    `client` fixture always overrides both env vars to tmp_path (so it
    never hits that code path on its own), so we reach for the same
    real-world shape directly: monkeypatch `config_path`/`overrides_path`
    themselves to return home-rooted paths, exactly like the un-overridden
    native install does, and confirm the report scrubs them instead of
    forwarding the account name."""
    home = Path.home()
    fake_config_path = home / "Library" / "Application Support" / "tldr" / "tldr.yaml"
    fake_overrides_path = home / "Library" / "Application Support" / "tldr" / "tldr.local.yaml"
    monkeypatch.setattr(config_mod, "config_path", lambda: fake_config_path)
    monkeypatch.setattr(config_mod, "overrides_path", lambda: fake_overrides_path)

    r = client.get("/diagnostics")
    assert r.status_code == 200
    body = r.json()

    assert str(home) not in r.text
    assert body["config"]["config_path"].startswith("~")
    assert body["config"]["overrides_path"].startswith("~")
    assert body["config"]["config_path"].endswith("tldr.yaml")
    assert body["config"]["overrides_path"].endswith("tldr.local.yaml")
