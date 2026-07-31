"""Integration tests for GET/PATCH /config and POST /config/test.

Each test points TLDR_CONFIG at a fresh tmp_path template and
TLDR_CONFIG_OVERRIDES at a sibling tldr.local.yaml, so PATCH writes never
touch the real template (or, worse, the checked-in
config/tldr.yaml.example used as the fallback by conftest.py). The
process-wide get_config()/llm.client caches are cleared before and after
every test so nothing leaks into other test modules sharing the process.
"""

from __future__ import annotations

import asyncio
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from src import config as config_mod
from src.llm import client as llm_client
from src.main import app
from src.storage.db import dispose_engine, init_engine
from src.storage.migrations import run_migrations

_MINIMAL_YAML = """
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
youtube: {}
storage:
  data_dir: /tmp
  db_filename: tldr.db
""".strip()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    config_file = tmp_path / "tldr.yaml"
    config_file.write_text(_MINIMAL_YAML)
    overrides_file = tmp_path / "tldr.local.yaml"

    monkeypatch.setenv("TLDR_CONFIG", str(config_file))
    monkeypatch.setenv("TLDR_CONFIG_OVERRIDES", str(overrides_file))
    config_mod.get_config.cache_clear()
    llm_client.reset_caches()

    db_path = tmp_path / "api.db"
    engine = init_engine(db_path)
    run_migrations(engine)

    # Whisper + retention workers → no-ops so lifespan starts cleanly.
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
    llm_client.reset_caches()


# ---------------------------------------------------------------------------
# GET /config
# ---------------------------------------------------------------------------


def test_get_config_reports_saved_values(client: TestClient) -> None:
    r = client.get("/config")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["llm"]["base_url"] == "http://127.0.0.1:1240/v1"
    assert body["llm"]["model"] == "test-model"
    assert body["llm"]["context_length"] == 32768
    assert body["llm"]["max_concurrent_calls"] == 1
    assert body["whisper"]["base_url"] == "http://127.0.0.1:1240/v1"
    assert body["output"]["language"] == "en"
    assert body["config_path"]
    assert body["overrides_path"]


def test_get_config_reports_inline_key_as_source_and_never_leaks_it(
    client: TestClient, tmp_path: Path
) -> None:
    # api_key: "dummy" is the LLMConfig default sentinel → treated as "none".
    # Switch to a real-looking inline key via the template directly.
    config_file = tmp_path / "tldr.yaml"
    config_file.write_text(_MINIMAL_YAML.replace("api_key: dummy\n  model: test-model", "api_key: sk-inline-AB12\n  model: test-model"))
    config_mod.get_config.cache_clear()

    r = client.get("/config")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["llm"]["api_key_set"] is True
    assert body["llm"]["api_key_source"] == "inline"
    assert body["llm"]["api_key_hint"] == "AB12"
    assert "sk-inline-AB12" not in r.text


def test_get_config_no_key_configured_reports_none(client: TestClient) -> None:
    r = client.get("/config")
    body = r.json()
    # api_key defaults to the "dummy" placeholder → no real key configured.
    assert body["llm"]["api_key_set"] is False
    assert body["llm"]["api_key_source"] == "none"
    assert body["llm"]["api_key_hint"] is None


# ---------------------------------------------------------------------------
# PATCH /config — writes overrides, not the template
# ---------------------------------------------------------------------------


def test_patch_writes_overrides_file_and_leaves_template_untouched(
    client: TestClient, tmp_path: Path
) -> None:
    template_before = (tmp_path / "tldr.yaml").read_text()

    r = client.patch("/config", json={"output": {"language": "ru"}})
    assert r.status_code == 200, r.text
    assert r.json()["output"]["language"] == "ru"

    assert (tmp_path / "tldr.yaml").read_text() == template_before
    overrides = yaml.safe_load((tmp_path / "tldr.local.yaml").read_text())
    assert overrides == {"output": {"language": "ru"}}

    # And GET reflects the merged result.
    assert client.get("/config").json()["output"]["language"] == "ru"


def test_patch_only_sent_fields_are_applied(client: TestClient) -> None:
    r = client.patch("/config", json={"llm": {"model": "new-model"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["llm"]["model"] == "new-model"
    # Sibling fields keep their template value.
    assert body["llm"]["base_url"] == "http://127.0.0.1:1240/v1"
    assert body["llm"]["context_length"] == 32768


def test_patch_overrides_file_and_key_file_are_0600(client: TestClient, tmp_path: Path) -> None:
    r = client.patch(
        "/config",
        json={"llm": {"api_key": "sk-file-secret", "api_key_storage": "file"}},
    )
    assert r.status_code == 200, r.text

    overrides_path = tmp_path / "tldr.local.yaml"
    assert stat.S_IMODE(overrides_path.stat().st_mode) == 0o600

    key_file = Path(yaml.safe_load(overrides_path.read_text())["llm"]["api_key_file"])
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    assert key_file.read_text() == "sk-file-secret"

    # Never echoed back.
    assert "sk-file-secret" not in r.text


def test_patch_api_key_storage_switch_cleans_previous_fields(
    client: TestClient, tmp_path: Path
) -> None:
    r = client.patch(
        "/config", json={"llm": {"api_key": "sk-first", "api_key_storage": "file"}}
    )
    assert r.status_code == 200, r.text
    overrides = yaml.safe_load((tmp_path / "tldr.local.yaml").read_text())
    assert "api_key_file" in overrides["llm"]

    r = client.patch(
        "/config", json={"llm": {"api_key": "sk-second", "api_key_storage": "inline"}}
    )
    assert r.status_code == 200, r.text
    overrides = yaml.safe_load((tmp_path / "tldr.local.yaml").read_text())
    assert overrides["llm"]["api_key"] == "sk-second"
    assert "api_key_file" not in overrides["llm"]
    assert "api_key_keychain" not in overrides["llm"]


def test_patch_api_key_storage_switch_deletes_our_managed_file(
    client: TestClient, tmp_path: Path
) -> None:
    """The file written via `api_key_storage='file'` is one we own and
    control the lifecycle of, so cleaning it up on a later switch to a
    different storage mode is safe (mirrors the "not ours" case below)."""
    r = client.patch(
        "/config", json={"llm": {"api_key": "sk-first", "api_key_storage": "file"}}
    )
    assert r.status_code == 200, r.text
    managed_file = config_mod.api_key_file_path()
    assert managed_file.exists()

    r = client.patch(
        "/config", json={"llm": {"api_key": "sk-second", "api_key_storage": "inline"}}
    )
    assert r.status_code == 200, r.text
    assert not managed_file.exists()


def test_patch_api_key_storage_switch_preserves_user_owned_file(
    client: TestClient, tmp_path: Path
) -> None:
    """`llm.api_key_file` is a plain config field — it could point at a file
    the user manages themselves (e.g. one shared with other tools) rather
    than one we wrote via api_key_storage='file'. Switching storage mode
    away from 'file' must never delete a file we didn't create, even though
    it's still referenced by `api_key_file` in the overrides we're about to
    replace."""
    user_file = tmp_path / "my-own-openai.key"
    user_file.write_text("sk-user-owned")

    overrides_file = tmp_path / "tldr.local.yaml"
    overrides_file.write_text(yaml.safe_dump({"llm": {"api_key_file": str(user_file)}}))
    config_mod.get_config.cache_clear()
    llm_client.reset_caches()

    r = client.patch(
        "/config", json={"llm": {"api_key": "sk-new", "api_key_storage": "inline"}}
    )
    assert r.status_code == 200, r.text
    assert user_file.exists()
    assert user_file.read_text() == "sk-user-owned"


def test_patch_api_key_storage_keychain_without_keyring_returns_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 'keyring' is an optional extra; force ImportError regardless of
    # whether it happens to be installed in the dev venv (it's a dev/test
    # dependency of other tests, not guaranteed absent here).
    import sys

    monkeypatch.setitem(sys.modules, "keyring", None)

    r = client.patch(
        "/config", json={"llm": {"api_key": "sk-x", "api_key_storage": "keychain"}}
    )
    assert r.status_code == 422, r.text
    assert "keyring" in r.text.lower()


def test_patch_malformed_field_type_returns_422_and_no_file_written(
    client: TestClient, tmp_path: Path
) -> None:
    r = client.patch("/config", json={"whisper": {"max_upload_mb": "not-a-number"}})
    assert r.status_code == 422, r.text
    assert not (tmp_path / "tldr.local.yaml").exists()


def test_patch_restart_required_true_when_max_concurrent_calls_changes(
    client: TestClient,
) -> None:
    r = client.patch("/config", json={"llm": {"max_concurrent_calls": 2}})
    assert r.status_code == 200, r.text
    assert r.json()["restart_required"] is True


def test_patch_restart_not_required_for_unrelated_field(client: TestClient) -> None:
    r = client.patch("/config", json={"output": {"language": "de"}})
    assert r.status_code == 200, r.text
    assert r.json()["restart_required"] is False


def test_patch_invalidates_llm_client_cache(client: TestClient) -> None:
    """After PATCH changes llm.base_url, the next _client() build must pick
    up the new value rather than an old cached AsyncOpenAI instance."""
    r = client.patch("/config", json={"llm": {"base_url": "http://example.test/v1"}})
    assert r.status_code == 200, r.text
    # AsyncOpenAI normalizes base_url with a trailing slash.
    assert str(llm_client._client().base_url).rstrip("/") == "http://example.test/v1"


# ---------------------------------------------------------------------------
# POST /config/test — probes without saving, always 200
# ---------------------------------------------------------------------------


class _FakeHttpResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or ""

    def json(self) -> dict[str, Any]:
        return self._payload


def _patch_models_probe(
    monkeypatch: pytest.MonkeyPatch, *, response: _FakeHttpResponse | None = None, raises: Exception | None = None
) -> None:
    class _FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _FakeAsyncClient:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def get(self, url: str, headers: dict[str, str] | None = None) -> _FakeHttpResponse:
            if raises is not None:
                raise raises
            assert response is not None
            return response

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)


def test_config_test_never_persists_anything(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_models_probe(monkeypatch, response=_FakeHttpResponse(200, {"data": []}))
    r = client.post("/config/test", json={})
    assert r.status_code == 200, r.text
    assert not (tmp_path / "tldr.local.yaml").exists()


def test_config_test_401_reports_verbatim_detail_and_ok_false(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_models_probe(
        monkeypatch, response=_FakeHttpResponse(401, text="Incorrect API key provided: sk-***")
    )
    r = client.post(
        "/config/test",
        json={"llm": {"base_url": "https://api.openai.com/v1", "model": "gpt-5", "api_key": "sk-bogus"}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is False
    assert body["step"] == "models"
    assert body["status_code"] == 401
    assert "Incorrect API key provided" in body["detail"]
    # The probed key must never come back in the response, in body OR detail.
    assert "sk-bogus" not in r.text


def test_config_test_connection_error_reports_models_step(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_models_probe(monkeypatch, raises=httpx.ConnectError("connection refused"))
    r = client.post("/config/test", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is False
    assert body["step"] == "models"
    assert body["status_code"] is None
    assert "connection refused" in body["detail"].lower()


def test_config_test_success_runs_both_steps(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_models_probe(
        monkeypatch, response=_FakeHttpResponse(200, {"data": [{"id": "test-model"}]})
    )

    class _FakeCompletions:
        async def create(self, **kwargs: Any) -> Any:
            return object()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeAsyncOpenAI:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.chat = _FakeChat()

    import src.api.config as config_api

    monkeypatch.setattr(config_api, "AsyncOpenAI", _FakeAsyncOpenAI)

    r = client.post("/config/test", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["step"] == "completion"
    assert body["status_code"] == 200
    assert body["models"] == ["test-model"]
    assert body["latency_ms"] is not None


def test_config_test_completion_adapts_dialect_instead_of_failing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cloud gpt-5/o-series backend rejects `max_tokens` on the first
    completion attempt. The probe must go through the same dialect
    adaptation as the real call path and report `ok: true` on the adapted
    retry, not fail the whole test on the first 400 — that first 400 is
    exactly the scenario (rejected/unsupported param on a cloud backend)
    this endpoint exists to get right."""
    _patch_models_probe(
        monkeypatch, response=_FakeHttpResponse(200, {"data": [{"id": "gpt-5"}]})
    )

    from openai import BadRequestError

    def _bad_request(param: str) -> BadRequestError:
        request = httpx.Request("POST", "http://example.test/v1/chat/completions")
        response = httpx.Response(400, request=request)
        message = f"Unsupported parameter: '{param}' is not supported with this model."
        return BadRequestError(message, response=response, body={"message": message, "param": param})

    class _FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise _bad_request("max_tokens")
            return object()

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class _FakeAsyncOpenAI:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.chat = _FakeChat()

    import src.api.config as config_api

    monkeypatch.setattr(config_api, "AsyncOpenAI", _FakeAsyncOpenAI)

    r = client.post(
        "/config/test",
        json={"llm": {"base_url": "https://api.openai.com/v1", "model": "gpt-5"}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["step"] == "completion"
