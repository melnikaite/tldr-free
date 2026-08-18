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
import re
import stat
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
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


def _minimal_yaml(data_dir: Path) -> str:
    # `data_dir` is always the test's OWN tmp_path — never a shared literal
    # like "/tmp" (see the incident writeup in `.claude/ops.md`: a shared
    # or unset data_dir here resolves to a real, possibly live, daemon data
    # directory once anything writes through `storage.data_dir`).
    return f"""
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
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    config_file = tmp_path / "tldr.yaml"
    config_file.write_text(_minimal_yaml(tmp_path))
    overrides_file = tmp_path / "tldr.local.yaml"

    monkeypatch.setenv("TLDR_CONFIG", str(config_file))
    monkeypatch.setenv("TLDR_CONFIG_OVERRIDES", str(overrides_file))
    config_mod.get_config.cache_clear()
    config_mod.keychain_backend_available.cache_clear()
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
    # A test may have monkeypatched this to a plain lambda (no cache to
    # clear) — guard rather than assume the lru_cache wrapper survived.
    if hasattr(config_mod.keychain_backend_available, "cache_clear"):
        config_mod.keychain_backend_available.cache_clear()
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
    config_file.write_text(
        _minimal_yaml(tmp_path).replace(
            "api_key: dummy\n  model: test-model", "api_key: sk-inline-AB12\n  model: test-model"
        )
    )
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


def test_get_config_reports_keychain_available_true(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_mod, "keychain_backend_available", lambda: True)
    r = client.get("/config")
    assert r.status_code == 200, r.text
    assert r.json()["keychain_available"] is True


def test_get_config_reports_keychain_available_false(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_mod, "keychain_backend_available", lambda: False)
    r = client.get("/config")
    assert r.status_code == 200, r.text
    assert r.json()["keychain_available"] is False


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


def test_patch_api_key_storage_keychain_unavailable_returns_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 'keyring' is now a base dependency (see pyproject.toml) — the 422 case
    # to cover is a keyring package with no *usable backend* (e.g. headless
    # Linux without a Secret Service in the session), not a missing package.
    monkeypatch.setattr(config_mod, "keychain_backend_available", lambda: False)

    r = client.patch(
        "/config", json={"llm": {"api_key": "sk-x", "api_key_storage": "keychain"}}
    )
    assert r.status_code == 422, r.text
    assert "keychain" in r.text.lower()
    assert "sk-x" not in r.text


def test_patch_keychain_write_failure_returns_422_not_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A usable backend (checked separately) doesn't guarantee the WRITE
    succeeds — e.g. a stale keychain entry from a previous install with an
    ACL that doesn't trust the current binary. That must surface as a
    clean 422, not an unhandled 500 crashing the request."""
    monkeypatch.setattr(config_mod, "keychain_backend_available", lambda: True)

    import sys

    def _raise_on_set(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Can't store password on keychain: (-25244, 'Unknown Error')")

    fake_keyring = type("FakeKeyring", (), {"set_password": staticmethod(_raise_on_set)})()
    monkeypatch.setitem(sys.modules, "keyring", fake_keyring)

    r = client.patch(
        "/config",
        json={"llm": {"api_key": "sk-write-fail-secret", "api_key_storage": "keychain"}},
    )
    assert r.status_code == 422, r.text
    assert "sk-write-fail-secret" not in r.text
    assert "keychain" in r.text.lower()


def _fake_keyring_module(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], str]:
    """Install an in-memory fake `keyring` module (mirrors the pattern in
    test_config.py) and return its backing store for assertions."""
    import sys

    store: dict[tuple[str, str], str] = {}
    fake_keyring = type(
        "FakeKeyring",
        (),
        {
            "set_password": staticmethod(
                lambda service, account, password: store.__setitem__((service, account), password)
            ),
            "get_password": staticmethod(lambda service, account: store.get((service, account))),
            "delete_password": staticmethod(
                lambda service, account: store.pop((service, account), None)
            ),
        },
    )()
    monkeypatch.setitem(sys.modules, "keyring", fake_keyring)
    return store


def test_patch_default_storage_is_keychain_when_available(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No `api_key_storage` given + keychain available → default to
    'keychain', not the old hardcoded 'file' default."""
    monkeypatch.setattr(config_mod, "keychain_backend_available", lambda: True)
    _fake_keyring_module(monkeypatch)

    r = client.patch("/config", json={"llm": {"api_key": "sk-auto-keychain"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["llm"]["api_key_source"] == "keychain"

    overrides = yaml.safe_load((tmp_path / "tldr.local.yaml").read_text())
    assert "api_key_keychain" in overrides["llm"]
    assert "api_key_file" not in overrides["llm"]
    assert "sk-auto-keychain" not in r.text


def test_patch_default_storage_is_file_when_keychain_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No `api_key_storage` given + keychain NOT available → falls back to
    'file', same as before this change."""
    monkeypatch.setattr(config_mod, "keychain_backend_available", lambda: False)

    r = client.patch("/config", json={"llm": {"api_key": "sk-auto-file"}})
    assert r.status_code == 200, r.text
    assert r.json()["llm"]["api_key_source"] == "file"

    overrides = yaml.safe_load((tmp_path / "tldr.local.yaml").read_text())
    assert "api_key_file" in overrides["llm"]


# ---------------------------------------------------------------------------
# PATCH /config — api_key_verified / api_key_verify_error (write-then-read-back)
# ---------------------------------------------------------------------------


def test_patch_api_key_verified_true_on_successful_save(client: TestClient) -> None:
    r = client.patch(
        "/config", json={"llm": {"api_key": "sk-verify-ok", "api_key_storage": "file"}}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["api_key_verified"] is True
    assert body["api_key_verify_error"] is None
    assert "sk-verify-ok" not in r.text


def test_patch_api_key_verified_true_when_key_not_touched(client: TestClient) -> None:
    """A PATCH that doesn't write the API key at all has nothing to
    verify — reported as verified rather than left ambiguous."""
    r = client.patch("/config", json={"output": {"language": "de"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["api_key_verified"] is True
    assert body["api_key_verify_error"] is None


def test_patch_api_key_verified_false_when_readback_raises(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read-back uses the exact accessor the daemon uses at call time
    (LLMConfig.effective_api_key) — if IT raises post-save, the save still
    stands but the failure is reported honestly, not rolled back."""

    def _boom(self: Any) -> str:
        raise RuntimeError("simulated read-back failure")

    monkeypatch.setattr(config_mod.LLMConfig, "effective_api_key", property(_boom))

    r = client.patch(
        "/config", json={"llm": {"api_key": "sk-verify-fail", "api_key_storage": "file"}}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["api_key_verified"] is False
    assert body["api_key_verify_error"]
    assert "sk-verify-fail" not in r.text

    # The save itself was NOT rolled back — GET still reports a key is set
    # (the effective_api_key property is still stubbed to raise here, so
    # the source can't be resolved, but the override was written).
    assert client.get("/config").json()["llm"]["api_key_source"] == "file"


def test_patch_api_key_verified_false_when_readback_mismatches(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        config_mod.LLMConfig, "effective_api_key", property(lambda self: "some-other-value")
    )

    r = client.patch(
        "/config", json={"llm": {"api_key": "sk-mismatch", "api_key_storage": "file"}}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["api_key_verified"] is False
    assert body["api_key_verify_error"]
    assert "sk-mismatch" not in r.text
    assert "some-other-value" not in r.text


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
# storage.retention_days — editable from the options page (Part 2)
# ---------------------------------------------------------------------------


def test_get_config_reports_default_retention_days(client: TestClient) -> None:
    r = client.get("/config")
    assert r.status_code == 200, r.text
    # StorageConfig default (config.py) — the fixture's _minimal_yaml() never
    # sets storage.retention_days, so this is the class default.
    assert r.json()["storage"]["retention_days"] == 365


def test_patch_retention_days_round_trips_through_get(
    client: TestClient, tmp_path: Path
) -> None:
    r = client.patch("/config", json={"storage": {"retention_days": 30}})
    assert r.status_code == 200, r.text
    assert r.json()["storage"]["retention_days"] == 30

    assert client.get("/config").json()["storage"]["retention_days"] == 30

    # Written to the overrides file, never the template.
    overrides = yaml.safe_load((tmp_path / "tldr.local.yaml").read_text())
    assert overrides["storage"]["retention_days"] == 30


def test_patch_retention_days_zero_round_trips_and_stays_in_overrides(
    client: TestClient, tmp_path: Path
) -> None:
    """0 means "disabled" and must stay expressible — not dropped as a
    falsy/unset value anywhere along the PATCH -> overrides-file -> GET path."""
    r = client.patch("/config", json={"storage": {"retention_days": 0}})
    assert r.status_code == 200, r.text
    assert r.json()["storage"]["retention_days"] == 0

    assert client.get("/config").json()["storage"]["retention_days"] == 0

    overrides = yaml.safe_load((tmp_path / "tldr.local.yaml").read_text())
    assert overrides["storage"]["retention_days"] == 0


def test_patch_retention_days_negative_rejected(client: TestClient, tmp_path: Path) -> None:
    r = client.patch("/config", json={"storage": {"retention_days": -1}})
    assert r.status_code == 422, r.text
    assert not (tmp_path / "tldr.local.yaml").exists()


def test_patch_retention_days_above_upper_bound_rejected(
    client: TestClient, tmp_path: Path
) -> None:
    r = client.patch("/config", json={"storage": {"retention_days": 100_000}})
    assert r.status_code == 422, r.text
    assert not (tmp_path / "tldr.local.yaml").exists()


def test_patch_retention_days_leaves_template_untouched(
    client: TestClient, tmp_path: Path
) -> None:
    template_before = (tmp_path / "tldr.yaml").read_text()
    r = client.patch("/config", json={"storage": {"retention_days": 7}})
    assert r.status_code == 200, r.text
    assert (tmp_path / "tldr.yaml").read_text() == template_before


# ---------------------------------------------------------------------------
# qa.web_search — Q&A's on/off switch for the DuckDuckGo search step
# ---------------------------------------------------------------------------


def test_get_config_reports_default_web_search_true(client: TestClient) -> None:
    """The fixture's `_minimal_yaml()` never sets a `qa:` section at all —
    this is exactly the "existing user's tldr.yaml predates this setting"
    case, and it must default to True silently (no behavior change for
    anyone who hasn't touched this)."""
    r = client.get("/config")
    assert r.status_code == 200, r.text
    assert r.json()["qa"]["web_search"] is True


def test_patch_web_search_false_round_trips_through_get(
    client: TestClient, tmp_path: Path
) -> None:
    r = client.patch("/config", json={"qa": {"web_search": False}})
    assert r.status_code == 200, r.text
    assert r.json()["qa"]["web_search"] is False

    assert client.get("/config").json()["qa"]["web_search"] is False

    # Written to the overrides file, never the template.
    overrides = yaml.safe_load((tmp_path / "tldr.local.yaml").read_text())
    assert overrides["qa"]["web_search"] is False


def test_patch_web_search_true_round_trips_through_get(
    client: TestClient, tmp_path: Path
) -> None:
    """Explicitly setting it back to True (e.g. after having turned it off)
    also round-trips — not just the implicit default."""
    client.patch("/config", json={"qa": {"web_search": False}})
    r = client.patch("/config", json={"qa": {"web_search": True}})
    assert r.status_code == 200, r.text
    assert r.json()["qa"]["web_search"] is True
    assert client.get("/config").json()["qa"]["web_search"] is True


def test_patch_web_search_leaves_template_untouched(
    client: TestClient, tmp_path: Path
) -> None:
    template_before = (tmp_path / "tldr.yaml").read_text()
    r = client.patch("/config", json={"qa": {"web_search": False}})
    assert r.status_code == 200, r.text
    assert (tmp_path / "tldr.yaml").read_text() == template_before


def test_patch_web_search_malformed_type_returns_422(client: TestClient, tmp_path: Path) -> None:
    r = client.patch("/config", json={"qa": {"web_search": "not-a-bool"}})
    assert r.status_code == 422, r.text
    assert not (tmp_path / "tldr.local.yaml").exists()


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


def _fake_completion(content: str, **extra: Any) -> Any:
    """A minimal stand-in for an OpenAI ``ChatCompletion`` — just enough
    shape for the code under test (`.choices[0].message.content`, plus
    optional extra fields reachable both as plain attributes and via
    ``.model_extra``, mirroring how the real SDK's pydantic models with
    ``extra="allow"`` expose unknown fields either way — see
    ``_reasoning_text`` in ``src/api/config.py``)."""
    message = SimpleNamespace(content=content, model_extra=dict(extra), **extra)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, create: Any) -> list[dict[str, Any]]:
    """Monkeypatch ``src.api.config.AsyncOpenAI`` to a fake client whose
    ``chat.completions.create`` is ``create`` (an async callable taking
    ``**kwargs``). Returns the list every call's kwargs gets appended to,
    for assertions on what was actually sent (e.g. ``extra_body``)."""
    calls: list[dict[str, Any]] = []

    class _FakeCompletions:
        async def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return await create(**kwargs)

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class _FakeAsyncOpenAI:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.chat = _FakeChat()

    import src.api.config as config_api

    monkeypatch.setattr(config_api, "AsyncOpenAI", _FakeAsyncOpenAI)
    return calls


def _prompt_content(kwargs: dict[str, Any]) -> str:
    return str(kwargs["messages"][0]["content"])


async def _default_ok_backend(**kwargs: Any) -> Any:
    """A backend that answers "ok" to everything and never errors — the
    huge context-probe request in particular just succeeds outright, so the
    context step reports "at least this many tokens" with no numeric
    suggestion. Good enough for tests that only care about the OTHER steps."""
    return _fake_completion("ok")


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
    steps = {s["step"]: s for s in body["steps"]}
    assert steps["reachable"]["ok"] is True
    assert steps["models"]["ok"] is False
    assert "401" in steps["models"]["detail"]
    assert "Incorrect API key provided" in steps["models"]["detail"]
    assert steps["completion"]["ok"] is None
    assert steps["thinking"]["ok"] is None
    assert steps["context"]["ok"] is None
    assert steps["translation"]["ok"] is None
    # The probed key must never come back in the response, in body OR detail.
    assert "sk-bogus" not in r.text


def test_config_test_connection_error_reports_reachable_step(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_models_probe(monkeypatch, raises=httpx.ConnectError("connection refused"))
    r = client.post("/config/test", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is False
    steps = {s["step"]: s for s in body["steps"]}
    assert steps["reachable"]["ok"] is False
    assert "connection refused" in steps["reachable"]["detail"].lower()
    for later in ("models", "completion", "thinking", "context", "translation"):
        assert steps[later]["ok"] is None


def test_config_test_success_runs_the_full_step_flow(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_models_probe(
        monkeypatch, response=_FakeHttpResponse(200, {"data": [{"id": "test-model"}]})
    )

    async def _backend(**kwargs: Any) -> Any:
        content = _prompt_content(kwargs)
        if "You translate transcripts" in content:
            # Return a plausible (non-echo, marker-preserving) translation
            # so the translation-contract step verifies cleanly.
            lines = content.rsplit("Input transcript:\n", 1)[-1].splitlines()
            out = []
            for line in lines:
                m = re.match(r"^(\[\d{1,2}:\d{2}(?::\d{2})?\])(.*)$", line)
                out.append(f"{m.group(1)} RU:{m.group(2)}" if m else line)
            return _fake_completion("\n".join(out))
        return _fake_completion("ok")

    _install_fake_openai(monkeypatch, _backend)

    r = client.post("/config/test", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["models"] == ["test-model"]
    assert body["latency_ms"] is not None

    steps = {s["step"]: s for s in body["steps"]}
    assert steps["reachable"]["ok"] is True
    assert steps["models"]["ok"] is True
    assert steps["completion"]["ok"] is True
    assert steps["thinking"]["ok"] is True
    assert "No thinking" in steps["thinking"]["detail"]
    assert steps["context"]["ok"] is True
    assert "at least" in steps["context"]["detail"]
    assert steps["translation"]["ok"] is True

    # The huge probe succeeded outright — no numeric ceiling to report.
    assert body["suggestions"]["context_length"] is None
    assert body["suggestions"]["reasoning_effort"] is None


def test_config_test_completion_adapts_dialect_instead_of_failing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cloud gpt-5/o-series backend rejects `max_tokens` on the first
    completion attempt. The probe must go through the same dialect
    adaptation as the real call path and report the completion step ok on
    the adapted retry, not fail the whole test on the first 400 — that
    first 400 is exactly the scenario (rejected/unsupported param on a
    cloud backend) this endpoint exists to get right."""
    _patch_models_probe(
        monkeypatch, response=_FakeHttpResponse(200, {"data": [{"id": "gpt-5"}]})
    )

    from openai import BadRequestError

    def _bad_request(param: str) -> BadRequestError:
        request = httpx.Request("POST", "http://example.test/v1/chat/completions")
        response = httpx.Response(400, request=request)
        message = f"Unsupported parameter: '{param}' is not supported with this model."
        return BadRequestError(message, response=response, body={"message": message, "param": param})

    seen = {"raised": False}

    async def _backend(**kwargs: Any) -> Any:
        if not seen["raised"]:
            seen["raised"] = True
            raise _bad_request("max_tokens")
        return _fake_completion("ok")

    _install_fake_openai(monkeypatch, _backend)

    r = client.post(
        "/config/test",
        json={"llm": {"base_url": "https://api.openai.com/v1", "model": "gpt-5"}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    steps = {s["step"]: s for s in body["steps"]}
    assert steps["completion"]["ok"] is True


# ---------------------------------------------------------------------------
# POST /config/test — thinking detection (step "thinking")
# ---------------------------------------------------------------------------


def test_config_test_thinking_detected_and_fixed_by_reasoning_effort_none(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty first reply with reasoning content present is exactly the
    Gemma-4-on-LocalAI failure mode from `.claude/llm.md`. Once
    reasoning_effort='none' is confirmed to fix it, that value must also be
    suggested AND carried forward into every later call (context/
    translation) — checked here via the recorded `extra_body`."""
    _patch_models_probe(monkeypatch, response=_FakeHttpResponse(200, {"data": [{"id": "gemma"}]}))

    async def _backend(**kwargs: Any) -> Any:
        extra_body = kwargs.get("extra_body") or {}
        if extra_body.get("reasoning_effort") == "none":
            return _fake_completion("ok")
        # No fix applied yet: thinking consumed the whole tiny token
        # budget, so content comes back empty with reasoning populated.
        return _fake_completion("", reasoning="the user wants...")

    calls = _install_fake_openai(monkeypatch, _backend)

    r = client.post("/config/test", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    steps = {s["step"]: s for s in body["steps"]}
    assert steps["thinking"]["ok"] is True
    assert "reasoning_effort='none'" in steps["thinking"]["detail"]
    assert body["suggestions"]["reasoning_effort"] == "none"

    # Every call made AFTER the fix was found (the individual per-line
    # translation retries, context probe) must carry the fixed value — not
    # just the one retry that discovered it.
    later_calls = calls[3:]  # 0=completion, 1=whole-group translate, 2=fixed retry
    assert later_calls, "expected further calls after the thinking fix"
    assert all(c.get("extra_body", {}).get("reasoning_effort") == "none" for c in later_calls)


def test_config_test_thinking_not_detected_on_trivial_prompt_but_detected_on_translation(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for the reported false negative: Gemma 4's thinking is
    ADAPTIVE — a trivial prompt genuinely doesn't trigger it, but a
    rule-heavy real prompt does (measured live: 0 reasoning on "reply with
    the single word ok", 1789 chars of `reasoning` plus truncated content
    — 3 of 4 lines — on the real translation prompt with reasoning_effort
    unset). A detector built on the trivial completion call would report
    "no thinking" and be wrong. This fake reproduces exactly that split —
    clean on the trivial prompt, polluted on the translation prompt — and
    the "completion" step (which only ever sends the trivial prompt) must
    stay clean/ok while "thinking" (which uses the translation call) must
    still catch it."""
    _patch_models_probe(monkeypatch, response=_FakeHttpResponse(200, {"data": [{"id": "gemma"}]}))

    async def _backend(**kwargs: Any) -> Any:
        content = _prompt_content(kwargs)
        extra_body = kwargs.get("extra_body") or {}
        if "You translate transcripts" not in content:
            # The trivial "completion" probe: Gemma 4 doesn't think here.
            return _fake_completion("ok")
        if extra_body.get("reasoning_effort") == "none":
            # Fixed retry: a full, clean, correctly-marked translation.
            lines = content.rsplit("Input transcript:\n", 1)[-1].splitlines()
            out = [
                f"{m.group(1)} RU:{m.group(2)}"
                for line in lines
                if (m := re.match(r"^(\[\d{1,2}:\d{2}(?::\d{2})?\])(.*)$", line))
            ]
            return _fake_completion("\n".join(out))
        # Unfixed: adaptive thinking fires on the real, rule-heavy prompt —
        # reasoning populated, content truncated (measured: 3 of 4 lines).
        lines = content.rsplit("Input transcript:\n", 1)[-1].splitlines()[:3]
        truncated = [
            f"{m.group(1)} RU:{m.group(2)}"
            for line in lines
            if (m := re.match(r"^(\[\d{1,2}:\d{2}(?::\d{2})?\])(.*)$", line))
        ]
        return _fake_completion("\n".join(truncated), reasoning="x" * 1789)

    _install_fake_openai(monkeypatch, _backend)

    r = client.post("/config/test", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    steps = {s["step"]: s for s in body["steps"]}
    # The trivial completion call never saw any reasoning — it must stay ok.
    assert steps["completion"]["ok"] is True
    # But thinking must be caught via the translation call, not missed.
    assert steps["thinking"]["ok"] is True
    assert "reasoning_effort='none'" in steps["thinking"]["detail"]
    assert body["suggestions"]["reasoning_effort"] == "none"
    # And the fixed retry produced a full, verifiable translation.
    assert steps["translation"]["ok"] is True


def test_config_test_thinking_detected_but_not_fixed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """reasoning_effort='none' doesn't help every backend — must be
    reported honestly (ok=False) rather than suggested anyway."""
    _patch_models_probe(monkeypatch, response=_FakeHttpResponse(200, {"data": [{"id": "gemma"}]}))

    async def _backend(**kwargs: Any) -> Any:
        return _fake_completion("", reasoning="still thinking no matter what")

    _install_fake_openai(monkeypatch, _backend)

    r = client.post("/config/test", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    steps = {s["step"]: s for s in body["steps"]}
    assert steps["thinking"]["ok"] is False
    assert body["suggestions"]["reasoning_effort"] is None


def test_config_test_no_thinking_detected_when_content_present(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_models_probe(monkeypatch, response=_FakeHttpResponse(200, {"data": [{"id": "qwen"}]}))
    _install_fake_openai(monkeypatch, _default_ok_backend)

    r = client.post("/config/test", json={})
    body = r.json()
    steps = {s["step"]: s for s in body["steps"]}
    assert steps["thinking"]["ok"] is True
    assert "No thinking" in steps["thinking"]["detail"]
    assert body["suggestions"]["reasoning_effort"] is None


# ---------------------------------------------------------------------------
# POST /config/test — real context length (step "context")
# ---------------------------------------------------------------------------


def test_config_test_context_parsed_from_error_message(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openai import BadRequestError

    _patch_models_probe(monkeypatch, response=_FakeHttpResponse(200, {"data": [{"id": "m"}]}))

    def _context_error() -> BadRequestError:
        request = httpx.Request("POST", "http://example.test/v1/chat/completions")
        response = httpx.Response(400, request=request)
        message = "request (60009 tokens) exceeds the available context size (32768 tokens)"
        return BadRequestError(message, response=response, body={"message": message})

    async def _backend(**kwargs: Any) -> Any:
        content = _prompt_content(kwargs)
        if len(content) > 5000:  # the huge context-probe prompt (~160k chars)
            raise _context_error()
        return _fake_completion("ok")

    _install_fake_openai(monkeypatch, _backend)

    r = client.post("/config/test", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    steps = {s["step"]: s for s in body["steps"]}
    assert steps["context"]["ok"] is True
    assert "32768" in steps["context"]["detail"]
    assert body["suggestions"]["context_length"] == 32768
    assert body["suggestions"]["single_pass_token_limit"] == int(32768 * 0.6)


def test_config_test_context_bisection_fallback_when_unparseable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The backend's error text doesn't mention any numbers at all — the
    probe must fall back to bisecting between a known-good floor and the
    known-bad huge size, bounded to a handful of calls, and land somewhere
    sane relative to the real (fake) ceiling."""
    from openai import BadRequestError

    from src.llm.tokens import count_tokens

    _patch_models_probe(monkeypatch, response=_FakeHttpResponse(200, {"data": [{"id": "m"}]}))

    fake_ceiling = 9000

    def _vague_error() -> BadRequestError:
        request = httpx.Request("POST", "http://example.test/v1/chat/completions")
        response = httpx.Response(400, request=request)
        message = "context window exceeded"
        return BadRequestError(message, response=response, body={"message": message})

    async def _backend(**kwargs: Any) -> Any:
        content = _prompt_content(kwargs)
        if len(content) > 5000 and count_tokens(content) > fake_ceiling:
            raise _vague_error()
        return _fake_completion("ok")

    calls = _install_fake_openai(monkeypatch, _backend)

    r = client.post("/config/test", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    steps = {s["step"]: s for s in body["steps"]}
    assert steps["context"]["ok"] is True
    assert body["suggestions"]["context_length"] is not None
    # Bisection converges from below — must never overshoot the real ceiling.
    assert body["suggestions"]["context_length"] <= fake_ceiling
    # Within the tolerance band, not wildly under-reporting either.
    assert body["suggestions"]["context_length"] > fake_ceiling - 4000
    # Bounded call count: at most 6 large-prompt context-probe attempts
    # (translation's own calls, made afterward, are small prompts and don't
    # count here — see `_CONTEXT_PROBE_MAX_ATTEMPTS`).
    large_prompt_calls = [c for c in calls if len(_prompt_content(c)) > 5000]
    assert len(large_prompt_calls) <= 6


def test_config_test_context_huge_probe_succeeds_reports_at_least(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_models_probe(monkeypatch, response=_FakeHttpResponse(200, {"data": [{"id": "m"}]}))
    _install_fake_openai(monkeypatch, _default_ok_backend)

    r = client.post("/config/test", json={})
    body = r.json()
    steps = {s["step"]: s for s in body["steps"]}
    assert steps["context"]["ok"] is True
    assert body["suggestions"]["context_length"] is None


def test_config_test_context_huge_probe_inconclusive_reports_unknown_not_a_guess(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the live run against gemma-4-e4b on LocalAI reported
    'approximately 35250 tokens' when the model's actual configured context
    was 131072 — the bisection loop was treating ANY failure (including a
    plain timeout on a huge, slow-to-prefill prompt) as proof of hitting the
    context wall. A timeout's message text carries no context/token-size
    complaint at all (see `_looks_like_context_overflow`), so it is NOT
    that proof. When even the FIRST (huge) probe can't get a message that
    reads as a size rejection, we must report `ok=None` and explain why,
    rather than fabricate a ceiling that later silently corrupts real
    jobs."""
    _patch_models_probe(monkeypatch, response=_FakeHttpResponse(200, {"data": [{"id": "m"}]}))

    async def _backend(**kwargs: Any) -> Any:
        content = _prompt_content(kwargs)
        if len(content) > 5000:  # the huge context-probe prompt
            raise TimeoutError("llm stream stalled: no chunk for 30s")
        return _fake_completion("ok")

    _install_fake_openai(monkeypatch, _backend)

    r = client.post("/config/test", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    steps = {s["step"]: s for s in body["steps"]}
    assert steps["context"]["ok"] is None
    assert "recognizable size complaint" in steps["context"]["detail"]
    assert "no chunk for 30s" in steps["context"]["detail"]
    assert body["suggestions"]["context_length"] is None
    assert body["suggestions"]["single_pass_token_limit"] is None


def test_config_test_context_overflow_recognized_via_message_even_as_http_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: measured live against qwen3-vl-8b-instruct on LocalAI —
    the real backend's context-overflow error ('rpc error: code = Internal
    desc = request (40008 tokens) exceeds the available context size
    (32768 tokens), try increasing it') arrived wrapped in an HTTP 500, not
    a 400. A status-code-only classifier threw this away as "inconclusive"
    on the one case where the signal was perfect and previously worked.
    Classification must key off the message TEXT (context/tokens + an
    overflow word — see `_looks_like_context_overflow`), independent of
    status code, and still extract the exact number when the message
    states one."""
    from openai import InternalServerError

    _patch_models_probe(monkeypatch, response=_FakeHttpResponse(200, {"data": [{"id": "m"}]}))

    def _real_context_error_as_500() -> InternalServerError:
        request = httpx.Request("POST", "http://example.test/v1/chat/completions")
        response = httpx.Response(500, request=request)
        message = (
            "Error code: 500 - {'error': {'code': 500, 'message': 'rpc error: code = "
            "Internal desc = request (40008 tokens) exceeds the available context size "
            "(32768 tokens), try increasing it'}}"
        )
        return InternalServerError(message, response=response, body={"message": message})

    async def _backend(**kwargs: Any) -> Any:
        content = _prompt_content(kwargs)
        if len(content) > 5000:
            raise _real_context_error_as_500()
        return _fake_completion("ok")

    _install_fake_openai(monkeypatch, _backend)

    r = client.post("/config/test", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    steps = {s["step"]: s for s in body["steps"]}
    assert steps["context"]["ok"] is True
    assert "32768" in steps["context"]["detail"]
    assert body["suggestions"]["context_length"] == 32768
    assert body["suggestions"]["single_pass_token_limit"] == int(32768 * 0.6)


def test_config_test_context_inconclusive_with_empty_exception_message_reports_legibly(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`asyncio.wait_for`'s own `TimeoutError()` (our client-side timeout
    firing before the backend responds at all) carries NO message —
    `str(TimeoutError())` is `""` by construction, not a bug losing content
    along the way. The reported detail must still read as a sentence, not
    trail off into a blank after a colon."""
    _patch_models_probe(monkeypatch, response=_FakeHttpResponse(200, {"data": [{"id": "m"}]}))

    async def _backend(**kwargs: Any) -> Any:
        content = _prompt_content(kwargs)
        if len(content) > 5000:
            raise TimeoutError()  # bare — empty str(), exactly like asyncio.wait_for's own
        return _fake_completion("ok")

    _install_fake_openai(monkeypatch, _backend)

    r = client.post("/config/test", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    steps = {s["step"]: s for s in body["steps"]}
    assert steps["context"]["ok"] is None
    detail = steps["context"]["detail"]
    assert detail is not None
    assert not detail.rstrip().endswith(":")
    assert "no error detail available" in detail


def test_config_test_context_bisection_stops_on_inconclusive_attempt_not_fabricate(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The huge probe gets a clean (but unparseable) rejection, so
    bisection starts — but one of the narrowing attempts times out instead
    of cleanly rejecting. That single inconclusive attempt must stop the
    bisection rather than being folded in as "also over the limit": the
    reported number must still be a genuine confirmed lower bound, never
    past the real (fake) ceiling."""
    from openai import BadRequestError

    from src.llm.tokens import count_tokens

    _patch_models_probe(monkeypatch, response=_FakeHttpResponse(200, {"data": [{"id": "m"}]}))

    fake_ceiling = 9000

    def _vague_error() -> BadRequestError:
        request = httpx.Request("POST", "http://example.test/v1/chat/completions")
        response = httpx.Response(400, request=request)
        message = "context window exceeded"
        return BadRequestError(message, response=response, body={"message": message})

    async def _backend(**kwargs: Any) -> Any:
        content = _prompt_content(kwargs)
        if len(content) > 5000:
            tokens = count_tokens(content)
            if tokens > fake_ceiling:
                raise _vague_error()
            # Anything below the real ceiling but still a large probe:
            # simulate a slow backend timing out rather than answering.
            raise TimeoutError("simulated slow backend")
        return _fake_completion("ok")

    _install_fake_openai(monkeypatch, _backend)

    r = client.post("/config/test", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    steps = {s["step"]: s for s in body["steps"]}
    assert steps["context"]["ok"] is True
    assert "inconclusive" in steps["context"]["detail"].lower()
    ceiling_guess = body["suggestions"]["context_length"]
    assert ceiling_guess is not None
    # Never overshoots the real (fake) ceiling — a confirmed lower bound,
    # not a number invented past the point where signal ran out.
    assert ceiling_guess <= fake_ceiling


def test_config_test_context_gets_its_own_dedicated_time_budget(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: on the live run that motivated this, the context probe
    consumed the entire 90s overall budget and the translation step (which
    used to run AFTER it) never got to execute at all. Context now runs
    LAST and gets its own tighter, dedicated deadline computed fresh right
    before it starts — verified here directly via the `deadline` handed to
    `_probe_context_length`, rather than racing real wall-clock time in a
    test."""
    import src.api.config as config_api
    from src.api.schemas import ConfigTestStepResult

    _patch_models_probe(monkeypatch, response=_FakeHttpResponse(200, {"data": [{"id": "m"}]}))

    async def _backend(**kwargs: Any) -> Any:
        content = _prompt_content(kwargs)
        if "You translate transcripts" in content:
            return _fake_completion(_translate_lines(content, lambda marker, rest: f"{marker} RU:{rest}"))
        return _fake_completion("ok")

    _install_fake_openai(monkeypatch, _backend)

    captured: dict[str, float] = {}

    async def _fake_probe_context_length(*, client: Any, model: Any, dialect: Any, deadline: float, api_key: Any) -> Any:
        captured["deadline"] = deadline
        return ConfigTestStepResult(step="context", ok=True, detail="stub"), None

    monkeypatch.setattr(config_api, "_probe_context_length", _fake_probe_context_length)

    before = time.monotonic()
    r = client.post("/config/test", json={})
    assert r.status_code == 200, r.text
    assert "deadline" in captured
    # The own-budget deadline must be MUCH tighter than the ~90s overall
    # ceiling — comfortably bounded by the dedicated budget, not the full
    # test timeout, regardless of how much of the overall budget remained.
    assert captured["deadline"] - before <= config_api._CONTEXT_PROBE_OWN_BUDGET_SECONDS + 2
    assert captured["deadline"] - before < config_api._TOTAL_TEST_TIMEOUT_SECONDS

    # And translation must have completed and been reported — it runs
    # before context in the new order, so it's never gated by context's
    # own budget at all.
    body = r.json()
    steps = {s["step"]: s for s in body["steps"]}
    assert steps["translation"]["ok"] is True


# ---------------------------------------------------------------------------
# POST /config/test — translation contract (step "translation")
# ---------------------------------------------------------------------------


def _translate_lines(content: str, transform: Any) -> str:
    lines = content.rsplit("Input transcript:\n", 1)[-1].splitlines()
    out = []
    for line in lines:
        m = re.match(r"^(\[\d{1,2}:\d{2}(?::\d{2})?\])(.*)$", line)
        out.append(transform(m.group(1), m.group(2)) if m else line)
    return "\n".join(out)


def test_config_test_translation_contract_passes_on_good_translation(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_models_probe(monkeypatch, response=_FakeHttpResponse(200, {"data": [{"id": "m"}]}))

    async def _backend(**kwargs: Any) -> Any:
        content = _prompt_content(kwargs)
        if "You translate transcripts" in content:
            return _fake_completion(_translate_lines(content, lambda marker, rest: f"{marker} RU:{rest}"))
        return _fake_completion("ok")

    _install_fake_openai(monkeypatch, _backend)

    r = client.post("/config/test", json={})
    body = r.json()
    steps = {s["step"]: s for s in body["steps"]}
    assert steps["translation"]["ok"] is True
    assert "first attempt" in steps["translation"]["detail"]


def test_config_test_translation_contract_group_echo_caught_at_whole_group_level(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model that echoes its whole-group input back unchanged (measured
    live — see `.claude/llm.md`: qwen3-1.7b returned 139/139 lines
    byte-identical and the OLD structural-only check accepted it) must NOT
    verify at the whole-group call — that's exactly what `_group_is_echo`
    exists to catch, reused here unmodified via `_align_translation`.

    It recovers to `ok=True` once retried at single-line granularity — a
    documented, ACCEPTED limitation of the reused production logic, not a
    bug in this probe: `.claude/llm.md`'s "Transcript translation" section
    calls this out by name ("a model that echoes UNCONDITIONALLY ... will,
    once bisection reaches single-line granularity, have those lines
    accepted"). This probe reuses the exact same check, so it reproduces
    the exact same limitation — asserted here so it's a documented fact,
    not a silent surprise."""
    _patch_models_probe(monkeypatch, response=_FakeHttpResponse(200, {"data": [{"id": "m"}]}))

    async def _backend(**kwargs: Any) -> Any:
        content = _prompt_content(kwargs)
        if "You translate transcripts" in content:
            # Echo: return the input lines completely unchanged, whether
            # called with the whole group or a single retried line.
            return _fake_completion(content.rsplit("Input transcript:\n", 1)[-1])
        return _fake_completion("ok")

    _install_fake_openai(monkeypatch, _backend)

    r = client.post("/config/test", json={})
    body = r.json()
    steps = {s["step"]: s for s in body["steps"]}
    assert steps["translation"]["ok"] is True
    assert "on retry" in steps["translation"]["detail"]


def test_config_test_translation_contract_fails_on_unusable_output(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model whose output can't be verified even at single-line
    granularity (no recognizable `[MM:SS]` marker at all, so
    `_align_translation` can't match it against the input by value) must be
    reported as a failed contract — the case a real translation would come
    back mostly or entirely stuck in the source language."""
    _patch_models_probe(monkeypatch, response=_FakeHttpResponse(200, {"data": [{"id": "m"}]}))

    async def _backend(**kwargs: Any) -> Any:
        content = _prompt_content(kwargs)
        if "You translate transcripts" in content:
            return _fake_completion("lorem ipsum dolor sit amet, no markers here at all")
        return _fake_completion("ok")

    _install_fake_openai(monkeypatch, _backend)

    r = client.post("/config/test", json={})
    body = r.json()
    steps = {s["step"]: s for s in body["steps"]}
    assert steps["translation"]["ok"] is False
    assert "FAILED" in steps["translation"]["detail"]


def test_config_test_translation_contract_reports_partial(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The combined call misaligns, but every line translates fine when
    retried individually — production would recover via bisection, so this
    must read as "works on retry", not a hard failure."""
    _patch_models_probe(monkeypatch, response=_FakeHttpResponse(200, {"data": [{"id": "m"}]}))

    async def _backend(**kwargs: Any) -> Any:
        content = _prompt_content(kwargs)
        if "You translate transcripts" in content:
            lines = content.rsplit("Input transcript:\n", 1)[-1].splitlines()
            if len(lines) > 1:
                # Whole-group call: scramble the marker order so alignment
                # (which matches markers forward-only, in order) breaks.
                translated = [
                    f"{m.group(1)} RU:{m.group(2)}"
                    for line in lines
                    if (m := re.match(r"^(\[\d{1,2}:\d{2}(?::\d{2})?\])(.*)$", line))
                ]
                return _fake_completion("\n".join(reversed(translated)))
            return _fake_completion(_translate_lines(content, lambda marker, rest: f"{marker} RU:{rest}"))
        return _fake_completion("ok")

    _install_fake_openai(monkeypatch, _backend)

    r = client.post("/config/test", json={})
    body = r.json()
    steps = {s["step"]: s for s in body["steps"]}
    assert steps["translation"]["ok"] is True
    assert "on retry" in steps["translation"]["detail"]


# ---------------------------------------------------------------------------
# Whisper API key parity: GET/PATCH report + persist it exactly like llm,
# with fully independent storage (own keychain service, own key file).
# ---------------------------------------------------------------------------


def test_get_config_reports_whisper_key_state_and_never_leaks_it(
    client: TestClient, tmp_path: Path
) -> None:
    config_file = tmp_path / "tldr.yaml"
    config_file.write_text(
        _minimal_yaml(tmp_path).replace(
            "whisper:\n  base_url: http://127.0.0.1:1240/v1\n  api_key: dummy\n  model: whisper",
            "whisper:\n  base_url: http://127.0.0.1:1240/v1\n  api_key: sk-whisper-CD34\n  model: whisper",
        )
    )
    config_mod.get_config.cache_clear()

    r = client.get("/config")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["whisper"]["api_key_set"] is True
    assert body["whisper"]["api_key_source"] == "inline"
    assert body["whisper"]["api_key_hint"] == "CD34"
    assert "sk-whisper-CD34" not in r.text


def test_get_config_no_whisper_key_configured_reports_none(client: TestClient) -> None:
    r = client.get("/config")
    body = r.json()
    assert body["whisper"]["api_key_set"] is False
    assert body["whisper"]["api_key_source"] == "none"
    assert body["whisper"]["api_key_hint"] is None


def test_patch_whisper_api_key_file_storage_writes_own_key_file(
    client: TestClient, tmp_path: Path
) -> None:
    r = client.patch(
        "/config",
        json={"whisper": {"api_key": "sk-whisper-file-secret", "api_key_storage": "file"}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["whisper"]["api_key_source"] == "file"
    assert "sk-whisper-file-secret" not in r.text

    overrides = yaml.safe_load((tmp_path / "tldr.local.yaml").read_text())
    key_file = Path(overrides["whisper"]["api_key_file"])
    assert key_file.name == "whisper.key"
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    assert key_file.read_text() == "sk-whisper-file-secret"
    # Whisper's key file is separate from llm's — must never collide.
    assert key_file != config_mod.api_key_file_path("llm")


def test_patch_whisper_api_key_storage_isolated_from_llm_keychain_entry(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Patching whisper.api_key via keychain storage must not touch llm's
    keychain entry or override fields, and vice versa."""
    monkeypatch.setattr(config_mod, "keychain_backend_available", lambda: True)
    store = _fake_keyring_module(monkeypatch)

    r = client.patch(
        "/config", json={"llm": {"api_key": "sk-llm-secret", "api_key_storage": "keychain"}}
    )
    assert r.status_code == 200, r.text
    assert store[("tldr-daemon-llm", "api_key")] == "sk-llm-secret"

    r = client.patch(
        "/config",
        json={"whisper": {"api_key": "sk-whisper-secret", "api_key_storage": "keychain"}},
    )
    assert r.status_code == 200, r.text
    assert store[("tldr-daemon-whisper", "api_key")] == "sk-whisper-secret"
    # The LLM entry survives untouched.
    assert store[("tldr-daemon-llm", "api_key")] == "sk-llm-secret"

    overrides = yaml.safe_load((tmp_path / "tldr.local.yaml").read_text())
    assert overrides["llm"]["api_key_keychain"] == "tldr-daemon-llm"
    assert overrides["whisper"]["api_key_keychain"] == "tldr-daemon-whisper"

    body = client.get("/config").json()
    assert body["llm"]["api_key_source"] == "keychain"
    assert body["whisper"]["api_key_source"] == "keychain"


def test_patch_whisper_api_key_storage_isolated_from_llm_key_file(
    client: TestClient, tmp_path: Path
) -> None:
    """Same isolation check for file storage: writing whisper's key file
    must not disturb llm's, and switching whisper away from file storage
    must not delete llm.key."""
    r = client.patch(
        "/config", json={"llm": {"api_key": "sk-llm-file", "api_key_storage": "file"}}
    )
    assert r.status_code == 200, r.text
    llm_key_file = config_mod.api_key_file_path("llm")
    assert llm_key_file.exists()
    assert llm_key_file.read_text() == "sk-llm-file"

    r = client.patch(
        "/config", json={"whisper": {"api_key": "sk-whisper-file", "api_key_storage": "file"}}
    )
    assert r.status_code == 200, r.text
    whisper_key_file = config_mod.api_key_file_path("whisper")
    assert whisper_key_file.exists()
    assert whisper_key_file.read_text() == "sk-whisper-file"
    # llm.key untouched by the whisper PATCH.
    assert llm_key_file.read_text() == "sk-llm-file"

    # Switching whisper's storage away from file must not delete llm.key.
    r = client.patch(
        "/config", json={"whisper": {"api_key": "sk-whisper-inline", "api_key_storage": "inline"}}
    )
    assert r.status_code == 200, r.text
    assert not whisper_key_file.exists()
    assert llm_key_file.exists()
    assert llm_key_file.read_text() == "sk-llm-file"


def test_patch_whisper_api_key_verified_true_on_successful_save(client: TestClient) -> None:
    r = client.patch(
        "/config", json={"whisper": {"api_key": "sk-whisper-verify-ok", "api_key_storage": "file"}}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["whisper_api_key_verified"] is True
    assert body["whisper_api_key_verify_error"] is None
    # The llm-scoped fields are unaffected by a whisper-only PATCH.
    assert body["api_key_verified"] is True
    assert body["api_key_verify_error"] is None
    assert "sk-whisper-verify-ok" not in r.text


def test_patch_whisper_api_key_verified_true_when_key_not_touched(client: TestClient) -> None:
    r = client.patch("/config", json={"output": {"language": "de"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["whisper_api_key_verified"] is True
    assert body["whisper_api_key_verify_error"] is None


def test_patch_whisper_api_key_verified_false_when_readback_mismatches(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        config_mod.WhisperConfig,
        "effective_api_key",
        property(lambda self: "some-other-value"),
    )

    r = client.patch(
        "/config", json={"whisper": {"api_key": "sk-whisper-mismatch", "api_key_storage": "file"}}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["whisper_api_key_verified"] is False
    assert body["whisper_api_key_verify_error"]
    assert "sk-whisper-mismatch" not in r.text
    assert "some-other-value" not in r.text
    # llm verification is untouched by this whisper-only PATCH.
    assert body["api_key_verified"] is True


# ---------------------------------------------------------------------------
# POST /config/test — target="whisper"
# ---------------------------------------------------------------------------


def test_config_test_default_target_still_tests_llm(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a body with no `target` at all must keep testing llm,
    exactly like before `target` was added to the contract."""
    _patch_models_probe(monkeypatch, response=_FakeHttpResponse(200, {"data": [{"id": "test-model"}]}))
    _install_fake_openai(monkeypatch, _default_ok_backend)

    r = client.post("/config/test", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    steps = {s["step"]: s for s in body["steps"]}
    assert steps["completion"]["ok"] is True


def test_config_test_whisper_401_reports_verbatim_detail(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_models_probe(
        monkeypatch, response=_FakeHttpResponse(401, text="Incorrect API key provided: sk-***")
    )
    r = client.post(
        "/config/test",
        json={
            "target": "whisper",
            "whisper": {
                "base_url": "https://api.openai.com/v1",
                "model": "whisper-1",
                "api_key": "sk-whisper-bogus",
            },
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is False
    assert body["step"] == "models"
    assert body["status_code"] == 401
    assert "Incorrect API key provided" in body["detail"]
    assert "sk-whisper-bogus" not in r.text


def test_config_test_whisper_success_only_runs_models_step(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whisper's probe never attempts a completion call (no audio file to
    upload for a real transcription probe) — `ok: true` after just the
    reachability step."""
    _patch_models_probe(
        monkeypatch, response=_FakeHttpResponse(200, {"data": [{"id": "whisper-1"}]})
    )

    r = client.post("/config/test", json={"target": "whisper"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["step"] == "models"
    assert body["models"] == ["whisper-1"]
    assert body["latency_ms"] is not None


def test_config_test_whisper_never_persists_anything(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_models_probe(monkeypatch, response=_FakeHttpResponse(200, {"data": []}))
    r = client.post("/config/test", json={"target": "whisper"})
    assert r.status_code == 200, r.text
    assert not (tmp_path / "tldr.local.yaml").exists()
