"""``LLMConfig.effective_api_key`` resolution + ``ensure_config_file`` perms.

Priority order under test: env var > keychain > file > inline. Each source
is exercised in isolation via freshly-constructed ``LLMConfig`` instances
(not the process-wide cached ``get_config()`` singleton) so tests don't leak
state into each other.
"""

from __future__ import annotations

import logging
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from src.config import LLMConfig, ensure_config_file


def _cfg(**overrides: Any) -> LLMConfig:
    return LLMConfig(base_url="http://localhost:1234/v1", model="test-model", **overrides)


# ---------------------------------------------------------------------------
# Priority order
# ---------------------------------------------------------------------------


def test_inline_api_key_is_the_fallback() -> None:
    cfg = _cfg(api_key="inline-secret")
    assert cfg.effective_api_key == "inline-secret"


def test_file_beats_inline(tmp_path: Path) -> None:
    key_file = tmp_path / "key.txt"
    key_file.write_text("file-secret\n")
    key_file.chmod(0o600)
    cfg = _cfg(api_key="inline-secret", api_key_file=str(key_file))
    assert cfg.effective_api_key == "file-secret"


def test_file_supports_tilde_expansion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    key_file = tmp_path / "key.txt"
    key_file.write_text("home-secret")
    key_file.chmod(0o600)
    cfg = _cfg(api_key_file="~/key.txt")
    assert cfg.effective_api_key == "home-secret"


def test_keychain_beats_file_and_inline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key_file = tmp_path / "key.txt"
    key_file.write_text("file-secret")
    key_file.chmod(0o600)

    fake_keyring = type(
        "FakeKeyring",
        (),
        {"get_password": staticmethod(lambda service, account: f"keychain:{service}:{account}")},
    )()
    monkeypatch.setitem(sys.modules, "keyring", fake_keyring)

    cfg = _cfg(
        api_key="inline-secret",
        api_key_file=str(key_file),
        api_key_keychain="tldr-llm",
        api_key_keychain_account="default",
    )
    assert cfg.effective_api_key == "keychain:tldr-llm:default"


def test_env_beats_everything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    key_file = tmp_path / "key.txt"
    key_file.write_text("file-secret")
    key_file.chmod(0o600)

    fake_keyring = type(
        "FakeKeyring", (), {"get_password": staticmethod(lambda *_: "keychain-secret")}
    )()
    monkeypatch.setitem(sys.modules, "keyring", fake_keyring)
    monkeypatch.setenv("TLDR__LLM__API_KEY", "env-secret")

    cfg = _cfg(
        api_key="inline-secret",
        api_key_file=str(key_file),
        api_key_keychain="tldr-llm",
    )
    assert cfg.effective_api_key == "env-secret"


# ---------------------------------------------------------------------------
# Error / warning cases
# ---------------------------------------------------------------------------


def test_missing_keyring_module_raises_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)  # forces ImportError on `import keyring`
    cfg = _cfg(api_key_keychain="tldr-llm")
    with pytest.raises(RuntimeError, match="keychain"):
        _ = cfg.effective_api_key


def test_keychain_entry_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_keyring = type("FakeKeyring", (), {"get_password": staticmethod(lambda *_: None)})()
    monkeypatch.setitem(sys.modules, "keyring", fake_keyring)
    cfg = _cfg(api_key_keychain="tldr-llm")
    with pytest.raises(RuntimeError, match="tldr-llm"):
        _ = cfg.effective_api_key


def test_missing_api_key_file_raises(tmp_path: Path) -> None:
    cfg = _cfg(api_key_file=str(tmp_path / "does-not-exist.txt"))
    with pytest.raises(RuntimeError, match="does not exist"):
        _ = cfg.effective_api_key


def test_empty_api_key_file_raises(tmp_path: Path) -> None:
    key_file = tmp_path / "key.txt"
    key_file.write_text("   \n")  # blank after strip
    cfg = _cfg(api_key_file=str(key_file))
    with pytest.raises(RuntimeError, match="empty"):
        _ = cfg.effective_api_key


def test_wide_permissions_on_api_key_file_warn_but_dont_fail(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    key_file = tmp_path / "key.txt"
    key_file.write_text("file-secret")
    key_file.chmod(0o644)  # group/world readable
    cfg = _cfg(api_key_file=str(key_file))

    with caplog.at_level(logging.WARNING, logger="src.config"):
        assert cfg.effective_api_key == "file-secret"

    assert any("api_key_file" in r.message and str(key_file) in r.message for r in caplog.records)


def test_tight_permissions_on_api_key_file_dont_warn(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    key_file = tmp_path / "key.txt"
    key_file.write_text("file-secret")
    key_file.chmod(0o600)
    cfg = _cfg(api_key_file=str(key_file))

    with caplog.at_level(logging.WARNING, logger="src.config"):
        assert cfg.effective_api_key == "file-secret"

    assert caplog.records == []


# ---------------------------------------------------------------------------
# ensure_config_file permissions
# ---------------------------------------------------------------------------


def test_ensure_config_file_creates_with_owner_only_permissions(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "tldr.yaml"
    assert ensure_config_file(target) is True
    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o600
