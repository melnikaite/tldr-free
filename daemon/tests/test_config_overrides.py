"""The ``tldr.local.yaml`` overrides layer in ``src/config.py``.

``tldr.yaml`` is a hand-edited, comment-heavy template — ``PATCH /config``
(tested separately in ``test_api_config.py``) never writes to it. This file
covers the lower-level plumbing in isolation: deep merge semantics, atomic
+ 0600 writes, env-beats-overrides precedence, and ``validate_full_config``.

Each test points ``TLDR_CONFIG``/``TLDR_CONFIG_OVERRIDES`` at a fresh
``tmp_path`` and clears the ``get_config()`` singleton around itself so
nothing leaks into other test modules that share the same process.
"""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from src import config as config_mod


def _minimal_yaml(data_dir: Path) -> str:
    # `data_dir` is unused by any assertion in this file (it's here only
    # because Config requires a value) — always the test's own tmp_path
    # rather than a shared literal like "/tmp" regardless, on general
    # principle (see the incident writeup in `.claude/ops.md`).
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
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fresh template at tmp_path/tldr.yaml; overrides at the sibling
    tldr.local.yaml (not yet created). Returns the template path."""
    config_file = tmp_path / "tldr.yaml"
    config_file.write_text(_minimal_yaml(tmp_path))
    overrides_file = tmp_path / "tldr.local.yaml"

    monkeypatch.setenv("TLDR_CONFIG", str(config_file))
    monkeypatch.setenv("TLDR_CONFIG_OVERRIDES", str(overrides_file))
    config_mod.get_config.cache_clear()
    yield config_file
    config_mod.get_config.cache_clear()


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_overrides_path_defaults_to_sibling_of_config_path(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TLDR_CONFIG_OVERRIDES", raising=False)
    assert config_mod.overrides_path() == isolated_config.parent / "tldr.local.yaml"


def test_overrides_path_respects_explicit_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = tmp_path / "somewhere-else.yaml"
    monkeypatch.setenv("TLDR_CONFIG_OVERRIDES", str(explicit))
    assert config_mod.overrides_path() == explicit


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------


def test_deep_merge_overlays_nested_keys_without_dropping_siblings() -> None:
    base = {"llm": {"base_url": "http://a", "model": "m", "context_length": 100}}
    overlay = {"llm": {"model": "m2"}}
    merged = config_mod._deep_merge(base, overlay)
    assert merged == {"llm": {"base_url": "http://a", "model": "m2", "context_length": 100}}
    # Base is untouched.
    assert base["llm"]["model"] == "m"


def test_deep_merge_scalar_overlay_replaces_dict_outright() -> None:
    base = {"llm": {"base_url": "http://a"}}
    overlay = {"llm": "not-a-dict-anymore"}
    assert config_mod._deep_merge(base, overlay) == {"llm": "not-a-dict-anymore"}


# ---------------------------------------------------------------------------
# get_config(): template + overrides + env, in that precedence order
# ---------------------------------------------------------------------------


def test_get_config_applies_overrides_on_top_of_template(isolated_config: Path) -> None:
    overrides_file = config_mod.overrides_path()
    overrides_file.write_text(yaml.safe_dump({"output": {"language": "ru"}}))
    config_mod.get_config.cache_clear()

    cfg = config_mod.get_config()
    assert cfg.output.language == "ru"
    # Untouched sibling fields from the template survive the merge.
    assert cfg.llm.model == "test-model"


def test_get_config_works_when_overrides_file_absent(isolated_config: Path) -> None:
    assert not config_mod.overrides_path().is_file()
    cfg = config_mod.get_config()
    assert cfg.output.language == "en"


def test_qa_web_search_defaults_true_when_section_absent(isolated_config: Path) -> None:
    """`_minimal_yaml()` (this file's template, like every existing user's
    real `tldr.yaml` written before this setting existed) has no `qa:`
    section at all — `QaConfig`'s `default_factory` on `Config.qa` must
    still produce `web_search=True` so nobody's behavior silently changes."""
    assert not config_mod.overrides_path().is_file()
    cfg = config_mod.get_config()
    assert cfg.qa.web_search is True


def test_env_override_still_wins_over_local_overrides(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overrides_file = config_mod.overrides_path()
    overrides_file.write_text(yaml.safe_dump({"output": {"language": "ru"}}))
    monkeypatch.setenv("TLDR__OUTPUT__LANGUAGE", "de")
    config_mod.get_config.cache_clear()

    cfg = config_mod.get_config()
    assert cfg.output.language == "de"


# ---------------------------------------------------------------------------
# write_overrides / read_overrides: atomic, 0600, round-trip
# ---------------------------------------------------------------------------


def test_write_overrides_is_0600_and_round_trips(isolated_config: Path) -> None:
    data: dict[str, Any] = {"llm": {"model": "new-model"}}
    config_mod.write_overrides(data)

    path = config_mod.overrides_path()
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600
    assert config_mod.read_overrides() == data


def test_write_overrides_does_not_touch_the_template(isolated_config: Path) -> None:
    original = isolated_config.read_text()
    config_mod.write_overrides({"output": {"language": "fr"}})
    assert isolated_config.read_text() == original


def test_write_api_key_file_is_0600(isolated_config: Path) -> None:
    path = config_mod.write_api_key_file("sk-secret-value")
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600
    assert path.read_text() == "sk-secret-value"


# ---------------------------------------------------------------------------
# validate_full_config: pre-write validation, no cache/disk side effects
# ---------------------------------------------------------------------------


def test_validate_full_config_accepts_valid_overrides(isolated_config: Path) -> None:
    cfg = config_mod.validate_full_config({"output": {"language": "es"}})
    assert cfg.output.language == "es"
    # Doesn't write anything.
    assert not config_mod.overrides_path().is_file()


def test_validate_full_config_rejects_missing_required_field(isolated_config: Path) -> None:
    # Blow away the "storage" section entirely — llm.Config requires it.
    with pytest.raises(ValidationError):
        config_mod.validate_full_config({"storage": "not-a-mapping"})
