"""GET/PATCH /config, POST /config/test — daemon settings editable from the
extension's options page instead of hand-editing ``tldr.yaml``.

Security: these endpoints accept and resolve secrets (LLM AND Whisper API
keys — see ``src/config.py#_ApiKeyConfigMixin``, shared by both sections).
The daemon only ever binds to 127.0.0.1 and CORS is restricted to
``chrome-extension://*`` (see ``src/main.py``) — do not relax either
without re-reading this module. A key is NEVER returned by ``GET``/``PATCH``
(only ``api_key_set``/``api_key_hint``/``api_key_source``, nested under
``llm``/``whisper`` respectively), never logged, and scrubbed out of any
provider error text before it reaches a response.

Override layer (see ``.claude/daemon.md`` / ``config.py`` docstrings):
``tldr.yaml`` is a hand-edited, comment-heavy template — writing to it with
``yaml.safe_dump`` would destroy those comments. All writes from this
router instead go to ``tldr.local.yaml`` (``config.overrides_path()``),
which ``get_config()`` deep-merges on top of the template before env
overrides are applied. Every PATCH is validated (``Config.model_validate``
via ``config.validate_full_config``) before anything is written to disk.

API key storage (``PATCH /config`` body ``llm.api_key_storage`` /
``whisper.api_key_storage`` — independent per section, same mechanics):

- ``"keychain"`` (default when ``config.keychain_backend_available()`` is
  true) — ``keyring.set_password(...)``; override gets
  ``<section>.api_key_keychain`` / ``<section>.api_key_keychain_account``.
  This is the recommended mode: the daemon itself writes the entry, so it's
  automatically in its own trusted-app ACL and reads it back with zero
  prompts (see ``.claude/llm.md``). 422 with an actionable message if
  ``keyring`` has no usable backend (headless Linux without a Secret
  Service in the session — see ``keychain_available`` below). ``llm`` and
  ``whisper`` use DIFFERENT keychain services (``tldr-daemon-llm`` /
  ``tldr-daemon-whisper``, see ``_KEYCHAIN_SERVICES``) so the two never
  collide even if both happen to hold the same underlying provider key.
- ``"file"`` (default when keychain isn't available) — the key is written
  to ``config.api_key_file_path(section)`` (``llm.key`` / ``whisper.key``,
  mode 0600) and the override points ``<section>.api_key_file`` at it.
  Also the right choice for Docker installs, which have neither macOS
  Keychain nor a Secret Service.
- ``"inline"`` — the key is written directly into the (0600) override file
  as ``<section>.api_key``.

Switching storage cleans up the fields (and best-effort the previous
file/keychain entry) used by the PREVIOUS mode, so only one source of
truth remains — see ``_apply_api_key_storage``. This is scoped per section:
patching ``whisper.api_key`` never touches ``llm``'s keychain entry/file or
vice versa.

``keychain_available`` (``GET``/``PATCH`` response, top level) reflects
``config.keychain_backend_available()`` — a real check that the keyring
backend is usable, not just that the package is importable. It's cached
for the process lifetime (the backend can't change while running) and is
the single place both the default-storage decision (above) and the
extension's options-page UI (disabling the keychain choice) read from.
Shared between ``llm`` and ``whisper`` — the OS keychain is either usable
on this machine or it isn't, regardless of which section is asking.

Write-then-read-back verification: whenever a PATCH writes a new API key,
the response includes ``api_key_verified``/``api_key_verify_error`` (for
``llm``) and ``whisper_api_key_verified``/``whisper_api_key_verify_error``
(for ``whisper``) — the freshly-saved config is read back via the EXACT
accessor the daemon uses at call time (``effective_api_key``, shared by
both sections via ``_ApiKeyConfigMixin``) and compared to what was just
written. A failed verification is reported, never rolled back — see
``_verify_saved_api_key``.

Cache invalidation after a successful PATCH: ``get_config.cache_clear()``
and ``llm_client.reset_caches()`` (client + dialect guess). The LLM
semaphore (``llm.client._llm_lock()``) is intentionally NOT touched — an
``asyncio.Semaphore`` is bound to the running event loop and can't be
resized in place, so a changed ``max_concurrent_calls`` instead reports
``restart_required: true``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import time
from dataclasses import replace as dataclasses_replace
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import APIRouter, HTTPException
from openai import APIStatusError, AsyncOpenAI
from pydantic import ValidationError

from src import config as config_module
from src.api.schemas import (
    ConfigPatchRequest,
    ConfigPatchResponse,
    ConfigResponse,
    ConfigTestLLMOverrides,
    ConfigTestRequest,
    ConfigTestResponse,
    ConfigTestStepResult,
    ConfigTestSuggestions,
    ConfigTestWhisperOverrides,
    LLMConfigOut,
    OutputConfigOut,
    StorageConfigOut,
    WhisperConfigOut,
)
from src.config import Config, _ApiKeyConfigMixin, get_config
from src.llm import client as llm_client
from src.llm.languages import Language, normalize_lang
from src.llm.tokens import make_filler_text

# Reused verbatim, not reimplemented — see the module docstring above and
# `.claude/llm.md`'s "Transcript translation" section: the whole point of
# the `translation` test step is to run the SAME verification the real
# translator applies, so a "test passed" report means what it says. Both
# are underscore-prefixed internals of `workers.translator`, imported
# across the module boundary deliberately (ruff's selected rule set has no
# private-access check, and there's no public wrapper worth adding for two
# functions used exactly as-is).
from src.workers.translator import _align_translation
from src.workers.translator import _load_prompt as _load_translate_prompt

log = logging.getLogger(__name__)

router = APIRouter(prefix="/config", tags=["config"])

# Provider error bodies can be verbose (HTML error pages, stack traces from
# misconfigured proxies); cap what we relay so the response stays sane.
_MAX_DETAIL_CHARS = 2000
# Generous but bounded — remote/cloud backends over a cold connection can be
# slower than local ones; this endpoint is interactive (user clicked "Test").
_TEST_TIMEOUT_SECONDS = 15.0
# Just enough budget for a reasoning model to clear its hidden thinking
# tokens and still emit a couple of visible ones — `max_tokens=1` starves a
# reasoning model before it can respond at all, which reads as a broken
# connection when it's actually a fine one. We only care that the call
# completes without an error, not about the content.
_TEST_COMPLETION_MAX_TOKENS = 16

# ---------------------------------------------------------------------------
# target="llm" step-by-step probe — tunables. See `.claude/llm.md` /
# CLAUDE.md's "Adding features" note for the design rationale.
# ---------------------------------------------------------------------------

# Per-LLM-call timeout for the lightweight steps (completion/translation —
# small prompts, a handful of output tokens).
_LLM_STEP_TIMEOUT_SECONDS = 20.0
# The context-length probe deliberately sends a large prompt (prefill of
# tens of thousands of tokens) — that's inherently slower than the other
# steps even on a fast backend, and slower still on a CPU-bound one.
_CONTEXT_PROBE_TIMEOUT_SECONDS = 30.0
# Hard ceiling on the WHOLE multi-step run — a slow/hanging backend must
# not leave the "Test setup" button spinning forever.
_TOTAL_TEST_TIMEOUT_SECONDS = 90.0
# Context gets its OWN dedicated time allocation, separate from (and never
# more than) whatever's left of the total above. Measured live: on a real
# LocalAI backend serving gemma-4-e4b, the context probe alone consumed the
# ENTIRE 90s budget (4 attempts, ~20s each) and starved the translation
# step that ran after it in the old step order — the step that actually
# answers "will my model work," left unattempted. Context now runs LAST
# (see the step order below) specifically so it can never again preempt a
# cheaper, more decisive step; this budget additionally stops it from
# running away on its own for the full 90s even in isolation.
_CONTEXT_PROBE_OWN_BUDGET_SECONDS = 60.0

# Deliberately-oversized prompt for the context-length probe. Chosen large
# enough to exceed every LOCAL default in config/tldr.yaml.example (32768-
# 131072) and so catch the common misconfiguration (declared context bigger
# than what's actually served), while staying far short of a multi-hundred-
# thousand-token cloud context — pushing a prefill that size on every "Test
# setup" click isn't worth it just to shave the last digits off a value that
# was very likely fine already. If this succeeds outright, we report "at
# least this many tokens" rather than hunting further.
_CONTEXT_PROBE_HUGE_TOKENS = 40_000
# Assumed-safe lower bound for the bisection fallback (see
# `_probe_context_length`) — step 3 (completion) already proved a ~10-token
# prompt works, so a few thousand tokens is a safe starting floor for any
# real backend.
_CONTEXT_PROBE_FLOOR_TOKENS = 2_000
# Bisection stops once the good/bad window is this narrow — no point
# chasing single-token precision out of a backend's own approximate report.
_CONTEXT_PROBE_TOLERANCE_TOKENS = 2_000
# Hard cap on LLM calls spent probing context (the huge probe counts as
# one), independent of the time budget above.
_CONTEXT_PROBE_MAX_ATTEMPTS = 6

# `single_pass_token_limit` suggestion ratio — matches the "~60% of
# context_length" comment attached to every backend example in
# config/tldr.yaml.example / daemon/src/assets/tldr.yaml.example.
_SINGLE_PASS_RATIO = 0.6

# Matches "<number> tokens" anywhere in a provider's context-length error
# text — e.g. "request (60009 tokens) exceeds the available context size
# (32768 tokens)" or "maximum context length is 32768 tokens... requested
# 60009 tokens". With 2+ matches, the SMALLER number is the real ceiling in
# both phrasings above (the probe is deliberately sized to exceed it), so
# no per-backend wording needs hardcoding.
_CONTEXT_ERROR_TOKEN_RE = re.compile(r"(\d[\d,]*)\s*tokens?", re.IGNORECASE)

# Fixed target language for the translation-contract probe (step 6) — not
# user-configurable. This step verifies a MECHANISM (line-for-line
# alignment via `workers.translator._align_translation`), not the user's
# real translation preference, so any well-supported, distinctly-non-English
# target works; Russian is the language this exact check was validated
# against historically (see `.claude/llm.md`'s "Transcript translation").
_TRANSLATION_PROBE_TARGET_LANG_CODE = "ru"
_TRANSLATION_PROBE_SOURCE_LANG_CODE = "en"
_TRANSLATION_PROBE_LINES = [
    "[00:01] This tool summarizes web pages, videos, and PDFs on your own machine.",
    "[00:05] It can also answer follow-up questions about anything it has processed.",
    "[00:10] Every request stays local unless you point it at a cloud backend yourself.",
    "[00:14] This line just checks that the translation contract still works end to end.",
]

# Separate keychain service per section so an llm key and a whisper key
# (even against the same cloud provider) never collide in the OS keychain.
_KEYCHAIN_SERVICES: dict[Literal["llm", "whisper"], str] = {
    "llm": "tldr-daemon-llm",
    "whisper": "tldr-daemon-whisper",
}
_KEYCHAIN_ACCOUNT = "api_key"


# ---------------------------------------------------------------------------
# GET /config — read-only view, secrets redacted to presence/hint/source
# ---------------------------------------------------------------------------


def _api_key_source(cfg: _ApiKeyConfigMixin) -> str:
    """Which of the four mechanisms will actually supply the key, in the
    same priority order as ``effective_api_key`` — for whichever section
    ``cfg`` belongs to (``cfg._section``)."""
    if os.environ.get(f"TLDR__{cfg._section.upper()}__API_KEY"):
        return "env"
    if cfg.api_key_keychain:
        return "keychain"
    if cfg.api_key_file:
        return "file"
    if cfg.api_key and cfg.api_key != "dummy":
        return "inline"
    return "none"


def _api_key_hint(cfg: _ApiKeyConfigMixin) -> str | None:
    """Last 4 characters of the resolved key, or None if unset/unresolvable.
    Never raises — a misconfigured file/keychain just means no hint."""
    try:
        key = cfg.effective_api_key
    except Exception:
        return None
    return key[-4:] if key else None


def _to_response(cfg: Config) -> ConfigResponse:
    llm_source = _api_key_source(cfg.llm)
    llm_hint = _api_key_hint(cfg.llm) if llm_source != "none" else None
    whisper_source = _api_key_source(cfg.whisper)
    whisper_hint = _api_key_hint(cfg.whisper) if whisper_source != "none" else None
    return ConfigResponse(
        llm=LLMConfigOut(
            base_url=cfg.llm.base_url,
            model=cfg.llm.model,
            context_length=cfg.llm.context_length,
            single_pass_token_limit=cfg.llm.single_pass_token_limit,
            max_concurrent_calls=cfg.llm.max_concurrent_calls,
            reasoning_effort=cfg.llm.reasoning_effort,
            api_key_set=llm_source != "none",
            api_key_hint=llm_hint,
            api_key_source=llm_source,  # type: ignore[arg-type]
        ),
        whisper=WhisperConfigOut(
            base_url=cfg.whisper.base_url,
            model=cfg.whisper.model,
            max_upload_mb=cfg.whisper.max_upload_mb,
            api_key_set=whisper_source != "none",
            api_key_hint=whisper_hint,
            api_key_source=whisper_source,  # type: ignore[arg-type]
        ),
        output=OutputConfigOut(language=cfg.output.language),
        storage=StorageConfigOut(retention_days=cfg.storage.retention_days),
        config_path=str(config_module.config_path()),
        overrides_path=str(config_module.overrides_path()),
        keychain_available=config_module.keychain_backend_available(),
    )


@router.get("", response_model=ConfigResponse)
async def get_config_route() -> ConfigResponse:
    return _to_response(get_config())


# ---------------------------------------------------------------------------
# PATCH /config — partial update, written to the overrides file
# ---------------------------------------------------------------------------


def _apply_api_key_storage(
    section_overrides: dict[str, Any],
    storage: str,
    key: str,
    *,
    section: Literal["llm", "whisper"],
) -> None:
    """Point ``section_overrides`` (the ``llm`` or ``whisper`` sub-dict of
    the overrides file) at ``key`` via ``storage``, clearing whatever fields
    the PREVIOUS mode used so exactly one source of truth remains. Raises
    ``HTTPException(422)`` for a bad/unsupported storage choice.

    ``section`` picks the keychain service (``_KEYCHAIN_SERVICES``) and the
    managed key file (``config.api_key_file_path(section)``) so ``llm`` and
    ``whisper`` never share or clobber each other's storage."""
    keychain_service = _KEYCHAIN_SERVICES[section]

    old_file = section_overrides.get("api_key_file")
    old_keychain = section_overrides.get("api_key_keychain")
    old_keychain_account = section_overrides.get("api_key_keychain_account", _KEYCHAIN_ACCOUNT)

    for field in ("api_key", "api_key_file", "api_key_keychain", "api_key_keychain_account"):
        section_overrides.pop(field, None)

    if storage == "inline":
        section_overrides["api_key"] = key
    elif storage == "file":
        path = config_module.write_api_key_file(key, section)
        section_overrides["api_key_file"] = str(path)
    elif storage == "keychain":
        if not config_module.keychain_backend_available():
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{section}.api_key_storage='keychain' requires a usable OS keychain "
                    "backend. On macOS/Windows this should always be available; on "
                    "Linux it needs a Secret Service (GNOME Keyring / KWallet) "
                    "running in the current session. Use api_key_storage='file' "
                    "instead (recommended for Docker installs and headless setups)."
                ),
            )
        import keyring

        try:
            keyring.set_password(keychain_service, _KEYCHAIN_ACCOUNT, key)
        except Exception as e:
            # A usable backend (checked above) doesn't guarantee the WRITE
            # succeeds — e.g. a stale entry left by a previous install with
            # an ACL that doesn't trust the current binary (a Python
            # minor-version bump invalidates it) can make the OS keychain
            # refuse the write outright rather than prompt, since this
            # runs headless (no session to show a permission dialog to).
            # Surface that as an actionable 422 instead of a raw 500.
            raise HTTPException(
                status_code=422,
                detail=(
                    "Failed to write the API key to the OS keychain "
                    f"(service={keychain_service!r}): {_redact(str(e), key)}. If a "
                    "previous version left a stale entry with an incompatible "
                    "ACL, remove it (e.g. `security delete-generic-password -s "
                    f"{keychain_service} -a {_KEYCHAIN_ACCOUNT}` on macOS) and retry, "
                    "or use api_key_storage='file' instead."
                ),
            ) from e
        section_overrides["api_key_keychain"] = keychain_service
        section_overrides["api_key_keychain_account"] = _KEYCHAIN_ACCOUNT
    else:
        raise HTTPException(
            status_code=422, detail=f"Unknown {section}.api_key_storage {storage!r}"
        )

    # Best-effort cleanup of whatever the previous mode left behind. Never
    # let this block the actual (already-applied) storage change. Only ever
    # delete `old_file` when it's OUR managed key file
    # (api_key_file_path(section)): `<section>.api_key_file` is a plain
    # user-editable config field, so it could instead point at a file the
    # user manages themselves (e.g. a `~/.config/openai.key` shared with
    # other tools) — silently deleting that on a storage-mode switch would
    # be a nasty surprise unrelated to what the user asked for.
    if storage != "file" and old_file:
        try:
            is_ours = Path(old_file).resolve() == config_module.api_key_file_path(section).resolve()
        except OSError:
            is_ours = False
        if is_ours:
            with contextlib.suppress(OSError):
                Path(old_file).unlink()
        else:
            log.info(
                "%s.api_key_storage changed away from 'file' but api_key_file %r is not "
                "our managed key file — leaving it in place",
                section,
                old_file,
            )
    if storage != "keychain" and old_keychain:
        with contextlib.suppress(Exception):
            import keyring

            keyring.delete_password(old_keychain, old_keychain_account)


@router.patch("", response_model=ConfigPatchResponse)
async def patch_config_route(body: ConfigPatchRequest) -> ConfigPatchResponse:
    old_cfg = get_config()
    old_max_concurrent_calls = old_cfg.llm.max_concurrent_calls

    overrides = config_module.read_overrides()
    llm_overrides = dict(overrides.get("llm") or {})
    whisper_overrides = dict(overrides.get("whisper") or {})
    output_overrides = dict(overrides.get("output") or {})
    storage_overrides = dict(overrides.get("storage") or {})

    # Set only when this PATCH actually (re)writes the API key for that
    # section — that's the one case worth a write-then-read-back check (see
    # `_verify_saved_api_key`). Left None otherwise.
    saved_llm_key: str | None = None
    saved_whisper_key: str | None = None

    if body.llm is not None:
        simple_fields = body.llm.model_dump(
            exclude={"api_key", "api_key_storage"}, exclude_unset=True
        )
        llm_overrides.update(simple_fields)

        if body.llm.api_key or body.llm.api_key_storage:
            storage = body.llm.api_key_storage or (
                "keychain" if config_module.keychain_backend_available() else "file"
            )
            new_key = body.llm.api_key
            if not new_key:
                # Storage changed but no new key given — migrate the
                # CURRENTLY effective key to the new location rather than
                # forcing the user to retype it.
                try:
                    new_key = old_cfg.llm.effective_api_key
                except Exception as e:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "llm.api_key_storage changed but no new llm.api_key was "
                            f"given, and the currently configured key could not be "
                            f"resolved to migrate it: {e}"
                        ),
                    ) from e
            _apply_api_key_storage(llm_overrides, storage, new_key, section="llm")
            saved_llm_key = new_key

    if body.whisper is not None:
        simple_fields = body.whisper.model_dump(
            exclude={"api_key", "api_key_storage"}, exclude_unset=True
        )
        whisper_overrides.update(simple_fields)

        if body.whisper.api_key or body.whisper.api_key_storage:
            storage = body.whisper.api_key_storage or (
                "keychain" if config_module.keychain_backend_available() else "file"
            )
            new_key = body.whisper.api_key
            if not new_key:
                # Storage changed but no new key given — migrate the
                # CURRENTLY effective key to the new location rather than
                # forcing the user to retype it.
                try:
                    new_key = old_cfg.whisper.effective_api_key
                except Exception as e:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "whisper.api_key_storage changed but no new whisper.api_key "
                            f"was given, and the currently configured key could not be "
                            f"resolved to migrate it: {e}"
                        ),
                    ) from e
            _apply_api_key_storage(whisper_overrides, storage, new_key, section="whisper")
            saved_whisper_key = new_key

    if body.output is not None:
        output_overrides.update(body.output.model_dump(exclude_unset=True))

    if body.storage is not None:
        storage_overrides.update(body.storage.model_dump(exclude_unset=True))

    new_overrides: dict[str, Any] = dict(overrides)
    for section_name, section_val in (
        ("llm", llm_overrides),
        ("whisper", whisper_overrides),
        ("output", output_overrides),
        ("storage", storage_overrides),
    ):
        if section_val:
            new_overrides[section_name] = section_val
        else:
            new_overrides.pop(section_name, None)

    try:
        new_cfg = config_module.validate_full_config(new_overrides)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    config_module.write_overrides(new_overrides)

    # Invalidate the process-wide singletons so the NEXT call picks up the
    # new file — the semaphore sized by max_concurrent_calls is the one
    # exception (see module docstring / restart_required below).
    get_config.cache_clear()
    llm_client.reset_caches()

    restart_required = new_cfg.llm.max_concurrent_calls != old_max_concurrent_calls
    api_key_verified, api_key_verify_error = _verify_saved_api_key(new_cfg.llm, saved_llm_key)
    whisper_api_key_verified, whisper_api_key_verify_error = _verify_saved_api_key(
        new_cfg.whisper, saved_whisper_key
    )

    response = _to_response(new_cfg)
    return ConfigPatchResponse(
        **response.model_dump(),
        restart_required=restart_required,
        api_key_verified=api_key_verified,
        api_key_verify_error=api_key_verify_error,
        whisper_api_key_verified=whisper_api_key_verified,
        whisper_api_key_verify_error=whisper_api_key_verify_error,
    )


def _verify_saved_api_key(
    cfg: _ApiKeyConfigMixin, saved_key: str | None
) -> tuple[bool, str | None]:
    """Read the just-saved API key back using the EXACT accessor the daemon
    uses at call time (``effective_api_key``, shared by ``LLMConfig`` and
    ``WhisperConfig`` via ``_ApiKeyConfigMixin``) and compare it to what was
    just written, so a broken storage choice (e.g. a keychain entry the
    daemon can't read back, a bad file path) is caught and reported
    immediately rather than surfacing as a mysterious 401 the next time a
    call is made.

    Never rolls back the save — by the time this runs, ``write_overrides``
    already succeeded and IS the source of truth; this only decides what
    to report. ``saved_key`` is None when this PATCH didn't touch this
    section's API key at all, in which case there's nothing to verify.

    Never lets the actual key value leak into the returned error string.
    """
    if saved_key is None:
        return True, None
    try:
        resolved = cfg.effective_api_key
    except Exception as e:
        return False, _redact(str(e), saved_key)
    if resolved != saved_key:
        return False, "Key read back from storage does not match the value just saved."
    return True, None


# ---------------------------------------------------------------------------
# POST /config/test — probe credentials without saving. Always 200: a
# 401/timeout/etc. IS the answer this endpoint exists to report.
# ---------------------------------------------------------------------------


def _redact(text: str, secret: str | None) -> str:
    if secret:
        text = text.replace(secret, "***")
    return text[:_MAX_DETAIL_CHARS]


# Fixed step order for target="llm" — every report lists all six, in this
# order, even when later ones are `ok=None` (never attempted). Keeping the
# order as one tuple (rather than repeating six literal strings at each
# call site) is what makes `_fill_remaining_as_skipped` a single loop.
_LLMTestStep = Literal["reachable", "models", "completion", "thinking", "context", "translation"]

# Order steps actually run in (and the order `_fill_remaining_as_skipped`
# fills in on an early exit). "thinking" and "translation" are computed
# TOGETHER from one call (see `_probe_translation_and_thinking`) — thinking
# detection needs a realistic, rule-heavy prompt to trigger Gemma 4's
# adaptive thinking at all (a trivial "reply with one word" prompt measured
# live as a false negative: 0 reasoning tokens on the trivial prompt, 1789
# chars of `reasoning` and truncated content on the real translation
# prompt), so it piggybacks on the translation probe's own call instead of
# spending a separate one. "context" runs LAST, deliberately: it's the
# slowest, least decisive step (see `_CONTEXT_PROBE_OWN_BUDGET_SECONDS`),
# so exhausting its budget can never again starve the cheap, decisive
# translation check the way it did before this ordering existed.
_LLM_STEP_ORDER: tuple[_LLMTestStep, ...] = (
    "reachable", "models", "completion", "thinking", "translation", "context",
)


def _fill_remaining_as_skipped(
    steps: list[ConfigTestStepResult], reason: str
) -> list[ConfigTestStepResult]:
    """Append `ok=None` entries for every step in `_LLM_STEP_ORDER` not
    already present in `steps` — so a report that stops early (an earlier
    step failed, or the time budget ran out) still lists all six steps
    rather than silently truncating the array."""
    done = {s.step for s in steps}
    for name in _LLM_STEP_ORDER:
        if name not in done:
            steps.append(ConfigTestStepResult(step=name, ok=None, detail=reason))
    return steps


def _reasoning_text(message: Any) -> str | None:
    """Best-effort extraction of hidden chain-of-thought from a chat
    completion message — the two field names actually seen in the wild
    (see `.claude/llm.md` / tldr.yaml.example): LocalAI/llama.cpp report
    `reasoning`, mlx-openai-server/mlx_vlm report `reasoning_content`.
    Neither is part of the OpenAI SDK's declared schema, but its pydantic
    models allow extra fields, so both plain attribute access and
    `model_extra` see them — checked both ways to also work against a
    plain test double that isn't a real SDK object at all."""
    for attr in ("reasoning", "reasoning_content"):
        val = getattr(message, attr, None)
        if val:
            return str(val)
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        for key in ("reasoning", "reasoning_content"):
            val = extra.get(key)
            if val:
                return str(val)
    return None


def _parse_context_from_error(text: str) -> int | None:
    """Extract a backend's real context ceiling from its own error message.

    Both observed phrasings mention the requested size and the ceiling as
    "<N> tokens" — the requested size is always the LARGER of the two
    (the probe is deliberately oversized), so the smaller of any 2+ matches
    is the ceiling, with no per-backend wording to hardcode. Returns None
    if the message doesn't contain at least two such numbers (the caller
    falls back to bisection)."""
    nums = [int(m.group(1).replace(",", "")) for m in _CONTEXT_ERROR_TOKEN_RE.finditer(text)]
    if len(nums) < 2:
        return None
    return min(nums)


async def _call_test_model(
    *,
    client: AsyncOpenAI,
    model: str,
    dialect: llm_client._Dialect,
    content: str,
    max_tokens: int,
    timeout: float,
) -> Any:
    """One throwaway chat completion against `client`/`model`/`dialect`,
    through the same dialect-adaptation logic as the real call path
    (`llm_client.call_with_dialect_adaptation`) but bounded by `timeout` —
    this endpoint is interactive and must never hang the UI on a stalled
    backend. Raises whatever `call_with_dialect_adaptation` raises, plus
    `TimeoutError` if the deadline is hit."""
    return await asyncio.wait_for(
        llm_client.call_with_dialect_adaptation(
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=False,
            client=client,
            model=model,
            dialect=dialect,
        ),
        timeout=timeout,
    )


# Outcome of one context-probe attempt:
# - "ok"           — the call succeeded outright.
# - "rejected"      — the error TEXT reads as a context/token-size
#                     complaint (see `_looks_like_context_overflow`).
#                     Treated as real evidence the size was too big —
#                     regardless of HTTP status code (see below).
# - "inconclusive"  — anything else: a timeout, a connection drop, a 5xx/4xx
#                     whose message doesn't mention size at all. None of
#                     these tell us the request was rejected FOR ITS SIZE —
#                     a slow CPU-bound backend just taking a long time to
#                     prefill a huge prompt looks identical to "hit the
#                     wall" from here if you only check "did it raise." An
#                     inconclusive result must never move `hi` down — it
#                     just stops the probe from claiming more than it
#                     actually knows.
#
# Classification is by MESSAGE CONTENT, not status code. Measured live: the
# exact same underlying failure — LocalAI relaying "request (40008 tokens)
# exceeds the available context size (32768 tokens)" from the backend it
# proxies — arrived as an HTTP 500, not 400. An earlier version of this
# classifier keyed off `status_code == 400` and threw away that message
# (and its number) as "inconclusive" purely because of the wrapping status
# code, on the one backend where the signal was perfect. Status code is
# gateway/proxy-dependent; the message text is where the actual evidence
# lives, so that's what's inspected now, whatever the status code says.
_ContextProbeOutcome = Literal["ok", "rejected", "inconclusive"]

# A context/token-size complaint mentions the SUBJECT (context/tokens) and
# an OVERFLOW verb (exceeds/too many/maximum/limit) — both observed live
# phrasings match ("request (N tokens) exceeds the available context size
# (N tokens)", "maximum context length is N tokens... you requested N").
# Deliberately independent of status code and of whether a number is
# present at all: a message like "context window exceeded" (no numbers)
# still IS a size rejection, just one `_parse_context_from_error` can't
# extract an exact ceiling from — that's what the bisection fallback below
# is for. Only a message with NEITHER half of this signature is treated as
# unrelated to size (a timeout, a dropped connection, an unrelated 5xx).
_CONTEXT_OVERFLOW_SUBJECT_RE = re.compile(r"\bcontext\b|\btokens?\b", re.IGNORECASE)
_CONTEXT_OVERFLOW_VERB_RE = re.compile(
    r"exceed\w*|too (?:many|long|large)|maximum|\blimit\b", re.IGNORECASE
)


def _looks_like_context_overflow(text: str) -> bool:
    """True when `text` reads as a context/token-size complaint — see the
    regexes above for what "reads as" means. This is the ONLY thing that
    makes a failed probe attempt "rejected" rather than "inconclusive"."""
    return bool(
        _CONTEXT_OVERFLOW_SUBJECT_RE.search(text) and _CONTEXT_OVERFLOW_VERB_RE.search(text)
    )


def _describe_probe_failure(text: str, api_key: str | None) -> str:
    """Redact + present a probe failure's error text, substituting a
    legible fallback when `text` is empty rather than reporting a blank.
    Genuinely empty is common here, not a bug to chase further:
    `asyncio.wait_for`'s own `TimeoutError()` (our client-side timeout
    firing before the backend responds at all) carries no message
    whatsoever — `str(TimeoutError())` is `""` by construction."""
    redacted = _redact(text, api_key) if text else ""
    return redacted or "(no error detail available — likely a timeout or a dropped connection)"


async def _probe_context_length(
    *,
    client: AsyncOpenAI,
    model: str,
    dialect: llm_client._Dialect,
    deadline: float,
    api_key: str | None,
) -> tuple[ConfigTestStepResult, int | None]:
    """Find the backend's REAL context ceiling by provoking it — the last
    step to run (see `_LLM_STEP_ORDER`) and bounded by `deadline` (the
    caller passes the tighter of the overall test deadline and this step's
    own dedicated budget, so it can never again starve an earlier step).

    Primary path: one deliberately oversized request
    (`_CONTEXT_PROBE_HUGE_TOKENS`); if the backend cleanly REJECTS it (a 4xx
    — see `_ContextProbeOutcome`), parse the ceiling straight out of its own
    error text (`_parse_context_from_error`). Only when that parse fails do
    we fall back to bisecting between a known-good floor and the known-bad
    huge size — bounded by `_CONTEXT_PROBE_MAX_ATTEMPTS` and `deadline`.

    If the huge probe doesn't fail at all, the backend's context is at
    least that large and we stop there rather than pushing further (see
    `_CONTEXT_PROBE_HUGE_TOKENS`'s docstring for why).

    If the huge probe — or, mid-bisection, every remaining attempt — comes
    back INCONCLUSIVE rather than a clean rejection, we do not fabricate a
    number: better to report `ok=None` (or narrow only as far as confirmed
    signals actually reach) than to hand back a value that quietly breaks
    real jobs later, the way a wrong `context_length` already has.
    """
    attempts = 0

    async def _attempt(token_count: int) -> tuple[_ContextProbeOutcome, str | None]:
        nonlocal attempts
        attempts += 1
        prompt = make_filler_text(token_count)
        try:
            await _call_test_model(
                client=client,
                model=model,
                dialect=dialect,
                content=prompt,
                max_tokens=_TEST_COMPLETION_MAX_TOKENS,
                timeout=_CONTEXT_PROBE_TIMEOUT_SECONDS,
            )
            return "ok", None
        except Exception as e:
            # Content decides "rejected" vs "inconclusive", not the
            # exception TYPE or status code — see `_looks_like_context_overflow`.
            # `str(e)` can genuinely be empty (e.g. `asyncio.wait_for`'s own
            # `TimeoutError()` on our client-side timeout carries no message
            # at all) — that's still correctly "inconclusive" (no signature
            # found in ""), just reported with a friendlier fallback string
            # by the caller rather than a blank.
            text = str(e)
            if _looks_like_context_overflow(text):
                return "rejected", text
            return "inconclusive", text

    outcome, err = await _attempt(_CONTEXT_PROBE_HUGE_TOKENS)
    if outcome == "ok":
        return (
            ConfigTestStepResult(
                step="context",
                ok=True,
                detail=(
                    f"No error at {_CONTEXT_PROBE_HUGE_TOKENS} tokens — the backend's real "
                    "context is at least that large. Not pushed further (see design notes); "
                    "no change suggested."
                ),
            ),
            None,
        )
    if outcome == "inconclusive":
        assert err is not None  # "inconclusive" always sets err (possibly "") — see _attempt
        return (
            ConfigTestStepResult(
                step="context",
                ok=None,
                detail=(
                    f"Could not determine the real context: the {_CONTEXT_PROBE_HUGE_TOKENS}-"
                    "token probe failed without a recognizable size complaint in the error "
                    f"text: {_describe_probe_failure(err, api_key)}. Reporting unknown rather "
                    "than guessing — this backend/model may just be slow with large prompts."
                ),
            ),
            None,
        )

    assert err is not None  # only "rejected" reaches here, which always sets err
    parsed = _parse_context_from_error(_redact(err, api_key))
    if parsed is not None:
        return (
            ConfigTestStepResult(
                step="context",
                ok=True,
                detail=f"Backend reports a real context of {parsed} tokens.",
            ),
            parsed,
        )

    # Fallback: bisect between a known-good floor and the known-bad huge
    # size. `lo` only ever moves up on a CONFIRMED "ok", `hi` only ever
    # moves down on a CONFIRMED "rejected" — an "inconclusive" attempt
    # moves neither and stops the loop outright, so whatever `lo` ends up
    # at is always a genuine lower bound, never a guess dressed up as one.
    lo, hi = _CONTEXT_PROBE_FLOOR_TOKENS, _CONTEXT_PROBE_HUGE_TOKENS
    stopped_early = False
    while (
        attempts < _CONTEXT_PROBE_MAX_ATTEMPTS
        and hi - lo > _CONTEXT_PROBE_TOLERANCE_TOKENS
        and time.monotonic() < deadline
    ):
        mid = (lo + hi) // 2
        mid_outcome, _ = await _attempt(mid)
        if mid_outcome == "ok":
            lo = mid
        elif mid_outcome == "rejected":
            hi = mid
        else:
            stopped_early = True
            break
    caveat = (
        " (stopped early — a later attempt was inconclusive rather than a clean rejection, "
        "so this doesn't narrow any further)"
        if stopped_early
        else ""
    )
    return (
        ConfigTestStepResult(
            step="context",
            ok=True,
            detail=(
                "The backend's error message didn't state a number, so we narrowed it by "
                f"trial and error: approximately {lo} tokens (in {attempts} attempt(s)){caveat}."
            ),
        ),
        lo,
    )


async def _probe_translation_and_thinking(
    *,
    client: AsyncOpenAI,
    model: str,
    dialect: llm_client._Dialect,
    api_key: str | None,
) -> tuple[ConfigTestStepResult, ConfigTestStepResult, llm_client._Dialect, str | None]:
    """Runs the REAL translation prompt on a small fixed probe, and gets
    BOTH the "thinking" and "translation" step results out of that single
    call — deliberately not two separate calls.

    Why thinking detection lives here instead of its own trivial call: a
    trivial "reply with the single word ok" prompt is a FALSE NEGATIVE for
    Gemma 4's adaptive thinking — measured live: 0 reasoning on the trivial
    prompt, but 1789 chars of `reasoning` and truncated content (3 of 4
    lines) on this exact translation prompt with `reasoning_effort` unset.
    Gemma 4 only thinks when the prompt actually has rules/a contract to
    reason about, so detecting on the SAME real, rule-heavy prompt this
    step already sends is both more accurate and strictly cheaper than a
    dedicated call.

    Verifies with the SAME code the production translator uses
    (`workers.translator._align_translation`, which itself embeds
    `_group_is_echo` — see that module and `.claude/llm.md`), rather than
    re-deriving what "a working translation" means here.

    Returns ``(thinking_step, translation_step, effective_dialect,
    suggested_reasoning_effort)``. ``effective_dialect`` is the dialect the
    CONTEXT step should use afterward: unchanged unless thinking was
    detected AND a `reasoning_effort="none"` retry confirmed it fixes the
    model — otherwise the context probe would inherit the same
    empty/polluted-content problem this step exists to catch.
    """
    lang = normalize_lang(_TRANSLATION_PROBE_TARGET_LANG_CODE)
    prompt_template = _load_translate_prompt()

    async def _translate(lines: list[str], call_dialect: llm_client._Dialect) -> tuple[list[str], Any]:
        prompt = prompt_template.format(
            target_language_name=lang.english_name, transcript="\n".join(lines),
        )
        response = await _call_test_model(
            client=client,
            model=model,
            dialect=call_dialect,
            content=prompt,
            max_tokens=400,
            timeout=_LLM_STEP_TIMEOUT_SECONDS,
        )
        message = response.choices[0].message
        content = message.content or ""
        return content.split("\n"), message

    try:
        whole_output, message = await _translate(_TRANSLATION_PROBE_LINES, dialect)
    except Exception as e:
        detail = f"Translation call failed: {_redact(str(e), api_key)}"
        return (
            ConfigTestStepResult(
                step="thinking", ok=None, detail="not attempted: translation call failed",
            ),
            ConfigTestStepResult(step="translation", ok=False, detail=detail),
            dialect,
            None,
        )

    content_empty = not (getattr(message, "content", None) or "").strip()
    reasoning_val = _reasoning_text(message)
    effective_dialect = dialect
    suggested_effort: str | None = None

    if reasoning_val or content_empty:
        fix_dialect = dataclasses_replace(dialect, reasoning_effort="none")
        try:
            fixed_output, fixed_message = await _translate(_TRANSLATION_PROBE_LINES, fix_dialect)
            fixed_content_empty = not (getattr(fixed_message, "content", None) or "").strip()
            fixed_reasoning = _reasoning_text(fixed_message)
            if not fixed_reasoning and not fixed_content_empty:
                thinking_step = ConfigTestStepResult(
                    step="thinking",
                    ok=True,
                    detail=(
                        "Thinking detected on the real translation prompt (reasoning content "
                        "and/or a truncated/empty response) — reasoning_effort='none' fixes it. "
                        "Suggested."
                    ),
                )
                suggested_effort = "none"
                effective_dialect = fix_dialect
                whole_output = fixed_output
            else:
                thinking_step = ConfigTestStepResult(
                    step="thinking",
                    ok=False,
                    detail=(
                        "Thinking detected on the real translation prompt, and "
                        "reasoning_effort='none' did not fix it — this backend/model may need "
                        "a different value, or doesn't support the field at all."
                    ),
                )
        except Exception as e:
            thinking_step = ConfigTestStepResult(
                step="thinking",
                ok=False,
                detail=(
                    "Thinking detected on the real translation prompt; the "
                    f"reasoning_effort='none' retry failed: {_redact(str(e), api_key)}"
                ),
            )
    else:
        thinking_step = ConfigTestStepResult(
            step="thinking", ok=True, detail="No thinking/reasoning detected.",
        )

    translation_step = await _verify_translation_output(
        whole_output,
        lang=lang,
        client=client,
        model=model,
        dialect=effective_dialect,
    )
    return thinking_step, translation_step, effective_dialect, suggested_effort


async def _verify_translation_output(
    whole_output: list[str],
    *,
    lang: Language,
    client: AsyncOpenAI,
    model: str,
    dialect: llm_client._Dialect,
) -> ConfigTestStepResult:
    """Verify (and, on mismatch, retry line-by-line to distinguish
    "recoverable" from "broken") a translation of `_TRANSLATION_PROBE_LINES`
    already produced by the caller — split out from
    `_probe_translation_and_thinking` purely so that function's job (get a
    translation AND detect thinking from one call) stays separate from
    this one's (verify a translation is a translation).
    """
    if (
        _align_translation(
            _TRANSLATION_PROBE_LINES,
            whole_output,
            source_lang=_TRANSLATION_PROBE_SOURCE_LANG_CODE,
            target_lang=lang.code,
        )
        is not None
    ):
        return ConfigTestStepResult(
            step="translation",
            ok=True,
            detail=(
                f"Translation contract verified — all {len(_TRANSLATION_PROBE_LINES)} lines "
                "aligned correctly on the first attempt."
            ),
        )

    prompt_template = _load_translate_prompt()

    async def _translate_one(line: str) -> list[str]:
        prompt = prompt_template.format(
            target_language_name=lang.english_name, transcript=line,
        )
        response = await _call_test_model(
            client=client,
            model=model,
            dialect=dialect,
            content=prompt,
            max_tokens=400,
            timeout=_LLM_STEP_TIMEOUT_SECONDS,
        )
        content = response.choices[0].message.content or ""
        return content.split("\n")

    # Whole-group call didn't verify — mirror the production recovery path
    # (workers.translator._translate_group) at single-line granularity,
    # without reimplementing its full bisection: enough to tell "some
    # lines recoverable" (partial) apart from "nothing aligns" (broken).
    resolved = 0
    for line in _TRANSLATION_PROBE_LINES:
        try:
            single_output = await _translate_one(line)
        except Exception:
            continue
        if (
            _align_translation(
                [line],
                single_output,
                source_lang=_TRANSLATION_PROBE_SOURCE_LANG_CODE,
                target_lang=lang.code,
            )
            is not None
        ):
            resolved += 1

    total = len(_TRANSLATION_PROBE_LINES)
    if resolved == total:
        return ConfigTestStepResult(
            step="translation",
            ok=True,
            detail=(
                "Translation contract verified on retry — the combined call misaligned, but "
                "every line aligns individually. Production translations against this "
                "backend/model would recover via retry/bisection, not lose text — expect "
                "occasional extra retries, not incorrect output."
            ),
        )
    if resolved > 0:
        return ConfigTestStepResult(
            step="translation",
            ok=False,
            detail=(
                f"Translation contract PARTIAL — only {resolved}/{total} lines verified even "
                "individually. A real translation against this backend/model would likely come "
                "back status=partial, with some lines left in the source language."
            ),
        )
    return ConfigTestStepResult(
        step="translation",
        ok=False,
        detail=(
            "Translation contract FAILED — the model's output could not be verified as a real "
            "translation (lost line alignment, a repetition loop, or it echoed the input back "
            "instead of translating). A real translation against this backend/model would "
            "likely fail or leave most of the transcript untranslated."
        ),
    )


async def _test_llm(overrides: ConfigTestLLMOverrides | None) -> ConfigTestResponse:
    """Six-step probe of a CANDIDATE (possibly unsaved) llm config — see
    `.claude/llm.md`'s "target=\"llm\" step-by-step probe" section for the
    full design. The report is always returned in full: an early failure
    doesn't shorten `steps`, it just marks the rest `ok=None` via
    `_fill_remaining_as_skipped`.
    """
    deadline = time.monotonic() + _TOTAL_TEST_TIMEOUT_SECONDS
    start_all = time.monotonic()
    cfg = get_config().llm
    base_url = (overrides.base_url if overrides else None) or cfg.base_url
    model = (overrides.model if overrides else None) or cfg.model

    steps: list[ConfigTestStepResult] = []

    if overrides and overrides.api_key:
        api_key: str | None = overrides.api_key
    else:
        try:
            api_key = cfg.effective_api_key
        except Exception as e:
            steps.append(
                ConfigTestStepResult(step="reachable", ok=False, detail=_redact(str(e), None))
            )
            _fill_remaining_as_skipped(steps, "not attempted: could not resolve an API key")
            return ConfigTestResponse(ok=False, models=[], latency_ms=None, steps=steps)

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    # --- Steps 1+2: reachability + model list ------------------------------
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=_TEST_TIMEOUT_SECONDS) as client:
            r = await client.get(f"{base_url}/models", headers=headers)
    except Exception as e:
        latency_ms = int((time.monotonic() - start) * 1000)
        steps.append(
            ConfigTestStepResult(step="reachable", ok=False, detail=_redact(str(e), api_key))
        )
        _fill_remaining_as_skipped(steps, "not attempted: backend unreachable")
        return ConfigTestResponse(ok=False, models=[], latency_ms=latency_ms, steps=steps)

    latency_ms = int((time.monotonic() - start) * 1000)
    steps.append(
        ConfigTestStepResult(
            step="reachable", ok=True, detail=f"HTTP {r.status_code} in {latency_ms} ms.",
        )
    )

    if r.status_code != 200:
        steps.append(
            ConfigTestStepResult(
                step="models",
                ok=False,
                detail=f"HTTP {r.status_code}: {_redact(r.text, api_key)}",
            )
        )
        _fill_remaining_as_skipped(steps, "not attempted: model list request failed")
        return ConfigTestResponse(ok=False, models=[], latency_ms=latency_ms, steps=steps)

    try:
        models = [m["id"] for m in r.json().get("data", []) if "id" in m]
    except Exception:
        models = []
    steps.append(
        ConfigTestStepResult(step="models", ok=True, detail=f"{len(models)} model(s) available.")
    )

    # --- Step 3: minimal chat completion against the target model ---------
    # Goes through the same dialect-adaptation logic as the real call path
    # (`llm_client.call_with_dialect_adaptation`), over a throwaway
    # client/model/dialect rather than the cached prod ones — otherwise a
    # cloud gpt-5/o-series backend would fail this probe on the very first
    # 400 (e.g. `max_tokens` unsupported) even though the real pipeline
    # would have adapted and worked fine. See llm/client.py for why there is
    # only one place that knows how to interpret these 400s.
    test_client = AsyncOpenAI(
        base_url=base_url, api_key=api_key or "dummy", max_retries=0,
        timeout=_CONTEXT_PROBE_TIMEOUT_SECONDS,
    )
    dialect = llm_client._new_dialect()
    try:
        # The response itself is discarded — this step only proves the
        # model answers SOMETHING at all. Thinking detection deliberately
        # does NOT use this trivial call (see `_probe_translation_and_thinking`).
        await _call_test_model(
            client=test_client,
            model=model,
            dialect=dialect,
            content="Reply with the single word: ok",
            max_tokens=_TEST_COMPLETION_MAX_TOKENS,
            timeout=_LLM_STEP_TIMEOUT_SECONDS,
        )
    except APIStatusError as e:
        steps.append(
            ConfigTestStepResult(
                step="completion",
                ok=False,
                detail=f"HTTP {e.status_code}: {_redact(str(e), api_key)}",
            )
        )
        _fill_remaining_as_skipped(steps, "not attempted: test completion failed")
        return ConfigTestResponse(ok=False, models=models, latency_ms=latency_ms, steps=steps)
    except Exception as e:
        steps.append(
            ConfigTestStepResult(step="completion", ok=False, detail=_redact(str(e), api_key))
        )
        _fill_remaining_as_skipped(steps, "not attempted: test completion failed")
        return ConfigTestResponse(ok=False, models=models, latency_ms=latency_ms, steps=steps)

    steps.append(
        ConfigTestStepResult(step="completion", ok=True, detail="Model responded to a test call.")
    )

    suggestions = ConfigTestSuggestions()

    # --- Steps 4+5: thinking + translation contract, from ONE real call ----
    # Order matters: this runs BEFORE context — see `_LLM_STEP_ORDER`'s
    # docstring and `_CONTEXT_PROBE_OWN_BUDGET_SECONDS` for why the cheap,
    # decisive step must never again be starved by the slow, diagnostic one.
    # The trivial step-3 completion above is deliberately never inspected
    # for thinking — see `_probe_translation_and_thinking`'s docstring for
    # why that was a measured false negative on Gemma 4.
    if time.monotonic() >= deadline:
        _fill_remaining_as_skipped(steps, "not attempted: overall test time budget exceeded")
        return ConfigTestResponse(
            ok=True, models=models, latency_ms=latency_ms, steps=steps, suggestions=suggestions,
        )
    thinking_step, translation_step, dialect, suggested_effort = (
        await _probe_translation_and_thinking(
            client=test_client, model=model, dialect=dialect, api_key=api_key,
        )
    )
    steps.append(thinking_step)
    steps.append(translation_step)
    if suggested_effort is not None:
        suggestions.reasoning_effort = suggested_effort

    # --- Step 6: real context length ----------------------------------------
    # Runs LAST and on its OWN dedicated budget (never more than what's left
    # of the overall deadline either) precisely so it can never again eat
    # into the time the decisive step above needs — see both budgets'
    # docstrings for the live incident that motivated this ordering.
    if time.monotonic() >= deadline:
        _fill_remaining_as_skipped(steps, "not attempted: overall test time budget exceeded")
    else:
        context_deadline = min(deadline, time.monotonic() + _CONTEXT_PROBE_OWN_BUDGET_SECONDS)
        context_step, context_value = await _probe_context_length(
            client=test_client,
            model=model,
            dialect=dialect,
            deadline=context_deadline,
            api_key=api_key,
        )
        steps.append(context_step)
        if context_value is not None:
            suggestions.context_length = context_value
            suggestions.single_pass_token_limit = int(context_value * _SINGLE_PASS_RATIO)

    total_latency_ms = int((time.monotonic() - start_all) * 1000)
    connectivity_ok = all(
        s.ok for s in steps if s.step in ("reachable", "models", "completion")
    )
    return ConfigTestResponse(
        ok=connectivity_ok,
        models=models,
        latency_ms=total_latency_ms,
        steps=steps,
        suggestions=suggestions,
    )


async def _test_whisper(overrides: ConfigTestWhisperOverrides | None) -> ConfigTestResponse:
    """Only probes reachability (``GET {base_url}/models``) — a real
    transcription probe would need an audio file to upload, which this
    endpoint doesn't have. ``ok: true`` here means "the backend is reachable
    and the key is accepted", not "transcription actually works"."""
    cfg = get_config().whisper
    base_url = (overrides.base_url if overrides else None) or cfg.base_url

    if overrides and overrides.api_key:
        api_key: str | None = overrides.api_key
    else:
        try:
            api_key = cfg.effective_api_key
        except Exception as e:
            return ConfigTestResponse(
                ok=False,
                step=None,
                status_code=None,
                detail=_redact(str(e), None),
                models=[],
                latency_ms=None,
            )

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=_TEST_TIMEOUT_SECONDS) as client:
            r = await client.get(f"{base_url}/models", headers=headers)
    except Exception as e:
        return ConfigTestResponse(
            ok=False,
            step="models",
            status_code=None,
            detail=_redact(str(e), api_key),
            models=[],
            latency_ms=int((time.monotonic() - start) * 1000),
        )
    latency_ms = int((time.monotonic() - start) * 1000)

    if r.status_code != 200:
        return ConfigTestResponse(
            ok=False,
            step="models",
            status_code=r.status_code,
            detail=_redact(r.text, api_key),
            models=[],
            latency_ms=latency_ms,
        )

    try:
        models = [m["id"] for m in r.json().get("data", []) if "id" in m]
    except Exception:
        models = []

    return ConfigTestResponse(
        ok=True,
        step="models",
        status_code=200,
        detail=None,
        models=models,
        latency_ms=latency_ms,
    )


@router.post("/test", response_model=ConfigTestResponse)
async def test_config_route(body: ConfigTestRequest) -> ConfigTestResponse:
    if body.target == "whisper":
        return await _test_whisper(body.whisper)
    return await _test_llm(body.llm)
