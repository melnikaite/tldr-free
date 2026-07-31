"""GET/PATCH /config, POST /config/test — daemon settings editable from the
extension's options page instead of hand-editing ``tldr.yaml``.

Security: these endpoints accept and resolve secrets (LLM API keys). The
daemon only ever binds to 127.0.0.1 and CORS is restricted to
``chrome-extension://*`` (see ``src/main.py``) — do not relax either
without re-reading this module. A key is NEVER returned by ``GET``/``PATCH``
(only ``api_key_set``/``api_key_hint``/``api_key_source``), never logged,
and scrubbed out of any provider error text before it reaches a response.

Override layer (see ``.claude/daemon.md`` / ``config.py`` docstrings):
``tldr.yaml`` is a hand-edited, comment-heavy template — writing to it with
``yaml.safe_dump`` would destroy those comments. All writes from this
router instead go to ``tldr.local.yaml`` (``config.overrides_path()``),
which ``get_config()`` deep-merges on top of the template before env
overrides are applied. Every PATCH is validated (``Config.model_validate``
via ``config.validate_full_config``) before anything is written to disk.

API key storage (``PATCH /config`` body ``llm.api_key_storage``):

- ``"keychain"`` (default when ``config.keychain_backend_available()`` is
  true) — ``keyring.set_password(...)``; override gets
  ``llm.api_key_keychain`` / ``llm.api_key_keychain_account``. This is the
  recommended mode: the daemon itself writes the entry, so it's
  automatically in its own trusted-app ACL and reads it back with zero
  prompts (see ``.claude/llm.md``). 422 with an actionable message if
  ``keyring`` has no usable backend (headless Linux without a Secret
  Service in the session — see ``keychain_available`` below).
- ``"file"`` (default when keychain isn't available) — the key is written
  to ``config.api_key_file_path()`` (mode 0600) and the override points
  ``llm.api_key_file`` at it. Also the right choice for Docker installs,
  which have neither macOS Keychain nor a Secret Service.
- ``"inline"`` — the key is written directly into the (0600) override file
  as ``llm.api_key``.

Switching storage cleans up the fields (and best-effort the previous
file/keychain entry) used by the PREVIOUS mode, so only one source of
truth remains — see ``_apply_api_key_storage``.

``keychain_available`` (``GET``/``PATCH`` response, top level) reflects
``config.keychain_backend_available()`` — a real check that the keyring
backend is usable, not just that the package is importable. It's cached
for the process lifetime (the backend can't change while running) and is
the single place both the default-storage decision (above) and the
extension's options-page UI (disabling the keychain choice) read from.

Write-then-read-back verification: whenever a PATCH writes a new API key,
the response includes ``api_key_verified`` (bool) and
``api_key_verify_error`` (str, redacted, or None) — the freshly-saved
config is read back via the EXACT accessor the daemon uses at call time
(``LLMConfig.effective_api_key``) and compared to what was just written.
A failed verification is reported, never rolled back — see
``_verify_saved_api_key``.

Cache invalidation after a successful PATCH: ``get_config.cache_clear()``
and ``llm_client.reset_caches()`` (client + dialect guess). The LLM
semaphore (``llm.client._llm_lock()``) is intentionally NOT touched — an
``asyncio.Semaphore`` is bound to the running event loop and can't be
resized in place, so a changed ``max_concurrent_calls`` instead reports
``restart_required: true``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from openai import APIStatusError, AsyncOpenAI
from pydantic import ValidationError

from src import config as config_module
from src.api.schemas import (
    ConfigPatchRequest,
    ConfigPatchResponse,
    ConfigResponse,
    ConfigTestRequest,
    ConfigTestResponse,
    LLMConfigOut,
    OutputConfigOut,
    WhisperConfigOut,
)
from src.config import Config, LLMConfig, get_config
from src.llm import client as llm_client

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

_KEYCHAIN_SERVICE = "tldr-daemon-llm"
_KEYCHAIN_ACCOUNT = "api_key"


# ---------------------------------------------------------------------------
# GET /config — read-only view, secrets redacted to presence/hint/source
# ---------------------------------------------------------------------------


def _api_key_source(cfg: LLMConfig) -> str:
    """Which of the four mechanisms will actually supply the key, in the
    same priority order as ``LLMConfig.effective_api_key``."""
    if os.environ.get("TLDR__LLM__API_KEY"):
        return "env"
    if cfg.api_key_keychain:
        return "keychain"
    if cfg.api_key_file:
        return "file"
    if cfg.api_key and cfg.api_key != "dummy":
        return "inline"
    return "none"


def _api_key_hint(cfg: LLMConfig) -> str | None:
    """Last 4 characters of the resolved key, or None if unset/unresolvable.
    Never raises — a misconfigured file/keychain just means no hint."""
    try:
        key = cfg.effective_api_key
    except Exception:
        return None
    return key[-4:] if key else None


def _to_response(cfg: Config) -> ConfigResponse:
    source = _api_key_source(cfg.llm)
    hint = _api_key_hint(cfg.llm) if source != "none" else None
    return ConfigResponse(
        llm=LLMConfigOut(
            base_url=cfg.llm.base_url,
            model=cfg.llm.model,
            context_length=cfg.llm.context_length,
            single_pass_token_limit=cfg.llm.single_pass_token_limit,
            max_concurrent_calls=cfg.llm.max_concurrent_calls,
            reasoning_effort=cfg.llm.reasoning_effort,
            api_key_set=source != "none",
            api_key_hint=hint,
            api_key_source=source,  # type: ignore[arg-type]
        ),
        whisper=WhisperConfigOut(
            base_url=cfg.whisper.base_url,
            model=cfg.whisper.model,
            max_upload_mb=cfg.whisper.max_upload_mb,
        ),
        output=OutputConfigOut(language=cfg.output.language),
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


def _apply_api_key_storage(llm_overrides: dict[str, Any], storage: str, key: str) -> None:
    """Point ``llm_overrides`` at ``key`` via ``storage``, clearing whatever
    fields the PREVIOUS mode used so exactly one source of truth remains.
    Raises ``HTTPException(422)`` for a bad/unsupported storage choice."""
    old_file = llm_overrides.get("api_key_file")
    old_keychain = llm_overrides.get("api_key_keychain")
    old_keychain_account = llm_overrides.get("api_key_keychain_account", _KEYCHAIN_ACCOUNT)

    for field in ("api_key", "api_key_file", "api_key_keychain", "api_key_keychain_account"):
        llm_overrides.pop(field, None)

    if storage == "inline":
        llm_overrides["api_key"] = key
    elif storage == "file":
        path = config_module.write_api_key_file(key)
        llm_overrides["api_key_file"] = str(path)
    elif storage == "keychain":
        if not config_module.keychain_backend_available():
            raise HTTPException(
                status_code=422,
                detail=(
                    "llm.api_key_storage='keychain' requires a usable OS keychain "
                    "backend. On macOS/Windows this should always be available; on "
                    "Linux it needs a Secret Service (GNOME Keyring / KWallet) "
                    "running in the current session. Use api_key_storage='file' "
                    "instead (recommended for Docker installs and headless setups)."
                ),
            )
        import keyring

        try:
            keyring.set_password(_KEYCHAIN_SERVICE, _KEYCHAIN_ACCOUNT, key)
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
                    f"(service={_KEYCHAIN_SERVICE!r}): {_redact(str(e), key)}. If a "
                    "previous version left a stale entry with an incompatible "
                    "ACL, remove it (e.g. `security delete-generic-password -s "
                    f"{_KEYCHAIN_SERVICE} -a {_KEYCHAIN_ACCOUNT}` on macOS) and retry, "
                    "or use api_key_storage='file' instead."
                ),
            ) from e
        llm_overrides["api_key_keychain"] = _KEYCHAIN_SERVICE
        llm_overrides["api_key_keychain_account"] = _KEYCHAIN_ACCOUNT
    else:
        raise HTTPException(status_code=422, detail=f"Unknown llm.api_key_storage {storage!r}")

    # Best-effort cleanup of whatever the previous mode left behind. Never
    # let this block the actual (already-applied) storage change. Only ever
    # delete `old_file` when it's OUR managed key file (api_key_file_path()):
    # `llm.api_key_file` is a plain user-editable config field, so it could
    # instead point at a file the user manages themselves (e.g. a
    # `~/.config/openai.key` shared with other tools) — silently deleting
    # that on a storage-mode switch would be a nasty surprise unrelated to
    # what the user asked for.
    if storage != "file" and old_file:
        try:
            is_ours = Path(old_file).resolve() == config_module.api_key_file_path().resolve()
        except OSError:
            is_ours = False
        if is_ours:
            with contextlib.suppress(OSError):
                Path(old_file).unlink()
        else:
            log.info(
                "llm.api_key_storage changed away from 'file' but api_key_file %r is not "
                "our managed key file — leaving it in place",
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

    # Set only when this PATCH actually (re)writes the API key — that's the
    # one case worth a write-then-read-back check (see
    # `_verify_saved_api_key`). Left None otherwise.
    saved_key: str | None = None

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
            _apply_api_key_storage(llm_overrides, storage, new_key)
            saved_key = new_key

    if body.whisper is not None:
        whisper_overrides.update(body.whisper.model_dump(exclude_unset=True))

    if body.output is not None:
        output_overrides.update(body.output.model_dump(exclude_unset=True))

    new_overrides: dict[str, Any] = dict(overrides)
    for section_name, section_val in (
        ("llm", llm_overrides),
        ("whisper", whisper_overrides),
        ("output", output_overrides),
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
    api_key_verified, api_key_verify_error = _verify_saved_api_key(new_cfg, saved_key)

    response = _to_response(new_cfg)
    return ConfigPatchResponse(
        **response.model_dump(),
        restart_required=restart_required,
        api_key_verified=api_key_verified,
        api_key_verify_error=api_key_verify_error,
    )


def _verify_saved_api_key(cfg: Config, saved_key: str | None) -> tuple[bool, str | None]:
    """Read the just-saved API key back using the EXACT accessor the daemon
    uses at call time (``LLMConfig.effective_api_key``) and compare it to
    what was just written, so a broken storage choice (e.g. a keychain
    entry the daemon can't read back, a bad file path) is caught and
    reported immediately rather than surfacing as a mysterious 401 the next
    time an LLM call is made.

    Never rolls back the save — by the time this runs, ``write_overrides``
    already succeeded and IS the source of truth; this only decides what
    to report. ``saved_key`` is None when this PATCH didn't touch the API
    key at all, in which case there's nothing to verify.

    Never lets the actual key value leak into the returned error string.
    """
    if saved_key is None:
        return True, None
    try:
        resolved = cfg.llm.effective_api_key
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


@router.post("/test", response_model=ConfigTestResponse)
async def test_config_route(body: ConfigTestRequest) -> ConfigTestResponse:
    cfg = get_config().llm
    overrides = body.llm
    base_url = (overrides.base_url if overrides else None) or cfg.base_url
    model = (overrides.model if overrides else None) or cfg.model

    if overrides and overrides.api_key:
        api_key = overrides.api_key
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

    # --- Step 1: GET {base_url}/models ------------------------------------
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

    # --- Step 2: minimal chat completion against the target model --------
    # Goes through the same dialect-adaptation logic as the real call path
    # (`llm_client.call_with_dialect_adaptation`), over a throwaway
    # client/model/dialect rather than the cached prod ones — otherwise a
    # cloud gpt-5/o-series backend would fail this probe on the very first
    # 400 (e.g. `max_tokens` unsupported) even though the real pipeline
    # would have adapted and worked fine. See llm/client.py for why there is
    # only one place that knows how to interpret these 400s.
    test_client = AsyncOpenAI(base_url=base_url, api_key=api_key or "dummy")
    start2 = time.monotonic()
    try:
        await llm_client.call_with_dialect_adaptation(
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=_TEST_COMPLETION_MAX_TOKENS,
            temperature=0.0,
            stream=False,
            client=test_client,
            model=model,
            dialect=llm_client._new_dialect(),
        )
    except APIStatusError as e:
        return ConfigTestResponse(
            ok=False,
            step="completion",
            status_code=e.status_code,
            detail=_redact(str(e), api_key),
            models=models,
            latency_ms=latency_ms + int((time.monotonic() - start2) * 1000),
        )
    except Exception as e:
        return ConfigTestResponse(
            ok=False,
            step="completion",
            status_code=None,
            detail=_redact(str(e), api_key),
            models=models,
            latency_ms=latency_ms + int((time.monotonic() - start2) * 1000),
        )

    return ConfigTestResponse(
        ok=True,
        step="completion",
        status_code=200,
        detail=None,
        models=models,
        latency_ms=latency_ms + int((time.monotonic() - start2) * 1000),
    )
