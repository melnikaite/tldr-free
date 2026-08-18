"""Loads daemon configuration from YAML + env overrides.

Single config file path is taken from TLDR_CONFIG. When unset, the default is
the container mount (/app/config/tldr.yaml) if present, otherwise the
platform-conventional path (see src/paths.py) — auto-created from the packaged
template on the first native (uv) run.

Individual fields can be overridden by env vars with double-underscore
separators, e.g.:
  TLDR__OUTPUT__LANGUAGE=ru
"""

from __future__ import annotations

import contextlib
import logging
import os
import stat
import tempfile
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from src import paths

log = logging.getLogger(__name__)

DAEMON_VERSION = "1.0.0"
DAEMON_API_VERSION = 4


class _ApiKeyConfigMixin(BaseModel):
    """Shared API-key fields + resolution logic for ``LLMConfig`` and
    ``WhisperConfig``. Both sections support the exact same four ways to
    supply a key (see ``effective_api_key``) — this is the single
    implementation both inherit rather than two copies that could diverge.

    ``_section`` (overridden per subclass to ``"llm"`` / ``"whisper"``)
    drives the two things that differ between sections: which env var wins
    (``TLDR__LLM__API_KEY`` vs ``TLDR__WHISPER__API_KEY``) and the
    field-name prefix in error messages (``llm.api_key_file`` vs
    ``whisper.api_key_file``).
    """

    _section: ClassVar[str] = "llm"

    api_key: str = "dummy"
    # Alternative ways to supply the API key, in priority order (see
    # ``effective_api_key``): a file on disk, or an OS keychain entry. Both
    # are optional — most local backends (mlx-server, LocalAI, Ollama) don't
    # check the key at all and the "dummy" default is fine.
    api_key_file: str | None = None
    api_key_keychain: str | None = None
    api_key_keychain_account: str = "api_key"

    @property
    def effective_api_key(self) -> str:
        """Resolve the API key to actually send to the backend.

        Strict priority order:

        1. Env var ``TLDR__<SECTION>__API_KEY`` — read directly from
           ``os.environ`` so it always wins. (``_apply_env_overrides``
           already maps this same var onto ``api_key`` before validation,
           so in practice this is usually a no-op re-confirmation — but
           reading it here directly makes the top-priority guarantee
           explicit and independent of that generic mechanism.)
        2. ``api_key_keychain`` — looked up via ``keyring.get_password``,
           with ``api_key_keychain_account`` as the account name.
        3. ``api_key_file`` — path read and stripped.
        4. Inline ``api_key`` field (default: the local-backend dummy value).

        Raises ``RuntimeError`` for any misconfiguration (missing optional
        dependency, missing/empty file, missing keychain entry) rather than
        silently falling through to an empty or wrong key.
        """
        env_key = os.environ.get(f"TLDR__{self._section.upper()}__API_KEY")
        if env_key:
            return env_key

        if self.api_key_keychain:
            try:
                import keyring
            except ImportError as e:
                # 'keyring' is a base dependency (see pyproject.toml) — this
                # only happens if it was somehow removed from the venv.
                raise RuntimeError(
                    f"{self._section}.api_key_keychain is set but the 'keyring' package "
                    "is not installed. Reinstall the daemon "
                    "('uv tool install --force .' / 'pip install .') to "
                    "restore it."
                ) from e
            password = keyring.get_password(self.api_key_keychain, self.api_key_keychain_account)
            if not password:
                raise RuntimeError(
                    "No password found in the OS keychain for "
                    f"service={self.api_key_keychain!r} "
                    f"account={self.api_key_keychain_account!r}."
                )
            return password

        if self.api_key_file:
            path = Path(self.api_key_file).expanduser()
            if not path.is_file():
                raise RuntimeError(f"{self._section}.api_key_file {path} does not exist.")
            mode = stat.S_IMODE(path.stat().st_mode)
            if mode & 0o077:
                log.warning(
                    "%s.api_key_file %s is readable beyond the owner (mode %o); "
                    "consider `chmod 600 %s`.",
                    self._section,
                    path,
                    mode,
                    path,
                )
            key = path.read_text().strip()
            if not key:
                raise RuntimeError(f"{self._section}.api_key_file {path} is empty.")
            return key

        return self.api_key


class LLMConfig(_ApiKeyConfigMixin):
    _section: ClassVar[str] = "llm"

    base_url: str
    model: str
    context_length: int = 32768
    single_pass_token_limit: int = 24000
    # Cap on concurrent in-flight LLM calls. Default 1 (single-user macOS box
    # — a parallel Qwen/Whisper inference will thrash the GPU and the fan).
    # Bump to 2-3 for beefy GPU servers or hosted backends.
    max_concurrent_calls: int = 1
    # Max seconds to wait for the next streaming chunk before giving up. Catches
    # backends that hang mid-stream (e.g. mlx-server unloading the model under
    # us). 60 s is generous for fast local backends and fast hosted ones; bump
    # for slow remote backends.
    stream_chunk_timeout_seconds: float = 60.0
    # Pass reasoning_effort to the backend. Set to "none" to disable thinking
    # mode on models like Gemma 4 in LM Studio — without it the model spends
    # its entire max_tokens budget on chain-of-thought and emits no content.
    # Leave null (the default) for backends that don't support this field;
    # they will simply ignore or error on the extra body parameter.
    reasoning_effort: str | None = None
    # Which kwarg carries the token limit. "auto" (default) starts with
    # "max_tokens" and lets llm.client auto-adapt to "max_completion_tokens"
    # on a 400 from the backend (OpenAI o-series/gpt-5). Pin it to skip the
    # detection round-trip once you know your backend's dialect.
    token_param: Literal["auto", "max_tokens", "max_completion_tokens"] = "auto"
    # Whether to send `temperature` at all. None (default) = auto-detect:
    # start by sending it, stop if the backend 400s on a non-default value
    # (OpenAI o-series/gpt-5 only accept the default). True/False pins it.
    send_temperature: bool | None = None
    # Extra tokens added to the requested limit when sending it as
    # `max_completion_tokens`, so hidden reasoning tokens (o-series/gpt-5)
    # don't eat the entire budget and leave `content` empty.
    reasoning_headroom_tokens: int = 4000
    # Optional hard ceiling on the token limit requested in any single call,
    # regardless of what the caller asked for. Useful to cap spend on a
    # metered cloud backend. None = no clamp.
    max_output_tokens: int | None = None


class WhisperConfig(_ApiKeyConfigMixin):
    _section: ClassVar[str] = "whisper"

    base_url: str
    model: str = "whisper"
    # Per-request upload ceiling (MB). Audio larger than this is split into
    # time-based chunks before transcription, then segments are stitched back
    # with their original timestamps. Defaults to 15 — LocalAI's default upload
    # limit; also safe for OpenAI's 25 MB cap. Raise it if your backend allows
    # bigger bodies (fewer requests = slightly faster, fewer seams).
    max_upload_mb: int = 15


# ISO 639-1 → English language name. Small enough to inline; covers the
# common cases. Anything else (a longer code, a full name, or something
# like "scientific English") flows through ``language_name`` unchanged so
# the LLM still sees what the user wrote.
_ISO_LANGUAGE_NAMES: dict[str, str] = {
    "ar": "Arabic",
    "cs": "Czech",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "zh": "Chinese",
}


class OutputConfig(BaseModel):
    """Output language for summaries and Q&A.

    Accepts an ISO 639-1 code (``en``, ``ru``, ``de``, …) which is expanded
    to the human-readable name when threading into LLM prompts. Smaller
    models follow ``"Russian"`` more reliably than ``"ru"`` on its own.
    Anything that isn't a known code is passed through verbatim.
    """
    language: str = "en"

    @property
    def language_name(self) -> str:
        s = self.language.strip()
        return _ISO_LANGUAGE_NAMES.get(s.lower(), s)


class YouTubeConfig(BaseModel):
    fast_path_max_attempts: int = 4
    fast_path_backoff_seconds: list[int] = Field(default_factory=lambda: [1, 4, 16, 60])
    # Max seconds of speech per [MM:SS] line. Transcript lines are split by
    # SENTENCE; this is only a safety cap so a punctuation-free auto-caption
    # track doesn't collapse into one giant line.
    segment_window_seconds: int = 30
    audio_format: str = "opus"
    audio_bitrate_max: int = 64
    ytdlp_sleep_interval: list[int] = Field(default_factory=lambda: [3, 8])
    # When the youtube-transcript-api fast path fails, we ask yt-dlp to fetch
    # YouTube's auto-generated captions before falling back to Whisper. The
    # original-language track is always tried first; this list adds further
    # acceptable language codes in priority order.
    subtitle_lang_preferences: list[str] = Field(default_factory=lambda: ["en", "ru"])


class StorageConfig(BaseModel):
    data_dir: str = "/data"
    db_filename: str = "tldr.db"
    # Periodic retention sweep — delete jobs older than this many days.
    # 0 disables the sweep entirely. Upper bound (10 years) is a sanity cap,
    # not a real-world expectation — matches api/schemas.py's
    # StorageConfigPatch so a value valid via PATCH is also valid read
    # straight from tldr.yaml/tldr.local.yaml.
    retention_days: int = Field(default=365, ge=0, le=3650)

    @model_validator(mode="after")
    def _resolve_data_dir(self) -> StorageConfig:
        # "/data" is the container default (also baked into the shipped config
        # template). On a native install /data doesn't exist — substitute the
        # platform data dir instead of failing or writing to the filesystem root.
        if self.data_dir == "/data" and not paths.CONTAINER_DATA.is_dir():
            self.data_dir = str(paths.platform_data_dir())
        return self

    @property
    def db_path(self) -> Path:
        return Path(self.data_dir) / self.db_filename


class WorkersConfig(BaseModel):
    # Wait this many seconds between consecutive background jobs to give the
    # CPU/GPU time to cool down. 0 = back-to-back. Useful when dumping a big
    # backlog overnight on a fanless laptop.
    cooldown_seconds: int = 0


class LoggingConfig(BaseModel):
    level: str = "INFO"


class QaConfig(BaseModel):
    """Follow-up Q&A behaviour (`llm/qa.py`'s plan -> look -> search ->
    synthesis flow)."""

    # When True (default), a Q&A turn that the PLAN step judges insufficient
    # runs a DuckDuckGo text search (`workers/search.py`) built from the
    # question, then fetches and cleans a handful of the resulting pages, to
    # enrich the answer — a search-engine query derived from your question,
    # plus requests to whatever pages DuckDuckGo returns, go out over the
    # internet. The side panel shows this via a "searching" stage so it's
    # never silent. Set to False to disable web search entirely for Q&A: no
    # DuckDuckGo query, no page fetch, ever. The daemon then answers only
    # from the processed material and the model's own training knowledge,
    # and is instructed to say plainly when something isn't covered rather
    # than guess (see `llm/qa.py`'s synthesis prompt) — for anyone who wants
    # follow-up questions to stay as local as the rest of the pipeline.
    web_search: bool = True


class Config(BaseModel):
    llm: LLMConfig
    whisper: WhisperConfig
    output: OutputConfig
    youtube: YouTubeConfig
    storage: StorageConfig
    workers: WorkersConfig = Field(default_factory=WorkersConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    qa: QaConfig = Field(default_factory=QaConfig)


def _apply_env_overrides(data: dict[str, Any], prefix: str = "TLDR") -> dict[str, Any]:
    """Override fields from env vars: TLDR__SECTION__KEY=value."""
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix + "__"):
            continue
        path = env_key.removeprefix(prefix + "__").lower().split("__")
        cursor: Any = data
        for key in path[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[path[-1]] = env_val
    return data


def ensure_config_file(config_path: Path) -> bool:
    """Create ``config_path`` from the packaged template if missing.

    Returns True when a new file was written. Backend URLs in the template
    point at ``host.docker.internal`` (the container's view of the host);
    natively the backends are plain localhost, so we rewrite them.

    Written with mode 0600 (owner read/write only) since the file may end up
    holding a cloud API key inline (``llm.api_key``) — the config directory
    isn't otherwise access-controlled.
    """
    if config_path.is_file():
        return False
    template = (resources.files("src.assets") / "tldr.yaml.example").read_text()
    template = template.replace("host.docker.internal", "127.0.0.1")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(template)
    return True


def config_path() -> Path:
    """Resolve the active ``tldr.yaml`` path: ``TLDR_CONFIG`` env override,
    else the platform-conventional path (see ``paths.py``). Also used to
    derive ``overrides_path()`` and reported verbatim by ``GET /config``."""
    env_path = os.environ.get("TLDR_CONFIG")
    return Path(env_path) if env_path else paths.default_config_path()


def overrides_path() -> Path:
    """Path to the user-writable overrides file, deep-merged on top of the
    (comment-only, hand-edited) ``tldr.yaml`` template by ``get_config()``.

    ``tldr.yaml`` is never written by the daemon itself — ``yaml.safe_dump``
    would destroy its comments, which is where the backend examples live.
    ``PATCH /config`` writes here instead. Defaults to a sibling of
    ``config_path()``; override with ``TLDR_CONFIG_OVERRIDES`` (mirrors
    ``TLDR_CONFIG``) so tests can point both at an isolated ``tmp_path``.
    """
    env_path = os.environ.get("TLDR_CONFIG_OVERRIDES")
    if env_path:
        return Path(env_path)
    return config_path().parent / "tldr.local.yaml"


def api_key_file_path(section: Literal["llm", "whisper"] = "llm") -> Path:
    """Where ``PATCH /config`` writes the ``section`` API key when
    ``api_key_storage="file"`` is selected. Lives next to
    ``overrides_path()`` (which the override file itself points at via
    ``<section>.api_key_file``) rather than next to ``config_path()``
    directly — this keeps it in the same, possibly test-isolated, writable
    directory as ``tldr.local.yaml`` instead of next to a possibly
    read-only ``tldr.yaml`` (e.g. the checked-in example used by the test
    suite). ``llm`` and ``whisper`` get separate files (``llm.key`` /
    ``whisper.key``) so switching one section's storage never touches the
    other's."""
    return overrides_path().parent / f"{section}.key"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base``, returning a new dict.
    Non-dict values in ``overlay`` replace the corresponding ``base`` value
    outright (including replacing a dict with a scalar, or vice versa)."""
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _atomic_write_text(path: Path, text: str, mode: int) -> None:
    """Write ``text`` to ``path`` atomically (temp file + ``os.replace``) with
    ``mode`` permissions, so a crash mid-write never leaves a partial file
    and a concurrent reader never sees one either."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def read_overrides() -> dict[str, Any]:
    """Load the raw overrides YAML (empty dict if the file doesn't exist)."""
    path = overrides_path()
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def write_overrides(data: dict[str, Any]) -> None:
    """Atomically persist ``data`` as the overrides YAML, mode 0600 (it may
    hold ``llm.api_key`` inline). Caller is responsible for validating the
    result first — see ``validate_full_config``."""
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    _atomic_write_text(overrides_path(), text, 0o600)


def write_api_key_file(key: str, section: Literal["llm", "whisper"] = "llm") -> Path:
    """Atomically write ``key`` to ``api_key_file_path(section)`` at mode
    0600. Returns the path so the caller can point ``<section>.api_key_file``
    at it."""
    path = api_key_file_path(section)
    _atomic_write_text(path, key, 0o600)
    return path


@lru_cache(maxsize=1)
def keychain_backend_available() -> bool:
    """Whether the OS keychain (macOS Keychain / Linux Secret Service /
    Windows Credential Locker) is actually usable on this machine — not
    merely whether the ``keyring`` package is importable, but whether a
    real backend is configured. ``keyring`` falls back to
    ``keyring.backends.fail.Keyring`` (every call raises) when no usable
    backend is found, e.g. headless Linux without a Secret Service running
    in the session — this returns ``False`` in that case.

    This is the single source of truth for "is keychain storage viable
    right now": ``GET /config`` reports it as ``keychain_available`` and
    ``PATCH /config`` uses it to pick the default ``api_key_storage``
    when the caller doesn't specify one — see ``src/api/config.py``.

    Cached for the process lifetime: the available backend can't change
    while the daemon is running (same rationale as the LLM client cache —
    see ``.claude/llm.md``).
    """
    try:
        import keyring
    except ImportError:
        return False
    try:
        backend = keyring.get_keyring()
    except Exception:
        return False
    # keyring falls back to the null "fail" backend when no real backend is
    # configured (e.g. headless Linux without a Secret Service in the
    # session) — every call on it raises. Compared by qualified class name
    # rather than isinstance-against-an-import so this stays robust under
    # test doubles that stub out the `keyring` module itself.
    backend_name = f"{type(backend).__module__}.{type(backend).__qualname__}"
    return backend_name != "keyring.backends.fail.Keyring"


def validate_full_config(overrides: dict[str, Any]) -> Config:
    """Validate ``overrides`` deep-merged onto the on-disk template, WITHOUT
    touching the cached ``get_config()`` singleton or writing anything.

    Used by ``PATCH /config`` to reject bad input with a 422 before
    persisting — raises ``pydantic.ValidationError`` on failure. Env
    overrides are re-applied on top so validation matches exactly what a
    subsequent ``get_config()`` call would produce.
    """
    path = config_path()
    raw = (yaml.safe_load(path.read_text()) or {}) if path.is_file() else {}
    merged = _deep_merge(raw, overrides) if overrides else raw
    merged = _apply_env_overrides(merged)
    return Config.model_validate(merged)


@lru_cache(maxsize=1)
def get_config() -> Config:
    path = config_path()
    env_path = os.environ.get("TLDR_CONFIG")
    if not path.is_file():
        if env_path:
            # An explicit path that doesn't exist is a user error — don't
            # silently create a default somewhere they pointed at.
            raise FileNotFoundError(
                f"Config file not found at {path}. "
                "Run 'task install' to copy from config/tldr.yaml.example."
            )
        ensure_config_file(path)
    raw = yaml.safe_load(path.read_text()) or {}
    overrides = read_overrides()
    if overrides:
        raw = _deep_merge(raw, overrides)
    raw = _apply_env_overrides(raw)
    return Config.model_validate(raw)
