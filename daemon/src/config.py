"""Loads daemon configuration from YAML + env overrides.

Single config file path is taken from TLDR_CONFIG (default: /app/config/tldr.yaml).
Individual fields can be overridden by env vars with double-underscore separators, e.g.:
  TLDR__OUTPUT__LANGUAGE=ru
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DAEMON_VERSION = "0.1.0"
DAEMON_API_VERSION = 3


class LLMConfig(BaseModel):
    base_url: str
    api_key: str = "dummy"
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


class WhisperConfig(BaseModel):
    base_url: str
    api_key: str = "dummy"
    model: str = "whisper"


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
    # 0 disables the sweep entirely.
    retention_days: int = 365

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


class Config(BaseModel):
    llm: LLMConfig
    whisper: WhisperConfig
    output: OutputConfig
    youtube: YouTubeConfig
    storage: StorageConfig
    workers: WorkersConfig = Field(default_factory=WorkersConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


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


@lru_cache(maxsize=1)
def get_config() -> Config:
    config_path = Path(os.environ.get("TLDR_CONFIG", "/app/config/tldr.yaml"))
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Run 'task install' to copy from config/tldr.yaml.example."
        )
    raw = yaml.safe_load(config_path.read_text()) or {}
    raw = _apply_env_overrides(raw)
    return Config.model_validate(raw)
