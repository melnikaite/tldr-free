"""GET /diagnostics — a self-contained report the USER pastes into a bug
report themselves. The daemon never sends this anywhere; see the options
page's "Diagnostics" section (extension/src/options) for the copy/save UI
this feeds.

We sell "your content never leaves this machine" as a feature (see
CLAUDE.md) — a diagnostics dump that leaks viewing history into a public
GitHub issue would be worse than no diagnostics at all. So everything
returned here is either structurally safe by construction (counts, enum
values, version strings) or has passed through ``_scrub`` below. Three
things ``_scrub`` removes, no exceptions:

- API keys — matched against the SAME resolved values ``effective_api_key``
  would send to a backend (llm AND whisper, independently), so even a
  fragment that leaked into a log line (e.g. inside a provider error
  string) never survives into the report.
- The home directory — tracebacks in the log tail routinely contain
  ``/Users/<name>/...`` (venv paths, project paths); replaced with ``~``.
- Any URL whose host ISN'T a loopback address. Loopback URLs
  (``127.0.0.1``/``localhost``/``::1``) are exactly the local-backend
  addresses a diagnosis needs to see (``http://127.0.0.1:1240/...``) and
  are kept; everything else — the page/video URL the user actually
  processed — is the one thing this report must never carry, so it's
  replaced with a placeholder wholesale. This matches BOTH a literal
  ``https://`` URL and a percent-encoded one (``https%3A%2F%2F...`` — how
  ``uvicorn.access`` used to write a request's query string before
  ``src/logging_setup.py``'s ``configure_logging`` started dropping query
  strings from that logger at the source; old, already-rotated log files
  can still have the encoded form on disk). See ``_redact_urls``.

What's deliberately NEVER included at all (not scrubbed — just never put in
the response in the first place): page/transcript content, job titles,
job URLs. ``DiagnosticsJobInfo`` (see schemas.py) only ever exposes
kind/status/progress_stage/error/transcript_source for a job.

The ``config`` block gets its own, narrower treatment — see
``_diagnostics_config`` and ``DiagnosticsConfigOut`` (schemas.py):
``api_key_hint`` (llm AND whisper) is dropped outright rather than
scrubbed — it carries zero diagnostic value beyond what ``api_key_set``
already reports, while being 4 real characters of a live secret — and
``config_path``/``overrides_path`` are run through the same home-directory
redaction as everything else (they're absolute paths on the reporter's own
machine, e.g. ``/Users/<name>/Library/Application Support/tldr/tldr.yaml``).
"""

from __future__ import annotations

import logging
import platform
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

from fastapi import APIRouter, HTTPException, status

from src.api.config import _to_response
from src.api.health import health as health_probe
from src.api.schemas import (
    DiagnosticsConfigOut,
    DiagnosticsJobInfo,
    DiagnosticsLLMConfigOut,
    DiagnosticsResponse,
    DiagnosticsWhisperConfigOut,
)
from src.config import DAEMON_VERSION, Config, get_config
from src.logging_setup import log_file_path
from src.storage import repo

log = logging.getLogger(__name__)

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])

# Lines from the tail of the rotating log file (src/logging_setup.py)
# included in the report. Generous enough to catch a recent failure without
# turning the report into a second copy of the whole log.
LOG_TAIL_LINES = 300

# Matches a literal scheme (``https://``) OR a percent-encoded one
# (``https%3A%2F%2F`` / ``http%3a%2f%2f``, any case — ``re.IGNORECASE``
# covers both the scheme letters and the hex digits). This is NOT an
# attempt to enumerate every possible encoding: uvicorn's access log
# (before the `configure_logging` fix in `logging_setup.py`) and query
# strings in general routinely carry a page URL percent-encoded wholesale
# (``GET /jobs?url=https%3A%2F%2Fexample.com%2F...``), and a literal-only
# regex lets that straight through — that's the exact leak this pattern
# closes. A URL that's already literal (``https://host/path%20with%20a%20space``,
# partial encoding in the PATH only) already matched the old pattern and
# still does; nothing about that case changes.
_URL_RE = re.compile(r"(?:https?://|https?%3a%2f%2f)[^\s\"'<>)\]]+", re.IGNORECASE)
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
_URL_PLACEHOLDER = "<redacted-url>"

# Fallback pattern in case a home path shows up spelled differently than
# `Path.home()` resolves to on THIS run (e.g. a log line written under a
# different $HOME, or before a user rename) — belt and suspenders on top of
# the exact-string replace in `_scrub`.
_HOME_PATTERNS = (
    re.compile(r"/Users/[^/\s]+"),
    re.compile(r"/home/[^/\s]+"),
    re.compile(r"C:\\Users\\[^\\\s]+"),
)


def _redact_urls(text: str) -> str:
    def _replace(m: re.Match[str]) -> str:
        raw = m.group(0)
        # `unquote` is a no-op on a URL that was already literal, so it's
        # always safe to run — this is what lets a single code path handle
        # a plain URL, a fully percent-encoded one, or anything in between
        # (partial encoding, mixed case) uniformly: decode first, THEN
        # look at the host. Anything that fails to parse as a URL at all
        # after decoding is treated as suspicious, not passed through.
        decoded = unquote(raw)
        try:
            host = urlsplit(decoded).hostname or ""
        except ValueError:
            return _URL_PLACEHOLDER
        # Return the ORIGINAL (possibly still-encoded) text for a kept
        # loopback match — we redact based on the decoded host, but never
        # rewrite an address we're choosing to keep. An empty/unparsable
        # host is never in `_LOOPBACK_HOSTS`, so it's redacted too.
        return raw if host in _LOOPBACK_HOSTS else _URL_PLACEHOLDER

    return _URL_RE.sub(_replace, text)


def _redact_home(text: str) -> str:
    home = str(Path.home())
    if home and home != "/":
        text = text.replace(home, "~")
    for pattern in _HOME_PATTERNS:
        text = pattern.sub("~", text)
    return text


def _redact_api_keys(text: str, config: Config) -> str:
    for cfg in (config.llm, config.whisper):
        try:
            key = cfg.effective_api_key
        except Exception:
            continue
        if key and key != "dummy":
            text = text.replace(key, "***")
    return text


def _scrub(text: str, config: Config) -> str:
    """Apply every redaction pass. Order matters: keys first (exact-string
    match, must run before anything else could split a key across a
    replacement boundary), then home paths, then URLs."""
    text = _redact_api_keys(text, config)
    text = _redact_home(text)
    text = _redact_urls(text)
    return text


def _diagnostics_config(config: Config) -> DiagnosticsConfigOut:
    """Build the ``config`` block of the report from ``_to_response()`` —
    reusing its API-key resolution/redaction logic rather than
    reimplementing it (same one ``GET /config`` uses) — but filtered down
    to ``DiagnosticsConfigOut``'s narrower shape. Two things NEVER reach the
    caller here even though ``_to_response()`` computed them:

    - ``api_key_hint`` (either section): dropped, not copied. See
      ``DiagnosticsConfigOut``'s docstring for why the hint is useless in
      this context while still being a live secret fragment.
    - the raw ``config_path``/``overrides_path`` strings: only their
      scrubbed (home-directory-redacted) form is copied.

    ``base_url`` goes through ``_redact_api_keys`` only, NOT
    ``_redact_urls`` / ``_redact_home`` — see ``DiagnosticsLLMConfigOut``.
    """
    full = _to_response(config)
    return DiagnosticsConfigOut(
        llm=DiagnosticsLLMConfigOut(
            base_url=_redact_api_keys(full.llm.base_url, config),
            model=full.llm.model,
            context_length=full.llm.context_length,
            single_pass_token_limit=full.llm.single_pass_token_limit,
            max_concurrent_calls=full.llm.max_concurrent_calls,
            reasoning_effort=full.llm.reasoning_effort,
            api_key_set=full.llm.api_key_set,
            api_key_source=full.llm.api_key_source,
        ),
        whisper=DiagnosticsWhisperConfigOut(
            base_url=_redact_api_keys(full.whisper.base_url, config),
            model=full.whisper.model,
            max_upload_mb=full.whisper.max_upload_mb,
            api_key_set=full.whisper.api_key_set,
            api_key_source=full.whisper.api_key_source,
        ),
        output=full.output,
        storage=full.storage,
        config_path=_redact_home(full.config_path),
        overrides_path=_redact_home(full.overrides_path),
        keychain_available=full.keychain_available,
    )


def _read_log_tail(config: Config) -> str:
    path = log_file_path(config)
    if not path.is_file():
        return ""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        log.warning("diagnostics: failed to read log file %s", path, exc_info=True)
        return ""
    return "\n".join(lines[-LOG_TAIL_LINES:])


@router.get("", response_model=DiagnosticsResponse)
async def get_diagnostics(job_id: str | None = None) -> DiagnosticsResponse:
    config = get_config()

    health = await health_probe()
    config_out = _diagnostics_config(config)
    log_tail = _scrub(_read_log_tail(config), config)
    job_status_summary = repo.count_jobs_by_status()

    job_info: DiagnosticsJobInfo | None = None
    if job_id is not None:
        job = repo.get_job(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"job {job_id} not found")
        job_info = DiagnosticsJobInfo(
            job_id=job.id,
            kind=job.kind,
            status=job.status,
            progress_stage=job.progress_stage,
            error=_scrub(job.error, config) if job.error else None,
            transcript_source=job.transcript_source,
        )

    return DiagnosticsResponse(
        daemon_version=DAEMON_VERSION,
        python_version=sys.version,
        platform=platform.platform(),
        health=health,
        config=config_out,
        log_tail=log_tail,
        job_status_summary=job_status_summary,
        job=job_info,
    )
