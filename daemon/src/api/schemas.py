"""API contract — Pydantic models shared by all routes.

This module is the single source of truth for HTTP request/response shapes.
The JSDoc mirror lives at extension/src/lib/api-types.js and MUST be kept
in sync manually whenever this file changes.

When you change a model here, update api-types.js and bump
DAEMON_API_VERSION in daemon/src/config.py.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class JobKind(StrEnum):
    PAGE = "page"
    YOUTUBE = "youtube"
    # Generic media URL: anything yt-dlp can extract — direct mp4/webm,
    # HLS (.m3u8), DASH (.mpd), iframe embeds (Vimeo, Dailymotion, Twitch
    # VOD, Bunny, Brightcove, JW Player, Wistia, Streamable, SoundCloud,
    # Spotify, …). Distinguished from YOUTUBE because there's no
    # subtitle/captions fast path — everything goes straight to Whisper.
    MEDIA = "media"
    # PDF documents. Text-first via pypdf; if pypdf returns ~nothing the
    # daemon assumes the PDF is scanned/image-only and falls back to
    # multimodal vision (pages rendered to PNG, sent to the LLM via
    # ``image_url`` content). For http(s) URLs the daemon fetches itself;
    # for ``file://`` the extension uploads bytes via ``pdf_bytes_b64``.
    PDF = "pdf"


class JobStatus(StrEnum):
    QUEUED = "queued"        # in deferred queue, awaiting worker
    RUNNING = "running"      # worker actively processing (extraction or summarization)
    DONE = "done"            # summary_md filled, ready
    FAILED = "failed"


class TranscriptSource(StrEnum):
    YOUTUBE_API = "youtube_api"
    YOUTUBE_AUTO_CAPTIONS = "youtube_auto_captions"  # via yt-dlp --write-auto-sub
    WHISPER = "whisper"
    PAGE_EXTRACT = "page_extract"     # extension extracted via Readability
    TRAFILATURA = "trafilatura"       # daemon fallback for pages without page_text
    PDF_TEXT = "pdf_text"             # pypdf — text-first path on a native PDF
    PDF_VISION = "pdf_vision"         # multimodal OCR — scanned/image-only fallback


# Sources whose text comes from speech (so it may contain ASR artefacts, and
# may legitimately carry [MM:SS] timecodes). Page/PDF sources are excluded.
AUDIO_TRANSCRIPT_SOURCES = frozenset(
    {
        TranscriptSource.WHISPER,
        TranscriptSource.YOUTUBE_AUTO_CAPTIONS,
        TranscriptSource.YOUTUBE_API,
    }
)


class DeferredReason(StrEnum):
    TRANSCRIPT_UNAVAILABLE = "transcript_unavailable"
    TRANSCRIPT_BLOCKED = "transcript_blocked"
    NETWORK_ERROR = "network_error"


# ---------------------------------------------------------------------------
# Cookies (forwarded from chrome.cookies.getAll)
# ---------------------------------------------------------------------------


class Cookie(BaseModel):
    """A single browser cookie. Mirrors chrome.cookies.Cookie shape."""
    name: str
    value: str
    domain: str
    path: str = "/"
    secure: bool = False
    http_only: bool = False
    expires: float | None = None  # epoch seconds


# ---------------------------------------------------------------------------
# Media candidate (one of several playable sources discovered on the page)
# ---------------------------------------------------------------------------


class MediaCandidate(BaseModel):
    """One playable media source found by the extension's page scanner.

    Pages can carry multiple playable items (lecture page with several
    talks, news article with embedded video + promo, podcast page with
    iframe embed + native ``<audio>`` download link, …). The extension
    auto-picks the top-scored one as ``media_url`` of the job; the
    remaining candidates ride along on the job under
    ``JobDetails.alt_media_candidates`` so the sidepanel can surface a
    "wrong source?" chip without re-running the extraction.

    Fields are deliberately minimal — we don't need to round-trip the DOM
    element, just enough to identify the source and label it in UI.
    """
    media_url: str
    kind: Literal["video", "audio", "iframe"]
    label: str


# ---------------------------------------------------------------------------
# POST /jobs (always async — work happens in the background, client subscribes
# to POST /ai/stream {job_id} to watch progress and receive summary tokens)
# ---------------------------------------------------------------------------


class JobCreateRequest(BaseModel):
    url: str
    kind: Literal["page", "youtube", "media", "pdf", "auto"] = "auto"
    page_text: str | None = None     # extension-extracted clean text (Readability)
    page_title: str | None = None
    # Direct media stream URL discovered by the extension on the page (a
    # <video src=…>, <audio src=…>, or known iframe embed). yt-dlp's
    # generic + site-specific extractors handle it. When present, the
    # daemon prefers ``JobKind.MEDIA`` regardless of host. ``url`` stays
    # the human-visible page URL — used for library dedup, display, and
    # the "open source" link.
    media_url: str | None = None
    # Other playable sources the extension saw on the same page. Stored
    # on the Job row for the sidepanel's "wrong source?" picker (see
    # JobDetails.alt_media_candidates). The daemon never reads these
    # itself — they're inert UI state that round-trips through the DB.
    alt_media_candidates: list[MediaCandidate] | None = None
    # PDF bytes, base64-encoded. Set only when the source is a ``file://``
    # URL the daemon can't reach itself (the extension reads the file via
    # ``fetch()`` and forwards the bytes). For http(s) PDFs leave None and
    # the daemon fetches the URL with the cookies below.
    pdf_bytes_b64: str | None = None
    cookies: list[Cookie] | None = None


class JobCreateResponse(BaseModel):
    """Returned for every POST /jobs. Always 202 Accepted.

    Subscribe to POST /ai/stream {job_id} to follow extraction + summarization.
    The deferred-to-whisper transition (if any) arrives there as a
    `stage("queued", detail=<reason>)` event.
    """
    id: str
    kind: JobKind
    status: JobStatus    # "running" for fast paths, "queued" for whisper deferred


# ---------------------------------------------------------------------------
# GET /jobs (list) and GET /jobs/{id} (detail)
# ---------------------------------------------------------------------------


class JobSummary(BaseModel):
    """Job entry as shown in lists. No raw_text, no summary_md."""
    id: str
    url: str
    kind: JobKind
    status: JobStatus
    title: str | None
    duration_seconds: int | None
    progress_stage: str | None     # "extracting" | "transcribing" | "summarizing" | None when idle/done
    transcript_source: TranscriptSource | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


class TranscriptTranslationSummary(BaseModel):
    """One translation entry as shown on a Job's detail response.

    Just enough for the sidepanel to render chips (cached languages with
    status) — full text comes through ``GET /jobs/{id}/transcript?lang=``.
    """
    language_code: str
    status: Literal["queued", "running", "done", "failed"]
    progress_percent: int = 0
    error: str | None = None


class JobDetails(JobSummary):
    """Full job with summary_md and partial_summary for reconnect replay."""
    summary_md: str | None
    error: str | None
    video_id: str | None
    # Accumulated delta text while the job is still running. Lets a client
    # that reconnects mid-generation (or restarts the browser) replay the
    # buffered content without waiting for future deltas. None once done.
    partial_summary: str | None = None
    # ISO-639-1 code of the transcript's source language. ``None`` for
    # PDF / HTML jobs where we don't run language detection on extracted
    # text — UI shows "Original" without a code in that case.
    transcript_language: str | None = None
    # Translations of this job's transcript that have been requested
    # (cached, in-flight, or failed). The original language is NOT in
    # this list — it's served from ``Job.raw_text`` directly. Empty list
    # for jobs no-one ever translated.
    transcript_translations: list[TranscriptTranslationSummary] = []
    # Other playable sources the extension discovered on the page at
    # job-creation time. Surfaced by the sidepanel as a "wrong source?"
    # chip when non-empty — clicking opens the list so the user can
    # switch to a different candidate (handled client-side: create a new
    # job for that URL, delete the current one). Empty for YouTube jobs
    # (URL is unambiguous), PAGE/PDF jobs, and media pages with exactly
    # one candidate. Backfilled as ``[]`` for legacy jobs created before
    # this field existed.
    alt_media_candidates: list[MediaCandidate] = []


class JobListResponse(BaseModel):
    items: list[JobSummary]
    total: int


# ---------------------------------------------------------------------------
# Chat messages (Q&A history per job)
# ---------------------------------------------------------------------------


class Message(BaseModel):
    id: int
    job_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class MessagesListResponse(BaseModel):
    items: list[Message]


# ---------------------------------------------------------------------------
# POST /ai/qa — Q&A streaming endpoint
# ---------------------------------------------------------------------------
#
# Request body: { job_id, question }
#
# Triggers a new QA call. Streams answer tokens, persists user + assistant
# messages (visible via GET /jobs/{id}/messages), emits done with message_id.
#
# Response is text/event-stream. Each frame is `data: <json>\n\n`.
# The stream ends with either `done` or `error`.


class AIStreamRequest(BaseModel):
    job_id: str
    question: str


class AIStageEvent(BaseModel):
    """Coarse-grained progress signal. Frontend uses this for badges.

    Stages (free-form so we don't have to bump the contract for every new step):
    - "queued"       waiting for a worker (whisper queue)
    - "extracting"   pulling page text or YouTube transcript
    - "transcribing" Whisper running (slow)
    - "ready"        extraction complete, summary about to start
    - "summarizing"  LLM call in progress for summary (deltas follow)
    - "thinking"     LLM call in progress for QA (deltas follow)
    """
    type: Literal["stage"] = "stage"
    stage: str
    detail: str | None = None


class AIDeltaEvent(BaseModel):
    """Token chunk from the LLM — append to the bubble being shown."""
    type: Literal["delta"] = "delta"
    delta: str


class AIDoneEvent(BaseModel):
    """Terminal success event — always sent last on a successful stream.

    `content` carries the full text (useful for caching on the client).
    `message_id` is set only for QA mode (points at the assistant Message row).
    """
    type: Literal["done"] = "done"
    content: str
    message_id: int | None = None


class AIErrorEvent(BaseModel):
    """Terminal failure event."""
    type: Literal["error"] = "error"
    error: str


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    queue_size: int
    queue_running: int
    llm_backend_reachable: bool      # any OpenAI-compatible /v1/models pingable
    llm_backend_models: list[str]
    # Set when the backend responded but rejected the request (401/403) —
    # distinguishes "unauthorized" (bad/missing llm.api_key) from a plain
    # network-level "unreachable". None when the probe succeeded or the
    # failure was a connection-level error.
    llm_backend_error: str | None = None
    version: str


# ---------------------------------------------------------------------------
# GET/PATCH /config, POST /config/test — daemon settings editable from the
# extension's options page instead of hand-editing tldr.yaml. Secrets are
# write-only: a key is never echoed back, only its presence/hint/source.
# ---------------------------------------------------------------------------


ApiKeySource = Literal["env", "keychain", "file", "inline", "none"]
ApiKeyStorage = Literal["file", "keychain", "inline"]


class LLMConfigOut(BaseModel):
    """LLM settings as reported by ``GET /config`` / ``PATCH /config``.

    ``api_key_set``/``api_key_hint``/``api_key_source`` are the only trace
    of the key that ever appears in a response — the key itself never is.
    """
    base_url: str
    model: str
    context_length: int
    single_pass_token_limit: int
    max_concurrent_calls: int
    reasoning_effort: str | None
    api_key_set: bool
    api_key_hint: str | None      # last 4 chars of the resolved key, or None
    api_key_source: ApiKeySource


class WhisperConfigOut(BaseModel):
    base_url: str
    model: str
    max_upload_mb: int


class OutputConfigOut(BaseModel):
    language: str


class ConfigResponse(BaseModel):
    llm: LLMConfigOut
    whisper: WhisperConfigOut
    output: OutputConfigOut
    config_path: str        # absolute path to tldr.yaml (read-only template)
    overrides_path: str     # absolute path to tldr.local.yaml (PATCH target)
    # Whether the OS keychain backend is actually usable (a real backend,
    # not keyring.backends.fail.Keyring) — not just whether the `keyring`
    # package is importable. Drives the default api_key_storage choice on
    # PATCH and the options-page UI (disable the keychain option + hint
    # when false). See config.keychain_backend_available().
    keychain_available: bool


class LLMConfigPatch(BaseModel):
    """Partial update for ``llm``. Only fields present in the request body
    are applied; everything else keeps its current value.

    ``api_key``/``api_key_storage`` are write-only side channels — see
    ``.claude/daemon.md`` / the ``/config`` route module docstring for the
    storage-selection logic. An absent or empty ``api_key`` leaves the
    currently configured key untouched even if ``api_key_storage`` changes
    (the existing key is migrated to the new storage instead).
    """
    base_url: str | None = None
    model: str | None = None
    context_length: int | None = None
    single_pass_token_limit: int | None = None
    max_concurrent_calls: int | None = None
    reasoning_effort: str | None = None
    api_key: str | None = None
    api_key_storage: ApiKeyStorage | None = None


class WhisperConfigPatch(BaseModel):
    base_url: str | None = None
    model: str | None = None
    max_upload_mb: int | None = None


class OutputConfigPatch(BaseModel):
    language: str | None = None


class ConfigPatchRequest(BaseModel):
    llm: LLMConfigPatch | None = None
    whisper: WhisperConfigPatch | None = None
    output: OutputConfigPatch | None = None


class ConfigPatchResponse(ConfigResponse):
    # True when a change (currently: llm.max_concurrent_calls) can't take
    # effect on the running process — the asyncio.Semaphore it sizes is
    # bound to the live event loop and can't be resized in place.
    restart_required: bool
    # Write-then-read-back check: whenever this PATCH (re)wrote the API
    # key, it's read back via the exact accessor the daemon uses at call
    # time (LLMConfig.effective_api_key) and compared to what was saved.
    # True when this PATCH didn't touch the API key at all (nothing to
    # verify) or when the read-back matched. A failed verification is
    # reported here but never rolls back the save.
    api_key_verified: bool
    # Human-readable reason when api_key_verified is False. Never contains
    # the API key value itself. None when api_key_verified is True.
    api_key_verify_error: str | None = None


class ConfigTestLLMOverrides(BaseModel):
    """Overrides probed by ``POST /config/test`` instead of the saved
    config, without persisting anything. All optional — unset fields fall
    back to the currently saved value."""
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None


class ConfigTestRequest(BaseModel):
    """Empty body (``{}``) tests the currently saved ``llm`` config."""
    llm: ConfigTestLLMOverrides | None = None


class ConfigTestResponse(BaseModel):
    """Always HTTP 200 — this endpoint reports probe failures in the body
    rather than raising, since a 401/timeout/etc. IS the useful answer.

    ``step`` marks which probe ran last: "models" (``GET {base_url}/models``)
    or "completion" (a minimal chat completion against ``model``). ``detail``
    carries the provider's error message verbatim (truncated), which is the
    whole point of this endpoint — never redacted except for the API key
    itself.
    """
    ok: bool
    step: Literal["models", "completion"] | None = None
    status_code: int | None = None
    detail: str | None = None
    models: list[str] = []
    latency_ms: int | None = None
