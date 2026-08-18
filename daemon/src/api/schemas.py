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

from pydantic import BaseModel, Field

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
    # When this row appeared ON THIS MACHINE — distinct from created_at (when
    # the material was processed). Equal to created_at for every normally
    # created job; differs only for a job brought in via POST /jobs/import,
    # which preserves the exporting machine's created_at but sets
    # added_at=now. The Library can show "imported <date>" on rows where the
    # two differ. See Job.added_at / repo.delete_jobs_older_than.
    added_at: datetime
    updated_at: datetime
    completed_at: datetime | None


class TranscriptTranslationSummary(BaseModel):
    """One translation entry as shown on a Job's detail response.

    Just enough for the sidepanel to render chips (cached languages with
    status) — full text comes through ``GET /jobs/{id}/transcript?lang=``.
    """
    language_code: str
    status: Literal["queued", "running", "done", "partial", "failed"]
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
# POST /jobs/export, POST /jobs/import — moving a library between machines,
# and letting a machine that can't run local models still read summaries/
# transcripts produced elsewhere. See ``storage.bundle`` for the pack/unpack
# implementation and the zip layout it reads and writes.
# ---------------------------------------------------------------------------


class JobExportRequest(BaseModel):
    """Body for ``POST /jobs/export``. The client is expected to have
    already filtered ``ids`` down to jobs it believes are exportable
    (``status == "done"``) — the daemon re-checks independently and
    silently skips anything that isn't, rather than erroring on a client
    list that went stale between the check and the request."""
    ids: list[str] = Field(min_length=1, max_length=1000)


class ImportedJob(BaseModel):
    """One job actually inserted by ``POST /jobs/import``, under a freshly
    minted id — the id it had on the exporting machine is never reused
    (see ``storage.bundle.import_bundle``)."""
    job_id: str
    url: str
    title: str | None


class ImportIssue(BaseModel):
    """One job from the bundle that was NOT inserted — either because a
    ``done`` job with the same URL already exists on this machine
    (``reason="duplicate"``) or because something about that job's entry
    raised while importing (``reason`` carries the exception message; the
    rest of the bundle still gets imported — each job is its own
    transaction, see ``storage.bundle.import_bundle``)."""
    url: str
    title: str | None
    reason: str


class JobImportResponse(BaseModel):
    """Always HTTP 200 — a bundle can be partially imported (some jobs
    duplicate, some malformed) without that being a request-level
    failure. Only a bundle that fails validation BEFORE any per-job work
    starts (bad zip, no manifest, wrong format/version, unsafe member
    names, oversized upload) is rejected with 400 instead — see
    ``POST /jobs/import``."""
    imported: list[ImportedJob]
    skipped: list[ImportIssue]
    failed: list[ImportIssue]


# ---------------------------------------------------------------------------
# POST /jobs/delete — bulk delete, e.g. from a Library multi-select.
# ---------------------------------------------------------------------------


class JobDeleteRequest(BaseModel):
    """Body for ``POST /jobs/delete``. Ids that no longer exist are counted
    as not-deleted rather than raising — same "client's own list may have
    gone stale" tolerance as ``JobExportRequest``."""
    ids: list[str] = Field(min_length=1, max_length=1000)


class JobDeleteResponse(BaseModel):
    """Always HTTP 200. ``deleted`` counts only ids that actually matched a
    row — each is removed via ``repo.delete_job`` (same audio/frame/event
    cleanup as a single ``DELETE /jobs/{id}``), an unknown id simply doesn't
    add to the count."""
    deleted: int


# ---------------------------------------------------------------------------
# Chat messages (Q&A history per job)
# ---------------------------------------------------------------------------


class FrameRef(BaseModel):
    """One video frame worth showing the user a thumbnail for.

    Two producers share this exact shape, so the client needs only ONE
    thumbnail renderer:

    - The QA LOOK step (``llm/qa.py``) — at most one ``FrameRef`` per
      inspected deixis moment, and ONLY when the vision model reported the
      frames as genuinely relevant to the question (see
      ``llm.qa.VisionResult`` / ``qa_frames.txt``). A moment that was
      looked at but found irrelevant contributes its finding text to the
      synthesis prompt same as before, but never produces a ``FrameRef`` —
      no thumbnail for "we checked and there was nothing to see".
    - ``POST /jobs/{id}/frames`` (see ``FrameFetchResponse``) — one
      ``FrameRef`` per extracted frame for the moment the user clicked a
      "look" affordance on, all sharing that moment's seconds/timecode/
      phrase. No vision call involved; the user looks themselves.

    ``frame_url`` is a path rooted at the daemon (``GET
    /jobs/{job_id}/frames/{rel_path}``), not an absolute URL — the client
    prefixes it with whatever base URL it's using to reach this daemon,
    the same way every other daemon-served resource works.

    QA-produced refs are persisted verbatim (as a JSON list) on the
    assistant ``Message`` row (``Message.frame_refs_json``) so reopening a
    job's chat history renders the identical thumbnail without redoing any
    of the LOOK step's work. ``POST /jobs/{id}/frames`` refs are not
    persisted anywhere — the client re-fetches (cheaply, see
    ``workers.frames.fetch_frames``'s ``reuse_existing``) if it wants them
    again after a reload.
    """
    seconds: float
    timecode: str
    phrase: str
    frame_url: str


class DeixisMoment(BaseModel):
    """One moment where this job's transcript speech points at the video's
    picture (see ``workers.deixis.DeixisCandidate``) — offered to the
    client so a summary line's ``[MM:SS]`` marker landing near one can show
    a "look" affordance (see ``GET /jobs/{id}/moments``).

    EXTERNAL candidates are never included here: they point OUTSIDE the
    video (a link, an article number in the description) and fetching a
    frame for one would show nothing relevant to what was said — same rule
    the QA LOOK step enforces daemon-side, independent of any client.
    """
    seconds: float
    timecode: str
    phrase: str
    category: Literal["action", "object"]


class MomentsListResponse(BaseModel):
    """``GET /jobs/{id}/moments`` — empty ``items`` (never an error) for a
    job that doesn't qualify for deixis candidates at all (page/PDF, no
    transcript, or a transcript with no candidates) — see
    ``workers.deixis.candidates_for_job``."""
    items: list[DeixisMoment]


class FrameFetchRequest(BaseModel):
    """Body for ``POST /jobs/{id}/frames`` — the moment (in seconds) the
    user clicked the "look" affordance for. Must match one of this job's
    own ``DeixisMoment`` entries (see ``GET /jobs/{id}/moments``); the
    route 404s otherwise, and rejects an EXTERNAL-category match the same
    way the QA LOOK step's daemon-side guard does (defence in depth — an
    EXTERNAL moment is never offered by ``GET /moments`` in the first
    place, but the route re-checks rather than trusting the caller sent
    back exactly what it was given).
    """
    seconds: float


class FrameFetchResponse(BaseModel):
    """Returned by ``POST /jobs/{id}/frames`` on success. ``items`` uses
    the SAME ``FrameRef`` shape the QA LOOK step returns (see ``FrameRef``)
    — one thumbnail-row renderer for both. Always non-empty here; the
    failure cases (unknown moment, EXTERNAL moment, per-job frame budget
    already spent, section-download failure after retries) are raised as
    HTTP errors instead of an empty/partial body — see the route.
    """
    items: list[FrameRef]


class Message(BaseModel):
    id: int
    job_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    # Empty for user messages and for assistant messages that never looked
    # at a frame, or looked but found nothing relevant. See `FrameRef`.
    frame_refs: list[FrameRef] = []


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


class AIFramesEvent(BaseModel):
    """Emitted once, after the LOOK step finishes, ONLY when at least one
    inspected moment turned out relevant (see `FrameRef`). Never emitted
    for a QA turn that ran no LOOK step, or where every inspected moment
    was irrelevant — a frame must have actually contributed to the answer
    to be worth showing the user a thumbnail for.

    Client renders `items` as a small thumbnail row under the finished
    answer bubble; the same list is persisted on the assistant `Message`
    row (`Message.frame_refs`) so history reload renders identically.
    """
    type: Literal["frames"] = "frames"
    items: list[FrameRef]


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
    """Whisper settings as reported by ``GET /config`` / ``PATCH /config``.

    Same write-only key-storage story as ``LLMConfigOut`` — see its
    docstring; the two sections' keys are fully independent (separate
    keychain entry, separate key file, separate env var).
    """
    base_url: str
    model: str
    max_upload_mb: int
    api_key_set: bool
    api_key_hint: str | None      # last 4 chars of the resolved key, or None
    api_key_source: ApiKeySource


class OutputConfigOut(BaseModel):
    language: str


class StorageConfigOut(BaseModel):
    """Retention setting as reported by ``GET /config`` / ``PATCH /config``.
    ``0`` means the retention sweep is disabled — see
    ``workers.retention.retention_worker``."""
    retention_days: int


class QaConfigOut(BaseModel):
    """Q&A web-search setting as reported by ``GET /config`` /
    ``PATCH /config``. See ``config.QaConfig`` for what ``web_search``
    actually gates."""
    web_search: bool


class ConfigResponse(BaseModel):
    llm: LLMConfigOut
    whisper: WhisperConfigOut
    output: OutputConfigOut
    storage: StorageConfigOut
    qa: QaConfigOut
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
    """Partial update for ``whisper``. Same write-only ``api_key``/
    ``api_key_storage`` side channel as ``LLMConfigPatch`` — see its
    docstring."""
    base_url: str | None = None
    model: str | None = None
    max_upload_mb: int | None = None
    api_key: str | None = None
    api_key_storage: ApiKeyStorage | None = None


class OutputConfigPatch(BaseModel):
    language: str | None = None


# 10 years — generous enough that no real user needs more, tight enough to
# reject a fat-fingered value (e.g. accidentally typing a year like "2025")
# that would effectively disable retention by accident.
_MAX_RETENTION_DAYS = 3650


class StorageConfigPatch(BaseModel):
    """Partial update for ``storage``. ``0`` disables the retention sweep
    entirely and must stay expressible (not treated as "unset")."""
    retention_days: int | None = Field(default=None, ge=0, le=_MAX_RETENTION_DAYS)


class QaConfigPatch(BaseModel):
    """Partial update for ``qa``."""
    web_search: bool | None = None


class ConfigPatchRequest(BaseModel):
    llm: LLMConfigPatch | None = None
    whisper: WhisperConfigPatch | None = None
    output: OutputConfigPatch | None = None
    storage: StorageConfigPatch | None = None
    qa: QaConfigPatch | None = None


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
    # Same write-then-read-back check as api_key_verified/api_key_verify_error
    # above, but for whisper.api_key. Parallel top-level fields (rather than
    # nesting) keep the existing llm-only fields meaning exactly what they
    # meant before this field was added — no breaking change for old clients
    # that only ever wrote llm.api_key.
    whisper_api_key_verified: bool
    whisper_api_key_verify_error: str | None = None


class ConfigTestLLMOverrides(BaseModel):
    """Overrides probed by ``POST /config/test`` instead of the saved
    config, without persisting anything. All optional — unset fields fall
    back to the currently saved value."""
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None


class ConfigTestWhisperOverrides(BaseModel):
    """Same shape as ``ConfigTestLLMOverrides``, for ``target="whisper"``."""
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None


class ConfigTestRequest(BaseModel):
    """Empty body (``{}``) tests the currently saved ``llm`` config — the
    default ``target`` preserves that exact old contract. Set
    ``target="whisper"`` (with optional ``whisper`` overrides) to probe the
    Whisper backend instead; only the reachability step runs for Whisper
    (see ``ConfigTestResponse`` — a transcription probe would need an audio
    file, so there's no "completion" equivalent)."""
    target: Literal["llm", "whisper"] = "llm"
    llm: ConfigTestLLMOverrides | None = None
    whisper: ConfigTestWhisperOverrides | None = None


class ConfigTestStepResult(BaseModel):
    """One step of a ``target="llm"`` ``POST /config/test`` run.

    ``ok=None`` means the step was never attempted — an earlier step it
    depends on failed (or the endpoint's overall time budget ran out) —
    NOT that it was skipped silently: the full step list is always
    returned (see ``ConfigTestResponse``), so the caller can show the user
    exactly how far the probe got. ``detail`` is a human-readable sentence,
    not a stack trace — provider error text is still relayed verbatim
    (truncated, key-redacted) inside it where relevant.
    """
    step: Literal["reachable", "models", "completion", "thinking", "context", "translation"]
    ok: bool | None
    detail: str | None = None


class ConfigTestSuggestions(BaseModel):
    """Aggregated proposals from a ``target="llm"`` test run. ``None`` on any
    field means "no suggestion" — either the step that would produce it
    never ran, or it ran and found nothing worth changing. Applying a
    suggestion is a separate, explicit ``PATCH /config`` the user triggers;
    this endpoint never writes anything itself."""
    reasoning_effort: str | None = None
    context_length: int | None = None
    single_pass_token_limit: int | None = None


class ConfigTestResponse(BaseModel):
    """Always HTTP 200 — this endpoint reports probe failures in the body
    rather than raising, since a 401/timeout/etc. IS the useful answer.

    Two shapes share this model:

    - ``target="whisper"``: the legacy flat shape — ``step`` ("models" is
      the only value ever reported, a transcription probe would need an
      audio file), ``status_code``, ``detail`` describe the single
      reachability probe. ``steps``/``suggestions`` are left empty/default.
    - ``target="llm"`` (default): the step-by-step probe described in
      ``.claude/llm.md`` — ``steps`` carries one ``ConfigTestStepResult``
      per stage (reachable → models → completion → thinking → context →
      translation, always in that order, always all six present even when
      later ones are ``ok=None``) and ``suggestions`` aggregates whatever
      the run learned. Top-level ``ok`` reflects only the three
      connectivity/model steps (reachable/models/completion) — thinking/
      context/translation are diagnostic, not pass/fail gates for the
      backend being usable at all. ``models``/``latency_ms`` stay populated
      (model list, total wall time) for both shapes.

    ``detail`` (top-level, legacy) carries the provider's error message
    verbatim (truncated), which is the whole point of this endpoint — never
    redacted except for the API key itself. Same redaction applies inside
    every ``ConfigTestStepResult.detail``.
    """
    ok: bool
    step: Literal["models", "completion"] | None = None
    status_code: int | None = None
    detail: str | None = None
    models: list[str] = []
    latency_ms: int | None = None
    steps: list[ConfigTestStepResult] = []
    suggestions: ConfigTestSuggestions = Field(default_factory=ConfigTestSuggestions)


# ---------------------------------------------------------------------------
# GET /diagnostics — a report meant to leave the machine (pasted into a
# GitHub issue), so its shape is dictated entirely by what must NEVER be in
# it. See src/api/diagnostics.py for the scrubbing this response is built
# through — this model only documents the result.
# ---------------------------------------------------------------------------


class DiagnosticsJobInfo(BaseModel):
    """Metadata for the single job requested via ``?job_id=`` — deliberately
    everything BUT the content: no ``title``, ``url``, ``raw_text``, or
    ``summary_md``. ``error`` is scrubbed the same way the log tail is (see
    ``diagnostics.py#_scrub``) since a transcript/page error message can
    itself contain a URL or a home-directory path."""
    job_id: str
    kind: str
    status: str
    progress_stage: str | None
    error: str | None
    transcript_source: str | None


class DiagnosticsLLMConfigOut(BaseModel):
    """``LLMConfigOut`` minus ``api_key_hint`` — see
    ``DiagnosticsConfigOut`` for why the hint is dropped rather than just
    scrubbed. ``base_url`` is passed through ``diagnostics.py``'s
    ``_redact_api_keys`` (but NOT ``_redact_urls``): a configured backend
    address is exactly the kind of thing a diagnosis needs (local or
    cloud), unlike a page/video URL the user processed."""
    base_url: str
    model: str
    context_length: int
    single_pass_token_limit: int
    max_concurrent_calls: int
    reasoning_effort: str | None
    api_key_set: bool
    api_key_source: ApiKeySource


class DiagnosticsWhisperConfigOut(BaseModel):
    """``WhisperConfigOut`` minus ``api_key_hint`` — see
    ``DiagnosticsLLMConfigOut``."""
    base_url: str
    model: str
    max_upload_mb: int
    api_key_set: bool
    api_key_source: ApiKeySource


class DiagnosticsConfigOut(BaseModel):
    """Same information as ``ConfigResponse``, minus what a report leaving
    the machine must never carry:

    - ``api_key_hint`` (either section) is DROPPED, not merely redacted.
      It exists in ``GET /config`` so the options-page UI can show "key
      ending in ...XXXX" — useful there because the user already knows
      their own key and is just confirming which one is saved. In a
      diagnostics report read by a stranger it answers no question
      ``api_key_set`` doesn't already answer, while being 4 real
      characters of a live secret. Dropping the field is also
      structurally stronger than nulling it — nothing has to remember to
      scrub it, because it was never on this model to begin with.
    - ``config_path``/``overrides_path`` are scrubbed (home directory →
      ``~``) rather than dropped — they're absolute filesystem paths on
      the reporter's own machine, useful for "which file did you edit",
      but the leading ``/Users/<name>`` leaks the account name otherwise.
    """
    llm: DiagnosticsLLMConfigOut
    whisper: DiagnosticsWhisperConfigOut
    output: OutputConfigOut
    storage: StorageConfigOut
    config_path: str
    overrides_path: str
    keychain_available: bool


class DiagnosticsResponse(BaseModel):
    """Everything ``GET /diagnostics`` hands back for the user to review and
    paste into a bug report themselves — nothing here is ever sent anywhere
    by the daemon. See the module docstring of ``src/api/diagnostics.py``
    for the full list of what's deliberately excluded.
    """
    daemon_version: str
    python_version: str
    platform: str
    health: HealthResponse
    config: DiagnosticsConfigOut
    # Already scrubbed (see diagnostics.py#_scrub) — home paths replaced with
    # "~", non-local URLs replaced with a placeholder, either API key
    # redacted if it somehow ended up in a log line.
    log_tail: str
    # status -> count, e.g. {"done": 12, "failed": 1}.
    job_status_summary: dict[str, int]
    # Populated only when the request carried ?job_id=; null otherwise.
    job: DiagnosticsJobInfo | None = None
