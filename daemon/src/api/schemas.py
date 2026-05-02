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
# POST /jobs (always async — work happens in the background, client subscribes
# to POST /ai/stream {job_id} to watch progress and receive summary tokens)
# ---------------------------------------------------------------------------


class JobCreateRequest(BaseModel):
    url: str
    kind: Literal["page", "youtube", "auto"] = "auto"
    page_text: str | None = None     # extension-extracted clean text (Readability)
    page_title: str | None = None
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


class JobDetails(JobSummary):
    """Full job with summary_md (raw_text length only, not the body)."""
    summary_md: str | None
    raw_text_length: int | None
    error: str | None
    video_id: str | None


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
# POST /ai/stream — unified streaming endpoint for ALL AI responses
# ---------------------------------------------------------------------------
#
# Request body:
#   { job_id, question? }
#
# Two modes:
#
#   Without `question` → SUBSCRIBE TO SUMMARY for this job. If extraction is
#   still running, wait for it (emitting "stage" events). When the summary
#   runs, emit "delta" tokens. If the summary is already cached, replay it
#   as a single delta + done. Multiple subscribers to the same job share
#   the same broker fanout.
#
#   With `question` → TRIGGER A NEW QA. Stream the answer tokens. On
#   completion, persist the user + assistant messages (visible via
#   GET /jobs/{id}/messages) and emit a `done` event with `message_id`
#   pointing at the assistant Message row.
#
# Response is text/event-stream. Each frame is a single line
# `data: <json>\n\n` where <json> is one of AIStreamEvent variants below.
# The stream ends with either `done` or `error`; the server closes after.

class AIStreamRequest(BaseModel):
    job_id: str
    question: str | None = None     # absent → summary mode; present → QA mode


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
    version: str
