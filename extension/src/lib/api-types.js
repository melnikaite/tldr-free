// API contract — JSDoc mirror of daemon/src/api/schemas.py.
// Kept in sync MANUALLY. Whenever schemas.py changes, update this file too.
//
// Other files reference these types via JSDoc:
//   /** @import { JobDetails, AIStreamEvent } from "./api-types.js" */
// then annotate variables/parameters with @type / @param.

// ---------------------------------------------------------------------------
// Enums (string literal unions in JSDoc-land)
// ---------------------------------------------------------------------------

/** @typedef {"page" | "youtube"} JobKind */

/** @typedef {"queued" | "running" | "done" | "failed"} JobStatus */

/** @typedef {"youtube_api" | "youtube_auto_captions" | "whisper" | "page_extract" | "trafilatura"} TranscriptSource */

/** @typedef {"transcript_unavailable" | "transcript_blocked" | "network_error"} DeferredReason */

// ---------------------------------------------------------------------------
// Cookie (forwarded from chrome.cookies.getAll)
// ---------------------------------------------------------------------------

/**
 * @typedef {object} Cookie
 * @property {string} name
 * @property {string} value
 * @property {string} domain
 * @property {string} path
 * @property {boolean} secure
 * @property {boolean} http_only
 * @property {number | null} expires
 */

// ---------------------------------------------------------------------------
// POST /jobs (always async — 202 Accepted; client subscribes via /ai/stream)
// ---------------------------------------------------------------------------

/**
 * @typedef {object} JobCreateRequest
 * @property {string} url
 * @property {"page" | "youtube" | "auto"} kind
 * @property {string | null} [page_text]
 * @property {string | null} [page_title]
 * @property {Cookie[] | null} [cookies]
 */

/**
 * @typedef {object} JobCreateResponse
 * @property {string} id
 * @property {JobKind} kind
 * @property {JobStatus} status                          - usually "running" or "queued"
 */

// ---------------------------------------------------------------------------
// GET /jobs
// ---------------------------------------------------------------------------

/**
 * @typedef {object} JobSummary
 * @property {string} id
 * @property {string} url
 * @property {JobKind} kind
 * @property {JobStatus} status
 * @property {string | null} title
 * @property {number | null} duration_seconds
 * @property {string | null} progress_stage
 * @property {TranscriptSource | null} transcript_source
 * @property {string} created_at  ISO datetime string
 * @property {string} updated_at
 * @property {string | null} completed_at
 */

/**
 * @typedef {JobSummary & {
 *   summary_md: string | null,
 *   raw_text_length: number | null,
 *   error: string | null,
 *   video_id: string | null
 * }} JobDetails
 */

/**
 * @typedef {object} JobListResponse
 * @property {JobSummary[]} items
 * @property {number} total
 */

// ---------------------------------------------------------------------------
// Chat history (per-job Q&A persistence)
// ---------------------------------------------------------------------------

/**
 * @typedef {object} ChatMessage
 * @property {number} id
 * @property {string} job_id
 * @property {"user" | "assistant"} role
 * @property {string} content
 * @property {string} created_at  ISO datetime string
 */

/**
 * @typedef {object} MessagesListResponse
 * @property {ChatMessage[]} items
 */

// ---------------------------------------------------------------------------
// POST /ai/stream — unified streaming endpoint for ALL AI responses
// ---------------------------------------------------------------------------
// Body shape:
//   { job_id, question? }
//
// Without `question` → SUMMARY mode: subscribe to the job's extraction +
// summarization lifecycle (live or replay cached).
//
// With `question` → QA mode: trigger a new QA call, persist the user +
// assistant messages, stream the answer.
//
// Response is text/event-stream. Each frame: `data: <json>\n\n`. Parse
// <json> as one of AIStreamEvent variants below. Stream ends with `done`
// or `error`.

/**
 * @typedef {object} AIStreamRequest
 * @property {string} job_id
 * @property {string} [question]
 */

/**
 * @typedef {object} AIStageEvent
 * @property {"stage"} type
 * @property {string} stage   - free-form: "queued" | "extracting" | "transcribing" | "ready" | "summarizing" | "thinking" | ...
 * @property {string | null} [detail]
 */

/**
 * @typedef {object} AIDeltaEvent
 * @property {"delta"} type
 * @property {string} delta - token chunk to append to the message bubble
 */

/**
 * @typedef {object} AIDoneEvent
 * @property {"done"} type
 * @property {string} content - full text (useful for caching)
 * @property {number | null} message_id - assistant Message row id (QA mode only)
 */

/**
 * @typedef {object} AIErrorEvent
 * @property {"error"} type
 * @property {string} error
 */

/** @typedef {AIStageEvent | AIDeltaEvent | AIDoneEvent | AIErrorEvent} AIStreamEvent */

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------

/**
 * @typedef {object} HealthResponse
 * @property {"ok" | "degraded"} status
 * @property {number} queue_size
 * @property {number} queue_running
 * @property {boolean} llm_backend_reachable
 * @property {string[]} llm_backend_models
 * @property {string} version
 */

// Marker export so editors recognise this as an ES module.
export {};
