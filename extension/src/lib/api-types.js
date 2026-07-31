// API contract — JSDoc mirror of daemon/src/api/schemas.py.
// Kept in sync MANUALLY. Whenever schemas.py changes, update this file too.
//
// Other files reference these types via JSDoc:
//   /** @import { JobDetails, AIStreamEvent } from "./api-types.js" */
// then annotate variables/parameters with @type / @param.

// ---------------------------------------------------------------------------
// Enums (string literal unions in JSDoc-land)
// ---------------------------------------------------------------------------

/** @typedef {"page" | "youtube" | "media" | "pdf"} JobKind */

/** @typedef {"queued" | "running" | "done" | "failed"} JobStatus */

/** @typedef {"youtube_api" | "youtube_auto_captions" | "whisper" | "page_extract" | "trafilatura" | "pdf_text" | "pdf_vision"} TranscriptSource */

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

/**
 * One playable media source discovered by extract.js. The top-scored one
 * drives ``media_url`` of the job; the rest ride along as
 * ``alt_media_candidates`` so the sidepanel can render a "wrong source?"
 * picker without re-running the page scanner.
 *
 * @typedef {object} MediaCandidate
 * @property {string} media_url
 * @property {"video" | "audio" | "iframe"} kind
 * @property {string} label   - Human-readable: <title> attr / aria-label / filename / "Video 2"
 */

// ---------------------------------------------------------------------------
// POST /jobs (always async — 202 Accepted; client subscribes via /ai/stream)
// ---------------------------------------------------------------------------

/**
 * @typedef {object} JobCreateRequest
 * @property {string} url
 * @property {"page" | "youtube" | "media" | "pdf" | "auto"} kind
 * @property {string | null} [page_text]
 * @property {string | null} [page_title]
 * @property {string | null} [media_url]       - direct media URL (yt-dlp-extractable). Sets kind=media when present under auto.
 * @property {MediaCandidate[] | null} [alt_media_candidates]  - other playable sources on the same page; populates JobDetails.alt_media_candidates for the "wrong source?" picker
 * @property {string | null} [pdf_bytes_b64]   - base64 PDF bytes (file:// only; http(s) PDFs are fetched daemon-side)
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
 * @typedef {object} TranscriptTranslationSummary
 * @property {string} language_code
 * @property {"queued" | "running" | "done" | "failed"} status
 * @property {number} progress_percent
 * @property {string | null} [error]
 */

/**
 * Response of GET /jobs/{id}/transcript?lang=…
 *
 * When ``is_pending`` is true, ``text`` is null and the UI should show a
 * placeholder + refetch on the next job_event. ``404`` is reserved for
 * "no such job" and "no such translation row" — in-flight transcripts /
 * translations return 200 with ``is_pending: true``.
 *
 * @typedef {object} TranscriptResponse
 * @property {string | null} text          - full raw_text (or translated text); null when is_pending
 * @property {string | null} language_code - ISO-639-1; null for PDF/HTML jobs
 * @property {boolean} is_original         - true when serving Job.raw_text
 * @property {boolean} [is_pending]        - true when work is still in flight
 */

/**
 * @typedef {JobSummary & {
 *   summary_md: string | null,
 *   error: string | null,
 *   video_id: string | null,
 *   partial_summary: string | null,
 *   transcript_language: string | null,
 *   transcript_translations: TranscriptTranslationSummary[],
 *   alt_media_candidates: MediaCandidate[]
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
 * @property {string | null} [llm_backend_error]
 * @property {string} version
 */

// ---------------------------------------------------------------------------
// GET/PATCH /config, POST /config/test — daemon settings editable from the
// options page instead of hand-editing tldr.yaml. Secrets are write-only:
// api_key is never echoed back, only api_key_set/api_key_hint/api_key_source.
// ---------------------------------------------------------------------------

/** @typedef {"env" | "keychain" | "file" | "inline" | "none"} ApiKeySource */

/** @typedef {"file" | "keychain" | "inline"} ApiKeyStorage */

/**
 * @typedef {object} LLMConfigOut
 * @property {string} base_url
 * @property {string} model
 * @property {number} context_length
 * @property {number} single_pass_token_limit
 * @property {number} max_concurrent_calls
 * @property {string | null} reasoning_effort
 * @property {boolean} api_key_set
 * @property {string | null} api_key_hint     - last 4 chars of the resolved key, or null
 * @property {ApiKeySource} api_key_source
 */

/**
 * @typedef {object} WhisperConfigOut
 * @property {string} base_url
 * @property {string} model
 * @property {number} max_upload_mb
 */

/**
 * @typedef {object} OutputConfigOut
 * @property {string} language
 */

/**
 * @typedef {object} ConfigResponse
 * @property {LLMConfigOut} llm
 * @property {WhisperConfigOut} whisper
 * @property {OutputConfigOut} output
 * @property {string} config_path      - absolute path to tldr.yaml (read-only template)
 * @property {string} overrides_path   - absolute path to tldr.local.yaml (PATCH target)
 * @property {boolean} keychain_available - whether the OS keychain backend is actually
 *   usable (real backend, not a null/fail one) — drives the default api_key_storage
 *   choice on PATCH and whether the options page offers "OS keychain" at all.
 */

/**
 * Partial update for `llm` in PATCH /config. Only fields present in the
 * request body are applied. `api_key`/`api_key_storage` are write-only —
 * an absent or empty `api_key` leaves the currently configured key
 * untouched even if `api_key_storage` changes (the existing key is
 * migrated to the new storage instead).
 *
 * @typedef {object} LLMConfigPatch
 * @property {string} [base_url]
 * @property {string} [model]
 * @property {number} [context_length]
 * @property {number} [single_pass_token_limit]
 * @property {number} [max_concurrent_calls]
 * @property {string | null} [reasoning_effort]
 * @property {string} [api_key]              - write-only; unset/empty = unchanged
 * @property {ApiKeyStorage} [api_key_storage]  - write-only; default "keychain" when
 *   the OS keychain backend is available (see ConfigResponse.keychain_available),
 *   else "file"
 */

/**
 * @typedef {object} WhisperConfigPatch
 * @property {string} [base_url]
 * @property {string} [model]
 * @property {number} [max_upload_mb]
 */

/**
 * @typedef {object} OutputConfigPatch
 * @property {string} [language]
 */

/**
 * @typedef {object} ConfigPatchRequest
 * @property {LLMConfigPatch} [llm]
 * @property {WhisperConfigPatch} [whisper]
 * @property {OutputConfigPatch} [output]
 */

/**
 * Same shape as ConfigResponse plus:
 *   - `restart_required`: true when a change (currently only
 *     `llm.max_concurrent_calls`) can't take effect on the running process
 *     and needs a daemon restart.
 *   - `api_key_verified` / `api_key_verify_error`: write-then-read-back
 *     check whenever this PATCH (re)wrote the API key — the freshly-saved
 *     config is read back the same way the daemon does at call time and
 *     compared to what was saved. `api_key_verified` is true when this
 *     PATCH didn't touch the key at all (nothing to verify) or the
 *     read-back matched; `api_key_verify_error` is a redacted reason
 *     string (never the key itself) when it's false, else null. A failed
 *     verification does NOT roll back the save.
 *
 * @typedef {ConfigResponse & {
 *   restart_required: boolean,
 *   api_key_verified: boolean,
 *   api_key_verify_error: string | null,
 * }} ConfigPatchResponse
 */

/**
 * @typedef {object} ConfigTestLLMOverrides
 * @property {string} [base_url]
 * @property {string} [model]
 * @property {string} [api_key]
 */

/**
 * Empty body ({}) tests the currently saved llm config.
 *
 * @typedef {object} ConfigTestRequest
 * @property {ConfigTestLLMOverrides} [llm]
 */

/**
 * Always HTTP 200 — probe failures are reported in the body (never thrown)
 * since a 401/timeout/etc. IS the useful answer this endpoint exists to
 * give. `step` marks which probe ran last: "models" (GET {base_url}/models)
 * or "completion" (a minimal chat completion). `detail` is the provider's
 * error message verbatim (truncated to 2000 chars), with the API key itself
 * scrubbed out if it happened to be echoed back.
 *
 * @typedef {object} ConfigTestResponse
 * @property {boolean} ok
 * @property {"models" | "completion" | null} step
 * @property {number | null} status_code
 * @property {string | null} detail
 * @property {string[]} models
 * @property {number | null} latency_ms
 */

// Marker export so editors recognise this as an ES module.
export {};
