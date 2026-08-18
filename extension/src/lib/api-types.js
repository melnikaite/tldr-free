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
 * @property {string} added_at    ISO datetime string - when this job appeared
 *   on THIS machine. Equal to created_at for everything processed locally;
 *   for a job that arrived through a bundle import it's the import moment.
 * @property {string} updated_at
 * @property {string | null} completed_at
 */

/**
 * @typedef {object} TranscriptTranslationSummary
 * @property {string} language_code
 * @property {"queued" | "running" | "done" | "partial" | "failed"} status
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
// POST /jobs/delete — bulk delete, used by the Library's selection-bar
// Delete action. Unlike single DELETE /jobs/{id}, this applies regardless
// of job status. The daemon emits the usual `job` deleted event per id, so
// an open Library/side-panel tab updates itself through /events; callers
// should still refetch() as a backstop the same way the import flow does.
// ---------------------------------------------------------------------------

/**
 * @typedef {object} JobDeleteRequest
 * @property {string[]} ids   - max 1000 per call
 */

/**
 * @typedef {object} JobDeleteResponse
 * @property {number} deleted
 */

// ---------------------------------------------------------------------------
// POST /jobs/export, POST /jobs/import — moving a library between machines,
// or letting a machine that can't run local models read summaries/
// transcripts imported from elsewhere.
//
// POST /jobs/export takes `{ids: string[]}` and responds with raw
// `application/zip` bytes (a `Content-Disposition: attachment` filename is
// set, but there's no JSON body to type — daemon-client.js hands callers a
// `Blob`). Only `status: "done"` jobs are actually exportable; the daemon
// silently skips the rest and 400s if nothing in the selection qualifies.
//
// POST /jobs/import takes the raw zip bytes as the request body
// (`Content-Type: application/zip` — NOT multipart, NOT JSON) and responds
// with JobImportResponse below.
// ---------------------------------------------------------------------------

/**
 * One job actually written into the daemon by a bundle import. `job_id` is
 * ALWAYS a freshly assigned id (never whatever id the job had in the
 * exporting daemon) — the importing daemon emits the usual `job` created
 * event for it, same as any other new job.
 *
 * @typedef {object} ImportedJob
 * @property {string} job_id
 * @property {string} url
 * @property {string | null} title
 */

/**
 * One bundle entry that was NOT imported as a new job — either it duplicated
 * a job already present (`reason: "duplicate"`) or the import failed for it
 * (`reason` is then a free-form human-readable message, not a fixed enum).
 *
 * @typedef {object} ImportIssue
 * @property {string} url
 * @property {string | null} title
 * @property {string} reason
 */

/**
 * Response of `POST /jobs/import`, HTTP 200. A malformed or oversized bundle
 * is rejected with 400 before this shape is ever produced.
 *
 * Imported jobs arrive with fresh ids and the daemon emits the usual `job`
 * created events, so an open Library tab updates itself through the
 * existing event stream — callers should still `refetch()` explicitly
 * afterwards in case an event was missed.
 *
 * @typedef {object} JobImportResponse
 * @property {ImportedJob[]} imported
 * @property {ImportIssue[]} skipped
 * @property {ImportIssue[]} failed
 */

// ---------------------------------------------------------------------------
// Chat history (per-job Q&A persistence)
// ---------------------------------------------------------------------------

/**
 * One video frame worth showing the user a thumbnail for. TWO producers
 * share this exact shape (one renderer, see lib/frame-thumbnails.js):
 *   - the QA LOOK step (daemon/src/llm/qa.py) — ONLY when the vision model
 *     reported the frames as genuinely relevant to the question. A moment
 *     that was looked at but found irrelevant still contributes its
 *     finding text to the answer, but never produces a FrameRef.
 *   - `POST /jobs/{id}/frames` (see FrameFetchResponse) — the on-demand
 *     "look" affordance next to a summary line; no vision call involved.
 *
 * `frame_url` is a path rooted at the daemon (`GET /jobs/{id}/frames/...`),
 * not an absolute URL — prefix it with `daemon.baseUrl()` before use, same
 * as every other daemon-served resource.
 *
 * @typedef {object} FrameRef
 * @property {number} seconds
 * @property {string} timecode      - "[MM:SS]"-style label, pre-formatted, no brackets
 * @property {string} phrase        - the deixis phrase that triggered this moment
 * @property {string} frame_url
 */

/**
 * One moment where a job's transcript speech points at the video's
 * picture — offered by `GET /jobs/{id}/moments` so the sidepanel can show
 * a "look" affordance next to a summary line's `[MM:SS]` marker that lands
 * near one. EXTERNAL candidates are never included (they point outside the
 * video — a link, an article number — so there's nothing to fetch).
 *
 * @typedef {object} DeixisMoment
 * @property {number} seconds
 * @property {string} timecode          - "[MM:SS]"-style label, no brackets
 * @property {string} phrase
 * @property {"action" | "object"} category
 */

/**
 * @typedef {object} MomentsListResponse
 * @property {DeixisMoment[]} items    - empty (never an error) for a job with no deixis moments
 */

/**
 * Body for `POST /jobs/{id}/frames` — echo back the exact `seconds` value
 * from a `DeixisMoment` (see MomentsListResponse). A value that doesn't
 * match one of the job's own moments 404s.
 *
 * @typedef {object} FrameFetchRequest
 * @property {number} seconds
 */

/**
 * Response for `POST /jobs/{id}/frames`. Same FrameRef shape the QA LOOK
 * step returns — one thumbnail-row renderer for both (see
 * lib/frame-thumbnails.js). A non-2xx response means the fetch failed;
 * the thrown Error's message (via daemon-client.js's `request()`) carries
 * the daemon's `detail` string.
 *
 * @typedef {object} FrameFetchResponse
 * @property {FrameRef[]} items
 */

/**
 * @typedef {object} ChatMessage
 * @property {number} id
 * @property {string} job_id
 * @property {"user" | "assistant"} role
 * @property {string} content
 * @property {string} created_at  ISO datetime string
 * @property {FrameRef[]} frame_refs  - empty unless a LOOK-step frame actually backed this answer
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

/**
 * Emitted once, after the LOOK step finishes, ONLY when at least one
 * inspected video moment turned out relevant (see FrameRef). Never emitted
 * when the LOOK step didn't run, or ran but found nothing relevant to show.
 *
 * @typedef {object} AIFramesEvent
 * @property {"frames"} type
 * @property {FrameRef[]} items
 */

/** @typedef {AIStageEvent | AIDeltaEvent | AIDoneEvent | AIErrorEvent | AIFramesEvent} AIStreamEvent */

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
 * Same write-only key-storage story as LLMConfigOut — see its comment.
 * llm and whisper keys are fully independent (separate keychain entry,
 * separate key file, separate env var).
 *
 * @typedef {object} WhisperConfigOut
 * @property {string} base_url
 * @property {string} model
 * @property {number} max_upload_mb
 * @property {boolean} api_key_set
 * @property {string | null} api_key_hint     - last 4 chars of the resolved key, or null
 * @property {ApiKeySource} api_key_source
 */

/**
 * @typedef {object} OutputConfigOut
 * @property {string} language
 */

/**
 * @typedef {object} StorageConfigOut
 * @property {number} retention_days   - days after a job is ADDED to this
 *   library (JobSummary.added_at, not created_at) before it's auto-deleted
 *   by the daemon's retention timer. 0 = automatic deletion is off.
 */

/**
 * @typedef {object} QaConfigOut
 * @property {boolean} web_search   - when true (default), a Q&A turn the plan step
 *   judges insufficient runs a DuckDuckGo search + page fetch to enrich the answer;
 *   when false, that step never runs at all and the model is instructed to say
 *   plainly when the material/its own knowledge don't cover something.
 */

/**
 * @typedef {object} ConfigResponse
 * @property {LLMConfigOut} llm
 * @property {WhisperConfigOut} whisper
 * @property {OutputConfigOut} output
 * @property {StorageConfigOut} storage
 * @property {QaConfigOut} qa
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
 * Same write-only api_key/api_key_storage side channel as LLMConfigPatch —
 * see its comment.
 *
 * @typedef {object} WhisperConfigPatch
 * @property {string} [base_url]
 * @property {string} [model]
 * @property {number} [max_upload_mb]
 * @property {string} [api_key]              - write-only; unset/empty = unchanged
 * @property {ApiKeyStorage} [api_key_storage]
 */

/**
 * @typedef {object} OutputConfigPatch
 * @property {string} [language]
 */

/**
 * Same shape and behaviour as OutputConfigPatch — see its comment.
 *
 * @typedef {object} StorageConfigPatch
 * @property {number} [retention_days]   - >= 0; 0 turns automatic deletion off
 */

/**
 * @typedef {object} QaConfigPatch
 * @property {boolean} [web_search]
 */

/**
 * @typedef {object} ConfigPatchRequest
 * @property {LLMConfigPatch} [llm]
 * @property {WhisperConfigPatch} [whisper]
 * @property {OutputConfigPatch} [output]
 * @property {StorageConfigPatch} [storage]
 * @property {QaConfigPatch} [qa]
 */

/**
 * Same shape as ConfigResponse plus:
 *   - `restart_required`: true when a change (currently only
 *     `llm.max_concurrent_calls`) can't take effect on the running process
 *     and needs a daemon restart.
 *   - `api_key_verified` / `api_key_verify_error`: write-then-read-back
 *     check whenever this PATCH (re)wrote the LLM API key — the freshly-saved
 *     config is read back the same way the daemon does at call time and
 *     compared to what was saved. `api_key_verified` is true when this
 *     PATCH didn't touch the key at all (nothing to verify) or the
 *     read-back matched; `api_key_verify_error` is a redacted reason
 *     string (never the key itself) when it's false, else null. A failed
 *     verification does NOT roll back the save.
 *   - `whisper_api_key_verified` / `whisper_api_key_verify_error`: the same
 *     check, but for `whisper.api_key` — fully independent of the llm-scoped
 *     fields above (patching one section's key never affects the other's
 *     verification result).
 *
 * @typedef {ConfigResponse & {
 *   restart_required: boolean,
 *   api_key_verified: boolean,
 *   api_key_verify_error: string | null,
 *   whisper_api_key_verified: boolean,
 *   whisper_api_key_verify_error: string | null,
 * }} ConfigPatchResponse
 */

/**
 * @typedef {object} ConfigTestLLMOverrides
 * @property {string} [base_url]
 * @property {string} [model]
 * @property {string} [api_key]
 */

/**
 * Same shape as ConfigTestLLMOverrides, for `target: "whisper"`.
 *
 * @typedef {object} ConfigTestWhisperOverrides
 * @property {string} [base_url]
 * @property {string} [model]
 * @property {string} [api_key]
 */

/**
 * Empty body ({}) tests the currently saved llm config — `target` defaults
 * to "llm", preserving that exact old contract. Set `target: "whisper"`
 * (with optional `whisper` overrides) to probe the Whisper backend instead.
 *
 * @typedef {object} ConfigTestRequest
 * @property {"llm" | "whisper"} [target]
 * @property {ConfigTestLLMOverrides} [llm]
 * @property {ConfigTestWhisperOverrides} [whisper]
 */

/**
 * One step of a `target: "llm"` POST /config/test run, always in the fixed
 * order reachable → models → completion → thinking → context → translation
 * — all six always present, even when later ones are `ok: null` because an
 * earlier step failed (or the overall time budget ran out). `detail` is a
 * human-readable sentence, not a stack trace, though provider error text is
 * still relayed verbatim (truncated, key-redacted) inside it where relevant.
 *
 * @typedef {object} ConfigTestStepResult
 * @property {"reachable" | "models" | "completion" | "thinking" | "context" | "translation"} step
 * @property {boolean | null} ok
 * @property {string | null} [detail]
 */

/**
 * Aggregated proposals from a `target: "llm"` test run. `null` on any field
 * means "no suggestion" — either the step that would produce it never ran,
 * or ran and found nothing worth changing. Applying a suggestion is a
 * separate, explicit PATCH /config the caller triggers; POST /config/test
 * never writes anything itself.
 *
 * @typedef {object} ConfigTestSuggestions
 * @property {string | null} [reasoning_effort]
 * @property {number | null} [context_length]
 * @property {number | null} [single_pass_token_limit]
 */

/**
 * Always HTTP 200 — probe failures are reported in the body (never thrown)
 * since a 401/timeout/etc. IS the useful answer this endpoint exists to
 * give. Two shapes share this type:
 *
 * - `target: "whisper"`: the legacy flat shape — `step` ("models" is the
 *   only value ever reported, a transcription probe would need an audio
 *   file), `status_code`, `detail` describe the single reachability probe.
 *   `steps`/`suggestions` stay empty/default.
 * - `target: "llm"` (default): the step-by-step probe — `steps` carries one
 *   ConfigTestStepResult per stage (see its typedef) and `suggestions`
 *   aggregates whatever the run learned. Top-level `ok` reflects only the
 *   three connectivity/model steps (reachable/models/completion) —
 *   thinking/context/translation are diagnostic, not pass/fail gates for
 *   the backend being usable at all.
 *
 * `models`/`latency_ms` stay populated (model list, total wall time) for
 * both shapes. `detail` (top-level, legacy) is the provider's error message
 * verbatim (truncated to 2000 chars), with the API key itself scrubbed out
 * if it happened to be echoed back — same redaction inside every
 * ConfigTestStepResult.detail.
 *
 * @typedef {object} ConfigTestResponse
 * @property {boolean} ok
 * @property {"models" | "completion" | null} [step]
 * @property {number | null} [status_code]
 * @property {string | null} [detail]
 * @property {string[]} models
 * @property {number | null} latency_ms
 * @property {ConfigTestStepResult[]} [steps]
 * @property {ConfigTestSuggestions} [suggestions]
 */

// ---------------------------------------------------------------------------
// GET /diagnostics — a report meant to be pasted into a bug report by the
// USER themselves; the daemon never sends it anywhere. Everything here has
// already been scrubbed daemon-side (see daemon/src/api/diagnostics.py):
// non-loopback URLs replaced with a placeholder, the home directory
// replaced with "~", either API key redacted even as a fragment (the
// api_key_hint field itself is dropped outright from `config`, not just
// scrubbed — see DiagnosticsLLMConfigOut). No page title, page/video URL,
// or transcript content is ever included at all.
// ---------------------------------------------------------------------------

/**
 * Metadata for the single job requested via `?job_id=` — deliberately
 * everything BUT the content: no title, url, raw_text, or summary_md.
 * `error` is scrubbed the same way `log_tail` is.
 *
 * @typedef {object} DiagnosticsJobInfo
 * @property {string} job_id
 * @property {JobKind} kind
 * @property {JobStatus} status
 * @property {string | null} progress_stage
 * @property {string | null} error
 * @property {TranscriptSource | null} transcript_source
 */

/**
 * `LLMConfigOut` minus `api_key_hint` — dropped outright (not just
 * scrubbed) since it's 4 real characters of the configured key and
 * `api_key_set` already answers "is one configured". `base_url` is
 * redacted only if it happens to contain the raw key itself (defense in
 * depth) — a configured backend address is diagnostic-relevant, unlike a
 * page/video URL, so it's NOT replaced just for being non-loopback.
 *
 * @typedef {object} DiagnosticsLLMConfigOut
 * @property {string} base_url
 * @property {string} model
 * @property {number} context_length
 * @property {number} single_pass_token_limit
 * @property {number} max_concurrent_calls
 * @property {string | null} reasoning_effort
 * @property {boolean} api_key_set
 * @property {ApiKeySource} api_key_source
 */

/**
 * `WhisperConfigOut` minus `api_key_hint` — see `DiagnosticsLLMConfigOut`.
 *
 * @typedef {object} DiagnosticsWhisperConfigOut
 * @property {string} base_url
 * @property {string} model
 * @property {number} max_upload_mb
 * @property {boolean} api_key_set
 * @property {ApiKeySource} api_key_source
 */

/**
 * Same information as `ConfigResponse`, minus `api_key_hint` (either
 * section, dropped) and with `config_path`/`overrides_path` scrubbed of
 * the home directory (kept, not dropped — still useful without the
 * username).
 *
 * @typedef {object} DiagnosticsConfigOut
 * @property {DiagnosticsLLMConfigOut} llm
 * @property {DiagnosticsWhisperConfigOut} whisper
 * @property {OutputConfigOut} output
 * @property {StorageConfigOut} storage
 * @property {string} config_path
 * @property {string} overrides_path
 * @property {boolean} keychain_available
 */

/**
 * @typedef {object} DiagnosticsResponse
 * @property {string} daemon_version
 * @property {string} python_version
 * @property {string} platform
 * @property {HealthResponse} health
 * @property {DiagnosticsConfigOut} config
 * @property {string} log_tail          - already scrubbed; tail of the daemon's rotating log file
 * @property {Record<string, number>} job_status_summary  - status -> count
 * @property {DiagnosticsJobInfo | null} [job]  - only set when ?job_id= was passed
 */

// Marker export so editors recognise this as an ES module.
export {};
