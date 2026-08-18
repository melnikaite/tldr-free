// HTTP client for the local daemon. Used by sidepanel, library, options.
// Shapes documented via JSDoc — see api-types.js.

/** @import {
 *   AIStreamEvent,
 *   AIStreamRequest,
 *   ChatMessage,
 *   DiagnosticsResponse,
 *   FrameFetchResponse,
 *   HealthResponse,
 *   JobCreateRequest,
 *   JobCreateResponse,
 *   JobDeleteResponse,
 *   JobDetails,
 *   JobImportResponse,
 *   JobListResponse,
 *   JobStatus,
 *   MessagesListResponse,
 *   MomentsListResponse
 * } from "./api-types.js" */

const DEFAULT_BASE_URL = "http://127.0.0.1:8765";

async function getBaseUrl() {
  const stored = await chrome.storage.local.get("daemonUrl");
  return stored.daemonUrl || DEFAULT_BASE_URL;
}

async function request(path, init) {
  const baseUrl = await getBaseUrl();
  // `init` (including `signal`) is passed straight to fetch. Callers can
  // hook up timeouts via `AbortSignal.timeout(ms)` or cancel via their own
  // AbortController.
  const res = await fetch(`${baseUrl}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  if (res.status === 204) return undefined;
  return res.json();
}

// If the daemon stops sending chunks for this many ms, we assume the stream
// is dead (network glitch, daemon crashed, or — most commonly here — the
// side panel was throttled/paused by Chrome while its window was minimised
// and the underlying fetch reader is hung). Throwing closes the generator,
// the caller's `finally` runs, and chat input gets re-enabled.
const SSE_CHUNK_TIMEOUT_MS = 120_000;

/**
 * SSE generator — POST `path` with `body` and yield each parsed `data:` frame
 * as the typed event union `T`. Used by both /ai/stream modes (summary, QA).
 *
 * @template T
 * @param {string} path
 * @param {object} body
 * @param {{ signal?: AbortSignal }} [opts]
 * @returns {AsyncGenerator<T, void, void>}
 */
async function* sseStream(path, body, opts = {}) {
  const baseUrl = await getBaseUrl();
  const res = await fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal: opts.signal,
  });
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  try {
    while (true) {
      // Per-chunk timeout: if the daemon stops sending for SSE_CHUNK_TIMEOUT_MS
      // we throw and let the caller's `finally` clean up. Needs Promise.race
      // (not AbortSignal.timeout on fetch) because the timeout resets on
      // every successful chunk.
      const { value, done } = await Promise.race([
        reader.read(),
        new Promise((_, reject) =>
          setTimeout(
            () => reject(new Error(`SSE stream stalled (no chunk for ${SSE_CHUNK_TIMEOUT_MS}ms)`)),
            SSE_CHUNK_TIMEOUT_MS,
          ),
        ),
      ]);
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        for (const line of frame.split("\n")) {
          if (line.startsWith("data: ")) {
            const json = line.slice(6);
            if (json) {
              try {
                yield /** @type {T} */ (JSON.parse(json));
              } catch (e) {
                console.warn("malformed SSE frame", json, e);
              }
            }
          }
        }
      }
    }
  } finally {
    // Release the underlying body stream so the fetch is fully torn down,
    // whether we exited normally, by timeout, by caller `break`, or by the
    // caller aborting via `opts.signal`.
    try { await reader.cancel(); } catch { /* ignore */ }
  }
}

export const daemon = {
  /**
   * Resolve the currently configured daemon base URL. Exposed directly
   * (not just through `request()`) so callers building a raw resource URL
   * — e.g. an `<img src>` for a QA frame thumbnail, see
   * `FrameRef.frame_url` — can prefix a server-relative path without
   * duplicating the `chrome.storage.local` read.
   *
   * @returns {Promise<string>}
   */
  baseUrl: () => getBaseUrl(),

  /**
   * @param {RequestInit} [init] standard fetch init — pass `{ signal }` for timeout/cancel
   * @returns {Promise<HealthResponse>}
   */
  health: async (init) => {
    const resp = await request("/health", init);
    // Reaching this line means the daemon answered at all (whatever its
    // `status` field says — "degraded" still counts, only a network
    // failure would have thrown above). Recorded so the sidepanel's
    // first-run welcome screen (sidepanel/welcome.js, gated in
    // sidepanel/app.js's `_gateIdleOnHealth`) can tell "never installed
    // the daemon" apart from "daemon crashed after working fine for a
    // year" the next time /health can't be reached — see
    // extension.md's "Side panel lifecycle" section. Every /health caller
    // (options, sidepanel, background) contributes to this, not just the
    // welcome-screen check, so the flag reflects the whole extension's
    // history, not just one surface's.
    chrome.storage.local.set({ daemonEverReachable: true }).catch(() => {});
    return resp;
  },

  /**
   * @param {JobCreateRequest} req
   * @returns {Promise<JobCreateResponse>}
   */
  createJob: (req) =>
    request("/jobs", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  /**
   * @param {{ status?: JobStatus[], kind?: string, tag?: string, url?: string, limit?: number, offset?: number }} [params]
   * @param {RequestInit} [init] standard fetch init — pass `{ signal }` for timeout/cancel
   * @returns {Promise<JobListResponse>}
   */
  listJobs: (params, init) => {
    const qs = new URLSearchParams();
    if (params?.status?.length) qs.set("status", params.status.join(","));
    if (params?.kind) qs.set("kind", params.kind);
    if (params?.url) qs.set("url", params.url);
    if (params?.limit !== undefined) qs.set("limit", String(params.limit));
    if (params?.offset !== undefined) qs.set("offset", String(params.offset));
    const q = qs.toString();
    return request(`/jobs${q ? `?${q}` : ""}`, init);
  },

  /**
   * @param {string} id
   * @param {RequestInit} [init]
   * @returns {Promise<JobDetails>}
   */
  getJob: (id, init) => request(`/jobs/${id}`, init),

  /**
   * @param {string} id
   * @returns {Promise<void>}
   */
  deleteJob: (id) => request(`/jobs/${id}`, { method: "DELETE" }),

  /**
   * Bulk delete jobs by id (max 1000 per call) — backs the Library's
   * selection-bar Delete action. Unlike single `deleteJob`, this applies
   * regardless of job status. The daemon emits the usual `job` deleted
   * event per id, so an open Library/side-panel tab updates itself
   * through /events; callers should still `refetch()` afterwards as a
   * backstop, same as the import flow does.
   *
   * @param {string[]} ids
   * @returns {Promise<JobDeleteResponse>}
   */
  deleteJobs: (ids) =>
    request("/jobs/delete", {
      method: "POST",
      body: JSON.stringify({ ids }),
    }),

  /**
   * Export a set of jobs as a downloadable zip bundle, for moving a library
   * between machines or handing summaries/transcripts to a machine that
   * can't run local models. Only `status: "done"` jobs are actually
   * included — the daemon silently skips the rest, and 400s if nothing in
   * `ids` qualifies. The response is raw zip bytes (not JSON), so this
   * bypasses `request()` (which forces `Content-Type: application/json` and
   * parses the body as JSON) and returns the raw `Blob` instead.
   *
   * @param {string[]} ids
   * @returns {Promise<Blob>}
   */
  exportJobs: async (ids) => {
    const baseUrl = await getBaseUrl();
    const res = await fetch(`${baseUrl}/jobs/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids }),
    });
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`${res.status} ${res.statusText}: ${body}`);
    }
    return res.blob();
  },

  /**
   * Import a previously-exported zip bundle. The request body is the raw
   * zip bytes with `Content-Type: application/zip` — NOT multipart, NOT
   * JSON — so this bypasses `request()` the same way `exportJobs` does.
   * Imported jobs arrive with fresh ids and the daemon emits the usual
   * `job` created events, so an open Library tab updates itself through the
   * existing event stream; callers should still `refetch()` afterwards in
   * case an event was missed.
   *
   * @param {Blob | File} fileOrBlob
   * @returns {Promise<JobImportResponse>}
   */
  importJobs: async (fileOrBlob) => {
    const baseUrl = await getBaseUrl();
    const res = await fetch(`${baseUrl}/jobs/import`, {
      method: "POST",
      headers: { "Content-Type": "application/zip" },
      body: fileOrBlob,
    });
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`${res.status} ${res.statusText}: ${body}`);
    }
    return res.json();
  },

  /**
   * Re-run the pipeline for a failed job. Preserves the job id (and any
   * cached audio file) so we don't accumulate duplicates in the library
   * and skip the slow yt-dlp step when possible.
   *
   * @param {string} id
   * @returns {Promise<JobCreateResponse>}
   */
  retryJob: (id) => request(`/jobs/${id}/retry`, { method: "POST" }),

  /**
   * Fetch the full transcript text for a job in the requested language.
   * Lazy — the sidepanel only calls this when the Transcript tab opens.
   *
   * Omit ``lang`` to get the original (Job.raw_text). Pass a language
   * code to get a cached translation — 404 if not cached.
   *
   * @param {string} id
   * @param {string} [lang]
   * @returns {Promise<import("./api-types.js").TranscriptResponse>}
   */
  getTranscript: (id, lang) => {
    const qs = lang ? `?lang=${encodeURIComponent(lang)}` : "";
    return request(`/jobs/${id}/transcript${qs}`);
  },

  /**
   * Enqueue a transcript translation. Dedup: a second call for an
   * already in-flight or completed translation is a no-op (returns the
   * existing status). The sidepanel uses /events to learn when the
   * translation finishes — no need to poll this endpoint.
   *
   * @param {string} id
   * @param {string} lang  - ISO-639-1 code, ISO-639-2, or English name
   * @returns {Promise<{language_code: string, status: string, progress_percent: number, is_source?: boolean}>}
   */
  translateTranscript: (id, lang) =>
    request(`/jobs/${id}/transcript/translate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lang }),
    }),

  /**
   * Re-enqueue every failed OR partial translation for this job (a
   * "partial" row has some lines that fell back to the source language —
   * as much a retry candidate as a fully failed one). Idempotent (no-op
   * when nothing is failed/partial).
   *
   * @param {string} id
   * @returns {Promise<{retried: import("./api-types.js").TranscriptTranslationSummary[]}>}
   */
  retryAllTranslations: (id) =>
    request(`/jobs/${id}/transcript/retry-all`, { method: "POST" }),

  /**
   * Background workers control: a single global pause covers both the
   * Whisper queue and the per-job pipeline (so it works regardless of
   * which LLM/Whisper backend is configured). In-flight work finishes;
   * the next task waits until resume. State is in-memory and resets on
   * daemon restart.
   *
   * @returns {Promise<{paused: boolean, queue_size: number, running: number}>}
   */
  workersStatus: () => request("/workers"),
  pauseWorkers: () => request("/workers/pause", { method: "POST" }),
  resumeWorkers: () => request("/workers/resume", { method: "POST" }),

  /**
   * @param {string} id
   * @returns {Promise<MessagesListResponse>}
   */
  listMessages: (id) => request(`/jobs/${id}/messages`),

  /**
   * The deixis moments for a job — feeds the summary's on-demand "look"
   * affordance (see sidepanel/app.js). Empty `items` (never an error) for
   * a job with no timestamped transcript or no deixis moments at all.
   *
   * @param {string} id
   * @returns {Promise<MomentsListResponse>}
   */
  getMoments: (id) => request(`/jobs/${id}/moments`),

  /**
   * Fetch (or reuse already-downloaded) frames for one of a job's own
   * deixis moments — `seconds` must be a value `getMoments()` actually
   * returned for this job, or the daemon 404s. No vision/LLM call: this
   * returns pictures, not descriptions. Throws (via `request()`) on any
   * non-2xx response — 404 (unknown job/moment), 400 (EXTERNAL moment),
   * 409 (per-job frame budget spent), 502 (download failed after retries).
   *
   * @param {string} id
   * @param {number} seconds
   * @returns {Promise<FrameFetchResponse>}
   */
  fetchMomentFrames: (id, seconds) =>
    request(`/jobs/${id}/frames`, {
      method: "POST",
      body: JSON.stringify({ seconds }),
    }),

  /**
   * Fetch the daemon's current configuration: LLM backend, Whisper
   * backend, output language, retention policy, and config file paths.
   * Neither API key is ever returned — only `api_key_set` (bool),
   * `api_key_hint` (last 4 chars or null), and `api_key_source`
   * (`"env" | "keychain" | "file" | "inline" | "none"`), reported
   * independently for `llm` and `whisper`. May 404 on an older daemon that
   * predates this endpoint — callers (options page) should treat that the
   * same as a network failure.
   *
   * Shape (subset):
   * ```
   * {
   *   llm: { base_url, model, context_length, single_pass_token_limit,
   *          max_concurrent_calls, reasoning_effort, api_key_set,
   *          api_key_hint, api_key_source },
   *   whisper: { base_url, model, max_upload_mb, api_key_set,
   *              api_key_hint, api_key_source },
   *   output: { language },
   *   storage: { retention_days },  // 0 = automatic deletion off
   *   config_path, overrides_path
   * }
   * ```
   *
   * @returns {Promise<object>}
   */
  getConfig: () => request("/config"),

  /**
   * Partially update the daemon configuration — send only the fields
   * that changed, nested under `llm` / `whisper` / `output` / `storage`
   * as returned by `getConfig()`. Two additional write-only fields under EACH of
   * `llm` and `whisper` (fully independent per section):
   *   - `api_key` — the raw new key string. Only send this when the user
   *     typed a new key; omit entirely to leave the stored key untouched
   *     (an empty string is NOT a valid way to say "no change").
   *   - `api_key_storage` — one of `"file" | "keychain" | "inline"`,
   *     defaults to `"keychain"` when available, else `"file"`.
   * A 422 response carries a `detail` field describing the validation
   * error (mirrored in the thrown Error's message by the `request()`
   * helper).
   *
   * @param {object} patch
   * @returns {Promise<object>} same shape as getConfig() plus `restart_required`,
   *   `api_key_verified`/`api_key_verify_error` (llm),
   *   `whisper_api_key_verified`/`whisper_api_key_verify_error` (whisper)
   */
  updateConfig: (patch) =>
    request("/config", {
      method: "PATCH",
      body: JSON.stringify(patch),
    }),

  /**
   * Validate LLM or Whisper credentials/backend without saving anything.
   * Pass `{ target: "whisper", whisper: { base_url, model, api_key } }` to
   * probe Whisper instead of the default `llm` target (only the
   * reachability step runs for Whisper — no completion-call equivalent).
   * Omitting `target`/`llm` tests the currently saved llm config. Always
   * resolves with HTTP 200, even when the test itself failed — check `ok`.
   *
   * @param {object} [overrides]
   * @returns {Promise<{
   *   ok: boolean,
   *   step: "models" | "completion",
   *   status_code?: number,
   *   detail?: string,
   *   models?: string[],
   *   latency_ms?: number,
   * }>}
   */
  testConfig: (overrides) =>
    request("/config/test", {
      method: "POST",
      body: JSON.stringify(overrides || {}),
    }),

  /**
   * Q&A streaming endpoint. Triggers a new QA call, persists user + assistant
   * messages, streams the answer tokens, emits done with message_id.
   *
   * Usage:
   *   for await (const ev of daemon.aiQa({ job_id, question })) {
   *     if (ev.type === "stage")  showStage(ev.stage);
   *     if (ev.type === "delta")  appendToBubble(ev.delta);
   *     if (ev.type === "done")   render(ev.content);
   *     if (ev.type === "error")  showError(ev.error);
   *   }
   *
   * @param {AIStreamRequest} req
   * @returns {AsyncGenerator<AIStreamEvent, void, void>}
   */
  aiQa: (req) => sseStream("/ai/qa", req),

  /**
   * Fetch a self-contained diagnostics report for the user to review and
   * paste into a bug report THEMSELVES — the daemon never sends this
   * anywhere on its own. Already scrubbed of anything privacy-sensitive
   * (see api-types.js's DiagnosticsResponse and
   * daemon/src/api/diagnostics.py) — no page/video URL, no title, no
   * transcript content, no API key. Pass `jobId` to also get metadata
   * (kind/status/progress_stage/error/transcript_source only) for one job;
   * 404s if that job doesn't exist.
   *
   * @param {string} [jobId]
   * @returns {Promise<DiagnosticsResponse>}
   */
  getDiagnostics: (jobId) =>
    request(`/diagnostics${jobId ? `?job_id=${encodeURIComponent(jobId)}` : ""}`),
};
