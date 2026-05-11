// HTTP client for the local daemon. Used by sidepanel, library, options.
// Shapes documented via JSDoc — see api-types.js.

/** @import {
 *   AIStreamEvent,
 *   AIStreamRequest,
 *   ChatMessage,
 *   HealthResponse,
 *   JobCreateRequest,
 *   JobCreateResponse,
 *   JobDetails,
 *   JobListResponse,
 *   JobStatus,
 *   MessagesListResponse
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
  /** @returns {Promise<HealthResponse>} */
  health: () => request("/health"),

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
   * Re-run the pipeline for a failed job. Preserves the job id (and any
   * cached audio file) so we don't accumulate duplicates in the library
   * and skip the slow yt-dlp step when possible.
   *
   * @param {string} id
   * @returns {Promise<JobCreateResponse>}
   */
  retryJob: (id) => request(`/jobs/${id}/retry`, { method: "POST" }),

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
};
