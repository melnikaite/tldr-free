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

/**
 * SSE generator — POST `path` with `body` and yield each parsed `data:` frame
 * as the typed event union `T`. Used by both /ai/stream modes (summary, QA).
 *
 * @template T
 * @param {string} path
 * @param {object} body
 * @returns {AsyncGenerator<T, void, void>}
 */
async function* sseStream(path, body) {
  const baseUrl = await getBaseUrl();
  const res = await fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
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
   * @returns {Promise<JobListResponse>}
   */
  listJobs: (params) => {
    const qs = new URLSearchParams();
    if (params?.status?.length) qs.set("status", params.status.join(","));
    if (params?.kind) qs.set("kind", params.kind);
    if (params?.tag) qs.set("tag", params.tag);
    if (params?.url) qs.set("url", params.url);
    if (params?.limit !== undefined) qs.set("limit", String(params.limit));
    if (params?.offset !== undefined) qs.set("offset", String(params.offset));
    const q = qs.toString();
    return request(`/jobs${q ? `?${q}` : ""}`);
  },

  /**
   * @param {string} id
   * @returns {Promise<JobDetails>}
   */
  getJob: (id) => request(`/jobs/${id}`),

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
   * Unified streaming endpoint for all AI responses.
   *
   * Summary mode (no `question`): subscribes to the job's extraction +
   * summarization lifecycle. Live or replay-cached. Yields stage / delta /
   * done / error events.
   *
   * QA mode (with `question`): triggers a new QA call. Persists the
   * user message, streams the answer tokens, persists the assistant
   * message, emits a final `done` event with the assistant message_id.
   *
   * Usage:
   *   for await (const ev of daemon.aiStream({ job_id })) {
   *     if (ev.type === "stage")  showStage(ev.stage);
   *     if (ev.type === "delta")  appendToBubble(ev.delta);
   *     if (ev.type === "done")   render(ev.content);
   *     if (ev.type === "error")  showError(ev.error);
   *   }
   *
   * @param {AIStreamRequest} req
   * @returns {AsyncGenerator<AIStreamEvent, void, void>}
   */
  aiStream: (req) => sseStream("/ai/stream", req),
};

/** Convenience type re-export so consumers don't have to dual-import. */
export const _types = { ChatMessage: /** @type {ChatMessage} */ (undefined) };
