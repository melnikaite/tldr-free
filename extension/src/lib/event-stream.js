// Single global SSE subscription to GET /events.
//
// Each page calls `openEventStream({types})` once and registers callbacks.
// Server-side filter narrows the firehose to the types the page actually
// uses — Library doesn't need delta tokens, sidebar does. Saves bandwidth
// and CPU on slow boxes.
//
// Browsers reconnect EventSource automatically with backoff; we don't
// need our own retry loop but do log on disconnect so a stuck stream is
// easy to spot in DevTools.
//
// Event types (mirror daemon/src/workers/broker.py):
//   { type: "job",     action: "created"|"updated"|"deleted", job: <JobSummary> }
//   { type: "workers", state: { paused, queue_size, running } }
//   { type: "stage",   job_id, stage, detail }
//   { type: "delta",   job_id, delta }
//   { type: "done",    job_id, content }
//   { type: "error",   job_id, error }

const DEFAULT_BASE_URL = "http://127.0.0.1:8765";

async function getBaseUrl() {
  const stored = await chrome.storage.local.get("daemonUrl");
  return stored.daemonUrl || DEFAULT_BASE_URL;
}

/** @typedef {(event: any) => void} Handler */

class EventStream {
  /** @param {string[]} types */
  constructor(types) {
    /** @type {Set<Handler>} */
    this.handlers = new Set();
    /** @type {EventSource | null} */
    this.es = null;
    this.types = types;
    this.connect();
  }

  async connect() {
    try {
      const baseUrl = await getBaseUrl();
      const qs = this.types?.length ? `?types=${this.types.join(",")}` : "";
      this.es = new EventSource(`${baseUrl}/events${qs}`);
      this.es.onmessage = (e) => {
        if (!e.data) return;
        let payload;
        try { payload = JSON.parse(e.data); }
        catch (err) { console.warn("[TLDR] /events: bad frame", e.data, err); return; }
        for (const h of this.handlers) {
          try { h(payload); }
          catch (err) { console.warn("[TLDR] /events handler threw", err); }
        }
      };
      this.es.onerror = () => {
        // EventSource auto-reconnects; log so a persistent stall is visible.
        console.warn("[TLDR] /events disconnected — browser will retry");
      };
    } catch (err) {
      console.error("[TLDR] /events failed to connect", err);
    }
  }

  /**
   * @param {Handler} handler
   * @returns {() => void} unsubscribe
   */
  subscribe(handler) {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }
}

/**
 * Open the global event stream for this page. Call once at module top.
 * @param {{ types?: string[] }} [opts]
 */
export function openEventStream(opts = {}) {
  return new EventStream(opts.types ?? []);
}
