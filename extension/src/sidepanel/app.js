// Side panel controller — action mode.
//
// Lifecycle:
//   - On open: read activeJobId from session storage, fetch job details.
//     If status=done → render cached summary_md. Otherwise subscribe to
//     /events filtered by job_id and pipe stage / delta / done into the
//     summary area in real time.
//   - On `job-created` broadcast (from background.js after a toolbar click,
//     or from Library on retry/open): if shouldSwitch=true, switch to the
//     new job. This is the primary path; storage.onChanged is a backup
//     (same-value sets aren't guaranteed to notify).
//   - On `tab-changed` broadcast (background.js noticed the active tab
//     changed): switch the panel to that tab's cached job, or render
//     "no summary yet" if it has none.
//   - Chat history persists in SQLite (per job). On job switch we GET
//     /jobs/{id}/messages and render the saved bubbles before any new
//     question.
//   - Processing-badge counter is driven by GET /events (no polling) — we
//     keep a local count of jobs in queued/running status and recompute on
//     every relevant event. One initial GET seeds the count.
//
// State:
//   chrome.storage.session.activeJobId  → currently shown job
//   chrome.storage.session.activeUrl    → URL the panel is following

import { daemon } from "../lib/daemon-client.js";
import { openEventStream } from "../lib/event-stream.js";
import { renderMarkdown } from "../lib/markdown.js";
import { escapeHtml, stringifyError } from "../lib/utils.js";
import { setActiveJob, getActiveJob, renderHistory, clearChat } from "./chat.js";

// Sidepanel needs every event type — stage/delta drive the active job's
// timeline + summary stream, job/workers drive the badge.
const eventStream = openEventStream();

// Module-level replay buffer: jobId → accumulated markdown text.
// Keeps growing as delta events arrive; lets the panel immediately show
// everything buffered so far when the user re-opens mid-generation, instead
// of starting from the current moment. Cleared on job completion or deletion.
/** @type {Map<string, string>} */
const streamAccCache = new Map();

const summaryEl = /** @type {HTMLElement} */ (document.getElementById("summary"));
const badgeEl = /** @type {HTMLElement} */ (document.getElementById("processing-badge"));
const badgeCountEl = /** @type {HTMLElement} */ (document.getElementById("processing-count"));
const stageBadgeEl = /** @type {HTMLElement} */ (document.getElementById("stage-badge"));
const openLibraryBtn = /** @type {HTMLButtonElement} */ (document.getElementById("open-library"));
const chatInput = /** @type {HTMLInputElement} */ (document.getElementById("chat-input"));
const chatSubmit = /** @type {HTMLButtonElement | null} */ (
  document.querySelector("#chat-form button[type='submit']")
);

/** Unsubscribe function for the currently-watched job's event subscription,
 *  or null when no job is being followed. */
let activeStreamUnsubscribe = /** @type {(() => void) | null} */ (null);
/** Set of job ids currently in queued/running — drives the badge counter. */
const activeJobIds = new Set();

openLibraryBtn?.addEventListener("click", () => {
  chrome.tabs.create({ url: chrome.runtime.getURL("src/library/index.html") });
});

// ---------------------------------------------------------------------------
// Timecode link handler — click on a [MM:SS] link in the summary.
// If the YouTube video is already open in a tab: focus that tab and seek
// the video element directly (no page reload). Otherwise open a new tab.
// Delegated on summaryEl so it works across dynamic rerenders.
// ---------------------------------------------------------------------------

summaryEl.addEventListener("click", (ev) => {
  const a = /** @type {HTMLElement} */ (ev.target).closest("a[data-tldr-seconds]");
  // Only intercept plain left-clicks — let ctrl/cmd/middle-click fall through
  // to the browser's native "open in new tab" behaviour.
  if (!a || ev.button !== 0 || ev.ctrlKey || ev.metaKey || ev.shiftKey) return;
  ev.preventDefault();
  const seconds = Number(/** @type {HTMLElement} */ (a).dataset.tldrSeconds);
  const videoId = /** @type {HTMLElement} */ (a).dataset.tldrVideoId || "";
  const fallbackUrl = /** @type {HTMLAnchorElement} */ (a).href;
  _openTimecode(videoId, seconds, fallbackUrl).catch((err) =>
    console.warn("[TLDR] timecode open failed:", err),
  );
});

/**
 * Find an already-open tab for ``url``.
 * YouTube URLs are matched by video ID so extra query params (&t=, &autoplay=,
 * etc.) on the open tab don't prevent the match.  All other URLs are matched
 * by exact string equality.
 * Returns the first matching Tab, or ``undefined`` if none is open.
 *
 * @param {string} url
 * @returns {Promise<chrome.tabs.Tab | undefined>}
 */
async function _findTab(url) {
  try {
    const parsed = new URL(url);
    if (parsed.hostname.endsWith("youtube.com") && parsed.searchParams.has("v")) {
      const videoId = parsed.searchParams.get("v");
      const ytTabs = await chrome.tabs.query({ url: "*://www.youtube.com/watch*" });
      return ytTabs.find((t) => {
        try {
          return new URL(t.url ?? "").searchParams.get("v") === videoId;
        } catch {
          return false;
        }
      });
    }
    const tabs = await chrome.tabs.query({});
    return tabs.find((t) => t.url === url);
  } catch {
    return undefined;
  }
}

/**
 * Focus ``tab`` (bring its window to the front and activate it).
 *
 * @param {chrome.tabs.Tab} tab
 */
async function _focusTab(tab) {
  if (tab.windowId !== undefined) {
    await chrome.windows.update(tab.windowId, { focused: true });
  }
  await chrome.tabs.update(/** @type {number} */ (tab.id), { active: true });
}

/**
 * Focus an existing YouTube tab playing this video and seek to ``seconds``,
 * or open a new tab if none is found.
 *
 * @param {string} videoId
 * @param {number} seconds
 * @param {string} fallbackUrl
 */
async function _openTimecode(videoId, seconds, fallbackUrl) {
  const existing = await _findTab(fallbackUrl);
  if (existing?.id !== undefined) {
    await _focusTab(existing);
    await chrome.scripting.executeScript({
      target: { tabId: existing.id },
      func: (t) => {
        const video = document.querySelector("video");
        if (video) video.currentTime = t;
      },
      args: [seconds],
    });
    return;
  }
  // No matching tab — open a new one at the correct timestamp.
  chrome.tabs.create({ url: fallbackUrl });
}

// ---------------------------------------------------------------------------
// Title link handler — click on the job title switches to the source tab if
// it is already open, otherwise opens a new tab.  Mirrors the timecode link
// behaviour.  Delegated on summaryEl so it works across dynamic rerenders.
// ---------------------------------------------------------------------------

summaryEl.addEventListener("click", (ev) => {
  const a = /** @type {HTMLElement} */ (ev.target).closest(".job-title a");
  // Only intercept plain left-clicks — let ctrl/cmd/middle-click fall through
  // to the browser's native "open in new tab" behaviour.
  if (!a || ev.button !== 0 || ev.ctrlKey || ev.metaKey || ev.shiftKey) return;
  ev.preventDefault();
  const url = /** @type {HTMLAnchorElement} */ (a).href;
  _openUrl(url).catch((err) =>
    console.warn("[TLDR] title open failed:", err),
  );
});

/**
 * Focus an existing tab already showing ``url``, or open a new tab.
 *
 * @param {string} url
 */
async function _openUrl(url) {
  const existing = await _findTab(url);
  if (existing?.id !== undefined) {
    await _focusTab(existing);
    return;
  }
  chrome.tabs.create({ url });
}

chrome.runtime.onMessage.addListener((msg) => {
  if (!msg || typeof msg !== "object") return;
  if (msg.type === "tab-switching") {
    // The active tab is changing. Abort any in-flight stream immediately so a
    // `done` event from the previous job can't render stale content while we
    // wait for `tab-changed` (which arrives after listJobs completes).
    abortActiveStream();
  } else if (msg.type === "tab-changed") {
    handleTabChanged(msg.url, msg.jobId).catch((e) =>
      console.error("[TLDR] tab-changed", e),
    );
  } else if (msg.type === "job-created") {
    handleJobCreated(msg).catch((e) =>
      console.error("[TLDR] job-created", e),
    );
  } else if (msg.type === "extraction-error") {
    renderError(msg.error || "Failed to extract page content.");
  }
});

/**
 * The user just submitted a job. background.js broadcasts this every time
 * POST /jobs returns, with `shouldSwitch=true` when the source tab is still
 * the active one (no hijack). Library also sends it on retry/open with
 * shouldSwitch=true to follow that job explicitly.
 *
 * Why this and not just `chrome.storage.onChanged` on activeJobId:
 *   storage.onChanged is debounced — setting the same value twice (e.g.
 *   re-clicking summarize on a deduped URL) won't fire a second time, so
 *   the panel could miss a re-show. The broadcast always fires.
 *
 * @param {{jobId?:string, shouldSwitch?:boolean}} msg
 */
async function handleJobCreated(msg) {
  if (!msg.jobId || !msg.shouldSwitch) return;
  const active = await getActiveJob();
  if (active?.id === msg.jobId) return;  // already showing it
  await loadAndRender(msg.jobId);
}

eventStream.subscribe((event) => {
  if (event.type === "job") {
    handleJobEvent(event);
  } else if (event.type === "done" || event.type === "error") {
    if (event.job_id) {
      activeJobIds.delete(event.job_id);
      setBadge(activeJobIds.size);
    }
  }
});

/** @param {{action: string, job: any}} event */
function handleJobEvent(event) {
  const j = event.job;
  if (!j?.id) return;
  if (event.action === "deleted") {
    activeJobIds.delete(j.id);
    streamAccCache.delete(j.id);
  } else if (j.status === "queued" || j.status === "running") {
    activeJobIds.add(j.id);
  } else {
    activeJobIds.delete(j.id);
  }
  setBadge(activeJobIds.size);
  // For YouTube the daemon seeds title with the video id and only fills the
  // canonical title once yt-dlp metadata returns mid-pipeline. That update
  // arrives as a job_event — patch the streaming view in place so the user
  // doesn't stare at "9Pipy0h0VJk" until the summary lands.
  patchActiveJobIfMatches(event).catch(() => {});
}

/** @param {{action: string, job: any}} event */
async function patchActiveJobIfMatches(event) {
  const j = event.job;
  if (!j?.id || event.action === "deleted") return;
  const active = await getActiveJob();
  if (!active || active.id !== j.id) return;
  const merged = { ...active, ...j };
  setActiveJob(merged);
  if (j.title) {
    const link = summaryEl.querySelector(".job-title a");
    if (link) link.textContent = j.title;
  }
}

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "session" && changes.activeJobId) {
    const id = changes.activeJobId.newValue;
    if (id) {
      loadAndRender(id).catch((e) => console.error("[TLDR] storage.onChanged", e));
    }
  }
});

bootstrap().catch((e) => {
  console.error("[TLDR] sidepanel bootstrap", e);
  renderError(stringifyError(e));
});

async function bootstrap() {
  const { activeJobId, activeUrl } = await chrome.storage.session.get([
    "activeJobId",
    "activeUrl",
  ]);
  await seedBadge();
  if (activeJobId) {
    await loadAndRender(activeJobId);
  } else if (activeUrl) {
    renderNoSummary(activeUrl);
    syncChatEnabled(false);
  }
}

/**
 * Tab follow: the active tab changed.
 * @param {string} url
 * @param {string | null} jobId
 */
async function handleTabChanged(url, jobId) {
  if (jobId) {
    const active = await getActiveJob();
    if (active && active.id === jobId) return;
    await loadAndRender(jobId);
  } else {
    abortActiveStream();
    setActiveJob(null);
    clearChat();
    renderNoSummary(url);
    syncChatEnabled(false);
    window.scrollTo({ top: 0 });
  }
}

/** @param {string} jobId */
async function loadAndRender(jobId) {
  abortActiveStream();
  renderLoading();
  syncChatEnabled(false);
  // Reset scroll to the top so the user sees the new title / summary from
  // the start (otherwise we may be parked deep in the previous job's chat).
  window.scrollTo({ top: 0 });

  // Safety net: if getJob hangs (daemon down, dead service worker, …) the
  // user gets a real error after 8s instead of an endless spinner.
  let job;
  try {
    job = await withTimeout(daemon.getJob(jobId), 8000, `getJob(${jobId})`);
  } catch (err) {
    console.error("[TLDR] getJob failed", err);
    renderError(stringifyError(err));
    return;
  }
  setActiveJob(job);

  // Pull chat history in parallel with summary rendering.
  loadHistory(jobId).catch((e) => console.warn("[TLDR] message history failed", e));

  if (job.status === "done" && job.summary_md) {
    renderSummary(job, job.summary_md);
    setStage(null);
    syncChatEnabled(true);
    return;
  }
  if (job.status === "failed") {
    renderError(job.error || "Job failed.", job);
    setStage(null);
    return;
  }

  // queued / running → subscribe to live /ai/stream.
  await streamSummaryFor(job);
}

/**
 * Drive the summary area for a queued/running job from /events.
 *
 * No second SSE connection is opened — we filter the global event stream by
 * job_id. This keeps each side panel at exactly one long-lived connection
 * and avoids running into Chrome's 6-per-origin HTTP/1.1 cap, which used
 * to leave subsequent fetches "stalled" with no entry in DevTools' Network
 * tab while a /ai/stream connection still hung from a prior pipeline.
 *
 * UI shape during streaming:
 *   <h2>title</h2>
 *   <ul class="timeline">
 *     <li class="phase phase--done">  ✓  Extracting page content
 *     <li class="phase phase--active"> dots Transcribing audio
 *   </ul>
 *   <div class="markdown-body" id="summary-stream"></div>  ← starts empty
 *
 * On the first delta the timeline collapses to a muted strip and tokens
 * begin flowing into #summary-stream. On done it's replaced with the
 * rendered markdown.
 *
 * @param {import("../lib/api-types.js").JobDetails} job
 */
function streamSummaryFor(job) {
  const titleHtml = renderTitleHtml(job);
  const initialStage = job.progress_stage || "queued";
  summaryEl.innerHTML =
    `${titleHtml}` +
    `<ul class="timeline" id="phase-timeline"></ul>` +
    `<div class="markdown-body" id="summary-stream"></div>`;
  const timelineEl = /** @type {HTMLElement} */ (document.getElementById("phase-timeline"));
  const streamEl = /** @type {HTMLElement} */ (document.getElementById("summary-stream"));

  /** @type {Array<{stage:string, detail:(string|undefined), status:"active"|"done"|"failed", error?:string}>} */
  const phases = [];
  pushOrUpdatePhase(phases, initialStage, undefined);
  renderTimeline(timelineEl, phases);

  // Restore any text accumulated before this subscription started.
  // Priority: module-level cache (exact, no gap — same browser session,
  // panel re-opened) > server-side partial_summary (works after browser
  // restart, may miss a few tokens between the getJob fetch and subscribe).
  let acc = streamAccCache.get(job.id) || job.partial_summary || "";
  let firstDelta = acc.length === 0;
  /** @type {number | null} */
  let rafId = null;
  setStage(initialStage);

  // If we already have buffered content, collapse the phase timeline and
  // render it right away so the user sees the full replay instead of a blank.
  if (acc) {
    timelineEl.classList.add("timeline--collapsed");
    streamEl.innerHTML = renderMarkdown(acc, job.video_id);
  }

  /**
   * Schedule a markdown re-render for the next animation frame.
   * Using rAF as a throttle: multiple delta events arriving in the same JS
   * task batch collapse into one DOM write per frame (~60 Hz max, but in
   * practice limited by the daemon's ~10 Hz delta flush rate). This gives
   * progressive formatted rendering during streaming instead of raw text.
   */
  const scheduleRender = () => {
    if (rafId !== null) return;          // already queued
    rafId = requestAnimationFrame(() => {
      rafId = null;
      if (acc) streamEl.innerHTML = renderMarkdown(acc, job.video_id);
    });
  };

  const cancelRender = () => {
    if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
  };

  // Subscribe to the global event stream filtered by this job's id.
  // The unsubscribe handle goes into `activeStreamUnsubscribe` so
  // loadAndRender / handleTabChanged can drop it when the user moves on.
  const unsubscribe = eventStream.subscribe((ev) => {
    if (ev.job_id !== job.id) return;
    if (ev.type === "stage") {
      setStage(ev.stage, ev.detail);
      pushOrUpdatePhase(phases, ev.stage, ev.detail || undefined);
      renderTimeline(timelineEl, phases);
    } else if (ev.type === "delta") {
      if (firstDelta) {
        markAllDone(phases);
        renderTimeline(timelineEl, phases);
        timelineEl.classList.add("timeline--collapsed");
        firstDelta = false;
      }
      acc += ev.delta;
      streamAccCache.set(job.id, acc);  // keep replay buffer current
      scheduleRender();
    } else if (ev.type === "done") {
      cancelRender();
      streamAccCache.delete(job.id);   // job finished — no longer need replay
      const content = ev.content || acc;
      markAllDone(phases);
      setStage(null);
      abortActiveStream();
      // Fetch the fresh job BEFORE rendering so video_id and the canonical
      // title are available. The job object captured at stream start has
      // video_id=null (set mid-pipeline by _set_extracted) and may still
      // carry the video-id placeholder as title. Rendering with the stale
      // object would leave [MM:SS] timecodes as plain text instead of links.
      daemon.getJob(job.id).then(fresh => {
        setActiveJob(fresh);
        renderSummary(fresh, content);
        syncChatEnabled(true);
      }).catch(() => {
        // Daemon unreachable — fall back to the stale job; timecodes won't
        // be links but at least the summary text is shown.
        renderSummary(job, content);
        syncChatEnabled(true);
      });
    } else if (ev.type === "error") {
      cancelRender();
      streamAccCache.delete(job.id);   // job failed — drop replay buffer
      markActiveFailed(phases, ev.error || "Error");
      renderTimeline(timelineEl, phases);
      renderError(ev.error, job);
      setStage(null);
      syncChatEnabled(false);
      abortActiveStream();
    }
  });

  activeStreamUnsubscribe = unsubscribe;
}

// ---------------------------------------------------------------------------
// Phase timeline — accumulating list of stages with done/active/failed icons.
// ---------------------------------------------------------------------------

function pushOrUpdatePhase(phases, stage, detail) {
  // Mark every previously-active phase as done before appending/updating.
  for (const p of phases) {
    if (p.status === "active") p.status = "done";
  }
  const existing = phases.find((p) => p.stage === stage);
  if (existing) {
    if (detail !== undefined) existing.detail = detail;
    existing.status = "active";
  } else {
    phases.push({ stage, detail, status: "active" });
  }
}

function markAllDone(phases) {
  for (const p of phases) {
    if (p.status === "active") p.status = "done";
  }
}

function markActiveFailed(phases, error) {
  const last = phases[phases.length - 1];
  if (last) {
    last.status = "failed";
    last.error = error;
  }
}

function renderTimeline(container, phases) {
  if (!container) return;
  container.innerHTML = "";
  for (const p of phases) {
    const li = document.createElement("li");
    li.className = `phase phase--${p.status}`;

    const icon = document.createElement("span");
    icon.className = "phase-icon";
    if (p.status === "done") {
      icon.textContent = "✓";
    } else if (p.status === "failed") {
      icon.textContent = "✕";
    } else {
      // active
      icon.innerHTML =
        `<span class="thinking-dots"><span></span><span></span><span></span></span>`;
    }
    li.appendChild(icon);

    const label = document.createElement("span");
    label.className = "phase-label";
    label.textContent = phaseLabel(p.stage);
    li.appendChild(label);

    if (p.status === "failed" && p.error) {
      const err = document.createElement("span");
      err.className = "phase-detail phase-detail--error";
      err.textContent = p.error;
      li.appendChild(err);
    } else if (p.detail) {
      const det = document.createElement("span");
      det.className = "phase-detail";
      det.textContent = p.detail;
      li.appendChild(det);
    }

    container.appendChild(li);
  }
}

/** Map a backend stage name to the phase row label. */
function phaseLabel(stage) {
  switch (stage) {
    case "extracting":         return "Fetching subtitles";
    case "fetching_captions":  return "Fetching captions via yt-dlp";
    case "queued":             return "Queued for transcription";
    case "downloading":        return "Downloading audio";
    case "transcribing":       return "Transcribing audio";
    case "ready":              return "Preparing summary";
    case "summarizing":        return "Summarising";
    case "thinking":           return "Thinking";
    default:                   return stage || "Working";
  }
}

function abortActiveStream() {
  if (activeStreamUnsubscribe) {
    activeStreamUnsubscribe();
    activeStreamUnsubscribe = null;
  }
}

/** @param {string} jobId */
async function loadHistory(jobId) {
  try {
    const { items } = await daemon.listMessages(jobId);
    renderHistory(items);
  } catch (err) {
    console.warn("[TLDR] listMessages failed", err);
    renderHistory([]);
  }
}

/**
 * @param {import("../lib/api-types.js").JobDetails} job
 * @param {string} markdown
 */
function renderSummary(job, markdown) {
  const titleHtml = renderTitleHtml(job);
  const html = renderMarkdown(markdown || "_(empty summary)_", job.video_id);
  summaryEl.innerHTML = `${titleHtml}<div class="markdown-body">${html}</div>`;
}

/** @param {import("../lib/api-types.js").JobDetails | null} job */
function renderTitleHtml(job) {
  if (!job?.title) return "";
  const safeUrl = escapeHtml(job.url);
  const safeTitle = escapeHtml(job.title);
  return `<h2 class="job-title"><a href="${safeUrl}" target="_blank" rel="noopener">${safeTitle}</a></h2>`;
}

function renderLoading() {
  summaryEl.innerHTML = `
    <div class="status-block">
      <div class="spinner" aria-hidden="true"></div>
      <p>Loading…</p>
    </div>
  `;
}

/** @param {string} url */
function renderNoSummary(url) {
  summaryEl.innerHTML = `
    <div class="placeholder-block">
      <p class="muted small url-line">${escapeHtml(url || "")}</p>
      <button class="summarize-btn" type="button">Summarize this page</button>
    </div>
  `;
  const btn = /** @type {HTMLButtonElement | null} */ (
    summaryEl.querySelector(".summarize-btn")
  );
  if (btn) {
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      btn.textContent = "Starting…";
      try {
        await chrome.runtime.sendMessage({ type: "summarize-active-tab" });
      } catch (err) {
        console.error("[TLDR] summarize-active-tab failed", err);
        renderError(stringifyError(err));
      }
    });
  }
  setStage(null);
}

/**
 * @param {string} message
 * @param {import("../lib/api-types.js").JobDetails} [job]
 */
function renderError(message, job) {
  const titleHtml = renderTitleHtml(job || null);
  const retryHtml = job?.id
    ? `<button class="retry-btn" data-retry-id="${escapeHtml(job.id)}">Retry</button>`
    : "";
  summaryEl.innerHTML = `
    ${titleHtml}
    <div class="status-block error">
      <p><strong>Error.</strong></p>
      <p class="muted small">${escapeHtml(message)}</p>
      ${retryHtml}
    </div>
  `;
  const btn = summaryEl.querySelector(".retry-btn");
  if (btn) {
    btn.addEventListener("click", async () => {
      const id = /** @type {HTMLElement} */ (btn).dataset.retryId;
      if (!id) return;
      btn.setAttribute("disabled", "true");
      try {
        await daemon.retryJob(id);
        await loadAndRender(id);
      } catch (err) {
        console.error("[TLDR] retry failed", err);
        btn.removeAttribute("disabled");
        renderError(stringifyError(err), job);
      }
    });
  }
}

/** @param {string | null} stage @param {string | null | undefined} [detail] */
function setStage(stage, detail) {
  if (!stageBadgeEl) return;
  if (!stage) {
    stageBadgeEl.classList.add("hidden");
    stageBadgeEl.textContent = "";
    return;
  }
  stageBadgeEl.classList.remove("hidden");
  stageBadgeEl.textContent = detail ? `${stage} · ${detail}` : stage;
}

/** @param {number} n */
function setBadge(n) {
  badgeCountEl.textContent = String(n);
  badgeEl.classList.toggle("hidden", n === 0);
}

/**
 * One-shot at startup: ask the daemon which jobs are queued/running so
 * the badge reflects state from before the panel opened. After this the
 * eventStream subscription keeps the count current — no polling.
 */
async function seedBadge() {
  try {
    const resp = await daemon.listJobs({ status: ["queued", "running"], limit: 50 });
    activeJobIds.clear();
    for (const j of resp.items || []) activeJobIds.add(j.id);
    setBadge(activeJobIds.size);
  } catch (err) {
    console.warn("[TLDR] seedBadge failed", err);
  }
}

/** @param {boolean} enabled */
function syncChatEnabled(enabled) {
  if (chatInput) chatInput.disabled = !enabled;
  if (chatSubmit) chatSubmit.disabled = !enabled;
}

/**
 * Race a promise against a timeout. Used as a safety net so a hung HTTP
 * call (daemon stopped, container restarting, MV3 service worker dead) shows
 * a real error instead of an indefinite spinner.
 *
 * @template T
 * @param {Promise<T>} promise
 * @param {number} ms
 * @param {string} label
 * @returns {Promise<T>}
 */
function withTimeout(promise, ms, label) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error(`${label} timed out after ${ms}ms`)),
      ms,
    );
    promise.then(
      (v) => { clearTimeout(timer); resolve(v); },
      (e) => { clearTimeout(timer); reject(e); },
    );
  });
}
