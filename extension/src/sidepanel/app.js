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
//   - On `set-active-tab` broadcast (background.js noticed the active tab
//     changed): switch the panel to that tab's cached job, or render
//     "no summary yet" if it has none. The message arrives twice per
//     switch — once before listJobs (jobId omitted) so the panel can
//     blank to a spinner, and once after (jobId resolved) so it can
//     render the final state. A monotonic `version` discards stale
//     phase-2 messages when the user switches again before listJobs
//     for the previous tab finishes.
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
import { classifyError, describeQueuedDetail, isDaemonUnreachable } from "../lib/error-hints.js";
import { openEventStream } from "../lib/event-stream.js";
import { buildFrameRow } from "../lib/frame-thumbnails.js";
import { renderMarkdown } from "../lib/markdown.js";
import { escapeHtml, stringifyError } from "../lib/utils.js";
import {
  setActiveJob as _chatSetActiveJob,
  getActiveJob,
  renderHistory,
  clearChat,
} from "./chat.js";
import * as transcript from "./transcript.js";
import { buildWelcomeView } from "./welcome.js";

/** Remember the id we last broadcast so we only reset the tab on real switches. */
let _lastBroadcastJobId = /** @type {string | null} */ (null);

/**
 * Set the active job everywhere that cares about it: chat (Q&A bubbles,
 * history rendering) and the transcript tab controller (lazy fetch +
 * polling lifecycle). Always go through this wrapper rather than calling
 * the chat-only setter directly — the transcript tab needs to know when
 * the job changes to reset its cache and stop the currentTime poll.
 *
 * When the user actually switches Chrome tabs to a different summarised
 * job (id A → id B), the visual tab is reset to Summary so they see
 * the canonical view of the new content. We deliberately do NOT reset
 * when going from "no job" → "job" (the user just clicked Process from
 * the Transcript tab; keep them there to watch the transcript stream
 * in) or for in-place patches (same id, e.g. yt-dlp title fill).
 *
 * @param {import("../lib/api-types.js").JobDetails | null} job
 */
function setActiveJob(job) {
  _chatSetActiveJob(job);
  transcript.setJob(job);
  const newId = job?.id ?? null;
  const wasNoJob = _lastBroadcastJobId == null;
  if (newId !== _lastBroadcastJobId) {
    _lastBroadcastJobId = newId;
    // Only force a summary-tab view when the user actually switched
    // between two existing jobs. Going from no-job → job is the
    // "just kicked off processing" path — leave the tab choice alone.
    if (!wasNoJob && typeof switchTab === "function") switchTab("summary");
  }
}

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
// Tab switching (Summary | Transcript)
//
// Two buttons in #tab-nav, two panes (#pane-summary / #pane-transcript). The
// Transcript tab button is hidden by transcript.js when the active job's
// kind doesn't have a meaningful transcript (page / pdf). When the user
// switches to it the first time for a given job, transcript.js lazy-loads
// the body via GET /jobs/{id}/transcript.
// ---------------------------------------------------------------------------

const summaryTabBtn = /** @type {HTMLButtonElement | null} */ (document.getElementById("tab-summary"));
const transcriptTabBtn = /** @type {HTMLButtonElement | null} */ (document.getElementById("tab-transcript"));
const summaryPaneEl = /** @type {HTMLElement | null} */ (document.getElementById("pane-summary"));
const transcriptPaneEl = /** @type {HTMLElement | null} */ (document.getElementById("pane-transcript"));

/** @param {"summary" | "transcript"} which */
function switchTab(which) {
  if (!summaryTabBtn || !transcriptTabBtn || !summaryPaneEl || !transcriptPaneEl) return;
  const wantSummary = which === "summary";
  summaryTabBtn.classList.toggle("tab--active", wantSummary);
  transcriptTabBtn.classList.toggle("tab--active", !wantSummary);
  summaryTabBtn.setAttribute("aria-selected", String(wantSummary));
  transcriptTabBtn.setAttribute("aria-selected", String(!wantSummary));
  summaryPaneEl.classList.toggle("tab-pane--active", wantSummary);
  transcriptPaneEl.classList.toggle("tab-pane--active", !wantSummary);
  // Show/hide via the active class — tab-pane is display:none by default.
  if (wantSummary) {
    transcript.onTabHide();
  } else {
    transcript.onTabShow();
  }
}

summaryTabBtn?.addEventListener("click", () => switchTab("summary"));
transcriptTabBtn?.addEventListener("click", () => switchTab("transcript"));

// ---------------------------------------------------------------------------
// Timecode link handler — click on a [MM:SS] link in the summary.
//
// Two flavours:
//   - YouTube (data-tldr-video-id): focus the YouTube tab if open, seek the
//     <video> element directly. Otherwise open a new tab at the timestamped
//     URL — YouTube's player honours the `t=` query param.
//   - Generic media (data-tldr-media-page-url): focus the page tab if open,
//     try to seek the first <video>/<audio> element on the page via
//     executeScript. Otherwise open the URL with the `#t=Ns` media-fragment
//     in the href — works for direct media file URLs natively; on regular
//     pages the user has to scrub manually but they at least land on the
//     right page.
//
// Delegated on summaryEl so it works across dynamic rerenders.
// ---------------------------------------------------------------------------

// Delegated on ``document.body`` so it works in BOTH panes: timecode links
// appear inside the summary (#summary), inside chat bubbles (#chat-messages),
// AND inside the transcript view (#pane-transcript). A handler bound to
// summaryEl misses the latter and the click falls through to the generic
// external-link handler — which opens a fresh tab instead of seeking the
// already-open one.
document.body.addEventListener("click", (ev) => {
  const a = /** @type {HTMLElement} */ (ev.target).closest("a[data-tldr-seconds]");
  if (!a || ev.button !== 0 || ev.ctrlKey || ev.metaKey || ev.shiftKey) return;
  ev.preventDefault();
  const el = /** @type {HTMLElement} */ (a);
  const seconds = Number(el.dataset.tldrSeconds);
  const videoId = el.dataset.tldrVideoId || "";
  const mediaPageUrl = el.dataset.tldrMediaPageUrl || "";
  const fallbackUrl = /** @type {HTMLAnchorElement} */ (a).href;
  _openTimecode({ videoId, mediaPageUrl, seconds, fallbackUrl }).catch((err) =>
    console.warn("[TLDR] timecode open failed:", err),
  );
});

/**
 * Find an already-open tab for ``url``.
 * YouTube URLs are matched by video ID so extra query params (&t=, &autoplay=,
 * etc.) on the open tab don't prevent the match. Other URLs are matched by
 * canonical form (fragment stripped on both sides), so a timecode anchor
 * like ``…/podcast#t=754`` matches the live tab at ``…/podcast``.
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
    parsed.hash = "";
    const canonical = parsed.toString();
    const tabs = await chrome.tabs.query({});
    return tabs.find((t) => {
      if (!t.url) return false;
      try {
        const tabUrl = new URL(t.url);
        tabUrl.hash = "";
        return tabUrl.toString() === canonical;
      } catch {
        return t.url === url;
      }
    });
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
 * Seek to ``seconds`` on the source page if it's open, else open a new tab.
 *
 * For YouTube: lookup is by video id (handled inside ``_findTab``), seek
 * targets ``video`` only.
 * For generic media: lookup is by canonical page URL, seek targets the first
 * ``video, audio`` element — covers HTML5 audio players, podcast pages,
 * direct .mp4/.webm files. Iframe-embedded players (YouTube-in-iframe,
 * Vimeo, etc.) won't seek because the iframe is a separate document scope;
 * we don't try to message-pass into them.
 *
 * @param {{ videoId: string, mediaPageUrl: string, seconds: number, fallbackUrl: string }} opts
 */
async function _openTimecode({ videoId, mediaPageUrl, seconds, fallbackUrl }) {
  const lookupUrl = videoId ? fallbackUrl : mediaPageUrl || fallbackUrl;
  const existing = await _findTab(lookupUrl);
  if (existing?.id !== undefined) {
    await _focusTab(existing);
    await chrome.scripting.executeScript({
      target: { tabId: existing.id },
      func: (t) => {
        const media = /** @type {HTMLMediaElement | null} */ (
          document.querySelector("video, audio")
        );
        if (media) media.currentTime = t;
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

// ---------------------------------------------------------------------------
// Generic external-link handler — catches every `<a href>` anywhere in the
// side panel that isn't a timecode marker or the job-title link (those have
// their own handlers above with custom focus/seek logic).
//
// Why this exists: links produced by `marked` (both in the summary body and
// in assistant chat bubbles) have no `target` attribute, and Chrome side
// panels can't navigate top-level, so a plain anchor click is a silent no-op
// in the side panel iframe. We translate every click into
// `chrome.tabs.create()` so external URLs reliably open in a new browser tab.
//
// Attached to ``document.body`` so the handler covers BOTH ``#summary`` and
// ``#chat-messages`` — markdown rendering happens in both surfaces.
//
// Skips ``javascript:``, ``#``-only, and empty hrefs — those are not
// external navigation and shouldn't be hijacked.
// ---------------------------------------------------------------------------

document.body.addEventListener("click", (ev) => {
  const target = /** @type {HTMLElement} */ (ev.target);
  const a = /** @type {HTMLAnchorElement | null} */ (target.closest("a[href]"));
  if (!a) return;
  // Skip anchors that the timecode / title handlers already own.
  if (a.dataset.tldrSeconds !== undefined) return;
  if (a.closest(".job-title")) return;
  // Honour modifier keys / non-left clicks — browser default already DTRT.
  if (ev.button !== 0 || ev.ctrlKey || ev.metaKey || ev.shiftKey) return;

  const raw = a.getAttribute("href") || "";
  // In-page anchors and javascript: shouldn't open a new tab.
  if (!raw || raw.startsWith("#") || raw.startsWith("javascript:")) return;
  const href = a.href;  // resolved absolute URL
  if (!href) return;
  ev.preventDefault();
  chrome.tabs.create({ url: href }).catch((err) =>
    console.warn("[TLDR] external link open failed:", err),
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

// Monotonic version of the latest tab switch the side panel has acknowledged.
// background.js increments its counter on every syncSidepanelForTab; we drop
// any message with an older version so out-of-order listJobs completions
// can't act on a tab the user has already left.
let lastTabVersion = 0;

chrome.runtime.onMessage.addListener((msg) => {
  if (!msg || typeof msg !== "object") return;
  if (msg.type === "set-active-tab") {
    if (typeof msg.version === "number" && msg.version < lastTabVersion) return;
    if (typeof msg.version === "number") lastTabVersion = msg.version;
    handleSetActiveTab(msg).catch((e) =>
      console.error("[TLDR] set-active-tab", e),
    );
  } else if (msg.type === "job-created") {
    handleJobCreated(msg).catch((e) =>
      console.error("[TLDR] job-created", e),
    );
  } else if (msg.type === "extraction-error") {
    handleExtractionError(msg.error).catch((e) =>
      console.error("[TLDR] extraction-error handling", e),
    );
  }
});

/**
 * A job failed to even get created (content-script injection failed, or the
 * background service worker's POST /jobs threw). Most causes deserve the
 * plain error box same as before — but "the daemon isn't running" is the
 * exact first-impression bug this file's welcome screen exists to fix: a
 * brand new install with nothing configured yet used to land here and show
 * the same red box a broken year-old install would. Route to the welcome
 * screen instead when this looks like that case AND we've never seen the
 * daemon answer /health before (see `_gateIdleOnHealth`'s docstring for why
 * a returning user still gets the classic error, not the welcome screen).
 *
 * @param {string | undefined} rawError
 */
async function handleExtractionError(rawError) {
  const message = rawError || "Failed to extract page content.";
  if (isDaemonUnreachable(message)) {
    const { daemonEverReachable } = await chrome.storage.local.get("daemonEverReachable");
    if (!daemonEverReachable) {
      renderState({ mode: "welcome", step: "daemon", health: null });
      return;
    }
  }
  renderState({ mode: "error", message });
}

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
  // Translation-progress pings are NOT job-state mutations — they piggyback
  // on the job-event channel for fan-out but their payload describes the
  // translation row (kind="transcript_translation", status="running", …).
  // Merging those fields into the Job would clobber kind and status,
  // hiding the Transcript tab and confusing the summary view. transcript.js
  // owns chip updates; the badge / patch logic here ignores them.
  if (event.action === "translation_updated") return;
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
  renderState({ mode: "error", message: stringifyError(e) });
});

async function bootstrap() {
  const { activeJobId, activeUrl } = await chrome.storage.session.get([
    "activeJobId",
    "activeUrl",
  ]);
  await seedBadge();
  if (activeJobId) {
    // An existing job takes precedence over the health gate below — if it's
    // cached and done, there's nothing to warn about; if loading it fails,
    // loadAndRender's own error path (existing behaviour) handles that.
    await loadAndRender(activeJobId);
    return;
  }
  const handled = await _gateIdleOnHealth(activeUrl);
  if (handled) return;
  if (activeUrl) renderState({ mode: "no-summary", url: activeUrl });
}

// ---------------------------------------------------------------------------
// First-run welcome screen — gates the idle ("no job for this tab") view
// behind GET /health so a fresh install sees setup guidance instead of the
// same error box a broken long-time install gets. See welcome.js for the
// rendered content and extension.md's "Side panel lifecycle" section for
// the full reasoning.
// ---------------------------------------------------------------------------

/**
 * Probe /health once. Never throws — an unreachable daemon IS one of the
 * three outcomes this module cares about, not something callers need to
 * catch separately.
 *
 * @returns {Promise<{ ok: true, health: import("../lib/api-types.js").HealthResponse } | { ok: false, error: unknown }>}
 */
async function _probeHealth() {
  try {
    const health = await daemon.health({ signal: AbortSignal.timeout(5000) });
    return { ok: true, health };
  } catch (error) {
    return { ok: false, error };
  }
}

/**
 * Gate an idle render (no active job for the current tab) behind GET
 * /health. Three-way split, all derived from one probe:
 *
 *   - daemon unreachable, `daemonEverReachable` never set → this is a
 *     fresh install with nothing configured yet: render the welcome
 *     screen's step 1 (install the daemon).
 *   - daemon unreachable, `daemonEverReachable` already set → a long-time
 *     user whose daemon just crashed/stopped. Showing "Welcome to TLDR"
 *     to them would be wrong (see welcome.js's module docstring), so this
 *     renders the SAME "error" state (and therefore the same
 *     `classifyError` "The daemon isn't running" hint) a failed job would
 *     have shown — no new copy, no new logic, just surfaced proactively
 *     instead of waiting for a click to fail first.
 *   - daemon reachable but `llm_backend_reachable === false` → welcome
 *     screen's step 2 (pick a model), regardless of history: the daemon
 *     answering at all already rules out "never installed anything".
 *   - everything fine → returns false; caller renders its normal idle view.
 *
 * @param {string | undefined} url
 * @returns {Promise<boolean>} true if this function already rendered a
 *   state (caller must not also render); false to proceed normally.
 */
async function _gateIdleOnHealth(url) {
  const probe = await _probeHealth();
  if (!probe.ok) {
    const { daemonEverReachable } = await chrome.storage.local.get("daemonEverReachable");
    if (daemonEverReachable) {
      renderState({ mode: "error", message: stringifyError(probe.error) });
    } else {
      renderState({ mode: "welcome", step: "daemon", health: null });
    }
    return true;
  }
  if (probe.health.llm_backend_reachable === false) {
    renderState({ mode: "welcome", step: "model", health: probe.health, url });
    return true;
  }
  return false;
}

/**
 * "Check again" button handler for the welcome screen — re-runs the same
 * gate and falls back to the normal "no summary yet" idle view once
 * everything's ready, without requiring the user to close/reopen the panel.
 *
 * @param {string | undefined} url
 */
async function _recheckIdleHealth(url) {
  const handled = await _gateIdleOnHealth(url);
  if (!handled) renderState({ mode: "no-summary", url: url || "" });
}

/**
 * Single entry point for tab-follow. Background fires this twice per switch:
 *
 *   phase 1 ({version, url}):           "moving to this tab, jobId unknown yet"
 *   phase 2 ({version, url, jobId}):    "listJobs done — here's the answer"
 *
 * The version filter above (`lastTabVersion`) discards stale phase-2 messages
 * if a newer switch already fired. Everything else is handled by
 * `renderState` idempotency: re-rendering the same state is a no-op, so we
 * don't need a side-flag like the old `domBlankedBySwitch` to remember
 * whether the DOM was wiped.
 *
 * @param {{ version?: number, url: string, jobId?: string | null }} msg
 */
async function handleSetActiveTab(msg) {
  const { url, jobId } = msg;
  const active = await getActiveJob();

  if (jobId === undefined) {
    // Phase 1: probe. background still resolving.
    if (active?.url === url) return;   // same tab — keep current view
    renderState({ mode: "loading" });
    return;
  }

  // Phase 2: definitive answer.
  if (jobId === null) {
    setActiveJob(null);
    clearChat();
    const handled = await _gateIdleOnHealth(url);
    if (!handled) renderState({ mode: "no-summary", url });
    window.scrollTo({ top: 0 });
    return;
  }

  if (active && active.id === jobId) {
    // Same job — render from in-memory. renderState idempotency keeps this
    // a no-op when nothing changed (window focus restored on the same tab
    // shouldn't flicker), and replaces a loading skeleton from phase 1.
    //
    // BUT: the cache can be stale in two ways — non-terminal status, or
    // terminal status with a missing payload (see `isCacheStale` for why).
    // Fall back to the cached view if the refresh fails.
    if (isCacheStale(active)) {
      const refreshed = await refreshActiveJob();
      if (refreshed) return;
    }
    renderFromJob(active);
    return;
  }
  await loadAndRender(jobId);
}

/**
 * Whether the cached active-job representation is too stale to render off
 * directly — caller should refresh from the daemon before rendering.
 *
 * Two cases count as stale:
 *
 * 1. **Non-terminal status.** The daemon may have advanced past the
 *    cached stage while the side panel was paused (window minimised
 *    throttles SSE), so the cache could under-report progress.
 *
 * 2. **Terminal status with missing payload.** This happens when
 *    `patchActiveJobIfMatches` receives a `job_event` (a lightweight
 *    JobSummary that has `status="done"` but no `summary_md`) and the
 *    per-job SSE "done" frame that *does* carry the content is dropped —
 *    typically because the side panel was throttled mid-stream by a
 *    window minimise. `renderFromJob` then sees `status="done" &&
 *    !summary_md` and falls through to the streaming placeholder,
 *    re-rendering the "Queued for transcription" view over a job that
 *    actually finished hours ago.
 *
 * @param {import("../lib/api-types.js").JobDetails | null | undefined} j
 */
function isCacheStale(j) {
  if (!j) return false;
  if (j.status !== "done" && j.status !== "failed") return true;
  if (j.status === "done" && !j.summary_md) return true;
  if (j.status === "failed" && !j.error) return true;
  return false;
}

/**
 * Re-fetch the currently-shown job from the daemon and re-render via
 * `renderFromJob`. Returns false if there's no active job or the fetch
 * failed (caller is responsible for any fallback).
 *
 * Why this exists: the side panel subscribes to GET /events for live
 * streaming state, but events that fire while the panel is throttled
 * (most commonly: window minimised → Chrome pauses SSE → /events "done"
 * arrives into a stalled fetch reader) are lost — EventSource has no
 * server-side replay. Without an explicit refresh, a streaming job
 * that finished during the pause is stuck on "queued"/"running" forever
 * even though the daemon long since wrote summary_md.
 *
 * Idempotent: `renderState` short-circuits when nothing changed, so
 * calling this when the job really is still streaming costs one HTTP
 * round-trip and zero DOM mutations.
 *
 * @returns {Promise<boolean>}
 */
async function refreshActiveJob() {
  const active = await getActiveJob();
  if (!active) return false;
  try {
    const fresh = await daemon.getJob(active.id, { signal: AbortSignal.timeout(8000) });
    setActiveJob(fresh);
    renderFromJob(fresh);
    return true;
  } catch (err) {
    console.warn("[TLDR] refresh active job failed", err);
    return false;
  }
}

// When the side panel becomes visible again (window restored from minimise,
// most often), any /events messages that landed during the pause are gone —
// the daemon only ever pushes live, no server-side replay. Two things may
// have gone stale:
//   1. The currently-shown job — could be stuck on "queued"/"running" even
//      though it actually finished while we were away.
//   2. The processing badge counter — could over-count if "done" events for
//      OTHER jobs were lost during the pause.
// Refresh both. Terminal cached state never changes so the refresh is
// skipped there for free.
//
// Independent from background.js's tab-sync because that path early-returns
// for non-summarizable tabs (Library, chrome://, …), so a user staring at
// Library when they un-minimise gets no `set-active-tab` and would
// otherwise stay stuck on the cached streaming view.
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState !== "visible") return;
  seedBadge().catch(() => {});
  getActiveJob().then((j) => {
    if (isCacheStale(j)) refreshActiveJob();
  });
});

/**
 * Pick the right renderState mode based on a job's current status.
 * Used wherever we have a job in hand and want "show this job, whatever
 * state it's in" without re-implementing the switch four times.
 *
 * @param {import("../lib/api-types.js").JobDetails} job
 */
function renderFromJob(job) {
  if (job.status === "done" && job.summary_md) {
    renderState({ mode: "done", job, content: job.summary_md });
  } else if (job.status === "failed") {
    renderState({ mode: "error", message: job.error || "Job failed.", job });
  } else {
    renderState({ mode: "streaming", job });
  }
}

/** @param {string} jobId */
async function loadAndRender(jobId) {
  renderState({ mode: "loading" });
  // Reset scroll to the top so the user sees the new title / summary from
  // the start (otherwise we may be parked deep in the previous job's chat).
  window.scrollTo({ top: 0 });

  // Safety net: if getJob hangs (daemon down, dead service worker, …) the
  // user gets a real error after 8s instead of an indefinite spinner.
  let job;
  try {
    job = await daemon.getJob(jobId, { signal: AbortSignal.timeout(8000) });
  } catch (err) {
    console.error("[TLDR] getJob failed", err);
    renderState({ mode: "error", message: stringifyError(err) });
    return;
  }
  setActiveJob(job);

  // Pull chat history in parallel with summary rendering.
  loadHistory(jobId).catch((e) => console.warn("[TLDR] message history failed", e));

  renderFromJob(job);
}

// ---------------------------------------------------------------------------
// Single source of truth for the summary pane.
//
// `renderState({mode, ...})` owns the triplet (`summaryEl.innerHTML`, the
// stage badge, chat enablement) for every state the pane can be in. Callers
// describe the new state declaratively; this function does the DOM writes,
// the secondary state plumbing, and the stream subscription teardown.
//
// Idempotent via `currentRender.key` — calling with the same state twice is
// a no-op, which matters for `streaming` (re-entering would drop the live
// subscription and reset the timeline) and `done` (would re-render the same
// markdown for no reason). State transitions still always render.
// ---------------------------------------------------------------------------

/** @typedef {(
 *   | { mode: "loading" }
 *   | { mode: "no-summary", url: string }
 *   | { mode: "welcome", step: "daemon" | "model", health: import("../lib/api-types.js").HealthResponse | null, url?: string }
 *   | { mode: "error", message: string, job?: import("../lib/api-types.js").JobDetails | null }
 *   | { mode: "done", job: import("../lib/api-types.js").JobDetails, content: string }
 *   | { mode: "streaming", job: import("../lib/api-types.js").JobDetails }
 * )} ViewState */

/** @type {{ key: string } | null} */
let currentRender = null;

/** @param {ViewState} state */
function renderState(state) {
  const key = _stateKey(state);
  if (currentRender?.key === key) return;
  currentRender = { key };

  // Every state transition drops any live subscription from the previous
  // state. Streaming re-attaches its own below.
  abortActiveStream();

  switch (state.mode) {
    case "loading":
      summaryEl.innerHTML = `
        <div class="status-block">
          <div class="spinner" aria-hidden="true"></div>
          <p>Loading…</p>
        </div>
      `;
      setStage(null);
      syncChatEnabled(false);
      return;

    case "no-summary":
      summaryEl.innerHTML = `
        <div class="placeholder-block">
          <p class="muted small url-line">${escapeHtml(state.url || "")}</p>
          <button class="summarize-btn" type="button">Summarize this page</button>
        </div>
      `;
      _bindSummarizeButton();
      setStage(null);
      syncChatEnabled(false);
      return;

    case "welcome": {
      const view = buildWelcomeView(state.step, state.health, {
        onCheckAgain: () => _recheckIdleHealth(state.url),
        onOpenOptions: () => chrome.runtime.openOptionsPage(),
      });
      summaryEl.innerHTML = "";
      summaryEl.appendChild(view);
      setStage(null);
      syncChatEnabled(false);
      return;
    }

    case "error": {
      const titleHtml = _titleHtml(state.job || null);
      const retryHtml = state.job?.id
        ? `<button class="retry-btn" data-retry-id="${escapeHtml(state.job.id)}">Retry</button>`
        : "";
      summaryEl.innerHTML = `
        ${titleHtml}
        <div class="status-block error">
          <p><strong>Error.</strong></p>
          <div class="error-hint"></div>
          ${retryHtml}
        </div>
      `;
      // Fire-and-forget: fills in the .error-hint div once GET /health
      // (best-effort) comes back. Never blocks the raw-text fallback,
      // which is built synchronously below.
      _renderErrorHint(state.message).catch((e) =>
        console.warn("[TLDR] error hint rendering failed:", e),
      );
      _bindRetryButton(state.job || null);
      setStage(null);
      syncChatEnabled(false);
      return;
    }

    case "done": {
      const titleHtml = _titleHtml(state.job);
      const html = renderMarkdown(state.content || "_(empty summary)_", state.job);
      summaryEl.innerHTML = `${titleHtml}<div class="markdown-body">${html}</div>`;
      setStage(null);
      syncChatEnabled(true);
      // Fire-and-forget: attaches a "look" affordance next to any [MM:SS]
      // marker that lands near one of this job's own deixis moments.
      // Never fetches a frame itself — only on click (see the function).
      _attachMomentAffordances(state.job).catch((err) =>
        console.warn("[TLDR] moment affordance setup failed:", err),
      );
      return;
    }

    case "streaming": {
      const titleHtml = _titleHtml(state.job);
      summaryEl.innerHTML =
        `${titleHtml}` +
        `<ul class="timeline" id="phase-timeline"></ul>` +
        `<div class="queued-hint" id="queued-hint" hidden></div>` +
        `<div class="markdown-body" id="summary-stream"></div>`;
      _attachStreamSubscription(state.job);
      // Chat is disabled while streaming: daemon /ai/qa requires status=done.
      syncChatEnabled(false);
      return;
    }
  }
}

/** @param {ViewState} state */
function _stateKey(state) {
  switch (state.mode) {
    case "loading":    return "loading";
    case "no-summary": return `no-summary:${state.url || ""}`;
    case "welcome":    return `welcome:${state.step}:${state.health?.llm_backend_error || ""}`;
    case "error":      return `error:${state.job?.id || ""}:${state.message}`;
    case "done":       return `done:${state.job.id}:${state.content?.length ?? 0}`;
    case "streaming":  return `streaming:${state.job.id}`;
  }
}

// ---------------------------------------------------------------------------
// "Look" affordance — on-demand video frames for a summary line whose
// [MM:SS] marker sits on a moment the speaker pointed at the video's
// picture (see daemon GET/POST /jobs/{id}/moments|frames,
// daemon/src/workers/deixis.py). Nothing is fetched until the user clicks.
//
// Only FEW summary lines should ever earn this — see
// MOMENT_MATCH_TOLERANCE_SECONDS below for why 10s, with the measured
// counts that justify it.
// ---------------------------------------------------------------------------

// A rendered [MM:SS] marker and the deixis moment it was drawn from rarely
// land on the exact same second (the summary LLM cites the transcript
// marker of whichever sentence/line it drew the fact from, which can start
// a few seconds before or after the exact phrase workers/deixis.py
// detected). This constant is how far apart the two are allowed to be and
// still count as "the same moment" for showing the affordance.
//
// MEASURED against 8 real video jobs in the owner's SQLite DB (real
// summary_md + real raw_segments_json), counting how many [MM:SS]-marked
// summary lines would get the affordance at each candidate window,
// out of every marked line that had ANY deixis moment on the job at all:
//
//   window(s):    3     5     8    10    15    20    30
//   hits/92:      2     3     4     5     7    11    14
//   percentage: 2.2%  3.3%  4.3%  5.4%  7.6% 12.0% 15.2%
//
// 10s keeps the overall rate low (5.4% — genuinely "few lines", matching
// the owner's "не пихать лишь бы пихать" rule) while still catching real
// matches in most measured jobs. 15s already pushes the worst single job
// to 3 of its 8 marked lines (38%) — i.e. "most" of that job's lines,
// which is exactly the noise threshold the rule rejects; 10s tops out at
// 2 of 8 (25%) for that same job. Segments in this DB run 1-5s apart and
// workers/deixis.py's own COLLAPSE_WINDOW_SECONDS (3s) already merges a
// gesture spanning consecutive segments into one moment, so 10s comfortably
// covers "summary cited the sentence's start, not the exact phrase" slack
// without reaching into unrelated nearby timestamps.
const MOMENT_MATCH_TOLERANCE_SECONDS = 10;

/**
 * Attach a "look" button next to every `[MM:SS]` timecode link in the
 * rendered summary that lands within `MOMENT_MATCH_TOLERANCE_SECONDS` of
 * one of this job's own deixis moments. Cheap no-op for jobs that can
 * never have moments (page/PDF) — skips the network call entirely.
 *
 * @param {import("../lib/api-types.js").JobDetails} job
 */
async function _attachMomentAffordances(job) {
  if (job.kind !== "youtube" && job.kind !== "media") return;
  const container = summaryEl.querySelector(".markdown-body");
  if (!container) return;
  const anchors = /** @type {HTMLAnchorElement[]} */ (
    Array.from(container.querySelectorAll("a[data-tldr-seconds]"))
  );
  if (anchors.length === 0) return;

  /** @type {import("../lib/api-types.js").DeixisMoment[]} */
  let moments;
  try {
    moments = (await daemon.getMoments(job.id)).items;
  } catch (err) {
    console.warn("[TLDR] failed to load deixis moments:", err);
    return;
  }
  if (!moments || moments.length === 0) return;

  for (const a of anchors) {
    const seconds = Number(a.dataset.tldrSeconds);
    if (!Number.isFinite(seconds)) continue;
    const match = _nearestMoment(moments, seconds);
    if (match) _insertLookAffordance(a, job, match);
  }
}

/**
 * @param {import("../lib/api-types.js").DeixisMoment[]} moments
 * @param {number} seconds
 * @returns {import("../lib/api-types.js").DeixisMoment | null}
 */
function _nearestMoment(moments, seconds) {
  let best = null;
  let bestDist = Infinity;
  for (const m of moments) {
    const dist = Math.abs(m.seconds - seconds);
    if (dist <= MOMENT_MATCH_TOLERANCE_SECONDS && dist < bestDist) {
      best = m;
      bestDist = dist;
    }
  }
  return best;
}

// Small camera-ish glyph (a well-worn open-source icon shape, redrawn here
// as plain inline SVG — no icon library, no emoji) for the "look" button.
const _LOOK_ICON_SVG =
  '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" ' +
  'stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
  'stroke-linejoin="round" aria-hidden="true">' +
  '<rect x="3" y="3" width="18" height="18" rx="2"></rect>' +
  '<circle cx="8.5" cy="8.5" r="1.5"></circle>' +
  '<polyline points="21 15 16 10 5 21"></polyline>' +
  "</svg>";

// Swapped in on a successful fetch. A plain checkmark, not the camera —
// the button STAYS next to the marker (see _handleLookClick) so its shape
// changing is the signal that this one already produced the row below it,
// distinct from every still-clickable camera icon elsewhere in the summary.
const _LOOK_DONE_ICON_SVG =
  '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" ' +
  'stroke="currentColor" stroke-width="2.5" stroke-linecap="round" ' +
  'stroke-linejoin="round" aria-hidden="true">' +
  '<polyline points="20 6 9 17 4 12"></polyline>' +
  "</svg>";

// How many of the fetched frames to actually render inline, per moment
// category — the full set workers/frames.py extracted (5 by default)
// always stays on disk and reachable via GET /jobs/{id}/frames/... either
// way; this only controls what gets shown in the summary.
//
// OBJECT candidates are, per workers/deixis.py's own category docs, "worth
// ONE good frame" — a ~1fps sample of the same still shot read as one
// picture printed several times over, which is exactly the noise the
// owner's rule rejects (and wrapped a bullet list apart in practice). Show
// exactly one — the QA LOOK step already does the same for this reason.
//
// ACTION candidates are documented as "worth several CONSECUTIVE frames" —
// a demonstrated motion genuinely reads better as a short sequence than a
// single freeze-frame. 3 is enough to show progression (start/mid/end of
// the fetched batch, via _pickRepresentativeFrames) without reintroducing
// the same wrap-the-layout-apart problem a full 5 caused.
const FRAMES_SHOWN_BY_CATEGORY = { object: 1, action: 3 };

/**
 * Pick a representative subset of `items` to render, per
 * FRAMES_SHOWN_BY_CATEGORY. For a single pick, takes the batch's middle
 * frame — workers/frames.py's section window is asymmetric (WINDOW_BEFORE/
 * AFTER) but centered close to the actual moment, so the middle of the
 * fetched sample is the best single stand-in. For a small handful, spreads
 * the picks evenly across the batch (first/mid/last-ish) to show
 * progression rather than clustering at one end.
 *
 * @param {import("../lib/api-types.js").FrameRef[]} items
 * @param {string} category
 * @returns {import("../lib/api-types.js").FrameRef[]}
 */
function _pickRepresentativeFrames(items, category) {
  const want = FRAMES_SHOWN_BY_CATEGORY[category] ?? 1;
  if (items.length <= want) return items;
  if (want <= 1) return [items[Math.floor((items.length - 1) / 2)]];
  const picked = new Set();
  for (let i = 0; i < want; i++) {
    picked.add(Math.round((i * (items.length - 1)) / (want - 1)));
  }
  return [...picked].sort((a, b) => a - b).map((i) => items[i]);
}

/**
 * @param {HTMLAnchorElement} anchor
 * @param {import("../lib/api-types.js").JobDetails} job
 * @param {import("../lib/api-types.js").DeixisMoment} moment
 */
function _insertLookAffordance(anchor, job, moment) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "look-affordance";
  btn.title = "Show the video frame here";
  btn.setAttribute("aria-label", "Show the video frame here");
  btn.innerHTML = _LOOK_ICON_SVG;
  btn.addEventListener("click", (ev) => {
    // Not nested inside the timecode <a>, so this wouldn't reach the
    // delegated seek handler anyway — stopPropagation is just cheap insurance.
    ev.preventDefault();
    ev.stopPropagation();
    _handleLookClick(btn, anchor, job, moment).catch((err) =>
      console.warn("[TLDR] look-affordance click failed:", err),
    );
  });
  anchor.insertAdjacentElement("afterend", btn);
}

/**
 * @param {HTMLButtonElement} btn
 * @param {HTMLAnchorElement} anchor
 * @param {import("../lib/api-types.js").JobDetails} job
 * @param {import("../lib/api-types.js").DeixisMoment} moment
 */
async function _handleLookClick(btn, anchor, job, moment) {
  if (btn.disabled) return;
  _clearLookError(btn);
  btn.disabled = true;
  btn.classList.add("look-affordance--pending");
  try {
    const { items } = await daemon.fetchMomentFrames(job.id, moment.seconds);
    const base = await daemon.baseUrl();
    const chosen = _pickRepresentativeFrames(items, moment.category);
    const row = buildFrameRow(job, chosen, base);
    row.classList.add("look-frame-row");
    // Insert under the WHOLE bullet/paragraph, not mid-sentence right
    // after the marker — the anchor can sit mid-sentence with more text
    // (and its closing punctuation) still to come. Falls back to the
    // anchor's parent for markup shapes without an enclosing li/p.
    const block = anchor.closest("li, p") || anchor.parentElement || anchor;
    block.insertAdjacentElement("afterend", row);

    // Deliberate end state: the button STAYS right next to the marker
    // (so the marker keeps visible context for what produced the row) but
    // is permanently disabled — a second click must not silently
    // re-fetch — and swaps to a checkmark so its own shape tells the user
    // this exact marker is the one that produced the row below the bullet.
    btn.classList.remove("look-affordance--pending");
    btn.classList.add("look-affordance--done");
    btn.innerHTML = _LOOK_DONE_ICON_SVG;
    btn.title = "Frame shown below";
    btn.setAttribute("aria-label", "Frame shown below");
    // No re-enable — see above.
  } catch (err) {
    btn.disabled = false;
    btn.classList.remove("look-affordance--pending");
    _showLookError(btn, stringifyError(err));
  }
}

/** @param {HTMLButtonElement} btn */
function _clearLookError(btn) {
  const next = btn.nextElementSibling;
  if (next?.classList.contains("look-affordance-error")) next.remove();
}

/**
 * @param {HTMLButtonElement} btn
 * @param {string} message
 */
function _showLookError(btn, message) {
  _clearLookError(btn);
  const span = document.createElement("span");
  span.className = "look-affordance-error";
  span.textContent = message;
  btn.insertAdjacentElement("afterend", span);
}

/**
 * @param {import("../lib/api-types.js").JobDetails | null} job
 */
function _titleHtml(job) {
  if (!job?.title) return "";
  const safeUrl = escapeHtml(job.url);
  const safeTitle = escapeHtml(job.title);
  const altHtml = _altSourcesHtml(job);
  return `<h2 class="job-title"><a href="${safeUrl}" target="_blank" rel="noopener">${safeTitle}</a></h2>${altHtml}`;
}

/**
 * "Wrong source?" chip for media jobs with multiple candidates on the
 * source page. Read-only for now — clicking expands a list of the other
 * candidates with their labels + URLs so the user can SEE we picked
 * something else, but switching requires manual "Process this page" on
 * another tab (Phase 2 wires actual one-click switch + old-job delete).
 *
 * @param {import("../lib/api-types.js").JobDetails | null} job
 * @returns {string}
 */
function _altSourcesHtml(job) {
  const alts = job?.alt_media_candidates || [];
  if (alts.length === 0) return "";
  const items = alts
    .map((c) => {
      const label = escapeHtml(c.label || c.media_url);
      const kind = escapeHtml(c.kind || "media");
      const url = escapeHtml(c.media_url);
      return `<li class="alt-source-item">
        <span class="alt-source-kind">${kind}</span>
        <span class="alt-source-label">${label}</span>
        <code class="alt-source-url">${url}</code>
      </li>`;
    })
    .join("");
  const count = alts.length;
  const word = count === 1 ? "source" : "sources";
  return `<details class="alt-sources">
    <summary>Wrong source? ${count} other ${word} on this page</summary>
    <ul class="alt-source-list">${items}</ul>
    <p class="alt-sources-hint muted small">
      Picker coming next — for now: open the right one in its own tab and re-run.
    </p>
  </details>`;
}

function _bindSummarizeButton() {
  const btn = /** @type {HTMLButtonElement | null} */ (
    summaryEl.querySelector(".summarize-btn")
  );
  if (!btn) return;
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "Starting…";
    try {
      await chrome.runtime.sendMessage({ type: "summarize-active-tab" });
    } catch (err) {
      console.error("[TLDR] summarize-active-tab failed", err);
      renderState({ mode: "error", message: stringifyError(err) });
    }
  });
}

/** @param {import("../lib/api-types.js").JobDetails | null} job */
function _bindRetryButton(job) {
  const btn = /** @type {HTMLButtonElement | null} */ (summaryEl.querySelector(".retry-btn"));
  if (!btn) return;
  btn.addEventListener("click", async () => {
    const id = btn.dataset.retryId;
    if (!id) return;
    btn.setAttribute("disabled", "true");
    try {
      await daemon.retryJob(id);
      await loadAndRender(id);
    } catch (err) {
      console.error("[TLDR] retry failed", err);
      btn.removeAttribute("disabled");
      renderState({ mode: "error", message: stringifyError(err), job });
    }
  });
}

/**
 * Fills the `.error-hint` div from the current "error" render with a
 * human-readable explanation (see lib/error-hints.js) above the raw text,
 * which always stays available under a "Technical details" `<details>`.
 *
 * GET /health feeds the "model backend unreachable"/"auth" branches — a
 * short timeout so a dead daemon (which can't answer /health either)
 * doesn't leave this hanging; a failed probe just means those two
 * branches can't fire, not that the whole hint is skipped.
 *
 * Built with textContent/createElement throughout, not innerHTML: `message`
 * and `health.llm_backend_error` both come from the daemon/backend, i.e.
 * untrusted input.
 *
 * @param {string} message
 */
async function _renderErrorHint(message) {
  const container = summaryEl.querySelector(".error-hint");
  if (!container) return; // state moved on before this ran

  /** @type {import("../lib/api-types.js").HealthResponse | null} */
  let health = null;
  try {
    health = await daemon.health({ signal: AbortSignal.timeout(6000) });
  } catch {
    health = null;
  }

  // The view may have moved on to a different render while /health was
  // in flight (retry, job switch, …) — bail rather than paint a stale
  // hint into a container that isn't the current one any more.
  if (summaryEl.querySelector(".error-hint") !== container) return;

  const hint = classifyError(message, health);
  container.textContent = "";

  if (hint) {
    const titleP = document.createElement("p");
    titleP.className = "error-hint-title";
    const strong = document.createElement("strong");
    strong.textContent = hint.title;
    titleP.appendChild(strong);
    container.appendChild(titleP);

    const explanationP = document.createElement("p");
    explanationP.className = "muted small";
    explanationP.textContent = hint.explanation;
    container.appendChild(explanationP);

    if (hint.action?.kind === "open-options") {
      const actionBtn = document.createElement("button");
      actionBtn.type = "button";
      actionBtn.className = "error-hint-action";
      actionBtn.textContent = hint.action.label;
      actionBtn.addEventListener("click", () => chrome.runtime.openOptionsPage());
      container.appendChild(actionBtn);
    }
  }

  const details = document.createElement("details");
  details.className = "error-hint-details";
  const summaryNode = document.createElement("summary");
  summaryNode.textContent = "Technical details";
  details.appendChild(summaryNode);
  const rawP = document.createElement("p");
  rawP.textContent = message;
  details.appendChild(rawP);
  container.appendChild(details);
}

/**
 * Wire up the live event-stream subscription for a streaming job. Expects
 * `summaryEl` to already contain the streaming skeleton (timeline + stream
 * div). The unsubscribe handle goes into `activeStreamUnsubscribe`; any
 * later `renderState(...)` (or explicit `abortActiveStream`) tears it down.
 *
 * No second SSE connection is opened — we filter the global event stream by
 * job_id. This keeps each side panel at exactly one long-lived connection
 * and avoids running into Chrome's 6-per-origin HTTP/1.1 cap.
 *
 * @param {import("../lib/api-types.js").JobDetails} job
 */
function _attachStreamSubscription(job) {
  const timelineEl = /** @type {HTMLElement} */ (document.getElementById("phase-timeline"));
  const streamEl = /** @type {HTMLElement} */ (document.getElementById("summary-stream"));
  const initialStage = job.progress_stage || "queued";

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
  // No detail is known yet at cold load — GET /jobs/{id} carries no trace
  // of a DeferredReason (see _renderQueuedHint's docstring). Only a LIVE
  // "stage" event below can ever populate this.
  _renderQueuedHint(initialStage, undefined);

  if (acc) {
    timelineEl.classList.add("timeline--collapsed");
    streamEl.innerHTML = renderMarkdown(acc, job);
  }

  // rAF throttle: multiple delta events in the same JS task batch collapse
  // into one DOM write per frame.
  const scheduleRender = () => {
    if (rafId !== null) return;
    rafId = requestAnimationFrame(() => {
      rafId = null;
      if (acc) streamEl.innerHTML = renderMarkdown(acc, job);
    });
  };
  const cancelRender = () => {
    if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
  };

  activeStreamUnsubscribe = eventStream.subscribe((ev) => {
    if (ev.job_id !== job.id) return;
    if (ev.type === "stage") {
      setStage(ev.stage, ev.detail);
      pushOrUpdatePhase(phases, ev.stage, ev.detail || undefined);
      renderTimeline(timelineEl, phases);
      _renderQueuedHint(ev.stage, ev.detail);
    } else if (ev.type === "delta") {
      if (firstDelta) {
        markAllDone(phases);
        renderTimeline(timelineEl, phases);
        timelineEl.classList.add("timeline--collapsed");
        firstDelta = false;
      }
      acc += ev.delta;
      streamAccCache.set(job.id, acc);
      scheduleRender();
    } else if (ev.type === "done") {
      cancelRender();
      streamAccCache.delete(job.id);
      const content = ev.content || acc;
      // Persist summary_md into the in-memory active-job cache: a later
      // renderFromJob(active) (window restore, tab switch back) would
      // otherwise see status="done" with summary_md=null (the daemon's
      // job_event publishes JobSummary without summary_md) and fall
      // through to the streaming placeholder. Merge into the *current*
      // active so mid-stream patches (yt-dlp title, etc.) aren't lost.
      getActiveJob().then((cur) => {
        if (cur?.id === job.id) {
          setActiveJob({ ...cur, status: "done", summary_md: content });
        }
      });
      renderState({ mode: "done", job, content });
    } else if (ev.type === "error") {
      cancelRender();
      streamAccCache.delete(job.id);
      const error = ev.error || "Error";
      // Same reasoning as the "done" branch — mirror status/error into the
      // cache so the failed-state render branch is taken on later renders.
      getActiveJob().then((cur) => {
        if (cur?.id === job.id) {
          setActiveJob({ ...cur, status: "failed", error });
        }
      });
      renderState({ mode: "error", message: error, job });
    }
  });
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

/**
 * Fills/hides the `#queued-hint` div (see the "streaming" render case)
 * with a human explanation of why a job is PARKED in the "queued" stage
 * — see lib/error-hints.js's `describeQueuedDetail` docstring for why
 * this is a distinct signal/function from the "error" hint, and why it
 * can only ever be populated from a live "stage" event's `detail`, never
 * from a cold `GET /jobs/{id}` load.
 *
 * Deliberately NOT the red `.status-block.error` styling — the job is
 * waiting, not dead. Built with textContent/createElement: this renders
 * from the daemon's own event stream, i.e. untrusted input.
 *
 * @param {string | null | undefined} stage
 * @param {string | null | undefined} detail
 */
function _renderQueuedHint(stage, detail) {
  const el = /** @type {HTMLElement | null} */ (document.getElementById("queued-hint"));
  if (!el) return;
  el.textContent = "";

  const info = stage === "queued" ? describeQueuedDetail(detail) : null;
  if (!info) {
    el.hidden = true;
    return;
  }
  el.hidden = false;

  const titleP = document.createElement("p");
  titleP.className = "queued-hint-title";
  const strong = document.createElement("strong");
  strong.textContent = info.title;
  titleP.appendChild(strong);
  el.appendChild(titleP);

  const explanationP = document.createElement("p");
  explanationP.className = "muted small";
  explanationP.textContent = info.explanation;
  el.appendChild(explanationP);

  if (info.action?.kind === "open-options") {
    const actionBtn = document.createElement("button");
    actionBtn.type = "button";
    actionBtn.className = "queued-hint-action";
    actionBtn.textContent = info.action.label;
    actionBtn.addEventListener("click", () => chrome.runtime.openOptionsPage());
    el.appendChild(actionBtn);
  }
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

