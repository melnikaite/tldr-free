// Service worker — toolbar click → content extraction → POST /jobs.
// Also keeps the side panel pointed at the active tab.
//
// We do NOT poll job status here. The side panel does that itself while it's
// open (see sidepanel/app.js); when the panel is closed nobody needs to know.

import { daemon } from "./lib/daemon-client.js";
import { getCookiesForDomain } from "./lib/cookies.js";
import { normalizeUrl } from "./lib/url.js";
import { stringifyError } from "./lib/utils.js";

/** @import {
 *   JobCreateRequest,
 *   JobCreateResponse
 * } from "./lib/api-types.js" */

const YT_HOST_RE = /^https?:\/\/(?:[^/]*\.)?(?:youtube\.com|youtu\.be)(?:\/|$)/i;

chrome.runtime.onInstalled.addListener(() => {
  // Keep openPanelOnActionClick=false so chrome.action.onClicked fires for our
  // custom flow. We open the panel ourselves inside the click handler.
  chrome.sidePanel
    .setPanelBehavior({ openPanelOnActionClick: false })
    .catch(console.error);
});

// ---------------------------------------------------------------------------
// Entry points: toolbar click + Summarize button in the side panel both call
// the same flow — open the panel, run the right content script for the page.
// ---------------------------------------------------------------------------

/**
 * Run the Readability/YouTube extractor on a tab. Side-effect: emits an
 * extracted-* message that handleExtracted{Page,YouTube} below picks up.
 *
 * @param {chrome.tabs.Tab} tab
 */
async function summarizeTab(tab) {
  if (!tab.id || !tab.url) return;
  const isYouTube = YT_HOST_RE.test(tab.url);
  try {
    if (isYouTube) {
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ["src/content/youtube.js"],
      });
    } else {
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ["vendor/readability.js", "src/content/extract.js"],
      });
    }
  } catch (err) {
    console.error("[TLDR] executeScript failed", err);
    await broadcast({ type: "extraction-error", error: stringifyError(err) });
  }
}

chrome.action.onClicked.addListener(async (tab) => {
  if (!tab.id) return;
  // Open side panel up-front so the user sees an immediate response. This
  // must happen inside the user gesture; awaits afterwards are fine.
  try {
    await chrome.sidePanel.open({ tabId: tab.id });
  } catch (err) {
    console.warn("[TLDR] sidePanel.open failed", err);
  }
  await summarizeTab(tab);
});

// ---------------------------------------------------------------------------
// Messages from content scripts (extracted-*) and from the side panel
// (summarize-active-tab — the in-panel "Summarize this page" button).
// ---------------------------------------------------------------------------

chrome.runtime.onMessage.addListener((msg, sender, _sendResponse) => {
  if (!msg || typeof msg !== "object") return false;

  // Content scripts run inside the tab; sender.tab.id is the source tab.
  // We use it (NOT a URL comparison) to decide whether the just-submitted
  // job should take over the side panel — see submitJob below.
  const sourceTabId = sender?.tab?.id ?? null;

  if (msg.type === "extracted-page") {
    handleExtractedPage(msg, sourceTabId).catch((e) =>
      console.error("[TLDR] handleExtractedPage", e),
    );
    return false;
  }
  if (msg.type === "extracted-youtube") {
    handleExtractedYouTube(msg, sourceTabId).catch((e) =>
      console.error("[TLDR] handleExtractedYouTube", e),
    );
    return false;
  }
  if (msg.type === "summarize-active-tab") {
    handleSummarizeActiveTab().catch((e) =>
      console.error("[TLDR] summarize-active-tab", e),
    );
    return false;
  }
  return false;
});

async function handleSummarizeActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  if (!tab) return;
  await summarizeTab(tab);
}

/**
 * @param {{url:string, text?:string, title?:string|null}} msg
 * @param {number|null} sourceTabId  tab.id of the page where the content
 *   script ran. Used by submitJob to decide whether to take over the side
 *   panel (only when the user is still on that tab).
 */
async function handleExtractedPage(msg, sourceTabId) {
  /** @type {JobCreateRequest} */
  const req = {
    url: normalizeUrl(msg.url),
    kind: "page",
    page_text: msg.text || "",
    page_title: msg.title || null,
  };
  await submitJob(req, sourceTabId);
}

/**
 * @param {{url:string, title?:string|null}} msg
 * @param {number|null} sourceTabId
 */
async function handleExtractedYouTube(msg, sourceTabId) {
  let cookies = [];
  try {
    cookies = await getCookiesForDomain(".youtube.com");
  } catch (err) {
    console.warn("[TLDR] cookies.getAll failed", err);
  }

  /** @type {JobCreateRequest} */
  const req = {
    url: normalizeUrl(msg.url),
    kind: "youtube",
    page_title: msg.title || null,
    cookies,
  };
  await submitJob(req, sourceTabId);
}

/**
 * Submit a job to the daemon. POST /jobs is async — always 202 with the new
 * id. The side panel (if open) subscribes to /ai/stream {job_id} to watch
 * progress and stream the summary.
 *
 * The broadcast goes out for every successful submit so the sidebar's badge
 * counter and Library's table can refresh. We only flip the global
 * ``activeJobId`` (which the sidepanel uses to decide what to display)
 * when the user is still on the source tab (the one where they clicked
 * summarize) — otherwise we'd hijack the panel away from whatever the user
 * is now looking at.
 *
 * Tab-id comparison (NOT URL comparison) — `sender.tab.id` from the content
 * script's message is the unambiguous identity of the source. URL comparison
 * fails for SPAs (location.href in the content script can drift from
 * Chrome's reported tab.url across awaits) and was the root cause of "I
 * clicked summarize but the panel never updated".
 *
 * The broadcast also carries `shouldSwitch` so the side panel can switch in
 * place even if `chrome.storage.onChanged` doesn't fire (e.g. value already
 * matches; same-value sets are not guaranteed to notify).
 *
 * @param {JobCreateRequest} req
 * @param {number|null} sourceTabId
 */
async function submitJob(req, sourceTabId) {
  /** @type {JobCreateResponse} */
  let resp;
  try {
    resp = await daemon.createJob(req);
  } catch (err) {
    console.error("[TLDR] createJob failed", err);
    await broadcast({ type: "extraction-error", error: stringifyError(err) });
    return;
  }

  let shouldSwitch = false;
  if (sourceTabId != null) {
    try {
      const [activeTab] = await chrome.tabs.query({
        active: true, lastFocusedWindow: true,
      });
      shouldSwitch = activeTab?.id === sourceTabId;
    } catch (err) {
      console.warn("[TLDR] tabs.query failed", err);
    }
  }

  await broadcast({
    type: "job-created",
    jobId: resp.id,
    url: req.url,
    shouldSwitch,
  });

  if (shouldSwitch) {
    await chrome.storage.session.set({ activeJobId: resp.id, activeUrl: req.url });
  }
}

// ---------------------------------------------------------------------------
// Tab tracking — sidepanel content follows the active tab.
//
// On any change to the active tab (switched, navigated, window focus moved),
// look up whether we have a cached job for the new URL. If yes, point the
// sidepanel at it; if no, point it at "empty / not yet summarized" state.
// ---------------------------------------------------------------------------

/** @type {Map<number, string>} */
const lastSyncedUrlByTab = new Map();

/** @param {chrome.tabs.Tab} tab */
async function syncSidepanelForTab(tab) {
  const url = tab?.url;
  if (!url) return;
  // Non-summarizable tabs (the extension's own Library page, chrome://,
  // about:blank, file://, etc.) must NOT disturb the side panel — otherwise
  // glancing at the Library while a job streams would yank the panel away
  // from the in-progress summary and replace it with "Summarize this page"
  // for the library URL itself.
  if (!/^https?:/i.test(url)) return;

  const normalized = normalizeUrl(url);

  let jobId = null;
  try {
    const resp = await daemon.listJobs({ url: normalized, limit: 1 });
    jobId = resp.items?.[0]?.id ?? null;
  } catch (err) {
    console.warn("[TLDR] tab sync listJobs failed", err);
    return;
  }

  await chrome.storage.session.set({ activeJobId: jobId, activeUrl: normalized });
  await broadcast({ type: "tab-changed", url: normalized, jobId });
}

chrome.tabs.onActivated.addListener(async ({ tabId }) => {
  try {
    const tab = await chrome.tabs.get(tabId);
    if (tab?.url) lastSyncedUrlByTab.set(tabId, normalizeUrl(tab.url));
    await syncSidepanelForTab(tab);
  } catch (err) {
    console.warn("[TLDR] tabs.onActivated", err);
  }
});

chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (!changeInfo.url) return;
  if (!tab.active) return;
  const normalized = normalizeUrl(changeInfo.url);
  if (lastSyncedUrlByTab.get(tabId) === normalized) return;
  lastSyncedUrlByTab.set(tabId, normalized);
  await syncSidepanelForTab(tab);
});

chrome.tabs.onRemoved.addListener((tabId) => {
  lastSyncedUrlByTab.delete(tabId);
});

chrome.windows.onFocusChanged.addListener(async (windowId) => {
  if (windowId === chrome.windows.WINDOW_ID_NONE) return;
  try {
    const [tab] = await chrome.tabs.query({ active: true, windowId });
    if (tab) await syncSidepanelForTab(tab);
  } catch (err) {
    console.warn("[TLDR] windows.onFocusChanged", err);
  }
});

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/** @param {object} msg */
async function broadcast(msg) {
  try {
    await chrome.runtime.sendMessage(msg);
  } catch {
    // Side panel / library may not be open — ignore receiver-not-found.
  }
}
