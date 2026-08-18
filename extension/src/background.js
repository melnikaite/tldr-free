// Service worker — toolbar click → content extraction → POST /jobs.
// Also keeps the side panel pointed at the active tab.
//
// We do NOT poll job status here. The side panel does that itself while it's
// open (see sidepanel/app.js); when the panel is closed nobody needs to know.

import { daemon } from "./lib/daemon-client.js";
import { getCookiesForDomain, getCookiesForUrl } from "./lib/cookies.js";
import { normalizeUrl } from "./lib/url.js";
import { stringifyError } from "./lib/utils.js";
import { setPanelBehavior, openSidePanel } from "./lib/browser-compat.js";

/** @import {
 *   JobCreateRequest,
 *   JobCreateResponse
 * } from "./lib/api-types.js" */

const YT_HOST_RE = /^https?:\/\/(?:[^/]*\.)?(?:youtube\.com|youtu\.be)(?:\/|$)/i;

// PDF tabs are special: Chrome renders them in its own viewer
// (a chrome-extension://… page) which we cannot inject content scripts
// into. Match on the URL path — covers ``file:///x.pdf``,
// ``https://host/foo.pdf?download=1``, etc. The actual parsing happens
// daemon-side via pypdf (with multimodal vision OCR fallback for
// scanned PDFs); the extension's job is just to detect the kind and,
// for ``file://`` URLs the daemon can't reach, upload the bytes.
const PDF_URL_RE = /\.pdf(?:$|[?#])/i;

// Upload cap for ``file://`` PDFs. The extension's service worker
// (a single MV3 background context) has a bounded heap, and
// base64-encoding hundreds of megabytes there is the fastest way to
// OOM-kill it — taking the entire extension down with no clear error.
// 50 MB covers virtually every PDF that isn't a scanned book; for
// those, OCR with ocrmypdf locally first and the daemon's text-first
// path will read the result instantly. http(s) PDFs aren't subject to
// this — the daemon fetches them itself with no extension memory cost.
const MAX_LOCAL_PDF_BYTES = 50 * 1024 * 1024;

chrome.runtime.onInstalled.addListener(() => {
  // Keep openPanelOnActionClick=false so chrome.action.onClicked fires for our
  // custom flow. We open the panel ourselves inside the click handler.
  // (No-op on Firefox — see lib/browser-compat.js.)
  setPanelBehavior().catch(console.error);
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

  // PDFs: bypass content-script extraction entirely. Chrome's PDF viewer
  // is a chrome-extension:// page that refuses script injection, and
  // parsing PDFs is a daemon concern anyway (pypdf with vision OCR
  // fallback for scanned PDFs). For http(s) the daemon fetches the URL
  // itself; for file:// we read the bytes here and ship them along.
  if (PDF_URL_RE.test(tab.url)) {
    await handlePdfTab(tab).catch((err) => {
      console.error("[TLDR] PDF submit failed", err);
      broadcast({ type: "extraction-error", error: stringifyError(err) });
    });
    return;
  }

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

/**
 * Submit a PDF tab to the daemon as ``kind=pdf``. For http(s) URLs we
 * just send the URL and any cookies — the daemon fetches the bytes
 * itself. For ``file://`` URLs the daemon can't reach the host
 * filesystem, so we read the file from the extension (where Chrome
 * grants access if the user has enabled "Allow access to file URLs")
 * and forward the bytes as base64.
 *
 * No content-script injection involved — Chrome's PDF viewer refuses
 * `chrome.scripting.executeScript` and parsing PDFs belongs in the
 * daemon anyway (pypdf + vision OCR fallback for scanned PDFs).
 *
 * @param {chrome.tabs.Tab} tab
 */
async function handlePdfTab(tab) {
  if (!tab.url) return;
  const isFileUrl = tab.url.startsWith("file://");

  let cookies = [];
  let pdfBytesB64 = null;
  if (isFileUrl) {
    // Daemon can't reach file:// — read the bytes here and upload.
    let buf;
    try {
      const resp = await fetch(tab.url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      buf = await resp.arrayBuffer();
    } catch (err) {
      throw new Error(
        'Could not read the local PDF. Open chrome://extensions, click "Details" ' +
          'on the TLDR card, and enable "Allow access to file URLs". Then reload ' +
          `the PDF tab and try again. (Underlying error: ${stringifyError(err)})`,
      );
    }
    if (buf.byteLength > MAX_LOCAL_PDF_BYTES) {
      const mb = (buf.byteLength / 1_048_576).toFixed(1);
      const cap = Math.round(MAX_LOCAL_PDF_BYTES / 1_048_576);
      throw new Error(
        `Local PDF is ${mb} MB — over the ${cap} MB upload cap. ` +
          "OCR the PDF locally first (e.g. ocrmypdf), or split it into smaller files.",
      );
    }
    pdfBytesB64 = _arrayBufferToBase64(buf);
  } else {
    // http(s) — forward cookies so signed/auth-protected PDFs work.
    try {
      cookies = await getCookiesForUrl(tab.url);
    } catch (err) {
      console.warn("[TLDR] cookies.getAll(url) failed", err);
    }
  }

  /** @type {JobCreateRequest} */
  const req = {
    url: normalizeUrl(tab.url),
    kind: "pdf",
    page_title: tab.title || null,
    pdf_bytes_b64: pdfBytesB64,
    cookies,
  };
  await submitJob(req, tab.id ?? null);
}

/**
 * Base64-encode an ArrayBuffer with bounded peak memory.
 *
 * The naive ``btoa(String.fromCharCode(...all_bytes))`` approach
 * accumulates the entire buffer as a UTF-16 JS string (each byte → 2
 * bytes of memory) BEFORE encoding — that's 3× the PDF size briefly,
 * which OOM-kills an MV3 service worker on multi-megabyte PDFs.
 *
 * Here we encode in 48 KB chunks (chosen as 49152 = 16384 × 3, a
 * multiple of 3 so every chunk except the last produces a clean base64
 * segment with no padding). Each chunk's intermediate binary string is
 * only ~48 KB and is freed before the next chunk begins, so peak
 * additional memory above the input is bounded by ``output.length``
 * (~4/3 × input) — not 3× as with the naive form.
 *
 * @param {ArrayBuffer} buf
 * @returns {string}
 */
function _arrayBufferToBase64(buf) {
  const bytes = new Uint8Array(buf);
  const CHUNK = 49152; // 48 KB, multiple of 3
  const parts = [];
  for (let i = 0; i < bytes.length; i += CHUNK) {
    const slice = bytes.subarray(i, Math.min(i + CHUNK, bytes.length));
    // Single-arg fromCharCode in a loop (NOT spread/apply) — keeps the
    // intermediate string bounded by the chunk size, doesn't blow the
    // call stack with hundreds of thousands of arguments.
    let bin = "";
    for (let j = 0; j < slice.length; j++) {
      bin += String.fromCharCode(slice[j]);
    }
    parts.push(btoa(bin));
  }
  return parts.join("");
}

chrome.action.onClicked.addListener(async (tab) => {
  if (!tab.id) return;
  // Open side panel up-front so the user sees an immediate response. This
  // must happen inside the user gesture; awaits afterwards are fine.
  try {
    await openSidePanel({ tabId: tab.id });
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
  if (msg.type === "extracted-media") {
    handleExtractedMedia(msg, sourceTabId).catch((e) =>
      console.error("[TLDR] handleExtractedMedia", e),
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
 * Generic media discovered on a non-YouTube page (native <video>/<audio>
 * with a real URL, or an iframe embed from a known media host). The daemon
 * routes this to ``kind=media`` and feeds ``mediaUrl`` straight to yt-dlp,
 * which has site-specific extractors for hundreds of hosts plus a generic
 * fallback for direct mp4/HLS/DASH links.
 *
 * Cookies: we forward exactly the cookies a real HTTP request to
 * ``mediaUrl`` would carry (URL-scoped, via chrome.cookies.getAll({url})).
 * That covers session cookies for player auth, CDN signing tokens, etc.,
 * without leaking unrelated cookies from sibling subdomains.
 *
 * @param {{url:string, mediaUrl:string, altCandidates?:{mediaUrl:string,kind:string,label:string}[], title?:string|null, text?:string}} msg
 * @param {number|null} sourceTabId
 */
async function handleExtractedMedia(msg, sourceTabId) {
  let cookies = [];
  try {
    cookies = await getCookiesForUrl(msg.mediaUrl);
  } catch (err) {
    console.warn("[TLDR] cookies.getAll(url) failed", err);
  }

  // Convert extension-side {mediaUrl, kind, label} to the snake_case shape
  // the daemon expects. Drop alternates with the same URL as the primary —
  // happens when the same <video> appears multiple times (e.g. mirrored
  // mobile/desktop sources).
  const altCandidates = (msg.altCandidates || [])
    .filter((c) => c.mediaUrl && c.mediaUrl !== msg.mediaUrl)
    .map((c) => ({
      media_url: c.mediaUrl,
      kind: c.kind,
      label: c.label,
    }));

  /** @type {JobCreateRequest} */
  const req = {
    url: normalizeUrl(msg.url),
    kind: "media",
    media_url: msg.mediaUrl,
    alt_media_candidates: altCandidates.length ? altCandidates : null,
    page_title: msg.title || null,
    // Best-effort page text extracted alongside the media candidate — the
    // daemon falls back to summarizing this when the media turns out too
    // short to contain speech, or its transcript comes back empty (see
    // workers/runner.py's page-text fallback).
    page_text: msg.text || "",
    cookies,
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

// Monotonic counter so the side panel can drop responses for tabs the user
// has already left. Every call to `syncSidepanelForTab` bumps it; the late
// completion of an older listJobs gets discarded both here (we skip the
// final broadcast if `version !== switchVersion`) and on the receiver
// side (it ignores messages with stale `version`).
//
// Versions are wall-clock (Date.now) floored to strict-monotonic, NOT a
// plain in-memory counter, so the version space survives service-worker
// restarts. MV3 routinely tears down idle backgrounds after ~30s; a fresh
// 0-reset counter would then issue versions smaller than the side panel's
// remembered `lastTabVersion`, and the receiver would silently drop every
// `set-active-tab` until the panel itself reloaded — most visibly: switch
// tabs after a quiet period and the panel keeps showing the previous job.
// `Math.max(prev + 1, Date.now())` guarantees the new value is strictly
// larger than both the local counter and any version any previous worker
// ever issued (assuming a sane clock).
let switchVersion = 0;

function nextSwitchVersion() {
  switchVersion = Math.max(switchVersion + 1, Date.now());
  return switchVersion;
}

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
  const version = nextSwitchVersion();

  // Phase 1: tell the panel which tab we're moving to *before* hitting the
  // daemon. jobId is omitted to mean "still resolving". The panel uses this
  // to decide whether to blank to a spinner (different URL) or stay put
  // (same URL — e.g. window focus restored on the same tab).
  await broadcast({ type: "set-active-tab", version, url: normalized });

  let jobId = null;
  try {
    const resp = await daemon.listJobs({ url: normalized, limit: 1 });
    jobId = resp.items?.[0]?.id ?? null;
  } catch (err) {
    console.warn("[TLDR] tab sync listJobs failed", err);
    // Fall through with jobId = null — phase 2 is always sent so the panel
    // stays in sync even when the daemon is temporarily unreachable.
  }

  // A newer syncSidepanelForTab fired after us — its phase 1 already moved
  // the panel forward. Don't clobber it with our late phase 2.
  if (version !== switchVersion) return;

  await chrome.storage.session.set({ activeJobId: jobId, activeUrl: normalized });
  await broadcast({ type: "set-active-tab", version, url: normalized, jobId });
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
