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
import {
  classifyStream,
  headerValue,
  lookupFrameTabs,
  pickSniffedMedia,
  pruneFrameTab,
  recordFrame,
  toSniffedStreams,
  upsertStream,
} from "./lib/stream-sniff.js";
import { scopeSniffed } from "./lib/page-videos.js";

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
// Network stream sniffer. Players inside cross-origin iframes (hls.js /
// MediaSource, blob: src) are invisible to the content script, but their
// manifests cross the network. Record HLS/DASH manifests and subtitle files
// per tab so the side panel can offer those players as video chips. Listeners are
// registered at top level so a request wakes the service worker; the lists
// live in chrome.storage.session to survive SW sleep.
// ---------------------------------------------------------------------------

const SNIFF_KEY_PREFIX = "sniff:";
const SNIFF_TYPES = /** @type {chrome.webRequest.ResourceType[]} */ ([
  "xmlhttprequest",
  "media",
  "other",
]);

/** @param {number} tabId */
const sniffKey = (tabId) => `${SNIFF_KEY_PREFIX}${tabId}`;
const FRAME_ORIGINS_KEY = "sniffFrames";

// requestId → request headers seen in onSendHeaders, consumed by
// onHeadersReceived (which is where the Content-Type and status are known).
// In memory only: both events fire within one request, and a lost entry
// just means a record without Referer/Origin (mediaHeadersFor falls back).
/** @type {Map<string, {referer: string|null, origin: string|null, userAgent: string|null}>} */
const pendingRequestHeaders = new Map();

// Serialise read-modify-write of each tab's list — manifests and subtitle
// files often arrive in bursts.
/** @type {Map<number, Promise<void>>} */
const sniffWriteChains = new Map();

/**
 * @param {number} tabId
 * @param {(list: import("./lib/stream-sniff.js").SniffedStream[]) => import("./lib/stream-sniff.js").SniffedStream[] | null} update
 *   returns the new list, or null to delete the key
 */
function updateSniffed(tabId, update) {
  const prev = sniffWriteChains.get(tabId) || Promise.resolve();
  const next = prev
    .then(async () => {
      const key = sniffKey(tabId);
      const stored = await chrome.storage.session.get(key);
      const list = update(stored[key] || []);
      if (list === null) await chrome.storage.session.remove(key);
      else await chrome.storage.session.set({ [key]: list });
    })
    .catch((err) => console.warn("[TLDR] sniff store", err))
    .finally(() => {
      if (sniffWriteChains.get(tabId) === next) sniffWriteChains.delete(tabId);
    });
  sniffWriteChains.set(tabId, next);
  return next;
}

/** @param {number} tabId */
function clearSniffed(tabId) {
  updateSniffed(tabId, () => null);
}

// Frame origin → tabs hosting a frame of that origin (most recent first).
// Lets us attribute service-worker requests (tabId -1) — some players
// register their own SW and fetch playlists from it. In memory for speed,
// mirrored to chrome.storage.session because the extension SW sleeps.
/** @type {import("./lib/stream-sniff.js").FrameOriginMap | null} */
let frameOrigins = null;
/** @type {Promise<import("./lib/stream-sniff.js").FrameOriginMap> | null} */
let frameOriginsLoad = null;
/** @type {Promise<void>} */
let frameOriginsWrite = Promise.resolve();

/** @returns {Promise<import("./lib/stream-sniff.js").FrameOriginMap>} */
function loadFrameOrigins() {
  if (frameOrigins) return Promise.resolve(frameOrigins);
  frameOriginsLoad ||= chrome.storage.session
    .get(FRAME_ORIGINS_KEY)
    .then((stored) => {
      frameOrigins ||= stored[FRAME_ORIGINS_KEY] || {};
      return /** @type {import("./lib/stream-sniff.js").FrameOriginMap} */ (frameOrigins);
    })
    .catch(() => (frameOrigins ||= {}));
  return frameOriginsLoad;
}

/** @param {(map: import("./lib/stream-sniff.js").FrameOriginMap) => import("./lib/stream-sniff.js").FrameOriginMap} update */
function updateFrameOrigins(update) {
  frameOriginsWrite = frameOriginsWrite
    .then(async () => {
      const map = update(await loadFrameOrigins());
      frameOrigins = map;
      await chrome.storage.session.set({ [FRAME_ORIGINS_KEY]: map });
    })
    .catch((err) => console.warn("[TLDR] frame origins store", err));
}

/**
 * Tab a service-worker request (tabId -1) belongs to: the most recently
 * seen still-open tab hosting a frame with the request's origin.
 *
 * @param {{initiator?: string, documentUrl?: string}} details
 * @returns {Promise<number | null>}
 */
async function attributeSwRequest(details) {
  const origin = details.documentUrl || details.initiator;
  if (!origin) return null;
  await frameOriginsWrite;
  for (const tabId of lookupFrameTabs(await loadFrameOrigins(), origin)) {
    try {
      await chrome.tabs.get(tabId);
      return tabId;
    } catch {
      // Tab is gone (onRemoved missed while the SW slept) — try the next.
    }
  }
  return null;
}

/** @param {number} tabId @returns {Promise<import("./lib/stream-sniff.js").SniffedStream[]>} */
async function getSniffed(tabId) {
  await sniffWriteChains.get(tabId);
  const key = sniffKey(tabId);
  const stored = await chrome.storage.session.get(key);
  return stored[key] || [];
}

/** @param {chrome.webRequest.WebRequestHeadersDetails} details */
function onSniffSendHeaders(details) {
  // tabId -1 (service-worker requests) is kept too: the requestId pairs it
  // with onHeadersReceived, which does the tab attribution.
  // Can't classify yet (Content-Type comes with the response), so keep
  // every request's headers until onHeadersReceived/onErrorOccurred.
  pendingRequestHeaders.set(details.requestId, {
    referer: headerValue(details.requestHeaders, "Referer"),
    origin: headerValue(details.requestHeaders, "Origin"),
    userAgent: headerValue(details.requestHeaders, "User-Agent"),
  });
  // Safety valve for requests that never reach onHeadersReceived.
  if (pendingRequestHeaders.size > 500) {
    const oldest = pendingRequestHeaders.keys().next().value;
    if (oldest !== undefined) pendingRequestHeaders.delete(oldest);
  }
}

/** @param {chrome.webRequest.WebResponseHeadersDetails} details */
async function onSniffHeadersReceived(details) {
  const sent = pendingRequestHeaders.get(details.requestId) || null;
  pendingRequestHeaders.delete(details.requestId);
  // documentUrl is Chrome 106+ only on some event types; initiator is the
  // requesting origin, good enough to derive Origin/Referer fallbacks.
  const d = /** @type {{documentUrl?: string}} */ (details);
  const viaSW = details.tabId < 0;
  const contentType = headerValue(details.responseHeaders, "Content-Type");
  const kind = classifyStream(details.url, contentType);
  if (!kind) return;
  if (details.statusCode < 200 || details.statusCode >= 300) return;
  const tabId = viaSW ? await attributeSwRequest(details) : details.tabId;
  if (tabId === null) return;
  const now = Date.now();
  const entry = {
    url: details.url,
    kind,
    frameUrl: d.documentUrl || details.initiator || null,
    referer: sent?.referer ?? null,
    origin: sent?.origin ?? null,
    userAgent: sent?.userAgent ?? null,
    firstSeen: now,
    lastSeen: now,
  };
  updateSniffed(tabId, (list) => upsertStream(list, entry));
}

{
  const filter = { urls: ["<all_urls>"], types: SNIFF_TYPES };
  try {
    // "extraHeaders" is what exposes Referer/Origin in Chrome.
    chrome.webRequest.onSendHeaders.addListener(onSniffSendHeaders, filter, [
      "requestHeaders",
      "extraHeaders",
    ]);
  } catch {
    // Firefox has no "extraHeaders" (and exposes those headers without it).
    chrome.webRequest.onSendHeaders.addListener(onSniffSendHeaders, filter, [
      "requestHeaders",
    ]);
  }
  chrome.webRequest.onHeadersReceived.addListener(onSniffHeadersReceived, filter, [
    "responseHeaders",
  ]);
  // A new top-level document means a new page: forget the old streams and
  // the old page's frames. Every frame (top or sub) is remembered by origin
  // for service-worker attribution.
  chrome.webRequest.onBeforeRequest.addListener(
    (details) => {
      if (details.tabId < 0) return;
      const now = Date.now();
      if (details.type === "main_frame") {
        clearSniffed(details.tabId);
        updateFrameOrigins((map) =>
          recordFrame(pruneFrameTab(map, details.tabId), details.url, details.tabId, now),
        );
      } else {
        updateFrameOrigins((map) => recordFrame(map, details.url, details.tabId, now));
      }
    },
    { urls: ["<all_urls>"], types: ["main_frame", "sub_frame"] },
  );
  chrome.webRequest.onErrorOccurred.addListener(
    (details) => pendingRequestHeaders.delete(details.requestId),
    filter,
  );
}

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
  if (msg.type === "analyze-video") {
    // From the side panel's video-chip CTA (sender has no tab — the target
    // tab travels in the message).
    handleAnalyzeVideo(msg).catch((e) => {
      console.error("[TLDR] analyze-video", e);
      broadcast({ type: "extraction-error", error: stringifyError(e) });
    });
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
  // Toolbar = this page, always. Videos found on the page (including a
  // sniffed HLS/DASH player) are offered as video chips in the side panel
  // and submitted via `analyze-video` — see handleAnalyzeVideo.
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
 * De-duplicate two cookie lists (as produced by ``getCookiesForUrl``),
 * keeping the first occurrence of any (name, domain, path) triple — that's
 * the identity Chrome itself uses to distinguish otherwise-same-named
 * cookies scoped to different domains/paths.
 *
 * @param {import("./lib/api-types.js").Cookie[]} primary
 * @param {import("./lib/api-types.js").Cookie[]} extra
 * @returns {import("./lib/api-types.js").Cookie[]}
 */
function _mergeCookies(primary, extra) {
  const seen = new Set(primary.map((c) => `${c.name}\0${c.domain}\0${c.path}`));
  const merged = primary.slice();
  for (const c of extra) {
    const key = `${c.name}\0${c.domain}\0${c.path}`;
    if (!seen.has(key)) {
      seen.add(key);
      merged.push(c);
    }
  }
  return merged;
}

/**
 * Generic media discovered on a non-YouTube page (native <video>/<audio>
 * with a real URL, or an iframe embed from a known media host). The daemon
 * routes this to ``kind=media`` and feeds ``mediaUrl`` straight to yt-dlp,
 * which has site-specific extractors for hundreds of hosts plus a generic
 * fallback for direct mp4/HLS/DASH links. On a site where the scraped
 * ``mediaUrl`` is just a bare CDN file (e.g. a broadcaster's <video> src),
 * yt-dlp may have no dedicated extractor for it even though it has one for
 * the *page* — the daemon then routes through ``url`` instead (see
 * pipeline.py's ``_run_media`` routing rule), so we can't assume in advance
 * which of the two URLs yt-dlp will actually fetch.
 *
 * Cookies: we forward the union of what a real HTTP request to ``mediaUrl``
 * would carry AND what one to ``url`` would carry (URL-scoped, via
 * chrome.cookies.getAll({url}) for each), merged and de-duplicated by
 * (name, domain, path). That covers session cookies for player auth, CDN
 * signing tokens, and the page's own site auth, without leaking unrelated
 * cookies from sibling subdomains.
 *
 * ``mediaHeaders`` is set only for network-sniffed streams (see
 * handleAnalyzeVideo): the Referer/Origin/User-Agent the player sent,
 * which pirate-ish CDNs check.
 *
 * ``jobUrl`` (video chips) overrides the job's ``url``; ``url`` stays the
 * page URL for cookies.
 *
 * @param {{url:string, jobUrl?:string|null, mediaUrl:string, altCandidates?:{mediaUrl:string,kind:string,label:string}[], mediaHeaders?:Record<string,string>|null, sniffedStreams?:import("./lib/api-types.js").SniffedStream[]|null, title?:string|null, text?:string}} msg
 * @param {number|null} sourceTabId
 */
async function handleExtractedMedia(msg, sourceTabId) {
  let mediaCookies = [];
  try {
    mediaCookies = await getCookiesForUrl(msg.mediaUrl);
  } catch (err) {
    console.warn("[TLDR] cookies.getAll(url) failed", err);
  }
  let pageCookies = [];
  try {
    pageCookies = await getCookiesForUrl(msg.url);
  } catch (err) {
    console.warn("[TLDR] cookies.getAll(url) failed", err);
  }
  const cookies = _mergeCookies(mediaCookies, pageCookies);

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
    // A video chip's key (`…#tldr-video=N`) is the job identity and must
    // reach the daemon verbatim — normalizeUrl would strip the fragment.
    url: msg.jobUrl || normalizeUrl(msg.url),
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
    media_headers: msg.mediaHeaders || null,
    sniffed_streams: msg.sniffedStreams?.length ? msg.sniffedStreams : null,
  };
  await submitJob(req, sourceTabId);
}

/**
 * Create the job for one side-panel video chip (lib/page-videos.js). The
 * chip's ``key`` is the job URL, so a repeat click reuses the daemon's job.
 *
 *   youtube → kind=youtube for the canonical watch URL
 *   embed   → kind=media, url = media_url = normalized embed URL
 *   native  → kind=media, media_url = element src, url = key
 *   stream  → kind=media from the sniffed manifests of that frame's origin
 *             (whole tab list if none match), same shape as before
 *
 * @param {{tabId: number, chip: import("./lib/page-videos.js").VideoChip}} msg
 */
async function handleAnalyzeVideo({ tabId, chip }) {
  const tab = await chrome.tabs.get(tabId);
  if (!tab?.url || !chip?.key) return;
  const title = [tab.title, chip.label].filter(Boolean).join(" — ") || null;
  if (chip.type === "youtube") {
    await handleExtractedYouTube({ url: chip.key, title: null }, tabId);
    return;
  }
  if (chip.type === "stream") {
    const list = await getSniffed(tabId);
    const scoped = scopeSniffed(list, chip.frameUrl);
    const use = scoped.some((e) => e.kind === "hls" || e.kind === "dash") ? scoped : list;
    const sniffed = pickSniffedMedia(use);
    if (!sniffed) throw new Error("No stream was captured for this player yet — press play and try again.");
    await handleExtractedMedia(
      {
        url: tab.url,
        jobUrl: chip.key,
        mediaUrl: sniffed.mediaUrl,
        altCandidates: sniffed.altCandidates,
        mediaHeaders: sniffed.mediaHeaders,
        sniffedStreams: toSniffedStreams(use),
        title,
      },
      tabId,
    );
    return;
  }
  if (!chip.mediaUrl) return;
  await handleExtractedMedia(
    { url: tab.url, jobUrl: chip.key, mediaUrl: chip.mediaUrl, title },
    tabId,
  );
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
// look up whether we have a cached job for the new URL (the PAGE job: the
// lookup is an exact url match, so video-chip jobs keyed
// `<page>#tldr-video=N` never answer it). `tabId` lets the panel tell a
// real tab switch from a focus change on the tab whose video chip it shows. If yes, point the
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
  await broadcast({ type: "set-active-tab", version, url: normalized, tabId: tab.id });

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
  await broadcast({ type: "set-active-tab", version, url: normalized, jobId, tabId: tab.id });
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
  clearSniffed(tabId);
  updateFrameOrigins((map) => pruneFrameTab(map, tabId));
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
