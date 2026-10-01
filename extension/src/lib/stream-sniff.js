// Pure helpers for the network stream sniffer in background.js.
//
// Some players live in a cross-origin iframe and play through MediaSource
// (hls.js, dash.js), so the DOM only ever shows a blob: <video> src. The
// manifest URL still crosses the network, though — background.js watches
// requests with chrome.webRequest and records manifests (and subtitle
// files) per tab. Everything here is chrome-free so it can be unit-tested
// with plain `node --test`.

/** @typedef {"hls" | "dash" | "subtitle"} SniffKind */

/**
 * @typedef {object} SniffedStream
 * @property {string} url
 * @property {SniffKind} kind
 * @property {string | null} frameUrl   - document (usually the player iframe) that made the request
 * @property {string | null} referer
 * @property {string | null} origin
 * @property {string | null} userAgent
 * @property {number} firstSeen          - ms epoch
 * @property {number} lastSeen
 */

// Per group: manifests and subtitle files are capped separately, so a
// player fetching WebVTT segment after segment can't evict the master.
export const SNIFF_CAP = 40;

/** @param {SniffKind} kind */
const sniffGroup = (kind) => (kind === "subtitle" ? "subtitle" : "manifest");

const EXT_KIND = /** @type {Record<string, SniffKind>} */ ({
  m3u8: "hls",
  m3u: "hls",
  mpd: "dash",
  vtt: "subtitle",
  srt: "subtitle",
  ass: "subtitle",
});

// Media segments, init fragments, keys — high-volume noise we never record
// even if the server labels them with a manifest-ish content type.
const SEGMENT_EXTS = new Set([
  "ts", "m4s", "m4a", "m4v", "mp4", "aac", "mp3", "ac3", "ec3", "webm",
  "cmfv", "cmfa", "fmp4", "key", "jpg", "jpeg", "png", "webp", "gif",
  "js", "css", "json", "html", "woff", "woff2",
]);

const CT_KIND = /** @type {Record<string, SniffKind>} */ ({
  "application/vnd.apple.mpegurl": "hls",
  "application/x-mpegurl": "hls",
  "audio/mpegurl": "hls",
  "audio/x-mpegurl": "hls",
  "application/dash+xml": "dash",
  "text/vtt": "subtitle",
  "application/x-subrip": "subtitle",
});

/** @param {string} url @returns {string} lowercase extension of the path, "" if none */
function pathExt(url) {
  let path;
  try {
    path = new URL(url).pathname;
  } catch {
    return "";
  }
  const last = path.slice(path.lastIndexOf("/") + 1);
  const dot = last.lastIndexOf(".");
  return dot > 0 ? last.slice(dot + 1).toLowerCase() : "";
}

/**
 * Classify a request. The path extension wins when it's telling (manifest
 * or known segment/asset); otherwise the response Content-Type decides —
 * pirate CDNs often serve playlists from extension-less or disguised paths.
 *
 * @param {string} url
 * @param {string | null | undefined} [contentType]
 * @returns {SniffKind | null}
 */
export function classifyStream(url, contentType) {
  if (!/^https?:/i.test(url)) return null;
  const ext = pathExt(url);
  if (EXT_KIND[ext]) return EXT_KIND[ext];
  if (SEGMENT_EXTS.has(ext)) return null;
  const ct = (contentType || "").split(";")[0].trim().toLowerCase();
  return CT_KIND[ct] || null;
}

/**
 * Case-insensitive header lookup on a webRequest header array.
 *
 * @param {{name: string, value?: string}[] | undefined} headers
 * @param {string} name
 * @returns {string | null}
 */
export function headerValue(headers, name) {
  const lower = name.toLowerCase();
  const h = (headers || []).find((x) => x.name.toLowerCase() === lower);
  return h?.value ?? null;
}

/**
 * Insert or refresh ``entry`` in a URL-deduped list ordered by first sight.
 * A repeat keeps its firstSeen/position, bumps lastSeen and fills header
 * fields that were missing the first time. Past ``cap`` entries of the same
 * group (manifests vs subtitles), the oldest of that group is dropped.
 *
 * @param {SniffedStream[]} list
 * @param {SniffedStream} entry
 * @param {number} [cap]
 * @returns {SniffedStream[]} new list
 */
export function upsertStream(list, entry, cap = SNIFF_CAP) {
  const idx = list.findIndex((e) => e.url === entry.url);
  if (idx >= 0) {
    const prev = list[idx];
    const merged = {
      ...prev,
      lastSeen: entry.lastSeen,
      frameUrl: prev.frameUrl || entry.frameUrl,
      referer: prev.referer || entry.referer,
      origin: prev.origin || entry.origin,
      userAgent: prev.userAgent || entry.userAgent,
    };
    return list.map((e, i) => (i === idx ? merged : e));
  }
  const next = [...list, entry];
  const group = sniffGroup(entry.kind);
  const inGroup = next.filter((e) => sniffGroup(e.kind) === group);
  if (inGroup.length <= cap) return next;
  const drop = new Set(inGroup.slice(0, inGroup.length - cap));
  return next.filter((e) => !drop.has(e));
}

/** @param {string | null} url @returns {string | null} */
function originOf(url) {
  if (!url) return null;
  try {
    const o = new URL(url).origin;
    return o && o !== "null" ? o : null;
  } catch {
    return null;
  }
}

/**
 * The tab's list in the daemon's ``sniffed_streams`` shape (http(s) only,
 * most recent ``max`` kept).
 *
 * @param {SniffedStream[]} list
 * @param {number} [max]
 * @returns {{url: string, kind: SniffKind, first_seen: number, last_seen: number}[]}
 */
export function toSniffedStreams(list, max = 100) {
  return (list || [])
    .filter((e) => /^https?:/i.test(e.url))
    .slice()
    .sort((a, b) => a.lastSeen - b.lastSeen)
    .slice(-max)
    .map((e) => ({
      url: e.url,
      kind: e.kind,
      first_seen: e.firstSeen,
      last_seen: e.lastSeen,
    }));
}

/**
 * Headers the daemon should replay when fetching the stream: what the
 * player actually sent, falling back to the player frame's origin (which is
 * what a cross-origin XHR under the default referrer policy sends anyway).
 *
 * @param {SniffedStream} entry
 * @returns {Record<string, string> | null}
 */
export function mediaHeadersFor(entry) {
  const frameOrigin = originOf(entry.frameUrl);
  /** @type {Record<string, string>} */
  const out = {};
  const referer = entry.referer || (frameOrigin ? `${frameOrigin}/` : null);
  const origin = entry.origin || frameOrigin;
  if (referer) out.Referer = referer;
  if (origin) out.Origin = origin;
  if (entry.userAgent) out["User-Agent"] = entry.userAgent;
  return Object.keys(out).length ? out : null;
}

// Path names packagers commonly give a master playlist / DASH manifest.
const MASTER_NAME_RE = /(?:^|\/)(?:master|playlist|manifest|main)[^/]*\.m3u8$|\.mpd$/i;

/** @param {string} url */
export function looksLikeMaster(url) {
  try {
    return MASTER_NAME_RE.test(new URL(url).pathname);
  } catch {
    return false;
  }
}

/**
 * Pick a media job out of a tab's sniffed list. Only a default: the
 * extension can't tell master from rendition playlists without their body
 * (all ``.m3u8``), so the whole list also goes to the daemon as
 * ``sniffed_streams`` and it makes the real choice (master, audio dub,
 * subtitle track). Default ``mediaUrl``: the most recently seen manifest
 * whose name looks like a master, else the earliest manifest (a master is
 * fetched before its renditions). Other manifests ride along as
 * alternates; subtitle entries don't — the "wrong source?" picker would
 * treat a .vtt as media.
 *
 * @param {SniffedStream[]} list
 * @returns {{
 *   mediaUrl: string,
 *   mediaHeaders: Record<string, string> | null,
 *   altCandidates: {mediaUrl: string, kind: "video", label: string}[],
 * } | null}
 */
export function pickSniffedMedia(list) {
  const manifests = (list || [])
    .filter((e) => e.kind === "hls" || e.kind === "dash")
    .slice()
    .sort((a, b) => a.firstSeen - b.firstSeen);
  if (!manifests.length) return null;
  const masterish = manifests.filter((e) => looksLikeMaster(e.url));
  const primary = masterish.length
    ? masterish.reduce((a, b) => (b.lastSeen >= a.lastSeen ? b : a))
    : manifests[0];
  const rest = manifests.filter((e) => e !== primary);
  return {
    mediaUrl: primary.url,
    mediaHeaders: mediaHeadersFor(primary),
    altCandidates: rest.map((e, i) => ({
      mediaUrl: e.url,
      kind: "video",
      label: `Stream ${i + 2} (${e.kind === "hls" ? "HLS" : "DASH"})`,
    })),
  };
}

// ---------------------------------------------------------------------------
// Service-worker attribution. A player that registers its own Service Worker
// fetches its playlists from there, and chrome.webRequest reports those
// requests with tabId -1. We remember which tabs host a frame of which
// origin and attribute a SW request to the most recently seen tab hosting a
// frame with the request's initiator origin.
// ---------------------------------------------------------------------------

/** @typedef {{tabId: number, ts: number}} FrameSighting */
/** @typedef {Record<string, FrameSighting[]>} FrameOriginMap  origin → sightings, most recent first */

// Sightings kept per origin — enough for a handful of tabs on one player host.
export const FRAME_TABS_PER_ORIGIN = 8;

/**
 * Record that ``tabId`` hosts a frame (top or sub) with ``url``'s origin.
 *
 * @param {FrameOriginMap} map
 * @param {string} url
 * @param {number} tabId
 * @param {number} ts
 * @returns {FrameOriginMap} new map (``map`` itself if the URL has no usable origin)
 */
export function recordFrame(map, url, tabId, ts) {
  const origin = /^https?:/i.test(url || "") ? originOf(url) : null;
  if (!origin || tabId < 0) return map;
  const rest = (map[origin] || []).filter((s) => s.tabId !== tabId);
  return { ...map, [origin]: [{ tabId, ts }, ...rest].slice(0, FRAME_TABS_PER_ORIGIN) };
}

/**
 * Tabs hosting a frame with ``origin``, most recently seen first.
 *
 * @param {FrameOriginMap} map
 * @param {string | null | undefined} origin  an origin or any URL on it
 * @returns {number[]}
 */
export function lookupFrameTabs(map, origin) {
  const o = originOf(origin || null);
  if (!o) return [];
  return (map[o] || []).map((s) => s.tabId);
}

/**
 * Forget every frame ``tabId`` hosts (tab closed or navigated top-level).
 *
 * @param {FrameOriginMap} map
 * @param {number} tabId
 * @returns {FrameOriginMap}
 */
export function pruneFrameTab(map, tabId) {
  /** @type {FrameOriginMap} */
  const out = {};
  for (const [origin, list] of Object.entries(map)) {
    const kept = list.filter((s) => s.tabId !== tabId);
    if (kept.length) out[origin] = kept;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Debug ring: recent requests classifyStream rejected, so a site where the
// sniffer records nothing can be diagnosed from storage. Never stores query
// strings — they carry tokens.
// ---------------------------------------------------------------------------

/**
 * @typedef {object} SniffDebugEntry
 * @property {string} host
 * @property {string} pathTail      - last 60 chars of the path, no query
 * @property {string} ext
 * @property {string | null} contentType
 * @property {number} status
 * @property {boolean} viaSW         - attributed from a tabId -1 (service worker) request
 * @property {number} ts
 */

export const SNIFF_DEBUG_CAP = 40;

const DEBUG_NOISE_EXTS = new Set([
  "jpg", "jpeg", "png", "webp", "gif", "svg", "ico", "avif", "bmp",
  "woff", "woff2", "ttf", "otf", "eot", "css", "js", "mjs", "map",
]);

/**
 * Build a debug-ring entry, or null for obvious noise / non-http URLs.
 *
 * @param {string} url
 * @param {string | null | undefined} contentType
 * @param {number} status
 * @param {boolean} viaSW
 * @param {number} ts
 * @returns {SniffDebugEntry | null}
 */
export function debugEntry(url, contentType, status, viaSW, ts) {
  let u;
  try {
    u = new URL(url);
  } catch {
    return null;
  }
  if (!/^https?:$/i.test(u.protocol)) return null;
  const ext = pathExt(url);
  if (DEBUG_NOISE_EXTS.has(ext)) return null;
  const ct = (contentType || "").split(";")[0].trim().toLowerCase() || null;
  if (ct && /^(image|font)\/|^text\/css$|javascript/.test(ct)) return null;
  return {
    host: u.host,
    pathTail: u.pathname.slice(-60),
    ext,
    contentType: ct,
    status,
    viaSW,
    ts,
  };
}

/**
 * Append to a ring, keeping the newest ``cap``.
 *
 * @param {SniffDebugEntry[]} ring
 * @param {SniffDebugEntry} entry
 * @param {number} [cap]
 * @returns {SniffDebugEntry[]}
 */
export function pushDebug(ring, entry, cap = SNIFF_DEBUG_CAP) {
  return [...(ring || []), entry].slice(-cap);
}

/**
 * One-line-per-entry summary for console.info.
 *
 * @param {SniffDebugEntry[]} ring
 * @returns {string[]}
 */
export function summarizeDebug(ring) {
  return (ring || []).map(
    (e) =>
      `${e.status} ${e.viaSW ? "[sw] " : ""}${e.host}${e.pathTail} ` +
      `ext=${e.ext || "-"} ct=${e.contentType || "-"}`,
  );
}
