// Video chips — the side panel's per-video tabs for the current browser tab.
//
// The panel shows one "Page" chip (the page job, keyed by the normalized tab
// URL) plus one chip per video that is live in the tab RIGHT NOW. Chips are
// ephemeral: rebuilt from a scan of every frame (`scanFrame`, injected via
// executeScript allFrames) plus the network sniffer's list for the tab
// (`sniff:<tabId>`, see stream-sniff.js). Nothing is restored from the DB —
// a chip only finds its job by its key (the job URL; POST /jobs and
// listJobs({url}) dedup on it).
//
// Chip sources:
//   - whitelisted embed iframes (EMBED_PATTERNS) — a frame whose own URL is
//     such an embed only enriches that chip (duration / playing / frameId),
//     it never yields a second chip;
//   - native <video>/<audio> with an http(s) src, any frame;
//   - blob: (MSE) players in a frame for which the sniffer saw a manifest
//     from the same origin.
// Anything with a known duration < MIN_VIDEO_SECONDS is dropped (ads, VAST
// pre-rolls, UI sounds). Native/sniffed items need a known duration — it is
// part of their key; an embed with unknown duration is kept.
//
// Keys:
//   - YouTube embed  → https://www.youtube.com/watch?v=<id>  (kind=youtube job)
//   - other embed    → normalized embed URL (url AND media_url, kind=media)
//   - native/sniffed → `${normalizeUrl(pageUrl)}#tldr-video=<round(duration)>`
//     Iframe/manifest URLs carry per-load tokens, the duration doesn't, and
//     it tells trailer from film. The fragment is part of the key, so it
//     must reach the daemon verbatim — never pass it through normalizeUrl().
//
// Pure except `scanFrame` (self-contained, serialized via toString()).

import { normalizeUrl } from "./url.js";

/** Known-duration items shorter than this never become chips. */
export const MIN_VIDEO_SECONDS = 60;

export const VIDEO_KEY_MARK = "#tldr-video=";

/**
 * Iframe-embed whitelist as `[source, flags]` pairs (plain data so it can
 * cross executeScript's JSON boundary). Only hosts yt-dlp has a dedicated
 * extractor for, so ads / comment widgets never become chips.
 *
 * @type {Array<[string, string]>}
 */
export const EMBED_PATTERNS = [
  /^https:\/\/player\.vimeo\.com\/video\/\d+/i,
  /^https:\/\/(?:www\.)?dailymotion\.com\/embed\/video\//i,
  /^https:\/\/(?:geo\.)?dailymotion\.com\/(?:embed|player)\//i,
  // Twitch VOD only — live streams (`?channel=`) never finish transcribing.
  /^https:\/\/player\.twitch\.tv\/\?video=/i,
  /^https:\/\/iframe\.mediadelivery\.net\/(?:embed|play)\//i,
  /^https:\/\/players\.brightcove\.net\/\d+\//i,
  /^https:\/\/(?:cdn|content)\.jwplayer\.com\/(?:players|previews|videos)\//i,
  /^https:\/\/fast\.wistia\.(?:com|net)\/embed\//i,
  /^https:\/\/(?:www\.)?streamable\.com\/[eo]\//i,
  /^https:\/\/w\.soundcloud\.com\/player\//i,
  /^https:\/\/open\.spotify\.com\/embed\//i,
  /^https:\/\/(?:www\.)?facebook\.com\/plugins\/video\.php/i,
  /^https:\/\/(?:www\.)?youtube(?:-nocookie)?\.com\/embed\/[\w-]{6,}/i,
].map((re) => /** @type {[string, string]} */ ([re.source, re.flags]));

const EMBED_RES = EMBED_PATTERNS.map(([s, f]) => new RegExp(s, f));
const YT_EMBED_RE = /^https:\/\/(?:www\.)?youtube(?:-nocookie)?\.com\/embed\/([\w-]{6,})/i;

/** @param {string | null | undefined} url */
export function isEmbedUrl(url) {
  return !!url && EMBED_RES.some((re) => re.test(url));
}

/** @param {string | null | undefined} url @returns {string | null} */
export function youtubeEmbedId(url) {
  const m = (url || "").match(YT_EMBED_RE);
  return m ? m[1] : null;
}

/**
 * Job URL for an embed: canonical watch URL for YouTube, else the
 * normalized embed URL.
 * @param {string} url
 */
export function embedKey(url) {
  const id = youtubeEmbedId(url);
  return id ? `https://www.youtube.com/watch?v=${id}` : normalizeUrl(url);
}

/**
 * Job URL for a native/sniffed video of ``pageUrl``.
 * @param {string} pageUrl
 * @param {number} duration seconds
 */
export function videoKey(pageUrl, duration) {
  return `${normalizeUrl(pageUrl)}${VIDEO_KEY_MARK}${Math.round(duration)}`;
}

/** Whether ``url`` is a native/sniffed video key (not a page URL). */
export function isVideoKey(url) {
  return typeof url === "string" && url.includes(VIDEO_KEY_MARK);
}

/** @param {number} d */
const known = (d) => Number.isFinite(d) && d > 0;

/** @param {string | null | undefined} url */
function originOf(url) {
  try {
    const o = new URL(url || "").origin;
    return o && o !== "null" ? o : null;
  } catch {
    return null;
  }
}

/**
 * Sniffed entries recorded from the same origin as ``frameUrl`` (an
 * entry's frameUrl is the documentUrl or just the initiator origin).
 *
 * @template {{frameUrl: string | null}} T
 * @param {T[]} list
 * @param {string | null | undefined} frameUrl
 * @returns {T[]}
 */
export function scopeSniffed(list, frameUrl) {
  const o = originOf(frameUrl);
  if (!o) return [];
  return (list || []).filter((e) => originOf(e.frameUrl) === o);
}

/** @param {number | null | undefined} sec */
export function formatDuration(sec) {
  if (!sec || !Number.isFinite(sec) || sec <= 0) return "";
  const s = Math.round(sec);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

/**
 * @typedef {object} ScannedMedia
 * @property {"video"|"audio"} kind
 * @property {string | null} src   http(s) URL, null for blob:/data:/none
 * @property {boolean} blob        MSE / blob: source
 * @property {number} duration     0 = unknown
 * @property {boolean} paused
 * @property {string} label        curated label ("" = none)
 */

/**
 * @typedef {object} FrameScan
 * @property {string} frameUrl
 * @property {boolean} isTop
 * @property {ScannedMedia[]} media
 * @property {Array<{src: string, label: string}>} embeds whitelisted iframes in this frame's DOM
 */

/**
 * @typedef {object} VideoChip
 * @property {string} key            job URL (identity)
 * @property {"youtube"|"embed"|"native"|"stream"} type
 * @property {string | null} mediaUrl  media_url for the job (null for youtube/stream)
 * @property {string} label
 * @property {number | null} duration seconds, null = unknown
 * @property {boolean} playing
 * @property {number | null} frameId   frame holding the element (null = not loaded/scannable)
 * @property {string | null} frameUrl
 */

/**
 * Merge per-frame scans and the tab's sniffed list into the chip list
 * (Page chip excluded — the panel adds it).
 *
 * @param {{
 *   pageUrl: string,
 *   frames: Array<{frameId: number, result?: FrameScan | null}>,
 *   sniffed?: Array<{kind: string, frameUrl: string | null}>,
 * }} input
 * @returns {VideoChip[]}
 */
export function buildChips({ pageUrl, frames, sniffed = [] }) {
  /** @type {Map<string, VideoChip>} */
  const chips = new Map();
  const ok = (frames || []).filter((f) => f && f.result);
  const manifests = (sniffed || []).filter((e) => e.kind === "hls" || e.kind === "dash");

  // 1. Embed iframes seen in any frame's DOM.
  for (const f of ok) {
    for (const e of f.result?.embeds || []) {
      if (!isEmbedUrl(e.src)) continue;
      const key = embedKey(e.src);
      if (!chips.has(key)) chips.set(key, _embedChip(key, e.src, e.label));
    }
  }

  for (const f of ok) {
    const r = /** @type {FrameScan} */ (f.result);
    // 2. A frame that IS an embed: its media enrich that embed's chip.
    if (isEmbedUrl(r.frameUrl)) {
      const key = embedKey(r.frameUrl);
      let chip = chips.get(key);
      if (!chip) {
        chip = _embedChip(key, r.frameUrl, "");
        chips.set(key, chip);
      }
      chip.frameId = f.frameId;
      chip.frameUrl = r.frameUrl;
      for (const m of r.media || []) {
        if (known(m.duration) && m.duration > (chip.duration || 0)) chip.duration = m.duration;
        if (!m.paused) chip.playing = true;
      }
      continue;
    }
    // 3. Native http(s) media / sniffed MSE players.
    const frameHasManifest = scopeSniffed(manifests, r.frameUrl).length > 0;
    for (const m of r.media || []) {
      if (!known(m.duration) || m.duration < MIN_VIDEO_SECONDS) continue;
      /** @type {VideoChip["type"] | null} */
      let type = null;
      if (m.src) type = "native";
      else if (m.blob && frameHasManifest) type = "stream";
      if (!type) continue;
      const key = videoKey(pageUrl, m.duration);
      const prev = chips.get(key);
      if (prev) {
        if (!m.paused) prev.playing = true;
        continue;
      }
      chips.set(key, {
        key,
        type,
        mediaUrl: type === "native" ? m.src : null,
        label: m.label || (type === "stream" ? _hostLabel(r.frameUrl) || "Player" : m.kind === "audio" ? "Audio" : "Video"),
        duration: Math.round(m.duration),
        playing: !m.paused,
        frameId: f.frameId,
        frameUrl: r.frameUrl,
      });
    }
  }

  const out = [...chips.values()].filter(
    (c) => !(c.duration != null && c.duration < MIN_VIDEO_SECONDS),
  );
  _disambiguate(out);
  return out;
}

/** @param {string} key @param {string} src @param {string} label @returns {VideoChip} */
function _embedChip(key, src, label) {
  const yt = !!youtubeEmbedId(src);
  return {
    key,
    type: yt ? "youtube" : "embed",
    mediaUrl: yt ? null : key,
    label: _short(label) || (yt ? "YouTube" : _hostLabel(src) || "Embed"),
    duration: null,
    playing: false,
    frameId: null,
    frameUrl: null,
  };
}

/** @param {string | null} url */
function _hostLabel(url) {
  try {
    return new URL(url || "").hostname.replace(/^(?:www|player|players|w|open|fast|iframe|cdn|content|geo)\./, "");
  } catch {
    return "";
  }
}

/** @param {string} s */
function _short(s) {
  const t = (s || "").replace(/\s+/g, " ").trim();
  return t.length > 40 ? `${t.slice(0, 39)}…` : t;
}

/** @param {VideoChip[]} chips */
function _disambiguate(chips) {
  const seen = new Map();
  for (const c of chips) {
    const n = (seen.get(c.label) ?? 0) + 1;
    seen.set(c.label, n);
    if (n > 1) c.label = `${c.label} ${n}`;
  }
}

/**
 * Pick the chip key to show for ``jobUrl`` among the tab's chips: the Page
 * chip (``"page"``) when it is the page job, a video chip's key, or null
 * when the job doesn't belong to this tab (e.g. opened from the Library).
 *
 * @param {string | null | undefined} jobUrl
 * @param {string} pageUrl normalized tab URL
 * @param {VideoChip[]} chips
 * @returns {string | null}
 */
export function chipKeyForJob(jobUrl, pageUrl, chips) {
  if (!jobUrl) return null;
  if (jobUrl === pageUrl) return "page";
  return chips.some((c) => c.key === jobUrl) ? jobUrl : null;
}

// ---------------------------------------------------------------------------
// Injected (self-contained)
// ---------------------------------------------------------------------------

/**
 * In-page: report this frame's media elements and whitelisted iframes.
 * @param {Array<[string, string]>} patterns EMBED_PATTERNS
 * @returns {FrameScan}
 */
export function scanFrame(patterns) {
  const res = (patterns || []).map(([s, f]) => new RegExp(s, f));
  /** @param {Element} el */
  const labelOf = (el) => {
    const t = el.getAttribute("title") || el.getAttribute("aria-label") || "";
    return t.trim().slice(0, 80);
  };
  const media = [];
  for (const el of /** @type {HTMLMediaElement[]} */ ([...document.querySelectorAll("video, audio")])) {
    let raw = el.currentSrc || el.src || "";
    if (!raw) {
      const s = /** @type {HTMLSourceElement | null} */ (el.querySelector("source[src]"));
      if (s) raw = s.src;
    }
    let src = null;
    if (raw && !raw.startsWith("blob:") && !raw.startsWith("data:")) {
      try {
        const u = new URL(raw, location.href);
        if (u.protocol === "http:" || u.protocol === "https:") src = u.toString();
      } catch {}
    }
    const d = el.duration;
    media.push({
      kind: el.tagName === "VIDEO" ? "video" : "audio",
      src,
      blob: raw.startsWith("blob:") || (!raw && !!(/** @type {any} */ (el).srcObject)),
      duration: Number.isFinite(d) && d > 0 ? d : 0,
      paused: el.paused,
      label: labelOf(el),
    });
  }
  const embeds = [];
  for (const f of /** @type {HTMLIFrameElement[]} */ ([...document.querySelectorAll("iframe[src]")])) {
    const src = f.src;
    if (src && res.some((re) => re.test(src))) embeds.push({ src, label: labelOf(f) });
  }
  return {
    frameUrl: location.href,
    isTop: window === window.top,
    media: /** @type {any} */ (media),
    embeds,
  };
}
