// Render markdown to sanitized HTML, then post-process [MM:SS] / [HH:MM:SS]
// markers into clickable links.
//
// Two link flavours are produced depending on the job:
//   - YouTube (kind=youtube, resolveVideoId returns a real id): a regular
//     `youtube.com/watch?v=…&t=Ns` URL. Click handler in app.js seeks an
//     already-open YouTube tab via executeScript.
//   - Generic media (kind=media): the JOB's url (the page where the media
//     was embedded) plus a `#t=Ns` fragment. Click handler tries
//     executeScript on any open tab with that page, falling back to opening
//     the URL fresh. HTML5 <video>/<audio> on the destination page honours
//     `#t=` natively where supported.
//
// Pages (HTML) and PDFs never get timecode markers from the LLM, so the
// short-circuit when no target is provided keeps them link-free.
//
// marked + DOMPurify are vendored as classic <script> tags by the consuming
// HTML page (sidepanel/library) and expose globals `marked` and `DOMPurify`.
//
// --- Trust boundary --------------------------------------------------
// `md` here is never something we typed: it is `summary_md` / chat text
// produced by an LLM whose context includes arbitrary page/video content
// chosen by whoever controls that page. Treat every byte as attacker-
// influenced. This function is the only thing standing between that text
// and `innerHTML` at the call sites (sidepanel app.js/chat.js). Two
// independent layers close the two distinct ways `marked` can hand us
// live markup, and BOTH are required — dropping either one reopens a hole:
//
//   1. `markedInstance` overrides the `html` renderer so *raw HTML typed
//      literally in the source text* (e.g. an article that contains a
//      literal `<select>`) is escaped to visible text instead of parsed
//      into a real element. Without this, marked's default behaviour is
//      to pass raw HTML straight through untouched — DOMPurify would still
//      strip the dangerous stuff, but incidentally: correctness (an
//      unclosed `<select>` swallows following content per HTML parsing
//      rules) breaks first, and it's needless surface area regardless.
//   2. The explicit `ALLOWED_TAGS`/`ALLOWED_ATTR`/`ALLOWED_URI_REGEXP`
//      passed to `DOMPurify.sanitize()` below is a fail-closed allowlist
//      of exactly what *legitimate markdown syntax* (not raw HTML) can
//      turn into, e.g. `![x](https://evil/beacon)` is real markdown that
//      marked legitimately renders as `<img src="https://evil/beacon">` —
//      layer 1 never sees this as "raw HTML" because marked itself
//      generates the tag, so only a tight DOMPurify allowlist stops the
//      beacon/overlay/phishing-form/media-autoplay vectors. `img`, all
//      form controls, all media tags, and `style` (tag and attribute) are
//      deliberately excluded here even though DOMPurify's default config
//      would allow some of them.
//
// If you're touching this function: layer 1 without layer 2 still leaks
// image/media beacons and full-panel overlays through legitimate markdown
// syntax; layer 2 without layer 1 still lets raw HTML corrupt rendering
// (dropped/misparsed tags, unclosed elements swallowing content) even
// though DOMPurify keeps it from being unsafe. Keep both.

import { resolveVideoId } from "./url.js";
import { escapeHtml } from "./utils.js";

/* global marked, DOMPurify */

/**
 * Dedicated marked instance (not the shared global `marked`) so the `html`
 * renderer override below can't leak into some other, future consumer of
 * the vendored bundle. `marked.Marked` is the class the UMD bundle exposes
 * alongside the convenience singleton; in v15 a renderer is installed via
 * the constructor (or `.use()`) — passing one to `.parse()` is no longer
 * supported.
 *
 * Both the block-level and inline `html` token types route through this
 * same `html(token)` method, so overriding it once covers e.g. a literal
 * `<select>` sitting in the middle of a paragraph as well as one on its
 * own line — both come out as escaped, visible text instead of a real
 * element (see the trust-boundary comment above for why).
 */
const markedInstance = new marked.Marked({
  renderer: {
    html(token) {
      return escapeHtml(token.text);
    },
  },
});

// Fail-closed allowlist for DOMPurify — see the trust-boundary comment
// above. Only tags/attributes that markdown syntax (as marked v15 renders
// it) legitimately produces. Notably absent: `img` (beacon vector via
// `![]()`), all form controls and media tags, and `style` (both the tag
// and the attribute — a `style` attribute is how a full-panel overlay
// would be built). `align` is included because marked emits it (not
// `style`) for table column alignment; it only ever holds
// "left"/"center"/"right"/"justify".
const ALLOWED_TAGS = [
  "p",
  "br",
  "hr",
  "strong",
  "em",
  "del",
  "s",
  "code",
  "pre",
  "blockquote",
  "ul",
  "ol",
  "li",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "a",
  "table",
  "thead",
  "tbody",
  "tr",
  "th",
  "td",
  "span",
];
const ALLOWED_ATTR = [
  "href",
  "title",
  "class",
  "colspan",
  "rowspan",
  "start",
  "align",
];
// http/https/mailto only — closes `javascript:`/`data:` URI vectors on
// links regardless of what DOMPurify's own default regexp would allow.
const ALLOWED_URI_REGEXP = /^(?:https?|mailto):/i;

// DOMPurify runs EVERY allowed attribute value through ALLOWED_URI_REGEXP
// unless the attribute name is in its (small, hardcoded) URI-safe list —
// not just conventional URI attributes like `href`. Its *default* regexp
// has a catch-all branch that happens to pass through non-URI-looking
// values (no scheme prefix), which is why this never shows up with the
// default config; our intentionally strict regexp above has no such
// escape hatch, so non-URI attribute values like `start="5"` or
// `align="left"` would otherwise get silently stripped. Declaring them
// here (not URI-bearing, so nothing to gain by exempting them) is the
// documented way to opt them out of that check.
const ADD_URI_SAFE_ATTR = ["colspan", "rowspan", "start", "align"];

/**
 * @typedef {{ video_id?: string | null, url?: string | null, kind?: string | null }} TimecodeJob
 */

/**
 * @param {string} md
 * @param {TimecodeJob | null | undefined} [job]
 * @returns {string} sanitized HTML
 */
export function renderMarkdown(md, job) {
  const html = DOMPurify.sanitize(markedInstance.parse(md ?? ""), {
    ALLOWED_TAGS,
    ALLOWED_ATTR,
    ALLOWED_URI_REGEXP,
    ADD_URI_SAFE_ATTR,
  });
  if (!job) return html;
  const videoId = resolveVideoId(job);
  const mediaPageUrl =
    !videoId && job.kind === "media" && job.url ? job.url : null;
  if (!videoId && !mediaPageUrl) return html;
  return injectTimecodeLinks(html, { videoId, mediaPageUrl });
}

// Match a bracket holding ONE OR MORE [MM:SS] / [HH:MM:SS] markers — the model
// sometimes groups several in one bracket, e.g. "[01:30, 04:30]". Non-global
// form for .test() so we don't have to worry about stateful lastIndex.
const TIMECODE_DETECT_RE =
  /\[\s*(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\s*[,;]\s*(?:\d{1,2}:)?\d{1,2}:\d{2})*\s*\]/;

// Tags whose text contents should NOT be transformed.
const SKIP_TAGS = new Set(["A", "CODE", "PRE", "SCRIPT", "STYLE", "TEXTAREA"]);

/**
 * @typedef {{ videoId: string | null, mediaPageUrl: string | null }} TimecodeTarget
 */

/**
 * Replace [MM:SS]/[HH:MM:SS] markers in text nodes with clickable links.
 * DOM-based to avoid breaking existing markup or links.
 *
 * @param {string} html
 * @param {TimecodeTarget} target
 * @returns {string}
 */
function injectTimecodeLinks(html, target) {
  const wrapper = new DOMParser()
    .parseFromString(`<div>${html}</div>`, "text/html")
    .body.firstElementChild;
  if (!wrapper) return html;

  const walker = wrapper.ownerDocument.createTreeWalker(
    wrapper,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode(node) {
        // Skip text inside tags we want to leave alone.
        for (let p = node.parentElement; p; p = p.parentElement) {
          if (SKIP_TAGS.has(p.tagName)) return NodeFilter.FILTER_REJECT;
        }
        return TIMECODE_DETECT_RE.test(node.nodeValue || "")
          ? NodeFilter.FILTER_ACCEPT
          : NodeFilter.FILTER_REJECT;
      },
    },
  );

  /** @type {Text[]} */
  const matches = [];
  let n;
  while ((n = walker.nextNode())) matches.push(/** @type {Text} */ (n));

  for (const textNode of matches) {
    replaceInTextNode(textNode, target);
  }

  return wrapper.innerHTML;
}

/**
 * Build a single clickable <a> for one timecode (h/m/s already parsed).
 *
 * @param {Document} doc
 * @param {string} label   text to show, e.g. "[01:30]"
 * @param {number} seconds
 * @param {TimecodeTarget} target
 * @returns {HTMLAnchorElement}
 */
function _makeTimecodeAnchor(doc, label, seconds, target) {
  const a = doc.createElement("a");
  if (target.videoId) {
    a.href = `https://www.youtube.com/watch?v=${encodeURIComponent(target.videoId)}&t=${seconds}s`;
    a.dataset.tldrVideoId = target.videoId;
  } else if (target.mediaPageUrl) {
    a.href = _withTimeFragment(target.mediaPageUrl, seconds);
    a.dataset.tldrMediaPageUrl = target.mediaPageUrl;
  }
  a.target = "_blank";
  a.rel = "noopener";
  a.dataset.tldrSeconds = String(seconds);
  a.textContent = label;
  return a;
}

/**
 * Split a text node, inserting <a> elements for each [MM:SS] / [HH:MM:SS]
 * marker found within. A bracket may hold several markers ("[01:30, 04:30]");
 * each becomes its own link, rendered as separate "[01:30] [04:30]" anchors.
 *
 * @param {Text} textNode
 * @param {TimecodeTarget} target
 */
function replaceInTextNode(textNode, target) {
  const text = textNode.nodeValue || "";
  const doc = textNode.ownerDocument;
  const frag = doc.createDocumentFragment();

  let lastIndex = 0;
  // Outer: a whole bracket holding one or more timecodes. New regex per call so
  // we don't share lastIndex state across nodes.
  const groupRe =
    /\[\s*(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\s*[,;]\s*(?:\d{1,2}:)?\d{1,2}:\d{2})*\s*\]/g;
  // Inner: each individual timecode within the bracket.
  const tcRe = /(?:(\d{1,2}):)?(\d{1,2}):(\d{2})/g;
  let m;
  while ((m = groupRe.exec(text)) !== null) {
    const before = text.slice(lastIndex, m.index);
    if (before) frag.appendChild(doc.createTextNode(before));

    let t;
    let first = true;
    tcRe.lastIndex = 0;
    while ((t = tcRe.exec(m[0])) !== null) {
      if (!first) frag.appendChild(doc.createTextNode(" "));
      first = false;
      const h = t[1] ? Number(t[1]) : 0;
      const seconds = h * 3600 + Number(t[2]) * 60 + Number(t[3]);
      frag.appendChild(_makeTimecodeAnchor(doc, `[${t[0]}]`, seconds, target));
    }

    lastIndex = m.index + m[0].length;
  }

  const tail = text.slice(lastIndex);
  if (tail) frag.appendChild(doc.createTextNode(tail));

  textNode.replaceWith(frag);
}

/**
 * Strip any existing fragment from ``url`` and append ``#t=Ns``. The browser
 * seek behaviour applies when the URL resolves directly to a media file
 * (HTML5 media fragment URI); on a regular page it's just a harmless hash.
 *
 * @param {string} url
 * @param {number} seconds
 * @returns {string}
 */
function _withTimeFragment(url, seconds) {
  const hashIdx = url.indexOf("#");
  const base = hashIdx === -1 ? url : url.slice(0, hashIdx);
  return `${base}#t=${seconds}`;
}
