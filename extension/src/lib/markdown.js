// Render markdown to sanitized HTML, then post-process [MM:SS] / [HH:MM:SS]
// markers into clickable YouTube links when a videoId is provided.
//
// marked + DOMPurify are vendored as classic <script> tags by the consuming
// HTML page (sidepanel/library) and expose globals `marked` and `DOMPurify`.

/* global marked, DOMPurify */

/**
 * @param {string} md
 * @param {string | null | undefined} [videoId]
 * @returns {string} sanitized HTML
 */
export function renderMarkdown(md, videoId) {
  const html = DOMPurify.sanitize(marked.parse(md ?? ""));
  if (!videoId) return html;
  return injectTimecodeLinks(html, videoId);
}

// Match [MM:SS] or [HH:MM:SS] markers in a text node. Non-global form for
// .test() so we don't have to worry about stateful lastIndex.
const TIMECODE_DETECT_RE = /\[(?:\d{1,2}:)?\d{1,2}:\d{2}\]/;

// Tags whose text contents should NOT be transformed.
const SKIP_TAGS = new Set(["A", "CODE", "PRE", "SCRIPT", "STYLE", "TEXTAREA"]);

/**
 * Replace [MM:SS]/[HH:MM:SS] markers in text nodes with clickable YouTube
 * links. DOM-based to avoid breaking existing markup or links.
 *
 * @param {string} html
 * @param {string} videoId
 * @returns {string}
 */
function injectTimecodeLinks(html, videoId) {
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
    replaceInTextNode(textNode, videoId);
  }

  return wrapper.innerHTML;
}

/**
 * Split a text node, inserting <a> elements for each [MM:SS] or [HH:MM:SS]
 * marker found within.
 *
 * @param {Text} textNode
 * @param {string} videoId
 */
function replaceInTextNode(textNode, videoId) {
  const text = textNode.nodeValue || "";
  const doc = textNode.ownerDocument;
  const frag = doc.createDocumentFragment();

  let lastIndex = 0;
  // New regex per call so we don't share lastIndex state across nodes.
  const re = /\[(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\]/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    const before = text.slice(lastIndex, m.index);
    if (before) frag.appendChild(doc.createTextNode(before));

    const h = m[1] ? Number(m[1]) : 0;
    const mm = Number(m[2]);
    const ss = Number(m[3]);
    const seconds = h * 3600 + mm * 60 + ss;

    const a = doc.createElement("a");
    a.href = `https://www.youtube.com/watch?v=${encodeURIComponent(videoId)}&t=${seconds}s`;
    a.target = "_blank";
    a.rel = "noopener";
    a.dataset.tldrSeconds = String(seconds);
    a.dataset.tldrVideoId = videoId;
    a.textContent = m[0];
    frag.appendChild(a);

    lastIndex = m.index + m[0].length;
  }

  const tail = text.slice(lastIndex);
  if (tail) frag.appendChild(doc.createTextNode(tail));

  textNode.replaceWith(frag);
}
