// Page content extraction — media-first, Readability fallback.
//
// Injected into non-YouTube pages by background.js via
// chrome.scripting.executeScript({ files: ["vendor/readability.js",
// "src/content/extract.js"] }) — order matters: Readability has to load
// before this script since we use the global on the fallback path.
//
// Flow:
//   1. Wait for DOM ready (SPAs / lazy-loaded players need a moment).
//   2. Scan for a transcribable media element:
//        a. <video> / <audio> with an extractable src (not blob:/data:,
//           not invisibly small).
//        b. iframe embed from a known media-host whitelist (yt-dlp's
//           site-specific extractors handle these — Vimeo, Dailymotion,
//           Twitch VOD, Bunny, Brightcove, JW Player, Wistia, Streamable,
//           SoundCloud, Spotify).
//   3. If media found → send `extracted-media { mediaUrl, title, url }`.
//      The background reads cookies for mediaUrl and POSTs a kind=media job.
//   4. Else → fall through to Readability and send `extracted-page` like
//      before.
//
// Why one script handles both cases: keeps the executeScript injection +
// the IPC ping-pong down to ONE round trip. A two-script orchestration
// from background would add latency that the user sees as a delay between
// click and "Queued for transcription".

/* global Readability */

(() => {
  function whenReady(fn) {
    if (document.readyState === "interactive" || document.readyState === "complete") {
      fn();
      return;
    }
    document.addEventListener("DOMContentLoaded", fn, { once: true });
  }

  // ----- media detection ----------------------------------------------------

  // Iframe-embed whitelist. Each entry is a regex against the iframe src;
  // when it matches, yt-dlp has a site-specific extractor that resolves the
  // underlying audio/video. We keep the list conservative — only domains
  // we know yt-dlp handles — so a random iframe (ads, comment widgets,
  // analytics) doesn't get misidentified as media.
  const IFRAME_WHITELIST = [
    /^https:\/\/player\.vimeo\.com\/video\/\d+/i,
    /^https:\/\/(?:www\.)?dailymotion\.com\/embed\/video\//i,
    /^https:\/\/(?:geo\.)?dailymotion\.com\/(?:embed|player)\//i,
    // Twitch VOD only — `?video=` form. Live streams (`?channel=`) are
    // unbounded duration and would never finish transcribing.
    /^https:\/\/player\.twitch\.tv\/\?video=/i,
    /^https:\/\/iframe\.mediadelivery\.net\/(?:embed|play)\//i,
    /^https:\/\/players\.brightcove\.net\/\d+\//i,
    /^https:\/\/(?:cdn|content)\.jwplayer\.com\/(?:players|previews|videos)\//i,
    /^https:\/\/fast\.wistia\.(?:com|net)\/embed\//i,
    /^https:\/\/(?:www\.)?streamable\.com\/[eo]\//i,
    /^https:\/\/w\.soundcloud\.com\/player\//i,
    /^https:\/\/open\.spotify\.com\/embed\//i,
    /^https:\/\/(?:www\.)?facebook\.com\/plugins\/video\.php/i,
  ];

  /**
   * @typedef {{mediaUrl: string, kind: "video"|"audio"|"iframe", label: string}} MediaCandidate
   * @typedef {{primary: MediaCandidate, alternates: MediaCandidate[]}} MediaFinding
   * @returns {MediaFinding | null}
   */
  function findMedia() {
    // Native <video>/<audio> wins over iframe embeds when both exist on the
    // same page (the native one is what the user is actually watching;
    // iframes are usually supplementary players). But we still surface the
    // iframe ones as alternates so the user can switch if we guessed wrong.
    const native = collectNativeMedia();
    const iframes = collectIframeEmbeds();
    const all = [...native, ...iframes];
    if (!all.length) return null;
    // Filter out chrome ("autoplay ad in sidebar", "related videos panel")
    // by keeping only candidates that live inside the page's main article
    // container — when the page has one AND at least one candidate is in
    // it. trafilatura does similar boilerplate stripping on the daemon
    // side for text; here we use cheap semantic selectors because we're
    // running in-page and want the scan fast.
    const filtered = _filterToMainContent(all);
    filtered.sort((a, b) => b.score - a.score);
    // Label disambiguation pass — "Video" → "Video 1", "Video 2" when there
    // are multiple. Iframes already carry the host name so they're unique.
    _disambiguateLabels(filtered);
    const [primary, ...alternates] = filtered.map(({ mediaUrl, kind, label }) => ({
      mediaUrl,
      kind,
      label,
    }));
    return { primary, alternates };
  }

  /**
   * Drop candidates outside the page's main article container — that's
   * where sidebars, recommendation rails, and autoplay ads live. The
   * filter is conservative: it only kicks in when (a) there's a clear
   * main container AND (b) at least one candidate is inside it. Pages
   * without semantic structure (`<article>` / `<main>`), and pages where
   * every candidate happens to live outside main, keep all candidates —
   * a heuristic guess is worse than showing everything.
   *
   * @template {{el: Element}} T
   * @param {T[]} candidates
   * @returns {T[]}
   */
  function _filterToMainContent(candidates) {
    // First semantic selector wins. Order: <article> (most specific) →
    // <main> → [role=main] (ARIA fallback for layout-only sites).
    const mainEl = document.querySelector("article, main, [role='main']");
    if (!mainEl) return candidates;
    const inside = candidates.filter((c) => mainEl.contains(c.el));
    return inside.length > 0 ? inside : candidates;
  }

  // Reject media whose *known* duration is below this many seconds — both
  // <video> and <audio>. This is deliberately NOT a visibility/controls
  // check: a hidden <audio> with no `controls`, driven by its own JS, is
  // the NORMAL way real audio players are built (SoundCloud, Bandcamp, any
  // custom podcast player), so filtering on visibility would break the
  // audio path entirely. Duration is the only reliable signal — UI
  // notification sounds ("ding") run 0.2-3s; the shortest meaningful
  // spoken clip is well above that. 12s sits comfortably above the former
  // and below the latter.
  const MIN_MEDIA_DURATION_SECONDS = 12;

  /**
   * @returns {{el: HTMLMediaElement, mediaUrl: string, kind: "video"|"audio",
   *           label: string, score: number}[]}
   */
  function collectNativeMedia() {
    /** @type {HTMLMediaElement[]} */
    const all = [...document.querySelectorAll("video, audio")];
    const out = [];
    let videoIdx = 0;
    let audioIdx = 0;
    for (const el of all) {
      const src = _readSrc(el);
      if (!src) continue;
      const rect = el.getBoundingClientRect();
      const isVideo = el.tagName === "VIDEO";
      // For video: require some real on-screen area (filters 0×0 hidden
      // players) OR known videoWidth from loaded metadata (covers
      // below-the-fold elements that haven't been laid out yet).
      const vw = /** @type {HTMLVideoElement} */ (el).videoWidth || 0;
      const area = rect.width * rect.height;
      if (isVideo && area < 5000 && vw < 100) continue;
      // Duration reject (video AND audio). Only fires on a known finite
      // number below the threshold — NaN/Infinity/unset (preload="none",
      // never played — completely normal for a script-driven hidden
      // player) must NOT be rejected here.
      const dur = el.duration;
      if (Number.isFinite(dur) && dur < MIN_MEDIA_DURATION_SECONDS) continue;
      const idx = isVideo ? ++videoIdx : ++audioIdx;
      out.push({
        el,
        mediaUrl: src,
        kind: isVideo ? "video" : "audio",
        label: _labelForElement(el, isVideo ? "video" : "audio", idx),
        score: (isVideo ? 1_000_000 : 100_000) + area,
      });
    }
    return out;
  }

  /**
   * Extract a real, fetchable URL from a media element. Returns null for
   * blob:/data: (browser-internal, not addressable from the daemon) and
   * for empty/relative-with-no-base values.
   *
   * @param {HTMLMediaElement} el
   * @returns {string | null}
   */
  function _readSrc(el) {
    /** @type {string} */
    let raw = el.currentSrc || el.src || "";
    if (!raw) {
      const source = el.querySelector("source[src]");
      if (source) raw = /** @type {HTMLSourceElement} */ (source).src;
    }
    if (!raw) return null;
    if (raw.startsWith("blob:") || raw.startsWith("data:")) return null;
    try {
      const u = new URL(raw, location.href);
      if (u.protocol !== "https:" && u.protocol !== "http:") return null;
      return u.toString();
    } catch {
      return null;
    }
  }

  /**
   * @returns {{el: HTMLIFrameElement, mediaUrl: string, kind: "iframe",
   *           label: string, score: number}[]}
   */
  function collectIframeEmbeds() {
    const out = [];
    const iframes = [...document.querySelectorAll("iframe[src]")];
    for (const f of iframes) {
      const src = /** @type {HTMLIFrameElement} */ (f).src;
      if (!src) continue;
      if (!IFRAME_WHITELIST.some((re) => re.test(src))) continue;
      try {
        const normalised = new URL(src).toString();
        const rect = f.getBoundingClientRect();
        const area = rect.width * rect.height;
        // Iframes score below native video (a real <video> on the page is
        // almost always the one the user is interacting with), but above
        // <audio> — embeds usually carry the primary content.
        out.push({
          el: /** @type {HTMLIFrameElement} */ (f),
          mediaUrl: normalised,
          kind: /** @type {"iframe"} */ ("iframe"),
          label: _labelForIframe(/** @type {HTMLIFrameElement} */ (f), normalised),
          score: 500_000 + area,
        });
      } catch {
        // malformed src — skip
      }
    }
    return out;
  }

  /**
   * Best-effort human-readable label for a native media element. Prefers
   * curated text (title attr, aria-label) over the filename, falling back
   * to a generic "Video N" / "Audio N" — _disambiguateLabels turns the
   * generic ones into numbered variants when there are duplicates.
   *
   * @param {HTMLMediaElement} el
   * @param {"video" | "audio"} kind
   * @param {number} idx  1-based position among elements of the same kind
   * @returns {string}
   */
  function _labelForElement(el, kind, idx) {
    const t = el.getAttribute("title")?.trim()
      || el.getAttribute("aria-label")?.trim();
    if (t) return t;
    // Look at parent figure caption / preceding heading — often holds the
    // talk / episode title on lecture / podcast pages.
    const fig = el.closest("figure")?.querySelector("figcaption")?.textContent?.trim();
    if (fig && fig.length <= 80) return fig;
    // Filename from URL — strip query and extension for readability.
    const fname = _filenameFromUrl(_readSrc(el) ?? "");
    if (fname) return fname;
    return kind === "video" ? `Video ${idx}` : `Audio ${idx}`;
  }

  /**
   * Label for a whitelisted iframe. We name by the host's recognisable
   * product (vimeo / dailymotion / twitch / …) plus the embed id when
   * present, so multiple embeds of the same provider stay distinguishable.
   *
   * @param {HTMLIFrameElement} f
   * @param {string} src
   * @returns {string}
   */
  function _labelForIframe(f, src) {
    const t = f.getAttribute("title")?.trim()
      || f.getAttribute("aria-label")?.trim();
    if (t) return t;
    try {
      const u = new URL(src);
      const host = u.hostname.replace(/^www\./, "");
      // Extract the embed id from the path tail when it looks like one
      // (numeric, hash-like). Falls back to just the host.
      const tail = u.pathname.replace(/\/$/, "").split("/").pop() || "";
      const id = /^[A-Za-z0-9_-]{4,}$/.test(tail) ? tail : "";
      return id ? `${host}: ${id}` : host;
    } catch {
      return "Embed";
    }
  }

  /**
   * Extract a filename from a URL, stripping query/hash and extension.
   * Returns null when there's nothing useful (e.g. URL ends with `/`).
   *
   * @param {string} url
   * @returns {string | null}
   */
  function _filenameFromUrl(url) {
    try {
      const u = new URL(url);
      const last = u.pathname.split("/").filter(Boolean).pop();
      if (!last) return null;
      const noExt = last.replace(/\.[^.]+$/, "");
      // Skip purely numeric / hash-like filenames — they're not useful as
      // labels (the user can't tell them apart at a glance).
      if (/^[0-9a-f]{8,}$/i.test(noExt)) return null;
      return decodeURIComponent(noExt).slice(0, 60);
    } catch {
      return null;
    }
  }

  /**
   * Append " 2", " 3", … to labels that repeat. Mutates the array in
   * place. Keeps the first occurrence un-numbered.
   *
   * @param {{label: string}[]} items
   */
  function _disambiguateLabels(items) {
    const seen = new Map();
    for (const item of items) {
      const base = item.label;
      const n = (seen.get(base) ?? 0) + 1;
      seen.set(base, n);
      if (n > 1) item.label = `${base} ${n}`;
    }
  }

  // Pick the best human title for the page: prefer og:title (curated by
  // the site for shares), then the first <h1>, then document.title.
  function extractTitle() {
    const og = /** @type {HTMLMetaElement | null} */ (
      document.querySelector('meta[property="og:title"], meta[name="og:title"]')
    );
    if (og?.content?.trim()) return og.content.trim();
    const h1 = document.querySelector("h1");
    const h1text = h1?.textContent?.trim();
    if (h1text) return h1text;
    return document.title || null;
  }

  /**
   * Readability article extraction, falling back to raw innerText on
   * failure. Shared by both messages below: the no-media path uses it as
   * the page's actual content; the media path uses it as a best-effort
   * fallback the daemon can summarize instead of audio (e.g. a hidden
   * notification-sound element that got past the duration filter, or a
   * media job whose transcript comes back empty).
   *
   * @returns {{title: string | null, text: string}}
   */
  function extractPageText() {
    let title = null;
    let text = "";
    try {
      const doc = document.cloneNode(true);
      const article = new Readability(doc).parse();
      title = (article && article.title) || document.title || null;
      text = (article && article.textContent) || "";
    } catch (err) {
      title = document.title || null;
      text = document.body ? document.body.innerText : "";
      console.warn("[TLDR] Readability failed, using fallback:", err);
    }
    return { title, text };
  }

  // ----- entry --------------------------------------------------------------

  whenReady(() => {
    const found = findMedia();
    if (found) {
      const { text } = extractPageText();
      chrome.runtime.sendMessage({
        type: "extracted-media",
        url: location.href,
        // Primary (top-scored) drives job creation; alternates surface in
        // the sidepanel as a "wrong source?" chip so the user can switch
        // without re-running the whole pipeline.
        mediaUrl: found.primary.mediaUrl,
        altCandidates: found.alternates,
        title: extractTitle(),
        // Best-effort page text so the daemon has something to fall back
        // to when the media turns out not to be summarizable (too short /
        // empty transcript) — see background.js's handleExtractedMedia.
        text,
      });
      return;
    }

    // No media on the page — fall back to Readability article extraction.
    const { title, text } = extractPageText();
    chrome.runtime.sendMessage({
      type: "extracted-page",
      url: location.href,
      title,
      text,
    });
  });
})();
