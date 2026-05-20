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
   * @typedef {{mediaUrl: string}} MediaFinding
   * @returns {MediaFinding | null}
   */
  function findMedia() {
    return findNativeMedia() || findIframeEmbed();
  }

  /** @returns {MediaFinding | null} */
  function findNativeMedia() {
    /** @type {HTMLMediaElement[]} */
    const all = [...document.querySelectorAll("video, audio")];
    // Score: prefer larger visible video, then visible audio, then anything
    // playable. This filters tiny notification chimes / hidden previews.
    /** @type {{el: HTMLMediaElement, src: string, score: number}[]} */
    const candidates = [];
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
      const score = (isVideo ? 1_000_000 : 100_000) + area;
      candidates.push({ el, src, score });
    }
    if (!candidates.length) return null;
    candidates.sort((a, b) => b.score - a.score);
    return { mediaUrl: candidates[0].src };
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

  /** @returns {MediaFinding | null} */
  function findIframeEmbed() {
    const iframes = [...document.querySelectorAll("iframe[src]")];
    for (const f of iframes) {
      const src = /** @type {HTMLIFrameElement} */ (f).src;
      if (!src) continue;
      if (IFRAME_WHITELIST.some((re) => re.test(src))) {
        try {
          return { mediaUrl: new URL(src).toString() };
        } catch {
          // malformed src — skip
        }
      }
    }
    return null;
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

  // ----- entry --------------------------------------------------------------

  whenReady(() => {
    const found = findMedia();
    if (found) {
      chrome.runtime.sendMessage({
        type: "extracted-media",
        url: location.href,
        mediaUrl: found.mediaUrl,
        title: extractTitle(),
      });
      return;
    }

    // No media on the page — fall back to Readability article extraction.
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
    chrome.runtime.sendMessage({
      type: "extracted-page",
      url: location.href,
      title,
      text,
    });
  });
})();
