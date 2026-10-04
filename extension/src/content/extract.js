// Page content extraction — Readability, innerText fallback.
//
// Injected into non-YouTube pages by background.js via
// chrome.scripting.executeScript({ files: ["vendor/readability.js",
// "src/content/extract.js"] }) — order matters: Readability has to load
// before this script since we use the global on the fallback path.
//
// Always produces a page job (`extracted-page`): the toolbar button means
// "this page". Videos on the page (native <video>/<audio>, whitelisted embed
// iframes, sniffed MSE players) are offered separately as video chips in the
// side panel — see lib/page-videos.js. YouTube watch pages use youtube.js.

/* global Readability */

(() => {
  function whenReady(fn) {
    if (document.readyState === "interactive" || document.readyState === "complete") {
      fn();
      return;
    }
    document.addEventListener("DOMContentLoaded", fn, { once: true });
  }

  /**
   * Readability article extraction, falling back to raw innerText on
   * failure.
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
    const { title, text } = extractPageText();
    chrome.runtime.sendMessage({
      type: "extracted-page",
      url: location.href,
      title,
      text,
    });
  });
})();
