// Page content extraction via Mozilla Readability.
// Injected into regular web pages by background.js via chrome.scripting.executeScript.
//
// Readability is vendored at extension/vendor/readability.js. Since this is a
// content script (classic, no ES modules), Readability must be loaded as a
// separate file via chrome.scripting.executeScript({ files: ["vendor/readability.js", "src/content/extract.js"] })
// — order matters: vendor first, then this script which uses the global.

/* global Readability */

(() => {
  // Wait for the DOM to be parsed before extracting. Without this, clicking
  // the toolbar (or Summarize button) on a tab that just started loading
  // gives Readability an empty / placeholder document, and the previous
  // page's title can leak in via document.title.
  function whenReady(fn) {
    if (document.readyState === "interactive" || document.readyState === "complete") {
      fn();
      return;
    }
    document.addEventListener("DOMContentLoaded", fn, { once: true });
  }

  whenReady(() => {
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
