// YouTube detection — extracts videoId and current page title.
// Injected into youtube.com pages.
//
// We wait for DOM ready so document.title and the watch-metadata <h1>
// reflect the current video, not the previous one (YouTube is an SPA and
// updates these asynchronously after navigation). The daemon double-checks
// the title via yt-dlp metadata anyway, but a correct hint here keeps the
// initial sidepanel render from flashing the wrong name.

(() => {
  function whenReady(fn) {
    if (document.readyState === "interactive" || document.readyState === "complete") {
      fn();
      return;
    }
    document.addEventListener("DOMContentLoaded", fn, { once: true });
  }

  whenReady(() => {
    const url = new URL(location.href);
    let videoId = url.searchParams.get("v");
    if (!videoId) {
      const m = url.pathname.match(/^\/(?:embed|shorts)\/([\w-]+)/);
      videoId = m ? m[1] : null;
    }
    if (!videoId && url.hostname.endsWith("youtu.be")) {
      videoId = url.pathname.replace(/^\//, "") || null;
    }

    // YouTube usually appends " - YouTube" suffix.
    let title = (document.title || "").replace(/\s*-\s*YouTube\s*$/, "");

    // For richer titles, prefer the canonical h1 if present.
    const h1 = document.querySelector("h1.title, h1.ytd-watch-metadata, ytd-watch-metadata h1");
    if (h1 && h1.textContent && h1.textContent.trim()) {
      title = h1.textContent.trim();
    }

    chrome.runtime.sendMessage({
      type: "extracted-youtube",
      url: location.href,
      videoId,
      title,
    });
  });
})();
