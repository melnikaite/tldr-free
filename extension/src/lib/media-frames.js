// Main-media selection across frames + self-rendered caption overlay.
//
// Video players often live in cross-origin iframes and run VAST ads in a
// second <video>, so "the first <video> in the top frame" is frequently
// wrong. Every caller (timecode seek, transcript time polling, captions)
// goes through probeTab(): `probeMedia` runs in ALL frames, each frame
// reports its locally-best media element (and marks it with
// `data-tldr-main`), and pickBestFrame() ranks the reports. The follow-up
// action is then executed only in the winning frameId and addresses the
// marked element.
//
// Functions passed to chrome.scripting.executeScript are serialized via
// toString(), so `probeMedia`, `seekMain` and `installCaptions` must stay
// self-contained (no imports, no module-scope references). The in-page
// ranking in `probeMedia` duplicates compareMedia() — keep them in sync;
// compareMedia() is the tested reference.
//
// All injected funcs run in the default ISOLATED world, which persists per
// frame across executeScript calls — the overlay state lives on its
// `window` and the probe can report whether it is still bound.

/**
 * @typedef {object} MediaInfo
 * @property {number} duration     finite seconds, 0 = unknown (NaN/Infinity/0)
 * @property {number} area         visible box area in CSS px²
 * @property {boolean} paused
 * @property {number} currentTime
 * @property {boolean} isVideo
 * @property {boolean} captionsBound overlay exists and is bound to this element
 * @property {boolean} matches      duration falls inside the expected range
 * @property {string} frameUrl      location.href of the reporting frame
 */

/**
 * Acceptable duration window for the job's media, ``max`` 0 = unbounded
 * (plain object: it crosses executeScript's JSON boundary).
 * @typedef {{ min: number, max: number }} DurationRange
 */

/**
 * Build the expected-duration window from what the side panel knows. The
 * job's own duration is near-exact, so ±3 s — a different rip, episode or
 * dub of the same film must not qualify. A transcript's last marker is
 * only a loose lower bound (transcripts may stop before the media does).
 * null = unknown.
 *
 * @param {number | null | undefined} jobDuration
 * @param {number | null | undefined} lastMarker
 * @returns {DurationRange | null}
 */
export function expectedRange(jobDuration, lastMarker) {
  if (jobDuration && jobDuration > 0) {
    return { min: Math.max(0.001, jobDuration - 3), max: jobDuration + 3 };
  }
  if (lastMarker && lastMarker > 0) return { min: lastMarker * 0.8, max: 0 };
  return null;
}

/**
 * Whether ``duration`` fits ``range``. Unknown (NaN/Infinity/0) never
 * matches — a film whose metadata hasn't loaded yet becomes eligible later.
 *
 * @param {number} duration
 * @param {DurationRange | null | undefined} range
 */
export function durationMatches(duration, range) {
  if (!range || !(Number.isFinite(duration) && duration > 0)) return false;
  return duration >= range.min && (!range.max || duration <= range.max);
}

/**
 * Whether the poll may silently re-bind lost captions to ``info``. Only
 * for a known duration, a matching element, and — once captions were
 * bound — the same frame URL as that binding (another player iframe on the
 * page is likely another rip with drifting timings). Explicit injections
 * (user picks a language) don't go through this.
 *
 * @param {Pick<MediaInfo, "isVideo" | "captionsBound" | "matches" | "frameUrl">} info
 * @param {DurationRange | null} range
 * @param {string | null} boundFrameUrl frame of the last successful bind
 */
export function canAutoRebind(info, range, boundFrameUrl) {
  if (!range || !info.isVideo || info.captionsBound || !info.matches) {
    return false;
  }
  return boundFrameUrl == null || boundFrameUrl === info.frameUrl;
}

/**
 * Ordering for media candidates: longest known duration first (ads are
 * short; live/unknown durations rank last), then larger area, then
 * playing over paused. With a known ``range``, a duration-matching element
 * beats everything else. Negative = ``a`` is better.
 *
 * @param {Pick<MediaInfo, "duration" | "area" | "paused">} a
 * @param {Pick<MediaInfo, "duration" | "area" | "paused">} b
 * @param {DurationRange | null} [range]
 * @returns {number}
 */
export function compareMedia(a, b, range = null) {
  if (range) {
    const ma = durationMatches(a.duration, range);
    const mb = durationMatches(b.duration, range);
    if (ma !== mb) return ma ? -1 : 1;
  }
  const da = _dur(a.duration);
  const db = _dur(b.duration);
  if (da !== db) return db - da;
  if (a.area !== b.area) return b.area - a.area;
  if (a.paused !== b.paused) return a.paused ? 1 : -1;
  return 0;
}

/** @param {number} d */
function _dur(d) {
  return Number.isFinite(d) && d > 0 ? d : 0;
}

/**
 * Pick the best frame from executeScript results (``result`` null = no
 * media in that frame). Ties keep the earlier result (top frame first).
 *
 * @template {Pick<MediaInfo, "duration" | "area" | "paused">} T
 * @param {Array<{ frameId: number, result?: T | null }>} results
 * @param {DurationRange | null} [range]
 * @returns {{ frameId: number, info: T } | null}
 */
export function pickBestFrame(results, range = null) {
  /** @type {{ frameId: number, info: T } | null} */
  let best = null;
  for (const r of results || []) {
    if (!r || !r.result) continue;
    if (!best || compareMedia(r.result, best.info, range) < 0) {
      best = { frameId: r.frameId, info: r.result };
    }
  }
  return best;
}

/**
 * Parse ``[MM:SS] text`` / ``[HH:MM:SS] text`` lines into cues. Each cue
 * spans marker → next marker; the last one gets +60s so the final line
 * stays visible during late playback.
 *
 * @param {string} text
 * @returns {Array<{ start: number, end: number, text: string }>}
 */
export function cuesFromText(text) {
  const re = /\[(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\]\s*([^\n]*)/g;
  /** @type {Array<{ start: number, text: string }>} */
  const raw = [];
  let m;
  while ((m = re.exec(text || "")) !== null) {
    const h = m[1] ? Number(m[1]) : 0;
    raw.push({
      start: h * 3600 + Number(m[2]) * 60 + Number(m[3]),
      text: (m[4] || "").trim(),
    });
  }
  return raw.map((c, i) => ({
    start: c.start,
    end: i + 1 < raw.length ? raw[i + 1].start : c.start + 60,
    text: c.text,
  }));
}

/**
 * Probe every frame of ``tabId`` and return the frame holding the main
 * media element (marked in-page), or null when there's none.
 *
 * @param {number} tabId
 * @param {DurationRange | null} [range]
 * @returns {Promise<{ frameId: number, info: MediaInfo } | null>}
 */
export async function probeTab(tabId, range = null) {
  const results = await chrome.scripting.executeScript({
    target: { tabId, allFrames: true },
    func: probeMedia,
    args: [range],
  });
  return pickBestFrame(
    /** @type {Array<{ frameId: number, result?: MediaInfo | null }>} */ (
      results
    ),
    range,
  );
}

/**
 * Seek the main media of ``tabId`` to ``seconds`` (play state untouched).
 * @param {number} tabId
 * @param {number} seconds
 * @param {DurationRange | null} [range]
 * @returns {Promise<boolean>}
 */
export async function seekTab(tabId, seconds, range = null) {
  const best = await probeTab(tabId, range);
  if (!best) return false;
  const [r] = await chrome.scripting.executeScript({
    target: { tabId, frameIds: [best.frameId] },
    func: seekMain,
    args: [seconds],
  });
  return !!r?.result;
}

/**
 * Install (or with empty ``cues`` remove) the caption overlay on the main
 * <video> of ``tabId``. Idempotent: replaces any prior overlay in that
 * frame.
 *
 * @param {number} tabId
 * @param {Array<{ start: number, end: number, text: string }>} cues
 * @param {string} lang
 * @param {DurationRange | null} [range] when known, only a matching
 *   element gets captions (a pre-roll ad never does); the poll's rebind
 *   picks the film up once its metadata loads
 * @param {string | null} [frameUrl] when set, bind only in a frame at
 *   this URL (auto-rebind into the originally bound player)
 * @returns {Promise<string | null>} bound frame's URL, null = not bound
 */
export async function installCaptionsInTab(
  tabId,
  cues,
  lang,
  range = null,
  frameUrl = null,
) {
  const best = await probeTab(tabId, range);
  if (!best || !best.info.isVideo) return null;
  if (range && !best.info.matches) return null;
  if (frameUrl != null && best.info.frameUrl !== frameUrl) return null;
  const [r] = await chrome.scripting.executeScript({
    target: { tabId, frameIds: [best.frameId] },
    func: installCaptions,
    args: [cues, lang, range],
  });
  return r?.result ? best.info.frameUrl : null;
}

// ---------------------------------------------------------------------------
// Injected (self-contained) functions
// ---------------------------------------------------------------------------

/**
 * In-page: pick this frame's best media element, move the
 * `data-tldr-main` mark onto it, report its state.
 * @param {DurationRange | null} range
 * @returns {MediaInfo | null}
 */
export function probeMedia(range) {
  const ATTR = "data-tldr-main";
  const els = /** @type {HTMLMediaElement[]} */ ([
    ...document.querySelectorAll("video, audio"),
  ]);
  if (!els.length) return null;
  /** @param {HTMLMediaElement} m */
  const info = (m) => {
    const d = m.duration;
    const r = m.getBoundingClientRect();
    const dur = Number.isFinite(d) && d > 0 ? d : 0;
    return {
      el: m,
      duration: dur,
      matches:
        !!range && dur > 0 && dur >= range.min && (!range.max || dur <= range.max),
      area: Math.max(0, r.width) * Math.max(0, r.height),
      paused: m.paused,
      marked: m.hasAttribute(ATTR),
    };
  };
  // Same order as compareMedia(); the already-marked element wins ties so
  // follow-up calls keep addressing the same element.
  /** @param {ReturnType<typeof info>} a @param {ReturnType<typeof info>} b */
  const cmp = (a, b) => {
    if (a.matches !== b.matches) return a.matches ? -1 : 1;
    if (a.duration !== b.duration) return b.duration - a.duration;
    if (a.area !== b.area) return b.area - a.area;
    if (a.paused !== b.paused) return a.paused ? 1 : -1;
    if (a.marked !== b.marked) return a.marked ? -1 : 1;
    return 0;
  };
  let best = info(els[0]);
  for (let i = 1; i < els.length; i++) {
    const c = info(els[i]);
    if (cmp(c, best) < 0) best = c;
  }
  for (const m of els) if (m !== best.el) m.removeAttribute(ATTR);
  best.el.setAttribute(ATTR, "");
  const st = /** @type {any} */ (window).__tldrCaptions;
  return {
    duration: best.duration,
    area: best.area,
    paused: best.paused,
    currentTime: best.el.currentTime,
    isVideo: best.el.tagName === "VIDEO",
    matches: best.matches,
    frameUrl: location.href,
    captionsBound: !!st && st.video === best.el && best.el.isConnected,
  };
}

/**
 * In-page: seek the marked main media element.
 * @param {number} t
 */
export function seekMain(t) {
  const m = /** @type {HTMLMediaElement | null} */ (
    document.querySelector("[data-tldr-main]")
    || document.querySelector("video, audio")
  );
  if (!m) return false;
  m.currentTime = t;
  return true;
}

/**
 * In-page: render ``cues`` over the marked main <video> in a shadow-DOM
 * overlay. Falls back to a native text track while the <video> element
 * itself is fullscreen (nothing can be layered over it then).
 *
 * @param {Array<{ start: number, end: number, text: string }>} cues
 * @param {string} lang
 * @param {DurationRange | null} range expected duration; drives the
 *   "ad loaded into the same element" check when known
 */
export function installCaptions(cues, lang, range) {
  const w = /** @type {any} */ (window);
  if (w.__tldrCaptions) w.__tldrCaptions.destroy();
  w.__tldrCaptions = null;
  if (!cues || !cues.length) return false;
  const video = /** @type {HTMLVideoElement | null} */ (
    document.querySelector("video[data-tldr-main]")
  );
  if (!video) return false;

  const host = document.createElement("div");
  host.setAttribute("data-tldr-captions", "");
  host.style.cssText =
    "all:initial;position:absolute;left:0;top:0;width:0;height:0;"
    + "pointer-events:none;z-index:2147483647;display:none;";
  const root = host.attachShadow({ mode: "closed" });
  root.innerHTML =
    "<style>"
    + ".box{position:absolute;left:0;right:0;bottom:7%;display:flex;"
    + "justify-content:center;padding:0 4%;box-sizing:border-box}"
    + ".line{max-width:92%;padding:.15em .5em;border-radius:4px;"
    + "background:rgba(0,0,0,.68);color:#fff;text-align:center;"
    + "font:500 var(--tldr-fs,20px)/1.35 system-ui,-apple-system,sans-serif;"
    + "white-space:pre-line;text-shadow:0 1px 2px #000,0 0 4px #000}"
    + ".line:empty{display:none}"
    + "</style><div class=\"box\"><div class=\"line\"></div></div>";
  const lineEl = /** @type {HTMLElement} */ (root.querySelector(".line"));

  // Native track for the "<video> itself is fullscreen" case. addTextTrack
  // needs no URL (no CSP / blob issues); tracks can't be removed, so one
  // per element is reused across installs.
  w.__tldrTracks = w.__tldrTracks || new WeakMap();
  /** @type {TextTrack} */
  let track = w.__tldrTracks.get(video);
  if (!track) {
    track = video.addTextTrack("subtitles", "TLDR", lang);
    w.__tldrTracks.set(video, track);
  }
  for (const c of [...(track.cues || [])]) track.removeCue(c);
  for (const c of cues) {
    try {
      track.addCue(new VTTCue(c.start, Math.max(c.end, c.start + 0.5), c.text));
    } catch {}
  }
  track.mode = "disabled";

  /** Duration the overlay belongs to; a much shorter one = an ad took over.
   *  The job's expected duration when known, else the max seen. */
  let boundDuration = range && range.min > 0
    ? range.min
    : Number.isFinite(video.duration) ? video.duration : 0;
  let shown = "";
  /** Stream identity: a new currentSrc (episode / dub switch, player
   *  reload; MSE players get a fresh blob: URL) ends this binding. */
  let boundSrc = video.currentSrc;

  const container = () => {
    const fe = document.fullscreenElement;
    if (fe && fe !== video && fe.contains(video)) {
      return video.parentElement && fe.contains(video.parentElement)
        ? video.parentElement
        : fe;
    }
    return video.parentElement || document.body;
  };

  const layout = () => {
    const parent = container();
    if (host.parentNode !== parent) parent.appendChild(host);
    const vr = video.getBoundingClientRect();
    host.style.left = "0px";
    host.style.top = "0px";
    const hr = host.getBoundingClientRect();
    host.style.left = `${vr.left - hr.left}px`;
    host.style.top = `${vr.top - hr.top}px`;
    host.style.width = `${vr.width}px`;
    host.style.height = `${vr.height}px`;
    host.style.setProperty(
      "--tldr-fs",
      `${Math.max(14, Math.round(vr.height * 0.045))}px`,
    );
  };

  /** @param {number} t */
  const cueAt = (t) => {
    let lo = 0;
    let hi = cues.length - 1;
    let idx = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (cues[mid].start <= t) {
        idx = mid;
        lo = mid + 1;
      } else hi = mid - 1;
    }
    return idx >= 0 && t < cues[idx].end ? cues[idx].text : "";
  };

  const update = () => {
    if (!video.isConnected) {
      st.destroy();
      return;
    }
    if (!boundSrc) boundSrc = video.currentSrc;
    else if (video.currentSrc !== boundSrc) {
      st.destroy();
      return;
    }
    const d = video.duration;
    if (!(range && range.min > 0) && Number.isFinite(d) && d > boundDuration) {
      boundDuration = d;
    }
    const adLike =
      Number.isFinite(d) && boundDuration > 0 && d < boundDuration * 0.5;
    const nativeFs = document.fullscreenElement === video;
    track.mode = nativeFs && !adLike ? "showing" : "disabled";
    if (nativeFs || adLike) {
      host.style.display = "none";
      return;
    }
    layout();
    host.style.display = "block";
    const text = cueAt(video.currentTime);
    if (text !== shown) {
      shown = text;
      lineEl.textContent = text;
    }
  };

  const events = ["emptied", "loadstart", "timeupdate", "seeked", "play", "pause", "loadedmetadata", "durationchange"];
  for (const e of events) video.addEventListener(e, update);
  document.addEventListener("fullscreenchange", update);
  window.addEventListener("resize", update);
  const ro = new ResizeObserver(() => update());
  ro.observe(video);

  const st = {
    video,
    destroy() {
      for (const e of events) video.removeEventListener(e, update);
      document.removeEventListener("fullscreenchange", update);
      window.removeEventListener("resize", update);
      ro.disconnect();
      host.remove();
      track.mode = "disabled";
      if (w.__tldrCaptions === st) w.__tldrCaptions = null;
    },
  };
  w.__tldrCaptions = st;
  update();
  return true;
}
