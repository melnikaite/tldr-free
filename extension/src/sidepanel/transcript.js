// Transcript tab controller.
//
// Responsibilities:
//   - Lazy-load the full transcript (Job.raw_text or a cached translation)
//     on first tab activation. Re-fetch when the user clicks a different
//     language chip.
//   - Render lines with [MM:SS] markers as <p data-tx-seconds="…">; the
//     existing summaryEl click handler in app.js works on these too once
//     we expose the same data-tldr-seconds attribute pattern.
//   - Poll media.currentTime via chrome.scripting.executeScript across all
//     frames (covers iframe-embedded players) every 500ms while the tab
//     is visible AND the media is playing. Highlight the matching line
//     via binary search; auto-scroll into view.
//   - Inject WebVTT <track> into the page's first <video> on language
//     click (track creation in main world so the blob URL resolves in
//     the page's context). Skip injection for audio-only pages (HTML5
//     <audio> has no native captions UI surface).
//
// State is module-local and reset on job switch. transcript.js doesn't
// know about job lifecycle directly — app.js calls onJobChange / onTabShow.

import { daemon } from "../lib/daemon-client.js";
import { openEventStream } from "../lib/event-stream.js";
import { resolveVideoId } from "../lib/url.js";
import { stringifyError } from "../lib/utils.js";

/** @import { JobDetails, TranscriptResponse } from "../lib/api-types.js" */

// ---------------------------------------------------------------------------
// DOM handles + module state
// ---------------------------------------------------------------------------

const tabBtn = /** @type {HTMLButtonElement | null} */ (
  document.getElementById("tab-transcript")
);
const paneEl = /** @type {HTMLElement | null} */ (
  document.getElementById("pane-transcript")
);
const langBarEl = /** @type {HTMLElement | null} */ (
  document.getElementById("lang-bar")
);
const bodyEl = /** @type {HTMLElement | null} */ (
  document.getElementById("transcript-body")
);

// Cache of {jobId+lang → TranscriptResponse} so re-clicking a chip
// is instant. Cleared on job switch.
/** @type {Map<string, TranscriptResponse>} */
const _textCache = new Map();

// Keys we currently have in pending state (server replied
// ``is_pending: true``). When a ``job_event("updated")`` arrives for the
// active job we flush these and refetch so the transcript appears as
// soon as the extraction / translation finishes — without the user
// having to click anything.
/** @type {Set<string>} */
const _pendingKeys = new Set();

/** @type {JobDetails | null} */
let _job = null;

/** Currently displayed language code (null = original). */
let _currentLang = null;

/** Sorted list of {sec, el} cues for binary-search highlight. */
/** @type {{ sec: number, el: HTMLElement }[]} */
let _cues = [];

/** Last highlight index — fast path for monotonic playback. */
let _lastCueIdx = -1;

/** setInterval handle for the currentTime poll. */
let _pollId = /** @type {number | null} */ (null);

/** Tab id we're polling. Re-resolved per poll if it goes stale. */
let _pollTabId = /** @type {number | null} */ (null);

/** Has the user clicked into this tab at least once for this job? */
let _opened = false;

// ---------------------------------------------------------------------------
// SSE listener — keep chips fresh while translations progress
//
// The daemon's translator worker emits ``job_event("translation_updated",
// {…})`` events on each chunk-progress tick and at status transitions.
// We mirror those into ``_job.transcript_translations`` and re-render the
// chips so the running spinner moves and a ``done`` chip becomes
// clickable the moment the worker finishes.
// ---------------------------------------------------------------------------

const _eventStream = openEventStream();
_eventStream.subscribe((event) => {
  if (event.type !== "job") return;
  // Translation progress / completion — owns the chip strip.
  if (event.action === "translation_updated") {
    const j = event.job;
    if (!_job || j?.id !== _job.id) return;
    const code = j.language_code;
    if (!code) return;
    const existing = _job.transcript_translations || [];
    const next = existing.filter((t) => t.language_code !== code);
    next.push({
      language_code: code,
      status: j.status,
      progress_percent: j.progress_percent ?? 0,
      error: j.error || null,
    });
    // Mutate in place so app.js's cached job reference sees the update
    // (we don't want to bother re-fetching JobDetails just for chips).
    _job.transcript_translations = next;
    _renderChips();
    // If the just-completed translation is the language the user is
    // looking at, refetch + re-render the body. /events doesn't carry
    // the text payload itself — translation bodies can be megabytes,
    // we keep them out of the broadcast for sanity.
    if ((j.status === "done" || j.status === "partial") && _currentLang === code) {
      _textCache.delete(`${_job.id}::${code}`);
      _pendingKeys.delete(`${_job.id}::${code}`);
      _showLanguage(code).catch(() => {});
    }
    return;
  }
  // Job state change (extraction → ready → summarizing → done). If we're
  // sitting on a pending transcript for THIS job, the extraction might
  // have just finished — refetch. set_extracted (raw_text saved) and
  // mark_done both publish job_event("updated"); the first event after
  // each is enough to flip a pending view into a live one.
  if (event.action === "updated") {
    const j = event.job;
    if (!_job || j?.id !== _job.id) return;
    if (_pendingKeys.size > 0) {
      // Snapshot + clear: we may add the key back if it's still pending
      // on the next response; otherwise the pending state ends here.
      const keys = [..._pendingKeys];
      _pendingKeys.clear();
      for (const key of keys) _textCache.delete(key);
      // Refetch the currently-displayed language so the body refreshes
      // for the user without them having to click anything.
      if (paneEl?.classList.contains("tab-pane--active") && _opened) {
        _showLanguage(_currentLang).catch(() => {});
      }
      return;
    }
    // Self-heal: text is rendered but polling didn't start (source tab
    // wasn't open when we first tried) — retry inject + poll. Idempotent:
    // _injectCaptionsIntoTab removes the prior <track> before adding the
    // new one; _startPoll is a no-op when _pollId is set.
    if (
      _opened
      && _cues.length > 0
      && _pollId === null
      && paneEl?.classList.contains("tab-pane--active")
    ) {
      const key = `${_job.id}::${_currentLang ?? ""}`;
      const data = _textCache.get(key);
      if (data && data.text) {
        _injectCaptionsIntoTab(data).catch(() => {});
        _startPoll();
      }
    }
  }
});

// ---------------------------------------------------------------------------
// Public API — called by app.js
// ---------------------------------------------------------------------------

/**
 * Notify that the active job changed. Resets cache + state when the job
 * id actually changed; for in-place patches (title fill-in, summary done,
 * etc.) we keep the user's transcript view intact and only refresh chips
 * from the new merged data.
 *
 * @param {JobDetails | null} job
 */
export function setJob(job) {
  const sameJob = !!(_job && job && _job.id === job.id);
  if (!sameJob) {
    // Real switch — tear down everything.
    _stopPoll();
    _textCache.clear();
    _pendingKeys.clear();
    _cues = [];
    _lastCueIdx = -1;
    _opened = false;
    _currentLang = null;
  }
  _job = job;
  _syncTabVisibility();
  if (!paneEl?.classList.contains("tab-pane--active") || !_shouldShowTab()) return;
  if (!sameJob) {
    // Re-open for the new job (fetches transcript, kicks polling, renders chips).
    _open().catch((err) => console.warn("[TLDR] transcript open failed:", err));
    return;
  }
  // Same job, just patched fields — re-render chips off the new data
  // without re-fetching the body.
  _renderChips();
}

/**
 * Notify that the user switched to the transcript tab. First call for a
 * job triggers the lazy fetch + render; without a job it renders the
 * "Process this page" entry point.
 */
export function onTabShow() {
  if (!_shouldShowTab()) return;
  // ``_open`` handles both states — with a job it lazy-loads the
  // transcript, without one it renders the no-job button.
  _open().catch((err) => console.warn("[TLDR] transcript open failed:", err));
}

/**
 * Notify that the user switched away from the transcript tab. Stops the
 * currentTime poll so we don't burn CPU on tab-script roundtrips.
 */
export function onTabHide() {
  _stopPoll();
}

// ---------------------------------------------------------------------------
// Visibility
// ---------------------------------------------------------------------------

function _shouldShowTab() {
  // No job yet → still show the tab so the user can kick off processing
  // from here without going to the Summary tab first. The body renders a
  // "Process this page" prompt in that case.
  if (!_job) return true;
  // Job exists — show only for timed media. Pages / PDFs have extracted
  // text but no [MM:SS] structure, and the Summary tab already shows
  // everything useful for them.
  return _job.kind === "youtube" || _job.kind === "media";
}

function _syncTabVisibility() {
  if (!tabBtn) return;
  tabBtn.classList.toggle("hidden", !_shouldShowTab());
}

// Run visibility sync once on module init: when the sidepanel boots
// without an active job (most common case — user just opened the
// browser), ``setJob`` never gets called and the tab button stays
// stuck with the ``hidden`` class from the HTML. Without this the
// "Process this page" entry point is unreachable.
_syncTabVisibility();
// Replace the HTML default placeholder ("Open this tab to load…") with
// the no-job entry point UI so the user sees an actionable button the
// first time they click into the Transcript tab — even before any
// setJob / onTabShow call has fired (e.g. on cold sidepanel open).
if (!_job) _renderNoJobState();

// ---------------------------------------------------------------------------
// Open / render
// ---------------------------------------------------------------------------

async function _open() {
  if (!_job) {
    _renderNoJobState();
    return;
  }
  _opened = true;
  // Default to the job's source language; null falls back to "original"
  // (which the daemon serves as Job.raw_text directly).
  const initialLang = _currentLang ?? _job.transcript_language ?? null;
  await _showLanguage(initialLang);
}

/**
 * Render the "no job for this tab" state on the Transcript pane: empty
 * language bar + a single "Process this page" button.
 *
 * Clicking the button sends ``summarize-active-tab`` to background.js,
 * which kicks off the full pipeline (extract + summarize). After that
 * the regular tab-follow + SSE plumbing picks the new job up and turns
 * the placeholder into a live pending → text view via the existing
 * job-updated event handler.
 *
 * Same message as the Summary tab's "Summarize this page" button — we
 * don't have a separate "transcribe-only" pipeline path (Whisper is the
 * slow part either way; saving the summary step would only shave ~10%
 * of total time and would require a new endpoint, schema field, and
 * state machine).
 */
function _renderNoJobState() {
  if (!bodyEl) return;
  if (langBarEl) langBarEl.innerHTML = "";
  bodyEl.innerHTML = `
    <div class="placeholder-block">
      <p class="muted small">No transcript yet — process this page to extract one.</p>
      <button class="summarize-btn" id="transcript-process-btn" type="button">Process this page</button>
    </div>
  `;
  const btn = /** @type {HTMLButtonElement | null} */ (
    document.getElementById("transcript-process-btn")
  );
  if (!btn) return;
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "Starting…";
    try {
      await chrome.runtime.sendMessage({ type: "summarize-active-tab" });
    } catch (err) {
      bodyEl.innerHTML = `<p class="placeholder">Couldn't start: ${stringifyError(err)}</p>`;
    }
  });
}

/**
 * Render the transcript for ``lang`` (null = original). Fetches from the
 * daemon (or in-memory cache), populates body + chips, kicks off polling.
 *
 * Pending responses (raw_text not extracted yet, or translation in
 * flight) get a placeholder. The key is added to ``_pendingKeys`` so the
 * SSE listener auto-refetches on the next job_event for this job.
 *
 * @param {string | null} lang
 */
async function _showLanguage(lang) {
  if (!_job || !bodyEl) return;

  _currentLang = lang;
  _renderChips();

  const key = `${_job.id}::${lang ?? ""}`;
  let data = _textCache.get(key);
  if (!data) {
    bodyEl.innerHTML = `<div class="transcript-translating"><div class="spinner"></div><p>Loading transcript…</p></div>`;
    try {
      data = await daemon.getTranscript(_job.id, lang ?? undefined);
    } catch (err) {
      bodyEl.innerHTML = `<p class="placeholder">Couldn't load transcript: ${stringifyError(err)}</p>`;
      return;
    }
    _textCache.set(key, data);
  }

  // Pending = extraction or translation still in flight. Show a
  // placeholder; the SSE listener above refetches when the job updates.
  if (data.is_pending || !data.text) {
    _pendingKeys.add(key);
    const label = data.is_original
      ? "Waiting for transcript to be extracted…"
      : "Translating — chip in the language bar shows progress…";
    bodyEl.innerHTML = `<div class="transcript-translating"><div class="spinner"></div><p>${label}</p></div>`;
    // Don't start polling / inject captions until we actually have text.
    _stopPoll();
    return;
  }

  _renderLines(data.text);
  _injectCaptionsIntoTab(data).catch((err) =>
    console.warn("[TLDR] caption injection failed:", err),
  );
  // Resume / start polling so the line highlight follows playback.
  _startPoll();
}

/**
 * Parse [MM:SS] markers out of ``rawText`` and render one <p> per line
 * with ``data-tx-seconds`` for binary-search lookup. Lines without a
 * leading marker are rendered as continuation paragraphs (no anchor).
 *
 * @param {string} rawText
 */
function _renderLines(rawText) {
  if (!bodyEl) return;
  bodyEl.innerHTML = "";
  _cues = [];
  _lastCueIdx = -1;

  // The transcript shape from build_marked_text is one [MM:SS] (or
  // [HH:MM:SS]) marker per line, followed by the bucket's text. Split
  // on newlines and parse the leading marker.
  const lines = rawText.split(/\r?\n/);
  const re = /^\[(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\]\s*(.*)$/;
  const frag = document.createDocumentFragment();
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const m = re.exec(trimmed);
    const p = document.createElement("p");
    p.className = "tx-line";
    if (m) {
      const h = m[1] ? Number(m[1]) : 0;
      const mm = Number(m[2]);
      const ss = Number(m[3]);
      const sec = h * 3600 + mm * 60 + ss;
      p.dataset.txSeconds = String(sec);
      // Make the marker clickable to seek — reuse the same click handler
      // pattern as the summary timecode links so app.js handles it.
      const a = document.createElement("a");
      a.className = "tx-mark";
      a.href = _seekHref(sec);
      a.dataset.tldrSeconds = String(sec);
      _setTimecodeTarget(a);
      a.target = "_blank";
      a.rel = "noopener";
      a.textContent = `[${m[0].slice(1, m[0].indexOf("]"))}]`;
      p.appendChild(a);
      p.appendChild(document.createTextNode(" " + (m[4] || "")));
    } else {
      p.textContent = trimmed;
    }
    frag.appendChild(p);
  }
  bodyEl.appendChild(frag);

  // Build cue index for highlight lookup.
  for (const el of bodyEl.querySelectorAll("p.tx-line[data-tx-seconds]")) {
    const sec = Number(/** @type {HTMLElement} */ (el).dataset.txSeconds);
    if (Number.isFinite(sec)) {
      _cues.push({ sec, el: /** @type {HTMLElement} */ (el) });
    }
  }
  // Defensive sort — build_marked_text already emits in order but bad
  // input shouldn't break the binary search invariant.
  _cues.sort((a, b) => a.sec - b.sec);
}

/**
 * Build the href used for the line's seek link. For YouTube we use the
 * canonical ``youtube.com/watch?v=…&t=Ns`` form; for media the page URL
 * with ``#t=N`` fragment. Same logic as ``markdown.js`` ``replaceInTextNode``.
 *
 * @param {number} seconds
 */
function _seekHref(seconds) {
  if (!_job) return "#";
  // ``resolveVideoId`` also parses the URL — important during the window
  // between job creation and the first set_extracted (which writes
  // ``video_id`` on the row). Without this fallback the transcript
  // renders during processing with hrefs like ``#`` that the click
  // handler can't seek, so anchors fall through to the generic
  // external-link handler which opens a *new* tab.
  const videoId = resolveVideoId(_job);
  if (videoId) {
    return `https://www.youtube.com/watch?v=${encodeURIComponent(videoId)}&t=${seconds}s`;
  }
  if (_job.kind === "media" && _job.url) {
    const u = _job.url;
    const i = u.indexOf("#");
    return `${i === -1 ? u : u.slice(0, i)}#t=${seconds}`;
  }
  return "#";
}

/**
 * Tag a seek anchor with the same dataset keys the sidepanel click
 * handler expects (data-tldr-video-id / data-tldr-media-page-url).
 *
 * @param {HTMLElement} a
 */
function _setTimecodeTarget(a) {
  if (!_job) return;
  const videoId = resolveVideoId(_job);
  if (videoId) {
    a.dataset.tldrVideoId = videoId;
  } else if (_job.kind === "media" && _job.url) {
    a.dataset.tldrMediaPageUrl = _job.url;
  }
}

// ---------------------------------------------------------------------------
// Language chips
// ---------------------------------------------------------------------------

function _renderChips() {
  if (!langBarEl || !_job) return;
  langBarEl.innerHTML = "";

  const source = _job.transcript_language;
  const cached = (_job.transcript_translations || []).filter(
    (t) =>
      t.status === "done" ||
      t.status === "partial" ||
      t.status === "running" ||
      t.status === "failed",
  );

  // Source language chip (original; always present even when null — UI
  // shows "Original" without a code).
  const srcCode = source || "original";
  const srcLabel = source ? source.toUpperCase() : "Original";
  langBarEl.appendChild(
    _makeChip({
      code: source || null,
      label: srcLabel,
      status: "done",
      current: (_currentLang ?? source ?? null) === (source ?? null),
    }),
  );

  for (const t of cached) {
    if (t.language_code === source) continue; // already shown
    langBarEl.appendChild(
      _makeChip({
        code: t.language_code,
        label: t.language_code.toUpperCase(),
        status: t.status,
        current: _currentLang === t.language_code,
        progress: t.progress_percent,
        error: t.error || null,
      }),
    );
  }

  // Free-form input: type a language code (ru, ja, en…) and press Enter
  // to enqueue a translation. The daemon dedups in-flight requests, so
  // mashing Enter doesn't spawn multiple workers.
  const input = document.createElement("input");
  input.type = "text";
  input.maxLength = 16;
  input.className = "lang-input";
  input.placeholder = "+ lang";
  input.title = "Type a language code (ru / ja / de…) and press Enter";
  input.addEventListener("keydown", (ev) => {
    if (ev.key !== "Enter") return;
    ev.preventDefault();
    const val = input.value.trim();
    if (!val) return;
    input.disabled = true;
    daemon.translateTranscript(_job.id, val)
      .then((result) => {
        input.value = "";
        // If the daemon told us this IS the source language, just switch
        // the view — no chip needs to appear (the source chip already
        // exists). A dedup response of "partial" means a previous run
        // already produced usable (if incomplete) text — same deal.
        if (result.is_source || result.status === "done" || result.status === "partial") {
          _showLanguage(result.language_code).catch(() => {});
          return;
        }
        // Optimistic chip insert: render the queued chip immediately
        // from the POST response so the user sees feedback even if the
        // /events SSE connection is currently broken (e.g. mid-reconnect
        // after a daemon restart). The follow-up translation_updated
        // event refreshes the same entry as soon as the worker emits one.
        if (_job && result.language_code) {
          const code = result.language_code;
          const existing = _job.transcript_translations || [];
          if (!existing.some((t) => t.language_code === code)) {
            _job.transcript_translations = [
              ...existing,
              {
                language_code: code,
                status: result.status,
                progress_percent: result.progress_percent ?? 0,
                error: null,
              },
            ];
            _renderChips();
          }
        }
      })
      .catch((err) => {
        // Show the error inline next to the input so the user knows it
        // didn't take.
        const note = document.createElement("span");
        note.className = "lang-error";
        note.textContent = stringifyError(err);
        input.parentElement?.appendChild(note);
        setTimeout(() => note.remove(), 4000);
      })
      .finally(() => {
        input.disabled = false;
      });
  });
  langBarEl.appendChild(input);

  // Retry-all button (appears when ≥1 chip is ``failed`` OR ``partial`` —
  // the daemon's retry-all endpoint re-queues both).
  const hasRetryable = cached.some(
    (t) => t.status === "failed" || t.status === "partial",
  );
  if (hasRetryable) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "lang-chip lang-chip--retry";
    btn.textContent = "Retry failed";
    btn.title =
      "Re-queue every failed or partially-translated language for this job.";
    btn.addEventListener("click", () => {
      btn.disabled = true;
      daemon.retryAllTranslations(_job.id)
        .catch((err) => {
          // Surface failures so the user knows the click did something
          // (or didn't) — earlier silent failures looked like a dead button.
          const note = document.createElement("span");
          note.className = "lang-error";
          note.textContent = stringifyError(err);
          btn.parentElement?.appendChild(note);
          setTimeout(() => note.remove(), 4000);
        })
        .finally(() => {
          btn.disabled = false;
        });
    });
    langBarEl.appendChild(btn);
  }
}

/**
 * @param {{
 *   code: string | null,
 *   label: string,
 *   status: "queued"|"running"|"done"|"partial"|"failed",
 *   current: boolean,
 *   progress?: number,
 *   error?: string | null,
 * }} opts
 */
function _makeChip(opts) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "lang-chip";
  if (opts.current) btn.classList.add("lang-chip--current");
  if (opts.status === "running" || opts.status === "queued") {
    btn.classList.add("lang-chip--running");
  }
  if (opts.status === "failed") btn.classList.add("lang-chip--failed");
  // "partial" is selectable like "done" (it has real text) but flagged
  // visually — some lines fell back to the source language.
  if (opts.status === "partial") btn.classList.add("lang-chip--partial");

  if (opts.status === "running" && opts.progress != null) {
    btn.append(opts.label, ` ${opts.progress}%`);
    const dot = document.createElement("span");
    dot.className = "spinner-dot";
    btn.appendChild(dot);
  } else {
    btn.textContent = opts.label;
  }
  if ((opts.status === "failed" || opts.status === "partial") && opts.error) {
    btn.title = opts.error;
  }

  btn.addEventListener("click", () => {
    // For "done" / "partial" / null (source) chips: switch to that
    // language. For in-flight or failed: no-op for now; Phase 3 wires
    // retry.
    if (opts.status !== "done" && opts.status !== "partial") return;
    if ((_currentLang ?? null) === (opts.code ?? null)) {
      // Re-clicking current chip re-injects captions (idempotent, handy
      // if the user reloaded the source tab).
      const key = `${_job?.id}::${opts.code ?? ""}`;
      const cached = _textCache.get(key);
      if (cached) _injectCaptionsIntoTab(cached).catch(() => {});
      return;
    }
    _showLanguage(opts.code).catch((err) =>
      console.warn("[TLDR] language switch failed:", err),
    );
  });
  return btn;
}

// ---------------------------------------------------------------------------
// currentTime polling + highlight
// ---------------------------------------------------------------------------

const POLL_INTERVAL_MS = 500;

async function _startPoll() {
  if (_pollId !== null) return;
  if (!_job) return;
  // Resolve once which tab to poll. ``_findSourceTab`` searches by
  // video_id (YouTube) or by canonical URL (media).
  _pollTabId = await _findSourceTab();
  if (_pollTabId == null) {
    // No matching tab open — nothing to follow. The user can still
    // click [MM:SS] links to open one.
    return;
  }
  _pollId = /** @type {number} */ (
    setInterval(() => _pollOnce().catch(() => {}), POLL_INTERVAL_MS)
  );
}

function _stopPoll() {
  if (_pollId !== null) {
    clearInterval(_pollId);
    _pollId = null;
  }
  _pollTabId = null;
}

async function _pollOnce() {
  if (_pollTabId == null) return;
  const result = await _readMediaState(_pollTabId);
  if (!result) return;
  if (result.paused) {
    // Paused → don't autoscroll; just leave the current highlight as it
    // is so the user can read freely.
    _highlight(result.currentTime, /* scroll */ false);
    return;
  }
  _highlight(result.currentTime, /* scroll */ true);
}

/**
 * executeScript into every frame, return {currentTime, paused} from the
 * first frame that has a <video> or <audio>. Returns null if no tab /
 * no media found (e.g. user reloaded the source page, killing playback).
 *
 * @param {number} tabId
 * @returns {Promise<{currentTime: number, paused: boolean} | null>}
 */
async function _readMediaState(tabId) {
  try {
    const results = await chrome.scripting.executeScript({
      target: { tabId, allFrames: true },
      func: () => {
        const m = document.querySelector("video, audio");
        if (!m) return null;
        return {
          currentTime: m.currentTime,
          paused: m.paused,
        };
      },
    });
    for (const r of results) {
      if (r.result) return r.result;
    }
  } catch {
    // Tab gone or extension lost permission — silently stop polling
    // so we don't spam the console.
    _stopPoll();
  }
  return null;
}

/**
 * Binary search ``_cues`` for the line that covers ``currentSec`` and
 * apply ``.tx-line--current`` (clearing any previous). If ``scroll`` is
 * true and the line is out of viewport, scroll it into view smoothly.
 *
 * @param {number} currentSec
 * @param {boolean} scroll
 */
function _highlight(currentSec, scroll) {
  if (!_cues.length) return;
  // Fast path: still inside the cached cue's range.
  if (
    _lastCueIdx >= 0 &&
    _lastCueIdx < _cues.length &&
    currentSec >= _cues[_lastCueIdx].sec &&
    (_lastCueIdx === _cues.length - 1 || currentSec < _cues[_lastCueIdx + 1].sec)
  ) {
    return;
  }
  // Binary search for the largest cue.sec <= currentSec.
  let lo = 0;
  let hi = _cues.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (_cues[mid].sec <= currentSec) lo = mid;
    else hi = mid - 1;
  }
  if (lo === _lastCueIdx) return;
  if (_lastCueIdx >= 0 && _lastCueIdx < _cues.length) {
    _cues[_lastCueIdx].el.classList.remove("tx-line--current");
  }
  _cues[lo].el.classList.add("tx-line--current");
  _lastCueIdx = lo;
  if (scroll) {
    _cues[lo].el.scrollIntoView({ block: "center", behavior: "smooth" });
  }
}

/**
 * Locate the tab playing the source media. For YouTube, match by video
 * id (so timestamp / autoplay params don't break the match). For media
 * jobs, match by canonical page URL (fragment stripped on both sides).
 *
 * @returns {Promise<number | null>}
 */
async function _findSourceTab() {
  if (!_job) return null;
  try {
    const videoId = resolveVideoId(_job);
    if (videoId) {
      const tabs = await chrome.tabs.query({ url: "*://www.youtube.com/watch*" });
      const hit = tabs.find((t) => {
        try {
          return new URL(t.url ?? "").searchParams.get("v") === videoId;
        } catch {
          return false;
        }
      });
      return hit?.id ?? null;
    }
    const targetUrl = _job.url;
    if (!targetUrl) return null;
    const canonical = _stripFragment(targetUrl);
    const tabs = await chrome.tabs.query({});
    const hit = tabs.find((t) => t.url && _stripFragment(t.url) === canonical);
    return hit?.id ?? null;
  } catch {
    return null;
  }
}

function _stripFragment(url) {
  try {
    const u = new URL(url);
    u.hash = "";
    return u.toString();
  } catch {
    return url;
  }
}

// ---------------------------------------------------------------------------
// WebVTT injection
// ---------------------------------------------------------------------------

/**
 * Convert ``text`` (one [MM:SS] line per cue) into WebVTT and inject a
 * <track> into the source tab's first <video>. Audio-only tabs and
 * iframe-embedded players are skipped (we can't surface captions in
 * those — for those cases the user reads the transcript here).
 *
 * @param {TranscriptResponse} data
 */
async function _injectCaptionsIntoTab(data) {
  if (!_job) return;
  const tabId = await _findSourceTab();
  if (tabId == null) return;
  const vtt = _toVtt(data.text);
  if (!vtt) return;
  const langCode = data.language_code || "x-tldr";

  try {
    await chrome.scripting.executeScript({
      target: { tabId },
      // MAIN world so the blob URL we create lives in the page's
      // context — a <track src=…> loaded by the page's <video> element
      // can't resolve extension-context blob URLs.
      world: "MAIN",
      func: (vttText, lang) => {
        const video = document.querySelector("video");
        if (!video) return false;
        // Remove any prior TLDR track so reapply is idempotent.
        for (const t of [...video.querySelectorAll('track[data-tldr]')]) {
          try {
            URL.revokeObjectURL(/** @type {HTMLTrackElement} */ (t).src);
          } catch {}
          t.remove();
        }
        const blob = new Blob([vttText], { type: "text/vtt" });
        const url = URL.createObjectURL(blob);
        const track = document.createElement("track");
        track.kind = "subtitles";
        track.label = `TLDR (${lang})`;
        track.srclang = lang;
        track.src = url;
        track.default = true;
        track.setAttribute("data-tldr", "1");
        video.appendChild(track);
        // Force-show — many players default new tracks to "disabled".
        const tt = video.textTracks[video.textTracks.length - 1];
        if (tt) tt.mode = "showing";
        return true;
      },
      args: [vtt, langCode],
    });
  } catch {
    // Permission denied (cross-origin iframe), no <video>, page CSP
    // blocked blob: — all expected on some sites. Silent.
  }
}

/**
 * Convert ``[MM:SS] text`` lines into a WebVTT body. Each cue spans
 * marker → next marker (last cue extends a generous +60s so the final
 * line stays visible during late playback). Marker parsing tolerates
 * [HH:MM:SS] too.
 *
 * @param {string} text
 * @returns {string}
 */
function _toVtt(text) {
  const re = /\[(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\]\s*([^\n]*)/g;
  const cues = [];
  let m;
  while ((m = re.exec(text)) !== null) {
    const h = m[1] ? Number(m[1]) : 0;
    const mm = Number(m[2]);
    const ss = Number(m[3]);
    cues.push({ start: h * 3600 + mm * 60 + ss, text: (m[4] || "").trim() });
  }
  if (!cues.length) return "";
  const lines = ["WEBVTT", ""];
  for (let i = 0; i < cues.length; i++) {
    const start = cues[i].start;
    const end = i + 1 < cues.length ? cues[i + 1].start : start + 60;
    lines.push(`${_vttTime(start)} --> ${_vttTime(end)}`);
    lines.push(cues[i].text);
    lines.push("");
  }
  return lines.join("\n");
}

function _vttTime(totalSec) {
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = Math.floor(totalSec % 60);
  return (
    String(h).padStart(2, "0") + ":" +
    String(m).padStart(2, "0") + ":" +
    String(s).padStart(2, "0") + ".000"
  );
}
