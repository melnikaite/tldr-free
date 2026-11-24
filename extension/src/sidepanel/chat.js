// Chat panel — Q&A scoped to the currently active job.
//
// Sends each question through POST /ai/stream {job_id, question}. The
// daemon persists user + assistant messages in SQLite, so chat history
// survives tab switches, browser restarts, and side-panel close. On job
// switch, the side panel calls renderHistory(jobId, items) below to redraw
// the stored bubbles before any new turn.
//
// Token streaming: append plain text to the assistant bubble as it arrives,
// then re-render via lib/markdown.js once the stream ends.
//
// Note — timecodes: qa.txt instructs the LLM to include [MM:SS] markers ONLY
// when (a) the answer came from the material (not web_search/general knowledge),
// (b) the material itself has those markers, and (c) they genuinely locate the
// relevant moment. Answers grounded in web_search must NOT include timestamps
// — there's no video to jump to and any marker would be a hallucination.
// Chat bubbles render WITH timecode-link injection (renderMarkdown(text, activeJob))
// so valid material references become clickable seek links just like in Summary.
//
// QA progress indicator strategy:
//   Problem A — spinner pauses: the assistant bubble lives inside #pane-summary
//     (display:none when Transcript tab is active), so CSS animations pause.
//   Problem B — badge conflict: #stage-badge is also owned by app.js; any
//     SSE job-update event calls setStage(null) and silently clears our QA state.
//   Problem C — repaint throttle: Chrome throttles repaints for sidepanels
//     that don't have focus; textNode.data updates happen but don't paint
//     until the panel regains focus.
//
// Solution:
//   - Toggle .tab--qa-active on the Summary TAB BUTTON (always visible,
//     even from the Transcript pane). The CSS adds a pulsing dot. No conflict
//     with app.js (which only toggles .tab--active).
//   - Keep module-level refs to the live text node + accumulated string so a
//     window.focus / visibilitychange listener can re-touch the node and
//     force Chrome to repaint the streaming text immediately.
//   - Scrolling uses lib/scroll-follow.js "follow" mode: auto-scroll only
//     while the reader is at the bottom; nothing ever jumps on answer done.
//
// Shared-DOM ownership — why every write below is guarded:
//   The side panel is one document per Chrome window. #chat-messages is a
//   single DOM list that every job's chat borrows in turn; it belongs to
//   whichever job app.js last called setActiveJob() for, not to whichever
//   turn happens to be running. A turn can outlive that: ask Q1 about video
//   A, ask Q2 while Q1 is still streaming (Q2 queues in pendingQuestions —
//   see the drain loop in handleAsk), then switch to video B before Q1's
//   answer lands. loadAndRender → renderHistory wipes #chat-messages for B
//   while Q1 is still in flight for A; when Q1 finishes, the drain loop
//   starts Q2 against A next. Neither turn is allowed to assume the DOM it
//   grabbed at the start is still theirs to write into — that would paint
//   A's answer into B's chat (and it did, before this fix).
//   `chatJobId` (below) is the authoritative "whose chat is on screen right
//   now" — app.js's setActiveJob() is the only writer. `_runQaTurn` checks
//   it before every DOM touch, not just once at the top, because ownership
//   can flip mid-stream in either direction: lost (user switches away — the
//   request keeps running so the answer still gets generated and persisted
//   under the right job, it just stops painting) and regained (user tabs
//   back — renderHistory re-attaches the live bubble from the accumulated
//   text instead of leaving a dead gap until the stream happens to finish).

import { daemon } from "../lib/daemon-client.js";
import { buildFrameRow } from "../lib/frame-thumbnails.js";
import { renderMarkdown } from "../lib/markdown.js";
import { chatDraftKey } from "../lib/active-job.js";
import { followScrollToEnd, setFollowing } from "../lib/scroll-follow.js";

/** @import { ChatMessage, FrameRef, JobDetails } from "../lib/api-types.js" */

/** @type {JobDetails | null} */
let activeJob = null;

// The job whose chat currently owns #chat-messages — see the ownership note
// above. `null` means no job's chat is on screen (e.g. the no-summary
// placeholder). Set only by setActiveJob() / clearChat(); everything else
// treats it as read-only.
/** @type {string | null} */
let chatJobId = null;

/** @param {JobDetails | null} job */
export function setActiveJob(job) {
  const newId = job?.id ?? null;
  if (newId !== chatJobId) _swapDraft(chatJobId, newId);
  // If we're switching away from the job that owns the in-flight turn (if
  // any), turn off its pulsing dot now — the turn keeps streaming in the
  // background, but nothing on screen should look like it's happening on
  // whatever job we're switching to. The turn re-enables it itself if the
  // user switches back (see renderHistory's reattach below).
  if (newId !== chatJobId && chatJobId !== null && chatJobId === _liveTurnJobId) {
    _setQaActive(false);
  }
  activeJob = job;
  chatJobId = newId;
  // After the assignments: a "done" here may start a queued turn, which
  // checks chat ownership synchronously.
  if (job) _noteStatus(job.id, job.status, job.error);
}

/**
 * Synchronous read of the in-memory active job. Deliberately NO fallback to
 * chrome.storage.session: that state is per window and the panel clears its
 * job on purpose (no-summary / video CTA) — resurrecting one from storage
 * would show another window's or a stale job. For callers that need the
 * freshest already-known job state (e.g. a live mid-stream title patch) without introducing an async gap a later event
 * could race past — see app.js's stream "done"/"error" handling.
 *
 * @returns {JobDetails | null}
 */
export function peekActiveJob() {
  return activeJob;
}

const form = /** @type {HTMLFormElement | null} */ (document.getElementById("chat-form"));
const input = /** @type {HTMLTextAreaElement | null} */ (document.getElementById("chat-input"));
const messages = /** @type {HTMLElement | null} */ (document.getElementById("chat-messages"));

// ---------------------------------------------------------------------------
// Composer draft — one per job, in chrome.storage.session (keyed by job id,
// not by window) so it survives panel reopen. Saved on input and on switch,
// restored on switch (empty when none), deleted on send. Typed text never
// follows the user to another job.
// ---------------------------------------------------------------------------

/** @type {Map<string, string>} in-memory mirror (authoritative once touched) */
const _drafts = new Map();
let _draftWriteTimer = /** @type {number | null} */ (null);

/** @param {string} jobId @param {string} text */
function _persistDraft(jobId, text) {
  _drafts.set(jobId, text);
  const key = chatDraftKey(jobId);
  (text ? chrome.storage.session.set({ [key]: text }) : chrome.storage.session.remove(key))
    .catch(() => {});
}

/** @param {string | null} oldId @param {string | null} newId */
function _swapDraft(oldId, newId) {
  if (!input) return;
  if (_draftWriteTimer !== null) {
    clearTimeout(_draftWriteTimer);
    _draftWriteTimer = null;
  }
  if (oldId) _persistDraft(oldId, input.value);
  const known = newId ? _drafts.get(newId) : "";
  input.value = known ?? "";
  _autoGrowChatInput();
  if (newId && known === undefined) {
    const key = chatDraftKey(newId);
    chrome.storage.session.get(key).then((got) => {
      const text = typeof got[key] === "string" ? got[key] : "";
      // Only if still on that job and the user hasn't started typing.
      if (chatJobId !== newId || _drafts.has(newId) || input.value) return;
      _drafts.set(newId, text);
      input.value = text;
      _autoGrowChatInput();
    }).catch(() => {});
  }
}

input?.addEventListener("input", () => {
  const jobId = chatJobId;
  if (!jobId) return;
  _drafts.set(jobId, input.value);
  if (_draftWriteTimer !== null) clearTimeout(_draftWriteTimer);
  _draftWriteTimer = /** @type {number} */ (setTimeout(() => {
    _draftWriteTimer = null;
    _persistDraft(jobId, _drafts.get(jobId) ?? "");
  }, 300));
});

// Summary tab button — we toggle .tab--qa-active on it while QA is in flight.
// This adds a pulsing dot via CSS that is visible regardless of which pane is
// active (the tab nav is always rendered above both panes).
// We deliberately do NOT touch #stage-badge (owned by app.js / setStage) to
// avoid the conflict where any SSE job-update clears our indicator.
const _summaryTabBtn = /** @type {HTMLButtonElement | null} */ (
  document.getElementById("tab-summary")
);

// Live streaming state — module-level so the focus-recovery listener can
// re-touch the text node and force Chrome to repaint throttled updates, AND
// so renderHistory() can re-attach a still-running turn's bubble when the
// user tabs back to the job it belongs to (see the ownership note up top).
/** @type {Text | null} */
let _liveTextNode = null;
let _liveAcc = "";
// The job the in-flight turn belongs to — independent of `chatJobId` (which
// job is on screen). The two agree while the turn owns the DOM and diverge
// the moment the user switches away.
/** @type {string | null} */
let _liveTurnJobId = null;
/** @type {HTMLElement | null} */
let _liveBubbleEl = null;
/** @type {HTMLElement | null} */
let _liveWrapEl = null;

/** Whether `jobId` is the job currently on screen — i.e. allowed to touch #chat-messages. */
function _ownsChatDom(jobId) {
  return jobId != null && jobId === chatJobId;
}

/** Toggle the pulsing-dot indicator on the Summary tab button. */
function _setQaActive(active) {
  _summaryTabBtn?.classList.toggle("tab--qa-active", active);
}

// When the sidepanel regains window focus or document visibility, Chrome's
// paint throttle lifts. Re-set the text node data to flush any accumulated
// tokens that were written but not painted while the panel was backgrounded.
// Guarded by ownership: if the user switched away from the turn's job,
// `_liveTextNode` still points at a node that renderHistory already ripped
// out of the document (or that belongs to a job no longer displayed) —
// touching it would be a silent no-op at best and is exactly the kind of
// write into detached DOM this whole fix exists to prevent.
function _repaintIfStreaming() {
  if (_liveTextNode !== null && _ownsChatDom(_liveTurnJobId)) {
    _liveTextNode.data = _liveAcc; // re-assign triggers a repaint
  }
}
window.addEventListener("focus", _repaintIfStreaming);
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") _repaintIfStreaming();
});

// Auto-grow the composer as the user types: it rests at one row (matching
// the old single-line <input>'s height) and grows up to CHAT_INPUT_MAX_ROWS,
// beyond which it scrolls internally instead of growing further. Called from
// the "input" event below and after every programmatic value clear so the
// box collapses back to one row rather than staying tall.
const CHAT_INPUT_MAX_ROWS = 6;
function _autoGrowChatInput() {
  if (!input) return;
  input.style.height = "auto"; // shrink first so scrollHeight reflects only the current content
  const cs = getComputedStyle(input);
  const lineHeight = parseFloat(cs.lineHeight) || 18;
  const vertical =
    parseFloat(cs.borderTopWidth) +
    parseFloat(cs.borderBottomWidth) +
    parseFloat(cs.paddingTop) +
    parseFloat(cs.paddingBottom);
  const maxHeight = lineHeight * CHAT_INPUT_MAX_ROWS + vertical;
  const next = Math.min(input.scrollHeight, maxHeight);
  input.style.height = `${next}px`;
  input.style.overflowY = input.scrollHeight > maxHeight ? "auto" : "hidden";
}
input?.addEventListener("input", _autoGrowChatInput);

// Enter submits the question; Shift+Enter inserts a newline (the textarea's
// default behaviour, left untouched). isComposing / keyCode 229 guards
// against submitting mid-keystroke while an IME (e.g. Japanese, Chinese,
// Korean input) is composing a word. requestSubmit() runs the real "submit"
// listener below (with its own near-simultaneous-submit guard) rather than
// bypassing it.
input?.addEventListener("keydown", (ev) => {
  if (ev.key !== "Enter" || ev.shiftKey) return;
  if (ev.isComposing || ev.keyCode === 229) return;
  ev.preventDefault();
  form?.requestSubmit();
});

if (form) {
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = input?.value.trim() || "";
    if (!text) return;
    handleAsk(text).catch((err) => console.error("[TLDR] ask failed", err));
  });
}

// Question queue — per job. Every submitted question becomes an entry
// {id, jobId, text} and its user bubble shows at once (marked pending). One
// turn runs at a time; the drain loop picks the oldest entry whose job is
// `done` (daemon QA 409s otherwise) and merges ONLY that job's entries
// accumulated meanwhile into one turn — questions for other jobs run as
// their own turns with their own job id. Entries for a job still being
// summarised wait until app.js reports it done (noteJobStatus); if it fails
// they are dropped with an error on their bubbles. Pending entries live
// only in this panel's memory (not in the DB yet), so renderHistory
// re-appends them for their job.
let qaInFlight = false;
let _pendingSeq = 0;
/** @type {{id: number, jobId: string, text: string}[]} */
let pendingQuestions = [];
/** Last known status per job id. @type {Map<string, string>} */
const _jobStatus = new Map();
/** Questions dropped because their job failed — shown with an error on re-render. @type {Map<string, {text: string, error: string}[]>} */
const _failedQuestions = new Map();

/**
 * Record a job's status (app.js calls this for every job event / stream
 * done|error of ANY job, so queued questions get sent even after the user
 * switched away). `done` drains, `failed` drops that job's queue.
 *
 * @param {string} jobId
 * @param {string | null | undefined} status
 * @param {string | null} [error]
 */
export function noteJobStatus(jobId, status, error) {
  _noteStatus(jobId, status, error);
}

/** @param {string} jobId @param {string | null | undefined} status @param {string | null} [error] */
function _noteStatus(jobId, status, error) {
  if (!status) return;
  const prev = _jobStatus.get(jobId);
  _jobStatus.set(jobId, status);
  if (status === prev) return;
  if (status === "done") {
    _drain().catch((err) => console.error("[TLDR] qa drain failed", err));
  } else if (status === "failed") {
    _failQueued(jobId, error || "The summary failed, so this question wasn't sent.");
  }
}

/** @param {string} jobId @param {string} error */
function _failQueued(jobId, error) {
  const dropped = pendingQuestions.filter((q) => q.jobId === jobId);
  if (!dropped.length) return;
  pendingQuestions = pendingQuestions.filter((q) => q.jobId !== jobId);
  const list = _failedQuestions.get(jobId) || [];
  for (const q of dropped) list.push({ text: q.text, error });
  _failedQuestions.set(jobId, list);
  if (!_ownsChatDom(jobId) || !messages) return;
  for (const q of dropped) {
    const el = messages.querySelector(`[data-pending-id="${q.id}"]`);
    if (el) _markBubbleFailed(/** @type {HTMLElement} */ (el), error);
  }
}

/** @param {HTMLElement} wrap @param {string} error */
function _markBubbleFailed(wrap, error) {
  wrap.classList.remove("chat-bubble--pending");
  wrap.classList.add("chat-bubble--failed");
  wrap.removeAttribute("data-pending-id");
  const note = document.createElement("div");
  note.className = "chat-bubble-note error";
  note.textContent = error;
  wrap.appendChild(note);
}

/** @param {string} question */
async function handleAsk(question) {
  // The job on screen in THIS panel — never a stored/global id.
  const jobId = chatJobId;
  if (!jobId) return;
  const entry = { id: ++_pendingSeq, jobId, text: question };
  pendingQuestions.push(entry);

  // Show the user's bubble and clear the input immediately. Keeping focus
  // lets the user keep typing follow-ups without a click. Submitting means
  // "I want to see the answer": follow the end again.
  const bubble = appendBubble("user", question);
  _markPending(bubble, entry.id);
  setFollowing(true);
  scrollMessagesToEnd();
  if (input) {
    input.value = "";
    _autoGrowChatInput();
    input.focus();
  }
  if (_draftWriteTimer !== null) {
    clearTimeout(_draftWriteTimer);
    _draftWriteTimer = null;
  }
  _persistDraft(jobId, "");

  if (_jobStatus.get(jobId) !== "done") {
    // Summary still running: wait for noteJobStatus("done"). Re-check once
    // in case the done event was missed (throttled panel).
    daemon.getJob(jobId).then((j) => _noteStatus(jobId, j.status, j.error)).catch(() => {});
    return;
  }
  await _drain();
}

/** Run queued turns until nothing sendable is left. One turn at a time. */
async function _drain() {
  if (qaInFlight) return;
  // Claimed synchronously (no await between check and set) so two
  // near-simultaneous calls can't both fire requests.
  qaInFlight = true;
  try {
    for (;;) {
      const head = pendingQuestions.find((q) => _jobStatus.get(q.jobId) === "done");
      if (!head) break;
      const batch = pendingQuestions.filter((q) => q.jobId === head.jobId);
      pendingQuestions = pendingQuestions.filter((q) => q.jobId !== head.jobId);
      if (messages && _ownsChatDom(head.jobId)) {
        for (const q of batch) {
          const el = messages.querySelector(`[data-pending-id="${q.id}"]`);
          if (el) {
            el.classList.remove("chat-bubble--pending");
            el.removeAttribute("data-pending-id");
          }
        }
      }
      await _runQaTurn(head.jobId, batch.map((q) => q.text).join("\n\n"));
    }
  } finally {
    qaInFlight = false;
  }
}

/** @param {HTMLElement} inner bubble inner @param {number} id */
function _markPending(inner, id) {
  const wrap = /** @type {HTMLElement | null} */ (inner.closest(".chat-bubble"));
  if (!wrap) return;
  wrap.classList.add("chat-bubble--pending");
  wrap.dataset.pendingId = String(id);
}

/**
 * (Re)build the DOM for the in-flight turn tracked by `_liveTurnJobId` and
 * point `_liveBubbleEl` / `_liveWrapEl` / `_liveTextNode` at it. Called from
 * two places: `_runQaTurn`, when a turn starts while its job already owns
 * the chat list, and `renderHistory`, when the user tabs back to a job that
 * has a turn still running. Both call sites have already verified
 * `chatJobId === _liveTurnJobId` — this function trusts that and does not
 * re-check ownership itself.
 *
 * Shows a spinner if no tokens have arrived yet, or the raw accumulated text
 * (`_liveAcc`) if the stream started/continued while the job was off screen
 * — otherwise reattaching would show a spinner even though the answer is
 * half-typed already.
 */
function _attachLiveTurnBubble() {
  const bubble = appendBubble("assistant", "");
  _liveBubbleEl = bubble;
  _liveWrapEl = /** @type {HTMLElement | null} */ (bubble.closest(".chat-bubble"));
  if (_liveAcc) {
    bubble.classList.add("chat-bubble-inner--streaming");
    _liveTextNode = document.createTextNode(_liveAcc);
    bubble.appendChild(_liveTextNode);
  } else {
    bubble.innerHTML =
      `<span class="thinking-dots"><span></span><span></span><span></span></span>`;
    _liveTextNode = null;
  }
  // Activate the pulsing-dot on the Summary tab button. Visible from any
  // pane — critical because the thinking-dots spinner inside #pane-summary
  // is invisible (and its CSS animation paused) when the Transcript tab is active.
  _setQaActive(true);
  scrollMessagesToEnd();
}

/** Reset the in-flight-turn bookkeeping once a turn finishes (done/error/thrown). */
function _clearLiveTurnState() {
  _liveTurnJobId = null;
  _liveBubbleEl = null;
  _liveWrapEl = null;
  _liveTextNode = null;
  _liveAcc = "";
}

/**
 * Run one Q&A turn end-to-end. Caller owns `qaInFlight` and the drain loop.
 *
 * The request always runs to completion regardless of what's on screen —
 * the daemon needs to generate and persist the answer under `jobId` no
 * matter which job the user is looking at. Only the DOM writes are
 * conditional on `_ownsChatDom(jobId)`, re-checked at every single one
 * (not just once up front) because ownership can flip mid-stream in either
 * direction — see the ownership note at the top of this file.
 *
 * @param {string} jobId
 * @param {string} question
 */
async function _runQaTurn(jobId, question) {
  // Reset live-turn state for this new turn.
  _liveTurnJobId = jobId;
  _liveBubbleEl = null;
  _liveWrapEl = null;
  _liveTextNode = null;
  _liveAcc = "";
  // Collected from a "frames" event, if the LOOK step found any moment
  // actually relevant to the question (see api-types.js AIFramesEvent).
  /** @type {FrameRef[]} */
  let frameRefs = [];

  // Create the assistant bubble only while this job's chat is the one on
  // screen. If it isn't, the turn still runs (below) — it just doesn't paint
  // anything until/unless renderHistory() re-attaches it (see that function).
  if (_ownsChatDom(jobId)) {
    _attachLiveTurnBubble();
  }

  try {
    for await (const ev of daemon.aiQa({ job_id: jobId, question })) {
      if (ev.type === "stage") {
        // Stage events (e.g. "thinking") arrive before first delta — the
        // pulsing dot already covers the "in progress" signal; no extra UI needed.
      } else if (ev.type === "frames") {
        frameRefs = ev.items || [];
      } else if (ev.type === "delta") {
        _liveAcc += ev.delta;
        if (!_ownsChatDom(jobId)) continue; // request keeps streaming, just not painting
        if (_liveTextNode === null && _liveBubbleEl) {
          // First token — replace spinner with streaming text. While this
          // text node is live the bubble holds RAW markdown, not rendered
          // HTML, so it needs pre-wrap to keep the model's line breaks; the
          // class is dropped again once "done" swaps in real markup.
          _liveBubbleEl.innerHTML = "";
          _liveBubbleEl.classList.add("chat-bubble-inner--streaming");
          _liveTextNode = document.createTextNode("");
          _liveBubbleEl.appendChild(_liveTextNode);
        }
        if (_liveTextNode) {
          _liveTextNode.data = _liveAcc;
          scrollMessagesToEnd();
        }
      } else if (ev.type === "done") {
        const final = ev.content || "";
        if (_ownsChatDom(jobId) && _liveBubbleEl) {
          _liveBubbleEl.classList.remove("chat-bubble-inner--streaming");
          // Render WITH timecode links — the QA prompt now ensures [MM:SS]
          // markers only appear when the answer came from the material, so any
          // marker the LLM emits is a real jump target (not a web_search hallucination).
          const job = activeJob?.id === jobId ? activeJob : null;
          _liveBubbleEl.innerHTML = renderMarkdown(final, job);
          _setQaActive(false);
          // No jump to the answer's start: a reader who scrolled away stays
          // put; a following reader just stays at the bottom.
          const wrap = _liveWrapEl;
          if (frameRefs.length && wrap) {
            // Clear the live state before awaiting so a renderHistory during
            // the await doesn't re-attach a finished turn.
            _clearLiveTurnState();
            await _appendFrameRow(wrap, frameRefs, job, jobId);
            return;
          }
          scrollMessagesToEnd();
        }
        _clearLiveTurnState();
        return;
      } else if (ev.type === "error") {
        if (_ownsChatDom(jobId) && _liveBubbleEl) {
          _liveBubbleEl.classList.remove("chat-bubble-inner--streaming");
          _setQaActive(false);
          renderErrorBubble(_liveBubbleEl, ev.error || "Error.");
        }
        _clearLiveTurnState();
        return;
      }
    }
  } catch (err) {
    console.error("[TLDR] aiStream qa failed", err);
    if (_ownsChatDom(jobId) && _liveBubbleEl) {
      _liveBubbleEl.classList.remove("chat-bubble-inner--streaming");
      _setQaActive(false);
      renderErrorBubble(_liveBubbleEl, err instanceof Error ? err.message : String(err));
    }
    _clearLiveTurnState();
  }
}

// ---------------------------------------------------------------------------
// History (called by app.js on job switch).
// ---------------------------------------------------------------------------

/**
 * Replace the current chat list with the persisted history for a job.
 * Built into a DocumentFragment so we hit the DOM once — long histories
 * (dozens of bubbles) would otherwise thrash layout per-append.
 *
 * Renders frame thumbnails (via `_appendFrameRow`) for any assistant
 * message that has `frame_refs` — the exact same helper the live-stream
 * path (`_runQaTurn`) uses, so a reloaded turn looks identical to when it
 * first streamed in.
 *
 * `jobId` must be the job this history was fetched *for* (app.js passes the
 * id it called `daemon.listMessages(jobId)` with), not read off `chatJobId`
 * at call time. app.js can have two `loadHistory` calls racing (one from the
 * `set-active-tab` message, one from `chrome.storage.onChanged`), and if
 * they land out of order the older response must not overwrite a newer
 * job's chat — hence the ownership check below, same rule as `_runQaTurn`.
 *
 * @param {string} jobId
 * @param {ChatMessage[]} items
 * @returns {Promise<void>}
 */
export async function renderHistory(jobId, items) {
  if (!messages) return;
  if (jobId !== chatJobId) return; // stale response — a different job is on screen now
  const job = activeJob?.id === jobId ? activeJob : null;
  messages.innerHTML = "";
  const frag = document.createDocumentFragment();
  /** @type {Promise<void>[]} */
  const framePromises = [];
  for (const m of items) {
    const bubble = appendBubble(m.role, "", frag);
    if (m.role === "assistant") {
      // Timecode links enabled — qa.txt prompt now guards against hallucinated
      // timestamps from web_search, so any [MM:SS] in stored answers is a
      // genuine material reference worth making clickable.
      bubble.innerHTML = renderMarkdown(m.content, job);
      if (m.frame_refs && m.frame_refs.length) {
        const wrap = /** @type {HTMLElement | null} */ (bubble.closest(".chat-bubble"));
        framePromises.push(_appendFrameRow(wrap, m.frame_refs, job, jobId));
      }
    } else {
      bubble.textContent = m.content;
    }
  }
  messages.appendChild(frag);
  await Promise.all(framePromises);
  // Ownership may have flipped during the await — a newer job's chat now
  // owns the list; don't append this job's live/pending bubbles to it.
  if (jobId !== chatJobId) return;

  // If this job has a turn still streaming (the user asked a question, tabbed
  // away before it finished, and just tabbed back), re-attach its bubble at
  // the end instead of leaving a dead gap until the stream happens to finish
  // — see `_attachLiveTurnBubble`. Otherwise just settle the scroll position.
  if (_liveTurnJobId === jobId) {
    _attachLiveTurnBubble();
  } else {
    scrollMessagesToEnd();
  }
  // Questions dropped because the job failed, then those still waiting
  // (not in the DB yet) — they must not vanish on re-render.
  for (const f of _failedQuestions.get(jobId) || []) {
    const inner = appendBubble("user", f.text);
    const wrap = /** @type {HTMLElement | null} */ (inner.closest(".chat-bubble"));
    if (wrap) _markBubbleFailed(wrap, f.error);
  }
  for (const q of pendingQuestions) {
    if (q.jobId === jobId) _markPending(appendBubble("user", q.text), q.id);
  }
  scrollMessagesToEnd();
}

/** Wipe all bubbles (called on tab-changed → no-job placeholder). */
export function clearChat() {
  if (messages) messages.innerHTML = "";
  if (chatJobId !== null) _swapDraft(chatJobId, null);
  chatJobId = null;
}

// ---------------------------------------------------------------------------
// Bubble helpers
// ---------------------------------------------------------------------------

/**
 * Append a chat bubble. Caller is responsible for scrolling (so batch
 * inserts in `renderHistory` don't trigger per-bubble layout).
 *
 * @param {"user" | "assistant"} who
 * @param {string} text
 * @param {Node} [container] target node; defaults to the live messages list
 * @returns {HTMLElement}
 */
function appendBubble(who, text, container) {
  const target = container || messages;
  if (!target) {
    const span = document.createElement("span");
    span.textContent = text;
    return span;
  }
  const wrap = document.createElement("div");
  wrap.className = `chat-bubble chat-bubble--${who}`;
  const inner = document.createElement("div");
  // Assistant bubbles get rendered markdown (see _runQaTurn / renderHistory,
  // both of which set .innerHTML via renderMarkdown()), so they need the
  // same .markdown-body rules the summary uses (headings, lists, code
  // blocks with overflow-x: auto, etc.) — otherwise those tags fall back to
  // unstyled UA defaults and a fenced code block overflows the bubble.
  // User bubbles stay plain text (set via textContent below) and must NOT
  // get this class.
  inner.className = who === "assistant" ? "chat-bubble-inner markdown-body" : "chat-bubble-inner";
  inner.textContent = text;
  wrap.appendChild(inner);
  target.appendChild(wrap);
  return inner;
}

// ---------------------------------------------------------------------------
// Frame thumbnails (LOOK-step visual findings — see api-types.js FrameRef)
// ---------------------------------------------------------------------------

/**
 * Insert a small thumbnail row right after `bubbleEl` (a `.chat-bubble`
 * wrapper) — one entry per `FrameRef`: image + `[MM:SS]` caption + phrase,
 * all on one quiet row. Row-building itself lives in
 * `lib/frame-thumbnails.js` (shared with the summary "look" affordance in
 * app.js) — this wrapper only owns bubble placement + fail-soft error
 * handling.
 *
 * Shared by both the live-stream path (`_runQaTurn`, on the "frames" SSE
 * event) and the history-reload path (`renderHistory`) so a reloaded turn
 * renders identically to when it first streamed in.
 *
 * @param {HTMLElement | null} bubbleEl
 * @param {FrameRef[] | null | undefined} frameRefs
 * @param {JobDetails | null} job the job these frames belong to (passed
 *   explicitly — `activeJob` may be another job by the time the await ends)
 * @param {string} jobId
 * @returns {Promise<void>}
 */
async function _appendFrameRow(bubbleEl, frameRefs, job, jobId) {
  if (!bubbleEl || !frameRefs || frameRefs.length === 0) return;
  // Fails soft, deliberately: this renders on top of an already-finished
  // answer bubble (live path) or an already-built history list (renderHistory,
  // whose callers never await it — see app.js's loadHistory). A thumbnail is
  // a nice-to-have; a rejected chrome.storage.local read (daemon.baseUrl())
  // or any other hiccup here must never surface as an unhandled promise
  // rejection, and must never take the rendered text down with it.
  try {
    const base = await daemon.baseUrl();
    if (!_ownsChatDom(jobId) || !bubbleEl.isConnected) return;
    const row = buildFrameRow(job, frameRefs, base);
    bubbleEl.insertAdjacentElement("afterend", row);
    scrollMessagesToEnd();
  } catch (err) {
    console.warn("[TLDR] failed to render frame thumbnails", err);
  }
}

/**
 * @param {HTMLElement} bubble
 * @param {string} message
 */
function renderErrorBubble(bubble, message) {
  bubble.innerHTML = "";
  const span = document.createElement("span");
  span.className = "error";
  span.textContent = message;
  bubble.appendChild(span);
}

function scrollMessagesToEnd() {
  // Single page-level scroll, "follow" mode (lib/scroll-follow.js): only
  // while the reader is at the bottom; any upward user scroll stops it.
  followScrollToEnd();
}
