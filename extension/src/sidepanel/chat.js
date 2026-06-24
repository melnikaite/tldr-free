// Chat panel — Q&A scoped to the currently active job.
//
// Sends each question through POST /ai/stream {job_id, question}. The
// daemon persists user + assistant messages in SQLite, so chat history
// survives tab switches, browser restarts, and side-panel close. On job
// switch, the side panel calls renderHistory(items) below to redraw the
// stored bubbles before any new turn.
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
//   - On answer done, scroll the bubble start into view (not the end) so the
//     user reads from the top, not the bottom of a long response.

import { daemon } from "../lib/daemon-client.js";
import { renderMarkdown } from "../lib/markdown.js";

/** @import { ChatMessage, JobDetails } from "../lib/api-types.js" */

/** @type {JobDetails | null} */
let activeJob = null;

/** @param {JobDetails | null} job */
export function setActiveJob(job) {
  activeJob = job;
}

/** @returns {Promise<JobDetails | null>} */
export async function getActiveJob() {
  if (activeJob) return activeJob;
  const { activeJobId } = await chrome.storage.session.get("activeJobId");
  if (!activeJobId) return null;
  try {
    activeJob = await daemon.getJob(activeJobId);
    return activeJob;
  } catch {
    return null;
  }
}

const form = /** @type {HTMLFormElement | null} */ (document.getElementById("chat-form"));
const input = /** @type {HTMLInputElement | null} */ (document.getElementById("chat-input"));
const messages = /** @type {HTMLElement | null} */ (document.getElementById("chat-messages"));

// Summary tab button — we toggle .tab--qa-active on it while QA is in flight.
// This adds a pulsing dot via CSS that is visible regardless of which pane is
// active (the tab nav is always rendered above both panes).
// We deliberately do NOT touch #stage-badge (owned by app.js / setStage) to
// avoid the conflict where any SSE job-update clears our indicator.
const _summaryTabBtn = /** @type {HTMLButtonElement | null} */ (
  document.getElementById("tab-summary")
);

// Summary pane element — consulted at answer-done time to decide whether to
// scroll to the bubble start (only when the pane is actually visible).
const _summaryPaneEl = /** @type {HTMLElement | null} */ (
  document.getElementById("pane-summary")
);

// Live streaming state — module-level so the focus-recovery listener can
// re-touch the text node and force Chrome to repaint throttled updates.
/** @type {Text | null} */
let _liveTextNode = null;
let _liveAcc = "";

/** Toggle the pulsing-dot indicator on the Summary tab button. */
function _setQaActive(active) {
  _summaryTabBtn?.classList.toggle("tab--qa-active", active);
}

// When the sidepanel regains window focus or document visibility, Chrome's
// paint throttle lifts. Re-set the text node data to flush any accumulated
// tokens that were written but not painted while the panel was backgrounded.
function _repaintIfStreaming() {
  if (_liveTextNode !== null) {
    _liveTextNode.data = _liveAcc; // re-assign triggers a repaint
  }
}
window.addEventListener("focus", _repaintIfStreaming);
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") _repaintIfStreaming();
});

if (form) {
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = input?.value.trim() || "";
    if (!text) return;
    handleAsk(text).catch((err) => console.error("[TLDR] ask failed", err));
  });
}

// While a Q&A stream is running, additional questions don't block the input
// or open a second request — they queue up here and get concatenated into a
// single follow-up turn when the current one finishes. Deliberately simple:
// no per-question ordering guarantees, no parallel requests, no UI for the
// pending state beyond the user bubbles already on screen.
let qaInFlight = false;
/** @type {string[]} */
let pendingQuestions = [];

/** @param {string} question */
async function handleAsk(question) {
  const { activeJobId } = await chrome.storage.session.get("activeJobId");
  if (!activeJobId) return;

  // Show the user's bubble and clear the input immediately, regardless of
  // whether we're already streaming. Keeping focus lets the user keep typing
  // follow-ups without a click.
  appendBubble("user", question);
  scrollMessagesToEnd();
  if (input) {
    input.value = "";
    input.focus();
  }

  if (qaInFlight) {
    // Another turn is in flight — stash this one. It'll be picked up (joined
    // with anything else that piled up) by the drain loop below.
    pendingQuestions.push(question);
    return;
  }
  // Claim the in-flight slot synchronously (no `await` between the check and
  // this assignment) so two near-simultaneous submits can't both decide
  // they're the first turn and fire parallel requests.
  qaInFlight = true;

  if (!activeJob) {
    // Fill once if app.js hasn't published the job yet — never overwrite a
    // freshly-set job from app.js (which may have a non-null `video_id` that
    // the daemon hasn't echoed back yet).
    try {
      activeJob = await daemon.getJob(activeJobId);
    } catch {
      // Continue without — the daemon will 404 if the job is gone.
    }
  }

  // Drain loop: run the user's turn, then keep running merged follow-ups
  // until the queue is empty. Iterative — no recursion in `finally`, so the
  // stack stays flat and any error in a follow-up surfaces here, not as a
  // swallowed `.catch(console.error)`.
  try {
    let next = question;
    while (next) {
      await _runQaTurn(activeJobId, next);
      if (pendingQuestions.length === 0) break;
      next = pendingQuestions.join("\n\n");
      pendingQuestions = [];
    }
  } finally {
    qaInFlight = false;
  }
}

/**
 * Run one Q&A turn end-to-end. Caller owns `qaInFlight` and the drain loop.
 *
 * @param {string} jobId
 * @param {string} question
 */
async function _runQaTurn(jobId, question) {
  const assistantBubble = appendBubble("assistant", "");
  const assistantWrap = /** @type {HTMLElement | null} */ (
    assistantBubble.closest(".chat-bubble")
  );
  assistantBubble.innerHTML =
    `<span class="thinking-dots"><span></span><span></span><span></span></span>`;
  scrollMessagesToEnd();

  // Activate the pulsing-dot on the Summary tab button. Visible from any
  // pane — critical because the thinking-dots spinner inside #pane-summary
  // is invisible (and its CSS animation paused) when the Transcript tab is active.
  _setQaActive(true);
  // Reset module-level streaming state.
  _liveTextNode = null;
  _liveAcc = "";

  try {
    for await (const ev of daemon.aiQa({ job_id: jobId, question })) {
      if (ev.type === "stage") {
        // Stage events (e.g. "thinking") arrive before first delta — the
        // pulsing dot already covers the "in progress" signal; no extra UI needed.
      } else if (ev.type === "delta") {
        if (_liveTextNode === null) {
          // First token — replace spinner with streaming text.
          assistantBubble.innerHTML = "";
          _liveTextNode = document.createTextNode("");
          assistantBubble.appendChild(_liveTextNode);
        }
        _liveAcc += ev.delta;
        _liveTextNode.data = _liveAcc;
        scrollMessagesToEnd();
      } else if (ev.type === "done") {
        _liveTextNode = null;
        _liveAcc = "";
        const final = ev.content || "";
        // Render WITH timecode links — the QA prompt now ensures [MM:SS]
        // markers only appear when the answer came from the material, so any
        // marker the LLM emits is a real jump target (not a web_search hallucination).
        assistantBubble.innerHTML = renderMarkdown(final, activeJob);
        _setQaActive(false);
        // Scroll to the START of the assistant bubble so the user reads from
        // the top, not the bottom of a potentially long answer. Only scroll
        // when Summary pane is visible — don't yank the user away from Transcript.
        if (_summaryPaneEl?.classList.contains("tab-pane--active") && assistantWrap) {
          assistantWrap.scrollIntoView({ block: "start", behavior: "smooth" });
        }
        return;
      } else if (ev.type === "error") {
        _liveTextNode = null;
        _liveAcc = "";
        _setQaActive(false);
        renderErrorBubble(assistantBubble, ev.error || "Error.");
        return;
      }
    }
  } catch (err) {
    _liveTextNode = null;
    _liveAcc = "";
    _setQaActive(false);
    console.error("[TLDR] aiStream qa failed", err);
    renderErrorBubble(assistantBubble, err instanceof Error ? err.message : String(err));
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
 * @param {ChatMessage[]} items
 */
export function renderHistory(items) {
  if (!messages) return;
  messages.innerHTML = "";
  const frag = document.createDocumentFragment();
  for (const m of items) {
    const bubble = appendBubble(m.role, "", frag);
    if (m.role === "assistant") {
      // Timecode links enabled — qa.txt prompt now guards against hallucinated
      // timestamps from web_search, so any [MM:SS] in stored answers is a
      // genuine material reference worth making clickable.
      bubble.innerHTML = renderMarkdown(m.content, activeJob);
    } else {
      bubble.textContent = m.content;
    }
  }
  messages.appendChild(frag);
  scrollMessagesToEnd();
}

/** Wipe all bubbles (called on tab-changed → no-job placeholder). */
export function clearChat() {
  if (messages) messages.innerHTML = "";
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
  inner.className = "chat-bubble-inner";
  inner.textContent = text;
  wrap.appendChild(inner);
  target.appendChild(wrap);
  return inner;
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
  // Single page-level scroll now. Auto-stick to the bottom only if the user
  // is already near it — otherwise they're reading the summary or earlier
  // history and shouldn't be yanked.
  const root = document.scrollingElement || document.documentElement;
  const distFromBottom = root.scrollHeight - root.scrollTop - window.innerHeight;
  if (distFromBottom < 120) {
    root.scrollTop = root.scrollHeight;
  }
}
