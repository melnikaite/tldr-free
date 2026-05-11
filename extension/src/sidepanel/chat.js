// Chat panel — Q&A scoped to the currently active job.
//
// Sends each question through POST /ai/stream {job_id, question}. The
// daemon persists user + assistant messages in SQLite, so chat history
// survives tab switches, browser restarts, and side-panel close. On job
// switch, the side panel calls renderHistory(items) below to redraw the
// stored bubbles before any new turn.
//
// Token streaming: append plain text to the assistant bubble as it arrives,
// then re-render via lib/markdown.js once the stream ends so [MM:SS] markers
// become clickable YouTube links.

import { daemon } from "../lib/daemon-client.js";
import { renderMarkdown } from "../lib/markdown.js";
import { resolveVideoId } from "../lib/url.js";

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
  assistantBubble.innerHTML =
    `<span class="thinking-dots"><span></span><span></span><span></span></span>`;
  scrollMessagesToEnd();

  /** @type {Text | null} */
  let textNode = null;
  let acc = "";
  try {
    for await (const ev of daemon.aiQa({ job_id: jobId, question })) {
      if (ev.type === "stage") {
        // Stage signals (e.g. "thinking") arrive before the first delta;
        // the dots indicator already covers that, no extra UI needed.
      } else if (ev.type === "delta") {
        if (textNode === null) {
          assistantBubble.innerHTML = "";
          textNode = document.createTextNode("");
          assistantBubble.appendChild(textNode);
        }
        acc += ev.delta;
        textNode.data = acc;
        scrollMessagesToEnd();
      } else if (ev.type === "done") {
        const final = ev.content || acc;
        const videoId = resolveVideoId(activeJob);
        assistantBubble.innerHTML = renderMarkdown(final, videoId);
        scrollMessagesToEnd();
      } else if (ev.type === "error") {
        renderErrorBubble(assistantBubble, ev.error || "Error.");
        return;
      }
    }
  } catch (err) {
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
  const videoId = resolveVideoId(activeJob);
  for (const m of items) {
    const bubble = appendBubble(m.role, "", frag);
    if (m.role === "assistant") {
      bubble.innerHTML = renderMarkdown(m.content, videoId);
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
