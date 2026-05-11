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

/** @param {string} question */
async function handleAsk(question) {
  const { activeJobId } = await chrome.storage.session.get("activeJobId");
  if (!activeJobId) return;

  if (!activeJob || activeJob.id !== activeJobId) {
    try {
      activeJob = await daemon.getJob(activeJobId);
    } catch {
      // Continue without — the daemon will 404 if the job is gone.
    }
  }

  appendBubble("user", question);
  if (input) {
    input.value = "";
    input.disabled = true;
  }

  const assistantBubble = appendBubble("assistant", "");
  assistantBubble.innerHTML =
    `<span class="thinking-dots"><span></span><span></span><span></span></span>`;
  scrollMessagesToEnd();

  /** @type {Text | null} */
  let textNode = null;
  let acc = "";
  try {
    for await (const ev of daemon.aiQa({ job_id: activeJobId, question })) {
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
        const videoId = activeJob?.video_id || null;
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
  } finally {
    if (input) {
      input.disabled = false;
      input.focus();
    }
  }
}

// ---------------------------------------------------------------------------
// History (called by app.js on job switch).
// ---------------------------------------------------------------------------

/**
 * Replace the current chat list with the persisted history for a job.
 * @param {ChatMessage[]} items
 */
export function renderHistory(items) {
  if (!messages) return;
  messages.innerHTML = "";
  for (const m of items) {
    const bubble = appendBubble(m.role, "");
    if (m.role === "assistant") {
      const videoId = activeJob?.video_id || null;
      bubble.innerHTML = renderMarkdown(m.content, videoId);
    } else {
      bubble.textContent = m.content;
    }
  }
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
 * @param {"user" | "assistant"} who
 * @param {string} text
 * @returns {HTMLElement}
 */
function appendBubble(who, text) {
  if (!messages) {
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
  messages.appendChild(wrap);
  scrollMessagesToEnd();
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
