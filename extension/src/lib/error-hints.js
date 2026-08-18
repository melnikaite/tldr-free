// Turns raw backend text into something a human can act on. Two unrelated
// signals come through here:
//
//   - classifyError(rawMessage, health) — job.error / a stringified
//     fetch-thrown error, for the sidepanel's "error" render state.
//   - describeQueuedDetail(detail) — a live "stage" event's `detail` when
//     `stage === "queued"`, for the sidepanel's "streaming" render state
//     (the job is PARKED, not dead — see that function's docstring for
//     why this can never be folded into classifyError).
//
// See extension.md's error-hint section (if you add an invariant here,
// mirror it there).
//
// Deliberately framework/DOM-free: no `chrome.*`, no innerHTML, nothing
// that would stop this from running under plain `node --check` or being
// unit-tested with a list of real strings. The sidepanel owns turning the
// `action` descriptor into an actual button.
//
// Classification is ALWAYS by message content, never by inventing meaning
// that isn't there — see `classifyError`'s final `return null`: an
// unrecognised message is shown as-is (the caller's existing raw-text
// fallback), never with a guessed diagnosis. A wrong diagnosis is worse
// than no diagnosis (see the task this module was written for).

/**
 * @typedef {object} ErrorHint
 * @property {string} title
 * @property {string} explanation
 * @property {{ label: string, kind: "open-options" } | null} action
 */

/**
 * @typedef {object} HealthLike
 * @property {boolean} [llm_backend_reachable]
 * @property {string | null} [llm_backend_error]
 */

// ---------------------------------------------------------------------------
// Pattern 1 — the daemon itself is unreachable. This is what a browser
// `fetch()` throws when nothing is listening on the configured port at all
// (daemon not started, wrong port, etc.) — distinct from a daemon-side
// error string, which always carries a daemon-generated prefix like
// "summarization failed: …" (see daemon/src/workers/pipeline.py). Chrome,
// Firefox and bare Node/undici each phrase a dead-socket fetch differently,
// so this matches all three observed phrasings.
// ---------------------------------------------------------------------------
const DAEMON_UNREACHABLE_RE =
  /failed to fetch|networkerror when attempting to fetch|load failed|fetch failed|err_connection_refused/i;

// ---------------------------------------------------------------------------
// Pattern 2 — context/token-size overflow. Mirrors
// `daemon/src/api/config.py::_looks_like_context_overflow` exactly: a
// size complaint mentions the SUBJECT (context/tokens) and an OVERFLOW
// verb (exceeds/too many/maximum/limit), independent of status code or
// exact wording — backends phrase this differently ("request (72196
// tokens) exceeds the available context size (32768 tokens)", "maximum
// context length is N tokens", "context window exceeded", …).
// ---------------------------------------------------------------------------
const CONTEXT_OVERFLOW_SUBJECT_RE = /\bcontext\b|\btokens?\b/i;
const CONTEXT_OVERFLOW_VERB_RE = /exceed\w*|too (?:many|long|large)|maximum|\blimit\b/i;

// ---------------------------------------------------------------------------
// Pattern 3 — model not found. Every OpenAI-compatible backend spells this
// differently ("The model `x` does not exist", Ollama's "model 'x' not
// found, try pulling it first", …) but all of them say "model" plus some
// flavour of "doesn't exist / wasn't found".
// ---------------------------------------------------------------------------
const MODEL_SUBJECT_RE = /\bmodel\b/i;
const MODEL_MISSING_RE =
  /not found|does not exist|no such model|unknown model|not available|try pulling/i;

// ---------------------------------------------------------------------------
// Pattern 4 — auth rejection from a cloud backend. A bare "401"/"403"
// (word-bounded, so it never matches inside an unrelated number like
// "34012") is strong enough evidence on its own; the keyword alternation
// catches phrasings that dropped the numeric code (redacted proxies, etc).
// ---------------------------------------------------------------------------
const AUTH_CODE_RE = /\b401\b|\b403\b/;
const AUTH_KEYWORD_RE = /unauthor|forbidden|incorrect api key|invalid api key|invalid credentials/i;

// ---------------------------------------------------------------------------
// Pattern 5 — stream stalled. Matches both the daemon-side per-chunk
// timeout (src/llm/client.py: "llm stream stalled: no chunk for {N}s") and
// the extension-side SSE watchdog (lib/daemon-client.js: "SSE stream
// stalled (no chunk for {N}ms)").
// ---------------------------------------------------------------------------
const STREAM_STALLED_RE = /stream stalled|no chunk for/i;

const OPEN_OPTIONS_TEST_SETUP = {
  label: 'Open Options and click "Test setup"',
  kind: "open-options",
};
const OPEN_OPTIONS_PICK_MODEL = {
  label: "Open Options and pick a model from the list",
  kind: "open-options",
};
const OPEN_OPTIONS_CHECK_KEY = {
  label: "Open Options and check the API key",
  kind: "open-options",
};
const OPEN_OPTIONS_CHECK_BACKEND = {
  label: "Open Options and check the backend address",
  kind: "open-options",
};
const OPEN_OPTIONS_CONFIGURE_WHISPER = {
  label: "Open Options and configure a Whisper backend",
  kind: "open-options",
};

/**
 * Classify a raw error message into a human hint, or `null` if nothing
 * recognisable matched (caller keeps showing the raw text alone).
 *
 * @param {string} rawMessage the text as shown today (job.error, or
 *   stringifyError() of a thrown/fetch error)
 * @param {HealthLike | null} [health] best-effort GET /health response,
 *   fetched by the caller at the same time it decided to show this error.
 *   `null`/omitted when /health itself couldn't be reached (which is
 *   itself part of the "daemon unreachable" story) or wasn't fetched.
 * @returns {ErrorHint | null}
 */
export function classifyError(rawMessage, health) {
  const text = String(rawMessage ?? "");

  // 1. Daemon unreachable — checked first because when this matches,
  // nothing else about the message (or `health`) can be trusted: the
  // fetch never reached the daemon, so there's no backend/job context to
  // read anything else out of.
  if (DAEMON_UNREACHABLE_RE.test(text)) {
    return {
      title: "The daemon isn't running",
      explanation:
        "The extension couldn't reach the local daemon at all (not a slow " +
        "response — no connection). GET /health would fail the same way " +
        "right now.",
      action: null, // starting the daemon is a terminal command, not an extension action
    };
  }

  // 2. Model backend unreachable — the daemon itself answered, but the
  // configured LLM/Whisper backend did not. Requires /health to say so
  // explicitly (a message alone can't distinguish this from a slow
  // backend hiccup that will resolve on retry).
  if (health && health.llm_backend_reachable === false) {
    const authish = health.llm_backend_error && AUTH_KEYWORD_RE.test(health.llm_backend_error);
    if (authish || (health.llm_backend_error && AUTH_CODE_RE.test(health.llm_backend_error))) {
      return {
        title: "The model backend rejected the API key",
        explanation: health.llm_backend_error || "The backend responded with 401/403.",
        action: OPEN_OPTIONS_CHECK_KEY,
      };
    }
    return {
      title: "The model backend is unreachable",
      explanation: health.llm_backend_error
        ? `GET /health reports: ${health.llm_backend_error}`
        : "GET /health couldn't reach the configured LLM backend at all.",
      action: OPEN_OPTIONS_CHECK_BACKEND,
    };
  }

  // 3. Auth rejection surfacing directly in the job error text (e.g. the
  // summarization call itself got a 401), independent of the current
  // /health snapshot.
  if (AUTH_CODE_RE.test(text) || AUTH_KEYWORD_RE.test(text)) {
    return {
      title: "The model backend rejected the API key",
      explanation: "The backend responded with an authorization error (401/403).",
      action: OPEN_OPTIONS_CHECK_KEY,
    };
  }

  // 4. Context/token-size overflow.
  if (CONTEXT_OVERFLOW_SUBJECT_RE.test(text) && CONTEXT_OVERFLOW_VERB_RE.test(text)) {
    return {
      title: "The request was bigger than the model's context",
      explanation:
        "The backend rejected the request for being too large for its " +
        "configured context window. Backends phrase this differently, but " +
        "the underlying problem is always the same: this page/video's " +
        "content didn't fit.",
      action: OPEN_OPTIONS_TEST_SETUP,
    };
  }

  // 5. Model not found.
  if (MODEL_SUBJECT_RE.test(text) && MODEL_MISSING_RE.test(text)) {
    return {
      title: "The configured model isn't available on this backend",
      explanation:
        "The backend doesn't recognise the model name in Options. It may " +
        "not be loaded/pulled, or the name may be spelled differently on " +
        "this backend.",
      action: OPEN_OPTIONS_PICK_MODEL,
    };
  }

  // 6. Stream stalled/timed out.
  if (STREAM_STALLED_RE.test(text)) {
    return {
      title: "The model stopped responding mid-stream",
      explanation:
        "No output arrived for longer than the configured stream timeout. " +
        "The model may be too slow for this backend, or it may have hung.",
      action: null,
    };
  }

  // Nothing matched — show the raw text with no invented diagnosis.
  return null;
}

// ---------------------------------------------------------------------------
// describeQueuedDetail — the OTHER signal (see the module docstring).
//
// api.schemas.DeferredReason (daemon/src/api/schemas.py:69) codes
// (transcript_unavailable / transcript_blocked / network_error) do NOT
// reach classifyError above, ever, in the current daemon: traced through
// daemon/src/workers/pipeline.py, a DeferredReason only feeds a log line
// and `stage_event("queued", detail=reason.value)` — it never reaches
// `mark_failed`, so no job.error string ever carries it. Verified
// empirically (not just by reading the source): a throwaway pytest run
// against the real pipeline (mocked network calls only, same fixture
// pattern as daemon/tests/test_api_jobs.py's
// test_post_jobs_youtube_without_transcript_defers) confirmed the
// broker really does publish `{type: "stage", stage: "queued", detail:
// "transcript_unavailable", ...}`, and confirmed a subsequent
// `GET /jobs/{id}` carries no trace of the reason anywhere (JobDetails
// has no field for it — see api/schemas.py). So a job that already sat
// down as "queued" before the panel (re)opened has no reason to show at
// all; this can only ever fire for a panel that's live-subscribed via
// GET /events at the moment the pipeline defers — exactly the "stage"
// event handling in sidepanel/app.js's `_attachStreamSubscription`.
//
// This is why it's a separate function from classifyError: the job is
// PARKED, waiting on the Whisper queue, not failed — the sidepanel must
// never paint this into the red "Error." status block.
//
/**
 * @param {string | null | undefined} detail a "queued"-stage event's
 *   `detail` field (only meaningful when `stage === "queued"`)
 * @returns {ErrorHint | null} same shape as classifyError's return value
 *   so the sidepanel can reuse one renderer for both, but conceptually a
 *   "why is this parked" explanation, never an error
 */
export function describeQueuedDetail(detail) {
  switch (detail) {
    case "transcript_unavailable":
      return {
        title: "Parked: no transcript available",
        explanation:
          "YouTube doesn't expose captions for this video. This job will " +
          "stay parked here forever unless a Whisper backend is configured " +
          "to transcribe the audio instead.",
        action: OPEN_OPTIONS_CONFIGURE_WHISPER,
      };
    case "transcript_blocked":
      return {
        title: "Parked: YouTube blocked the transcript request",
        explanation:
          "YouTube rate-limited or blocked the caption request after " +
          "retrying. It may work if you try again later, or updating " +
          "yt-dlp may help if this keeps happening.",
        action: null,
      };
    case "network_error":
      return {
        title: "Parked: a network error interrupted the transcript fetch",
        explanation:
          "The connection to YouTube failed while fetching captions. " +
          "Check your network connection and try again.",
        action: null,
      };
    default:
      return null;
  }
}
