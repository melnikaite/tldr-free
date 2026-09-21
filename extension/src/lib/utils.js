// Tiny shared helpers for sidepanel / library / background.
//
// Kept to genuinely-shared one-liners; UI-specific rendering still lives in
// the surface that owns it.

/**
 * Escape a string for safe insertion into an HTML attribute or text node.
 * @param {unknown} s
 * @returns {string}
 */
export function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/**
 * Best-effort human-readable string from any thrown value. Errors and
 * promise rejections from `fetch` give us heterogeneous shapes — this lets
 * the UI display something useful without inspecting types at every call.
 * @param {unknown} err
 * @returns {string}
 */
export function stringifyError(err) {
  if (err instanceof Error) return err.message;
  try {
    return JSON.stringify(err);
  } catch {
    return String(err);
  }
}

/**
 * Coarse, human-readable duration for a summary badge — "~5 min", "~45s" —
 * NOT a timecode (see the sidepanel's [MM:SS] markers for that). Used for
 * ``transcript_missing_seconds`` (JobSummary/JobDetails): the point is a
 * rough sense of how much material Whisper never reliably recovered, not a
 * precise figure, so this rounds to whole minutes above 60s and whole
 * seconds below it. Shared by the sidepanel's transcript header badge and
 * the Library row badge so the two surfaces never drift in wording.
 * @param {number} seconds
 * @returns {string}
 */
export function formatApproxDuration(seconds) {
  if (seconds < 60) return `~${Math.round(seconds)}s`;
  return `~${Math.round(seconds / 60)} min`;
}
