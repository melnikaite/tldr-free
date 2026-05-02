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
