// Per-window "what the side panel shows" state in chrome.storage.session.
//
// Every Chrome window has its own side-panel document, so the followed job is
// keyed by windowId — a tab switch / navigation / Library "open" in one window
// must never re-point the panel of another. Writers: background.js (tab sync,
// submitJob) and library/app.js ("open in side panel", the Library tab's
// window only). Reader: sidepanel/app.js, which only looks at its own key.
// background.js removes both keys on windows.onRemoved.

/** @param {number} windowId → `{jobId: string|null, url: string|null}` */
export const activeJobKey = (windowId) => `activeJob:${windowId}`;

/** @param {number} windowId → one-shot `{jobId, query}` Library search hand-over */
export const searchHandoffKey = (windowId) => `transcriptSearchHandoff:${windowId}`;

/** chrome.storage.session key prefix for a per-job chat composer draft. */
export const chatDraftKey = (/** @type {string} */ jobId) => `chatDraft:${jobId}`;

/**
 * `kind` filter for "the PAGE job of this tab URL" lookups (GET /jobs accepts
 * a comma list): legacy kind=media jobs stored under a plain page URL must
 * not be taken for the page's job. Not for video-chip keys (those ARE media).
 */
export const PAGE_JOB_KINDS = "page,pdf,youtube";
