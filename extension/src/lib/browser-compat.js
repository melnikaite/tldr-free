// Chrome ⇄ Firefox adapter for the side-panel surface. Chrome exposes
// `chrome.sidePanel` (per-tab/per-window panel); Firefox exposes
// `browser.sidebarAction` (one global sidebar per window). Everything else
// we use (`chrome.tabs`, `chrome.storage`, `chrome.scripting`, …) exists
// under the `chrome.*` namespace in both browsers, so only the panel needs
// a shim.
//
// IMPORTANT Firefox constraint: `browser.sidebarAction.open()` must be
// invoked SYNCHRONOUSLY inside a user-gesture handler — any `await` before
// the call consumes the gesture and the open is rejected. Callers must
// therefore call `openSidePanel()` before any other async work in click
// handlers. (In Chrome `chrome.sidePanel.open` has the same gesture
// requirement but tolerates prior awaits less strictly; calling first is
// correct for both.)

// `browser.sidebarAction` only exists in Firefox AND only when the manifest
// declares `sidebar_action` — exactly the Firefox build produced by
// scripts/build-firefox.sh. In Chrome `browser` is undefined.
const HAS_SIDEBAR_ACTION =
  typeof browser !== "undefined" &&
  typeof browser.sidebarAction?.open === "function";

/**
 * Chrome: keep openPanelOnActionClick=false so chrome.action.onClicked
 * fires for our custom flow. Firefox: no equivalent (toolbar click always
 * reaches action.onClicked; the sidebar never auto-opens) — no-op.
 *
 * @returns {Promise<void>}
 */
export function setPanelBehavior() {
  if (HAS_SIDEBAR_ACTION) return Promise.resolve();
  return chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: false });
}

/**
 * Open the side panel / sidebar. Must be called synchronously within a
 * user-gesture handler (see module comment).
 *
 * - Firefox: opens the global per-window sidebar; `tabId` is ignored
 *   (Firefox sidebars are not per-tab).
 * - Chrome with `tabId`: `chrome.sidePanel.open({ tabId })`.
 * - Chrome without `tabId`: resolve the current window first, then
 *   `chrome.sidePanel.open({ windowId })` — the Library-page path.
 *
 * @param {{ tabId?: number }} [opts]
 * @returns {Promise<void>}
 */
export function openSidePanel({ tabId } = {}) {
  if (HAS_SIDEBAR_ACTION) {
    // Synchronous invocation is the load-bearing part — do not add awaits
    // before this line in callers.
    return browser.sidebarAction.open();
  }
  if (tabId != null) {
    return chrome.sidePanel.open({ tabId });
  }
  return chrome.windows.getCurrent().then((win) => {
    if (win?.id !== undefined) {
      return chrome.sidePanel.open({ windowId: win.id });
    }
    return undefined;
  });
}
