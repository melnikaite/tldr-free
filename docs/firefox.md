# Firefox port

The Chrome extension in `extension/` also runs on Firefox (128+). The
Firefox variant differs only in the manifest; all JavaScript is shared,
with `extension/src/lib/browser-compat.js` selecting the panel API at
runtime (`chrome.sidePanel` on Chrome, `browser.sidebarAction` on Firefox).

## Build

```bash
task install                  # once — populates extension/vendor/
./scripts/build-firefox.sh    # → dist/firefox/
```

The script rsyncs `extension/` to `dist/firefox/` and swaps in
`extension/manifest.firefox.json` as `manifest.json`. Idempotent — re-run
after any code change.

## Load as a temporary add-on

1. Open `about:debugging#/runtime/this-firefox` in Firefox.
2. Click **Load Temporary Add-on…**.
3. Pick `dist/firefox/manifest.json`.
4. The sidebar opens via the toolbar button (TLDR icon), or View → Sidebar.

Temporary add-ons are removed when Firefox exits — reload after restart.
After editing code: re-run `build-firefox.sh`, then click **Reload** on the
add-on card in `about:debugging`.

## Why Firefox 128+ (`strict_min_version`)

| Requirement | Firefox version |
|---|---|
| MV3 + `action`, `scripting` | 109 / 102 |
| `background.type: "module"` (ES-module event page) | 112 |
| `storage.session` | 115 |
| `scripting.executeScript({ world: "MAIN" })` — used by the transcript `<track>` injection | **128** |

Firefox does not support `background.service_worker` (bug 1573659); the
Firefox manifest uses an event page (`background.scripts`) instead, which
since FF 112 can be an ES module — no loader shim needed.

## Manifest differences vs Chrome

- `sidebar_action` instead of `side_panel`; the `sidePanel` permission is
  removed (it is not a Firefox permission and would be rejected).
- `background.scripts` + `type: "module"` instead of `service_worker`.
- `browser_specific_settings.gecko`: stable id `tldr@melnikaite.github.io`,
  `strict_min_version: 128.0`, `data_collection_permissions: none`
  (required for AMO submission; harmless for temporary loading).
- `options_ui` (`open_in_tab: true`) instead of the deprecated
  `options_page`.

## Known limitations on Firefox (documented, not worked around)

- **`file://` PDFs do not work.** Firefox extensions cannot `fetch()`
  `file://` URLs (there is no "Allow access to file URLs" toggle), so the
  local-PDF upload path in `background.js` fails with a clear error.
  http(s) PDFs are unaffected — the daemon fetches those itself.
- **Sidebar is global per window, not per tab.** `browser.sidebarAction`
  has no `tabId` scoping; the sidebar shows the same document across all
  tabs of a window. In practice this matches the extension's behavior
  anyway (the panel follows the active tab via background tab-tracking),
  but Chrome's per-tab panel nuances don't apply.
- **`sidebarAction.open()` must run synchronously in a user gesture.**
  Handled in code: `chrome.action.onClicked` opens the panel before any
  other async work, and the Library's "open in side panel" button issues
  the open call first. Any new open-panel call site must follow the same
  rule (see `browser-compat.js` module comment).
- **`storage.session` is in-memory, max 10 MB, cleared on browser exit**
  (same as Chrome), but `setAccessLevel()` is not implemented in Firefox —
  irrelevant here since no content script reads session storage.
- **Event page vs service worker.** Firefox suspends idle event pages like
  Chrome kills idle service workers; the existing restart-safety design
  (wall-clock `switchVersion`, no background SSE) covers both.
- **`storage.session.setAccessLevel` absent** — see above; not used.

## Manual test checklist

1. Build (`./scripts/build-firefox.sh`), load via about:debugging — no
   manifest errors on the add-on card.
2. Toolbar click on an article page → sidebar opens immediately, summary
   streams in.
3. Toolbar click on a YouTube video → summary with timecodes streams.
4. Click a `[MM:SS]` timecode link in the summary → the YouTube tab seeks
   to that position (focuses existing tab, doesn't open a duplicate).
5. Transcript tab appears for the YouTube job; clicking a transcript line
   seeks; the current line live-highlights while the video plays.
6. Transcript language switch → translated transcript renders; for a
   native `<video>` page, captions `<track>` appears in the player
   (MAIN-world injection — the FF-128 requirement).
7. Switch to another tab with no summary → sidebar shows the "no summary
   yet" placeholder with that URL; switch back → summary returns.
8. Ask a follow-up question in chat → answer streams; reload the sidebar →
   chat history persists.
9. Library page (link from sidebar) lists jobs; in-flight badge counts
   match; "open in side panel" from Library opens/switches the sidebar
   (gesture-sensitive path).
10. Delete a job from Library → row disappears, sidebar resets if it was
    showing that job.
11. Retry a failed job from Library → job re-queues and streams.
12. http(s) PDF tab → summarize works (daemon fetches the URL).
13. `file://` PDF tab → fails with the explanatory error (expected
    limitation on Firefox).
14. Cookie-gated page (e.g. members-only video / authed PDF): summarize a
    page whose media requires session cookies → daemon receives cookies
    and the job succeeds.
15. Quit and restart Firefox (or just idle >30s), then switch tabs →
    sidebar still follows the active tab (event-page restart safety).
