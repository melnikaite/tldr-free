# Extension permissions — audit & justification

Audited 2026-06-11 against actual code usage (every claim below is backed by
a call site). Written for Chrome Web Store / AMO review and for contributors.

## Minimizations applied in the audit

- **Removed `web_accessible_resources`** (`src/library/index.html` exposed to
  `<all_urls>`): the Library page is only ever opened via
  `chrome.tabs.create(chrome.runtime.getURL(...))` from extension contexts —
  no web page loads it. Pure attack-surface reduction, zero behavior change.
- **Removed redundant host entries** (`https://*.youtube.com/*`,
  `https://www.youtube.com/*`): subsumed by `<all_urls>`.
- No declarative `content_scripts` exist — all injection is user-initiated
  (`chrome.scripting.executeScript` after a toolbar click or an explicit
  button press in the side panel).

## Permission-by-permission justification

| Permission | Used for (call sites) | Why it can't be narrower |
|---|---|---|
| `sidePanel` | The product UI (`side_panel`, `chrome.sidePanel.open`) | Core surface |
| `tabs` | Side panel follows the active tab: `tabs.onActivated`/`onUpdated` + `tabs.query` (background.js), finding the YouTube/media tab to seek on timecode click (sidepanel/app.js, transcript.js) | `activeTab` grants access only after a toolbar click; tab-following and "find the tab playing this video" need continuous `tabs.query`/events |
| `activeTab` | Grants temporary host access on toolbar click for extraction | Used as the *preferred* grant for the click flow |
| `scripting` | Injecting the extractor (`Readability`/`youtube.js`) on user request; seeking `<video>` to a clicked timecode; caption-track injection | All injections are user-initiated; no automatic injection exists |
| `cookies` | Reading cookies for the processed site/media URL and passing them **to the user's local daemon only** so yt-dlp can download access-restricted media (age-gated YouTube, membership content, signed CDN URLs, auth-protected PDFs) | Media files live on arbitrary CDN domains that differ from the page origin — a fixed domain list is impossible |
| `storage` | `storage.session` (active job/tab state), `storage.local` (daemon URL option) | — |
| host: `localhost`/`127.0.0.1` | All daemon HTTP + SSE | The product's only data channel |
| host: `<all_urls>` | (a) `fetch()` of PDFs on arbitrary domains and `file://` PDFs from the background; (b) `chrome.cookies.getAll({url})` for arbitrary media/CDN URLs; (c) extraction triggered from the **side-panel button**, which is not an `activeTab`-qualifying gesture on the tab | The product's core promise is "summarise anything on any site"; the content (PDFs, podcasts, embedded players, CDNs) lives on unpredictable domains |

## Alternatives considered (and why deferred)

- **Drop `<all_urls>`, rely on `activeTab` only**: breaks (1) the side-panel
  "Process this page" button (its click is not a gesture on the tab, so no
  `activeTab` grant), (2) cookie reads for cross-origin media CDNs, (3)
  PDFs fetched from domains other than the active tab.
- **`optional_host_permissions` + `permissions.request()` at first use**:
  viable privacy-friendlier UX; deferred as a follow-up because it adds a
  permission-prompt flow to every first-time site and needs careful UX.
  Tracked for a future release.

## Data-flow guarantee

No page content, cookie, or transcript ever leaves the machine via the
extension: the only network destination in extension code is the local
daemon (`http://127.0.0.1:8765`). See [`PRIVACY.md`](../PRIVACY.md) for the
canonical privacy policy, and [`store-listing.md`](store-listing.md) for the
same justifications phrased for the Chrome Web Store dashboard fields.
