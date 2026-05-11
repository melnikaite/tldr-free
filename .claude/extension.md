# Extension source layout

Manifest V3, vanilla JavaScript, ES modules. **No build step, no bundler,
no TypeScript compiler.** Vendored libs (`marked`, `DOMPurify`, `Readability`)
live in `extension/vendor/` and are loaded as classic `<script>` tags
exposing globals.

## Layout

```
extension/
├─ manifest.json                # MV3, permissions, side panel, options
├─ public/icons/                # 16 / 48 / 128 PNG
├─ vendor/                      # gitignored; populated by `task install` (`task ext:vendor`)
│  ├─ marked.min.js             # global `marked`
│  ├─ purify.min.js             # global `DOMPurify`
│  └─ readability.js            # global `Readability`
└─ src/
   ├─ background.js             # service worker: action click, content-script injection,
   │                              POST /jobs (async), tab tracking
   ├─ content/
   │  ├─ extract.js             # Readability runner (injected per page)
   │  └─ youtube.js             # videoId + title detection (injected on YouTube)
   ├─ sidepanel/
   │  ├─ index.html             # loads vendor scripts then ./app.js as a module
   │  ├─ app.js                  # follows the active job via /events (no per-job /ai/stream)
   │  ├─ chat.js                 # SSE consumer for /ai/stream QA mode; renders history
   │  └─ style.css
   ├─ library/
   │  ├─ index.html
   │  ├─ app.js                  # list, filters, search, delete, retry; reactive to /events
   │  └─ style.css
   ├─ options/
   │  ├─ index.html
   │  └─ options.js              # daemonUrl in chrome.storage.local
   └─ lib/
      ├─ api-types.js            # JSDoc mirror of daemon `schemas.py` — READ-ONLY contract
      ├─ daemon-client.js        # the only place that talks HTTP to the daemon
      ├─ event-stream.js         # singleton EventSource over /events (auto-reconnect)
      ├─ utils.js                # escapeHtml, stringifyError — shared across surfaces
      ├─ cookies.js              # chrome.cookies.getAll → daemon Cookie[]
      ├─ markdown.js             # marked + DOMPurify + DOM-walk to linkify [MM:SS]
      └─ url.js                  # normalizeUrl: canonicalize for matching/storage
```

## Lifecycle of a job in the side panel

```
toolbar click
   ↓
background.js
   ↓
chrome.sidePanel.open  (opens an empty / loading panel)
   ↓
content script injection (Readability or YouTube extractor)
   ↓
POST /jobs → 202 {id}
   ↓
chrome.storage.session.activeJobId = id   (only when the just-submitted
broadcast "job-created"                    URL matches the active tab)
   ↓
sidepanel/app.js loadAndRender(id):
   ├─ if job.status === "done"  → render cached summary_md, enable chat
   └─ else                      → subscribe to the open /events SSE filtered
      by job_id; pipe stage/delta/done into the summary area; on done,
      re-render markdown with timecode links, enable chat
   ↓
sidepanel/app.js loadHistory(id) (in parallel):
   GET /jobs/{id}/messages → render saved bubbles
```

The Side panel keeps **one** SSE connection (the global /events) instead
of opening a fresh /ai/stream per job. With Chrome's 6-per-origin HTTP/1.1
cap and several Library refreshes flying around, dedicating that
connection to /ai/stream while running pipelines stalled subsequent
fetches in DevTools-invisible ways. /events with a job_id filter on the
client gets the same per-job stream over the same one socket.

## Tab tracking

`background.js` listens to `tabs.onActivated`, `tabs.onUpdated`
(URL change in active tab, after dedupe by normalizeUrl), and
`windows.onFocusChanged`. On any of those:
`normalizeUrl(tab.url)` → `daemon.listJobs({ url, limit: 1 })` → broadcast
`{type:"tab-changed", url, jobId}`. The side panel reacts:

- `jobId` present → `loadAndRender(jobId)` (which itself decides whether
  to render cached summary or open `/ai/stream`)
- `jobId === null` → render the "no summary yet" placeholder with the URL

Non-summarizable tabs (`chrome-extension://` — including our own Library
page — `chrome://`, `about:blank`, `file://`) are **ignored** by the sync.
They neither set `activeJobId` nor broadcast `tab-changed`, so glancing at
the Library while a job streams keeps the side panel attached to the
in-progress summary instead of yanking it away.

We dedupe via `lastSyncedUrlByTab` so a `?t=754s` (timecode click) doesn't
re-trigger a lookup for what's clearly the same article.

## Chat persistence

Each Q&A turn is persisted server-side in the `Message` table. The side
panel's `chat.js` calls `daemon.aiStream({ job_id, question })` to:
1. Persist the user message.
2. Stream the answer tokens into the assistant bubble.
3. Persist the assistant message; the `done` event carries `message_id`.

On every job switch, `app.js` calls `daemon.listMessages(jobId)` and
`chat.renderHistory(items)` to redraw the saved bubbles. So chat is
preserved across tab switches, browser restarts, and side-panel close.

There's no "clear chat" — deleting the job from the Library drops its
`Message` rows via FK cascade.

## Side panel state

- `chrome.storage.session.activeJobId` — currently shown job (cleared on browser close)
- `chrome.storage.session.activeUrl` — last normalized URL synced

## In-flight badge

The badge counter is a `Set<jobId>` of jobs currently in
`queued / running`, recomputed from `/events`:
- one `seedBadge()` call on bootstrap pulls the initial set from
  `daemon.listJobs({status:["queued","running"]})`
- after that, `job` events (created / updated / deleted) and `done` /
  `error` events keep the set current. No polling, no intervals.

Library is the same shape: one `refetch()` on load, then `/events` drives
every subsequent insert / update / remove. It subscribes with
`?types=job,workers,done,error` so the high-volume per-token `delta`
events from a running pipeline don't reach it.

## Reload

`chrome://extensions` → click the reload icon (circular arrow) on the TLDR
card. Chrome does NOT auto-reload unpacked extensions on file change.
Service-worker errors are visible in the "Service worker" devtools link on
the same card; side-panel errors are in the side panel's own devtools.
