# Extension

Chrome MV3 service worker + side panel + library page + options. Vanilla
JavaScript with ES modules — **no build step, no bundler, no TypeScript
compiler**. Vendored libs (`marked`, `DOMPurify`, `Readability`) load as
classic `<script>` tags exposing globals; `task install` populates
`extension/vendor/`.

For the tree itself: `ls extension/src/`. This doc covers what isn't
obvious from filenames.

## Surfaces and how they talk to the daemon

| Surface | File | Daemon connection |
|---|---|---|
| Service worker | `background.js` | HTTP only (POST /jobs, GET /jobs, retry/delete). No SSE — the SW is killed by Chrome after ~30s idle, an EventSource would die with it. |
| Side panel | `sidepanel/app.js` | One global `/events` SSE via `lib/event-stream.js`, filtered by active `job_id` for stage/delta/done. Plus per-question `/ai/stream` via `chat.js`. |
| Library | `library/app.js` | Same global `/events` SSE, `?types=job,workers,done,error` — skips per-token `delta` chatter. |
| Options | `options/options.js` | None — writes `daemonUrl` to `chrome.storage.local`. |

**Why one SSE per surface, not per job.** Chrome has a 6-per-origin HTTP/1.1
connection cap. Dedicating a connection to `/ai/stream` while running
pipelines stalls subsequent fetches in DevTools-invisible ways. The global
`/events` stream with a client-side `job_id` filter gets the same per-job
view over one socket.

## PDF tabs bypass content-script extraction

Chrome's built-in PDF viewer is a `chrome-extension://…` page that
refuses script injection, so the Readability path doesn't apply. PDF
parsing is also fundamentally a daemon concern (pypdf + vision OCR
fallback — see [workers.md](workers.md) and [llm.md](llm.md)), so the
extension's job here is minimal:

- http(s) PDFs → submit `{kind:"pdf", url, cookies}`. The daemon
  fetches the bytes itself.
- `file://` PDFs → the daemon can't reach the host filesystem, so
  `background.js` does the fetch (Chrome grants `file://` access only
  when the user enables "Allow access to file URLs" in extension
  details), base64-encodes the bytes, and submits
  `{kind:"pdf", url, pdf_bytes_b64}`.

Either way: no client-side parsing, no pdf.js, no extra round-trips.
The side panel just renders the streaming summary like any other job.

## Summary / Transcript tabs

The side panel shows two tabs: **Summary** (always) and **Transcript**
(only for jobs with a meaningful transcript — `kind in (youtube, media)`,
hidden for `page` / `pdf`).

`sidepanel/transcript.js` lazy-loads the full `raw_text` via
`GET /jobs/{id}/transcript` only when the user clicks the tab. The
payload can be megabytes for hour-long podcasts so it's kept out of
`JobDetails`. Each `[MM:SS]` line becomes a `<p data-tx-seconds="…">`
which doubles as the click target for seeking (handled by the same
`app.js` handler as the summary's timecode links) AND as the anchor
for live-highlight via binary search.

Live highlight: every 500 ms while the tab is visible + the source
media is playing, the controller calls `chrome.scripting.executeScript
({allFrames: true})` to read `<video, audio>.currentTime` from any
frame (including cross-origin iframes like YouTube embeds — extension's
host permissions cover those). Binary search picks the matching line,
applies `tx-line--current`, scrolls into view if media is playing.

`<track>` injection (WebVTT): on language switch, the controller builds
a VTT body from the displayed text and injects a `<track>` into the
page's first `<video>` via `executeScript({world: "MAIN"})`. World
`MAIN` because the blob URL holding the VTT must resolve in the page's
context; the extension's ISOLATED-world blobs can't be loaded by
page-context `<video>`. Skipped for `<audio>` (no native captions UI)
and for iframe-embedded players (we can't inject into their internal
DOM safely — the sidepanel transcript itself serves as the captions
surface).

The language switcher is a sticky-positioned bar with chips for cached
languages (source + each translation) plus a free-form input. Enter
triggers `POST /jobs/{id}/transcript/translate`. Chips update live via
the existing `/events` SSE — the translator worker publishes
`translation_updated` job_events as it progresses. See
[llm.md](llm.md) → "Transcript translation".

## Side panel lifecycle of a job

```
toolbar click → background.js
   ↓ chrome.sidePanel.open    (panel renders empty/loading)
   ↓ inject content script    (extract.js or youtube.js)
   ↓ POST /jobs → 202 {id}
   ↓ chrome.storage.session.activeJobId = id   ← only if source URL still matches active tab
   ↓ broadcast `job-created`

sidepanel/app.js loadAndRender(id):
   ├─ job.status === "done"   → render cached summary_md, enable chat
   └─ else                    → render skeleton; pipe stage/delta/done
                                from /events into the summary area;
                                on done, re-render markdown with timecode
                                links, enable chat
sidepanel/app.js loadHistory(id) (parallel):
   GET /jobs/{id}/messages → render saved bubbles
```

## Tab tracking — when the panel switches jobs

`background.js` listens to `tabs.onActivated`, `tabs.onUpdated` (URL change
in active tab), `windows.onFocusChanged`. On any of those:
`normalizeUrl(tab.url)` → `daemon.listJobs({ url, limit: 1 })` → broadcast
`{type: "set-active-tab", url, jobId, version}`. The side panel reacts:

- `jobId` resolved → `loadAndRender(jobId)`
- `jobId === null` → render the "no summary yet" placeholder with the URL

**Non-summarizable tabs are skipped**: `chrome-extension://` (including our
own Library), `chrome://`, `about:blank`, `file://`. Glancing at the Library
while a job streams keeps the side panel attached to the in-progress
summary instead of yanking it away.

`version` is a monotonic counter (wall-clock-floored — `Math.max(prev + 1,
Date.now())`) so MV3 service-worker restarts can't fire an "older" version
than what the side panel already acknowledged. Dedupes via
`lastSyncedUrlByTab` so a `?t=754s` (timecode click) doesn't re-trigger a
lookup for the same article.

## Chat persistence

Per Q&A turn: `chat.js` calls `daemon.aiStream({ job_id, question })` which
(a) persists the user message, (b) streams answer tokens, (c) persists the
assistant message. On job switch, `app.js` calls `daemon.listMessages(jobId)`
and `chat.renderHistory(items)` — bubbles survive tab switches, browser
restarts, side-panel close. No "clear chat" UI; deleting the Job from the
Library drops its `Message` rows via FK cascade.

## In-flight badge — no polling

Counter is a `Set<jobId>` of jobs in `queued/running`, computed from `/events`:

- One `seedBadge()` on bootstrap pulls the initial set via
  `daemon.listJobs({ status: ["queued", "running"] })`.
- After that, `job` events (created/updated/deleted) and `done`/`error`
  keep the set current. No polling, no intervals.
- Library follows the same pattern.

## When the side panel may go stale, and what fixes it

Chrome throttles SSE in backgrounded surfaces. A minimised window can miss
the `done` event for a streaming job. Two defences in `app.js`:

- `document.visibilitychange` listener: when the panel becomes visible
  again, re-seeds the badge + refreshes the active job if its cache looks
  stale (`isCacheStale`).
- SSE `done` and `error` handlers proactively write the final
  `summary_md`/`error` into the in-memory active-job cache, so a later
  `renderFromJob(active)` (e.g. tab switch back) doesn't see
  `status=done && summary_md=null` and re-render the streaming placeholder.

If you find a way to lose the final state again, add a third defence in
the same place. Don't paper over with polling.

## State (chrome.storage)

- `chrome.storage.session.activeJobId` — currently shown job (clears on browser close)
- `chrome.storage.session.activeUrl` — last normalized URL synced
- `chrome.storage.local.daemonUrl` — daemon endpoint (default `http://localhost:8765`)

## Reloading after edits

In short: hit the reload icon in `chrome://extensions`. Manifest changes
sometimes need full Remove + Load unpacked. Full reload matrix in
[runbook.md](runbook.md).
