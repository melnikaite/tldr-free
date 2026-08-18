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

## Media detection: duration-based reject, not visibility-based

`content/extract.js`'s `collectNativeMedia()` accepts any `<video>`/`<audio>`
with a resolvable src — including elements with no `controls` attribute and
zero on-screen size. That's deliberate: a hidden `<audio>` driven entirely by
its own JS is the NORMAL way real audio players are built (SoundCloud,
Bandcamp, any custom podcast widget), and filtering on visibility/`controls`
would break exactly the sites the audio path exists for.

The one filter that IS applied to both `<video>` and `<audio>` is
**duration**: `MIN_MEDIA_DURATION_SECONDS` (12s) rejects an element only when
`el.duration` is a **known finite number** below the threshold. `NaN` /
`Infinity` / unset (e.g. `preload="none"`, never played — normal for a
script-driven hidden player) is never rejected on this basis — an unplayed
element simply hasn't reported a duration yet, which is not evidence it's
short. This exists to stop invisible zero-duration UI sounds ("ding" on
notification, click, etc.) from being treated as summarizable content; the
existing on-screen-size filter for `<video>` (area/videoWidth check) is
unrelated and untouched.

The daemon has a matching, independently-gated probe before it ever
downloads a `kind=media` job's audio — see [workers.md](workers.md).

Because a hidden/short media element can still slip past this filter (a
probe run server-side can disagree with the DOM value, or duration is simply
unknown client-side), `extract.js` now also does a best-effort page-text
extraction (`extractPageText()`, the same Readability-or-innerText logic the
no-media `extracted-page` branch has always used) and includes it as a
`text` field on the `extracted-media` message. `background.js`'s
`handleExtractedMedia` forwards it as `page_text` on the `JobCreateRequest`
so the daemon has something to summarize instead of audio if the media turns
out not to be speech, or Whisper returns nothing.

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

## First-run welcome screen — idle view is gated behind GET /health

Any render of the idle view (no job for the current tab — bootstrap with
no `activeJobId`, or `handleSetActiveTab`'s `jobId === null` phase) first
goes through `app.js`'s `_gateIdleOnHealth(url)`, which probes `GET
/health` once and picks one of four outcomes:

- **daemon unreachable, `chrome.storage.local.daemonEverReachable` never
  set** → this is a fresh install with nothing configured yet. Renders
  `sidepanel/welcome.js`'s step **"daemon"**: what the daemon is, the
  native-install one-liner (copy button) mirrored from README.md's
  "Install — native, no Docker" section, and a link to the full
  instructions.
- **daemon unreachable, `daemonEverReachable` already set** → a
  previously-working install whose daemon just died. Renders the SAME
  `"error"` state (and therefore `error-hints.js`'s `classifyError` "The
  daemon isn't running" hint) a failed job would show — proactively,
  instead of waiting for a click to fail first. Deliberately NOT the
  welcome screen: see `welcome.js`'s module docstring for why showing
  "Welcome to TLDR" to a year-old install would be dishonest.
- **daemon reachable but `health.llm_backend_reachable === false`** →
  `welcome.js`'s step **"model"**: the daemon (already confirmed running)
  stays in the loop either way; the choice is local (free/private, needs
  memory — numbers mirrored from README.md) vs. cloud (your key, your
  account, content leaves the machine). Shown regardless of
  `daemonEverReachable` — the daemon answering at all already rules out
  "nothing installed".
- **everything reachable** → `_gateIdleOnHealth` returns `false` and the
  caller renders the normal `"no-summary"` idle placeholder as before.

`daemon.health()` (lib/daemon-client.js) sets `daemonEverReachable = true`
on every successful response (whatever `status` says — "degraded" still
counts, only a thrown fetch means "unreachable"), from ANY caller
(options/library/sidepanel), so the flag reflects the whole extension's
history, not just this one check.

The same daemon-unreachable classification also intercepts the OTHER path
that used to hit the raw error box first: `background.js`'s
`extraction-error` broadcast (content-script injection failure, or a
failed `POST /jobs` — the literal "click the toolbar button, get an
error" bug this screen exists to fix). `app.js`'s `handleExtractionError`
reuses `error-hints.js`'s exported `isDaemonUnreachable(text)` (the same
regex `classifyError`'s first branch matches on — never duplicated) to
decide welcome-vs-classic-error the same way, without a second `/health`
round-trip.

Each welcome step has its own "Check again" button
(`_recheckIdleHealth(url)`) that re-runs the same gate and falls back to
`"no-summary"` once things are ready — no panel reload needed. `"welcome"`
is its own `renderState` mode (own `ViewState` variant, own `_stateKey`),
not a branch inside `case "error"` — the two are rendered, and mean,
different things.

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

## Error hints — raw backend text isn't the whole story

`lib/error-hints.js` turns raw backend text into something a human can
act on, for TWO unrelated signals — keep them separate, they render into
different UI and mean different things:

- **`classifyError(rawMessage, health)`** — for the `"error"` render
  state (`app.js`'s `_renderErrorHint()`). Pattern-matches `job.error` (or
  a stringified fetch/thrown error), plus a best-effort `GET /health`,
  into `{ title, explanation, action }` for the failures that account for
  almost everything seen on localhost: daemon unreachable, model backend
  unreachable/unauthorized, context overflow, model not found, stream
  stall. Rendered inside the red `.status-block.error` box — this is
  always a genuinely failed/dead job.
- **`describeQueuedDetail(detail)`** — for the `"streaming"` render
  state's `#queued-hint` div (`app.js`'s `_renderQueuedHint()`, called
  from the live "stage" event handler in `_attachStreamSubscription`).
  Explains a job PARKED in `stage === "queued"` with a `detail` matching
  one of `api.schemas.DeferredReason`'s three codes
  (`daemon/src/api/schemas.py:69`: `transcript_unavailable`,
  `transcript_blocked`, `network_error` — the transcript fast path
  deferred to Whisper, or the retry loop feeding it gave up). Rendered
  in a neutral `.queued-hint` box, never the error styling: the job is
  waiting, not dead.

Both fetch a raw signal, classify it, and render via
`textContent`/`createElement` (never `innerHTML` — the raw message,
`health.llm_backend_error`, and the stage `detail` are all
backend-controlled strings), with `app.js` owning the
`action.kind === "open-options"` → `chrome.runtime.openOptionsPage()`
wiring in both cases. `classifyError` additionally keeps the raw text
visible under a "Technical details" `<details>` (the queued-hint has no
raw-text fallback to preserve — the compact stage badge/timeline still
shows the bare `detail` code alongside it, same as before).

**Why these two are different functions, not one:** traced through
`daemon/src/workers/pipeline.py`, a `DeferredReason` only ever feeds a
log line and `stage_event("queued", detail=reason.value)` — it is never
passed to `mark_failed`, so no `job.error` string can ever carry it.
Verified empirically, not just by reading the source: a throwaway pytest
run against the real pipeline (mocked network calls only, same fixture
pattern as `daemon/tests/test_api_jobs.py`'s
`test_post_jobs_youtube_without_transcript_defers`) confirmed the broker
really does publish `{type: "stage", stage: "queued", detail:
"transcript_unavailable", ...}`, and confirmed a subsequent
`GET /jobs/{id}` carries no trace of the reason anywhere — `JobDetails`
has no field for it. So these codes can only ever be shown to a panel
that is live-subscribed via `GET /events` at the exact moment the
pipeline defers; a panel that opens/reopens after a job has already
settled into `queued` has no way to know why (this is a real, currently
unfixed gap — not something the extension can paper over without the
daemon persisting the reason somewhere `GET /jobs/{id}` can read).

Invariants if you touch this:

- **No guessed diagnosis.** Both functions return `null` when nothing
  matches; the caller's existing fallback (raw text, or a hidden
  `#queued-hint`) is what renders then. Never widen a pattern just to
  "cover" an unrecognized string — a wrong diagnosis is worse than none.
- **Classify by message content, not status code** — mirrors
  `daemon/src/api/config.py::_looks_like_context_overflow` for the context
  bucket (context/tokens + an overflow word), since backends relay the
  same failure at different HTTP status codes.
- **`error-hints.js` stays framework-free** (no `chrome.*`, no DOM) so it
  can run under plain `node --check`/a Node script for regression-checking
  against real captured strings.
- **`isDaemonUnreachable(rawMessage)`** exports `classifyError`'s first
  regex test standalone — the first-run welcome screen (see above) reuses
  it to decide welcome-vs-classic-error from a raw `extraction-error`
  string, instead of duplicating the pattern.
- **Never let a `DeferredReason` code reach `classifyError`.** If the
  daemon ever changes to surface these through `job.error` instead of (or
  in addition to) the stage `detail`, that's a deliberate design change —
  update this doc and decide then whether `classifyError` needs its own
  branch, rather than silently duplicating `describeQueuedDetail`'s logic.

## State (chrome.storage)

- `chrome.storage.session.activeJobId` — currently shown job (clears on browser close)
- `chrome.storage.session.activeUrl` — last normalized URL synced
- `chrome.storage.local.daemonUrl` — daemon endpoint (default `http://localhost:8765`)
- `chrome.storage.local.daemonEverReachable` — set `true` by
  `daemon.health()` (lib/daemon-client.js) on its first-ever successful
  `/health` response from any surface; distinguishes a fresh install from
  a returning user in the first-run welcome screen (see above)

## Reloading after edits

In short: hit the reload icon in `chrome://extensions`. Manifest changes
sometimes need full Remove + Load unpacked. Full reload matrix in
[runbook.md](runbook.md).
