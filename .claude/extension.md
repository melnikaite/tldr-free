# Extension

Chrome MV3 service worker + side panel + library page + options. Vanilla
JavaScript with ES modules — **no build step, no bundler, no TypeScript
compiler**. Vendored libs (`marked`, `DOMPurify`, `Readability`) load as
classic `<script>` tags exposing globals; `task install` populates
`extension/vendor/`.

For the tree itself: `ls extension/src/`. This doc covers what isn't
obvious from filenames.

**Rendering trust boundary:** everything that reaches `innerHTML` via
`lib/markdown.js`'s `renderMarkdown()` (summary + chat bubbles) is
attacker-influenced — it's LLM output whose context includes arbitrary
page/video content. `renderMarkdown` defends in two independent layers:
a dedicated `marked` instance whose `html` renderer escapes raw HTML
tokens to text (so literal tags in the source material can't become live
elements or corrupt parsing), plus a fail-closed `DOMPurify` allowlist
(no `img`, no form/media tags, no `style`) that catches what *legitimate*
markdown syntax like `![]()` can still turn into. Both layers are needed —
see the header comment in `markdown.js` before changing either one.

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

## Network stream sniffing (players in cross-origin iframes)

`extract.js` only sees the top frame, and hls.js/dash.js players play from a
`blob:` src, so a player inside a cross-origin iframe yields `extracted-page`.
`background.js` therefore records, via `chrome.webRequest` (listeners at SW top
level so requests wake it), every HLS/DASH manifest and subtitle file each
tab fetches — classification/dedup/selection are pure functions in
`lib/stream-sniff.js` (tested in `tests/stream-sniff.test.mjs`). Request
headers (Referer/Origin/User-Agent, needs `extraHeaders`) come from
`onSendHeaders`; Content-Type and status from `onHeadersReceived`, which is
the single recording point (non-2xx and segments are never recorded). Lists
live in `chrome.storage.session` under `sniff:<tabId>` (cap 40, oldest
dropped), cleared on a `main_frame` request and on tab close.

`handleExtractedPage` checks that list first: if it has a manifest, it
submits a `kind=media` job instead (earliest manifest = `media_url`, the
others as `alt_media_candidates` "Stream N (HLS/DASH)"), with
`media_headers` built from the recorded headers (frame origin as fallback).
Subtitle entries are recorded but not yet used — they'd be wrong in the
"wrong source?" picker, which hands its pick to yt-dlp as media.

The extension can't tell a master playlist from a rendition playlist (all
`.m3u8`), so the whole sniffed list also goes to the daemon as
`sniffed_streams` and the daemon makes the real choice (master, audio dub,
subtitle track — see `.claude/workers.md`). `media_url` is just a default:
the latest-seen manifest whose name looks like a master
(`looksLikeMaster`), else the earliest. The 40-entry cap is per group
(manifests vs subtitles) so WebVTT segment spam can't evict the master.

**Service-worker requests.** Some embedded players (hls.js in a cross-origin
iframe) register their own Service Worker and fetch playlists from it; webRequest
reports those with `tabId -1`. `onBeforeRequest` for `main_frame`/`sub_frame`
records origin → tabs hosting a frame of that origin (most recent first;
in memory, mirrored to `chrome.storage.session["sniffFrames"]` because the
extension SW sleeps). A tab's entries are pruned on its `main_frame`
navigation and on tab close. `onHeadersReceived` attributes a `tabId -1`
request by `documentUrl`/`initiator` origin to the latest-seen tab that
`chrome.tabs.get` still finds; otherwise it's dropped. Two tabs with the same
player origin: the latest-seen frame wins (a SW request carries no tab
info, so it can be misattributed until the other tab's frame reloads).
Pure helpers: `recordFrame`/`lookupFrameTabs`/`pruneFrameTab`.

**Debug ring.** Requests `classifyStream` rejected (xhr/media/other, minus
images/fonts/css/js) from player contexts only — service-worker requests
(`viaSW`) or sub-frames (`frameId > 0`); top-frame requests are dropped by a
sync check before attribution or storage access (`shouldDebugRecord`) — go to `chrome.storage.session["sniffdbg:<tabId>"]`
(newest 40, `{host, pathTail, ext, contentType, status, viaSW, ts}`, never
the query string), cleared together with `sniff:`. When `handleExtractedPage`
falls back to a page job, it `console.info`s that ring and the sniff list.
To inspect: `chrome://extensions` → TLDR → "service worker" → Console, then
`chrome.storage.session.get(null)`.

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

Main-media selection (`lib/media-frames.js`): timecode seeks, live
highlight and captions all target the page's *main* media element.
`probeMedia` runs via `executeScript({allFrames: true})` (host permissions
reach cross-origin player iframes), each frame marks its best
`<video, audio>` with `data-tldr-main` and reports it; `pickBestFrame`
ranks frames (longest finite duration → larger area → playing), and the
action runs only in the winning `frameIds: [id]`. Duration-first is what
skips VAST pre-/mid-roll ads living in a second `<video>`. When the
expected duration is known (`expectedRange`: job `duration_seconds` ±3 s;
else 0.8× the transcript's last marker as a loose lower bound),
a duration-matching element wins outright, and captions bind *only* to a
matching element — during a pre-roll the film may not exist yet or have
unknown duration, so nothing binds until the poll's rebind sees it. Injected funcs
must stay self-contained (serialized via `toString()`); the in-page
comparator mirrors the tested `compareMedia`.

Live highlight: every 500 ms while the tab is visible the controller
probes as above and binary-searches the main element's `currentTime` to
the matching line, applies `tx-line--current`, scrolls into view if media
is playing.

Captions: on language switch the displayed `[MM:SS]` lines become plain
`{start,end,text}` cues passed as args (no blob URLs → no page-CSP
issues) to `installCaptions`, which renders a closed-shadow-DOM overlay
(pointer-events none, max z-index) positioned over the main `<video>`
inside its parent, re-laid out on `timeupdate`/`seeked`/resize/
`ResizeObserver`/`fullscreenchange`. Custom players often hide native
text tracks, hence self-rendering — for YouTube too. When the `<video>`
itself is the fullscreen element (nothing can overlay it) a native
`addTextTrack` track is shown instead until exit. The overlay hides
while the bound element's duration drops below half of the expected
minimum (or of the max seen, if unknown) — an ad loaded into the same
element. State lives on the ISOLATED world's
`window`; the probe reports `captionsBound`, and the poll re-injects
(throttled, 3 s) when the player swapped its `<video>` or the page
reloaded — but only with a known duration, a matching element, and in the
same frame URL as the last bind (`canAutoRebind`); explicit injections
(language pick) reset that frame. A `currentSrc` change on the bound
element (episode/dub switch, reload) destroys the overlay immediately.
With a known duration and no matching element the time-read returns null
(highlight stops following); seek falls back to the plain ranking. Skipped for `<audio>`.

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
and `chat.renderHistory(jobId, items)` — bubbles survive tab switches, browser
restarts, side-panel close. `renderHistory` is a no-op if `jobId` is no longer
the job on screen (guards against two racing `loadHistory` calls landing out
of order) and re-attaches a still-streaming turn's bubble if one is in flight
for that job — see the shared-DOM ownership note atop `chat.js`. No "clear
chat" UI; deleting the Job from the Library drops its `Message` rows via FK
cascade.

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
- **`describeQueuedDetail({detail, whisperConfigured, whisperQueuePosition})`**
  — for the `"streaming"` render state's `#queued-hint` div (`app.js`'s
  `_renderQueuedHint()`, called both from cold load and from the live
  "stage" event handler in `_attachStreamSubscription`). Explains a job
  PARKED in `stage === "queued"` with a `detail` matching one of
  `api.schemas.DeferredReason`'s three codes (`daemon/src/api/schemas.py:69`:
  `transcript_unavailable`, `transcript_blocked`, `network_error`). Rendered
  in a neutral `.queued-hint` box, never the error styling: the job is
  waiting, not dead.

  All three `DeferredReason` codes fall through the SAME "defer to
  Whisper" branch in `workers/pipeline.py` — they only explain why the
  YouTube-captions fast path failed, never what happens next. So what's
  actually true depends on live state, not on which code fired:
  `HealthResponse.whisper_configured` (is a Whisper backend even set up?)
  and `JobDetails.whisper_queue_position` (1-based FIFO position while
  waiting, `null` once a pool worker picks the job up). Only when
  `whisperConfigured === false` is "parked here until you configure
  Whisper" true — the other branches say "waiting in line" (position
  known), "transcription starting" (position `null`, still picked up), or
  a deliberately noncommittal "handed off for Whisper transcription" when
  `/health` itself couldn't be reached (never guess `whisperConfigured`
  either way). `_renderQueuedHint` fetches both fields itself — via a
  single `GET /health` + `GET /jobs/{id}` pair triggered once per "queued"
  transition, guarded by a monotonic token against out-of-order resolution
  — since neither field travels on the live "stage" SSE event
  (`workers/broker.py`'s `stage_event` only ever carries `{stage,
  detail}`), and the daemon never re-fires "queued" for the same job as
  its position in the FIFO changes. This is a one-shot fetch tied to a
  state transition, not a new polling loop.

  The one gap a "queued"-triggered fetch alone can't close: a job's own
  position can keep moving (other jobs ahead of it draining) with no new
  "stage" event ever firing for THIS job. `refreshActiveJob()` (the
  existing `visibilitychange` staleness refresh — see "When the side panel
  may go stale" below) already re-fetches the active `JobDetails` whenever
  the panel regains visibility; it now also calls `_renderQueuedHint`
  directly when the refreshed job is still `"queued"`, bypassing
  `renderState`'s `_stateKey` idempotency (which would otherwise no-op a
  same-job "streaming" re-render and leave the position stale). Still no
  new polling — it rides the same lifecycle hook that already existed for
  a different staleness reason.

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
"transcript_unavailable", ...}`.

A cold-load gap used to exist here — `GET /jobs/{id}` carried no trace of
the reason, so a panel that opened/reopened after a job had already
settled into `queued` couldn't say why. That's fixed: `Job.queued_reason`
(daemon `storage/db.py`, migration v9) persists the same string the
`stage` event carries, and `JobDetails.queued_reason` (`api/schemas.py`)
surfaces it on `GET /jobs/{id}`. `_attachStreamSubscription` now passes
`job.queued_reason` into `pushOrUpdatePhase`/`setStage`/
`_renderQueuedHint` at cold load, the same way it passes a live event's
`detail` — one function renders both. `DeferredReason` still never
reaches `job.error`/`classifyError`, which is why these remain two
functions, not one.

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

### Find in transcript

A find bar (`#transcript-search`, in the sticky lang-bar row) searches
whatever `_renderLines` last rendered — original or the selected
translation — and re-runs after every re-render. Matching is
case-insensitive and accent-folding (NFD, marks stripped, ß→ss) via the
pure helpers in `lib/text-search.js` (`foldWithMap` keeps a folded →
original offset map so highlights land on the right characters; unit
tests: `task test:extension`). Only `span.tx-text` (the line's spoken
text) is searched and rewritten with `<mark>`s — never the `[MM:SS]`
link, ⚠ flag, frame rows or gap rows — so clearing restores the exact
DOM and `_cues` / click-to-seek are untouched. While a query is active
the playing line is still highlighted but not auto-scrolled to. Cmd/Ctrl+F
in the panel is claimed only when the job has a transcript. The query is
cleared on every real job switch.

**Library → panel hand-over:** opening a job from a library list with a
non-empty search box writes `transcriptSearchHandoff = {jobId, query}`
into `chrome.storage.session` in the same `set()` as `activeJobId`. The
panel (`_receiveSearchHandoff` in `sidepanel/app.js`) removes the key on
read (one-shot), waits until `setActiveJob` shows that job, then opens the
Transcript tab with the query prefilled; it is applied after the lazy
transcript render. Ignored for page / pdf jobs. Storage rather than the
`job-created` message because the panel is often not open yet when the
library writes — storage is what a freshly opened panel reads on boot.

## State (chrome.storage)

- `chrome.storage.session.activeJobId` — currently shown job (clears on browser close)
- `chrome.storage.session.activeUrl` — last normalized URL synced
- `chrome.storage.session.transcriptSearchHandoff` — one-shot
  `{jobId, query}` from the library, consumed by the side panel (see
  "Find in transcript")
- `chrome.storage.session["sniff:<tabId>"]` — network-sniffed manifests/subtitles for that tab (see "Network stream sniffing")
- `chrome.storage.session["sniffdbg:<tabId>"]` / `["sniffFrames"]` — sniffer debug ring / frame-origin → tab map for service-worker attribution
- `chrome.storage.local.daemonUrl` — daemon endpoint (default `http://localhost:8765`)
- `chrome.storage.local.daemonEverReachable` — set `true` by
  `daemon.health()` (lib/daemon-client.js) on its first-ever successful
  `/health` response from any surface; distinguishes a fresh install from
  a returning user in the first-run welcome screen (see above)

## Reloading after edits

In short: hit the reload icon in `chrome://extensions`. Manifest changes
sometimes need full Remove + Load unpacked. Full reload matrix in
[runbook.md](runbook.md).
