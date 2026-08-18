# Privacy Policy — TLDR

*Last updated: August 18, 2026*

TLDR is a free, open-source Chrome extension built and maintained by one
individual (not a company). It pairs with a small daemon program that runs
on your own computer. This document explains, as precisely as the code
allows, what data the extension and daemon touch, where it goes, what gets
stored, and how to remove it.

If anything here doesn't match what the code actually does, that's a bug —
please [open an issue](https://github.com/melnikaite/tldr-free/issues).

## The short version

- TLDR does not have a server. There is no "TLDR cloud." Everything runs
  on your machine: the extension in your browser, the daemon as a local
  process you install and start yourself.
- The extension talks only to the local daemon (`localhost`/`127.0.0.1`).
  The daemon talks to: (1) the LLM/Whisper backend you configured, (2) the
  website or video you ask it to process, and, for follow-up questions,
  (3) DuckDuckGo and pages it finds, when it decides a web search would
  help answer your question (see "Web search for Q&A" below — this is the
  one outbound connection that isn't obviously "the thing you asked about"
  and deserves a clear callout).
- There is no analytics, no crash reporting service, no update-ping
  telemetry, and no account or sign-up of any kind.
- Everything TLDR produces (summaries, transcripts, chat history) is
  stored in a single local SQLite database on your disk. You can delete
  it at any time.

## What the extension can access, and why

TLDR's Chrome permissions, from `extension/manifest.json`:

| Permission | Why it's needed |
|---|---|
| `sidePanel` | Renders the summary/chat UI in Chrome's side panel instead of a popup. (Firefox has no equivalent permission — it uses a `sidebar_action` instead; see `extension/manifest.firefox.json`.) |
| `tabs` | Reads the active tab's URL and title so the daemon knows what to summarize, and follows tab switches so the side panel shows the right job. |
| `activeTab` | Grants temporary access to the page you're currently looking at, scoped to when you actually invoke the extension. |
| `scripting` | Injects a content script into the current page to extract its readable text (via Readability) before sending it to the daemon — this is how "summarize this page" gets the article text instead of just a URL. |
| `storage` | Saves your own settings (daemon URL, preferences) inside the browser's local extension storage. |
| `cookies` | Reads browser cookies for the specific site being processed, and forwards them to your own local daemon so it can fetch content that requires you to be logged in (e.g. a members-only video). See the dedicated section below — this is the most sensitive permission TLDR asks for, and it deserves a direct explanation. |
| Host permission: `http://localhost/*`, `http://127.0.0.1/*` | Lets the extension talk to the daemon running on your own machine. |
| Host permission: `<all_urls>` | Lets the extension read and act on whatever page you're actively asking it to summarize — which could be any site, since TLDR is a general-purpose page/video summarizer, not one scoped to a fixed list of domains. |

### Cookies — exactly what happens

This is worth being direct about, because "an extension reads your
cookies" sounds alarming out of context.

- **What triggers it**: cookies are only read and sent when you ask TLDR
  to process something — a YouTube video, a PDF behind a login, or any
  page/media URL. Nothing happens in the background without you invoking
  the extension.
- **Which cookies**: `extension/src/lib/cookies.js` reads cookies through
  Chrome's `chrome.cookies` API in two ways, both scoped to the content
  you're processing, never your whole cookie jar:
  - For YouTube jobs, `getCookiesForDomain(".youtube.com")` — every cookie
    for that domain and its subdomains, so an authenticated request (e.g.
    a members-only or age-restricted video) can succeed.
  - For a generic media URL or an http(s) PDF, `getCookiesForUrl(url)` —
    exactly the cookies a real browser request to that specific URL would
    carry (host, path, and Secure/HttpOnly rules all respected by Chrome
    itself), so no cookies from unrelated sibling subdomains leak in.
- **Where they go**: the cookies are attached to the `POST /jobs` request
  the extension already makes to your local daemon (see the `Cookie`
  model and `cookies` field in `daemon/src/api/schemas.py`). That request
  goes to `127.0.0.1`/`localhost` — your own daemon process, nothing
  external.
- **What the daemon does with them**: the daemon passes the cookies
  through to yt-dlp / its own HTTP fetch (`daemon/src/workers/pdf.py`'s
  `_fetch`, and the YouTube/media worker paths) so it can download the
  specific video or PDF you asked about, exactly as your browser would
  have. The cookies are used for that one fetch and are not written to
  the database, logged, or forwarded anywhere else — the daemon's own
  diagnostics export (see below) never includes them either.
- **Bottom line**: your cookies for the site you're processing go from
  your browser, to your own daemon, to that same site (via yt-dlp/httpx)
  — never to a third party, never to the extension author, never anywhere
  off your machine.

## Two backend modes — and what each actually means for your data

TLDR needs an OpenAI-compatible LLM backend (and optionally a Whisper
backend for audio/video without captions), configured in `daemon/src/config.py`
via `llm.base_url` / `whisper.base_url` in `tldr.yaml`. Which backend you
point it at determines where page/video content actually goes:

- **Local backend** (the default — e.g. mlx-openai-server, Ollama,
  LM Studio, llama.cpp, LocalAI running on your own machine): the text and
  images TLDR sends for summarization/Q&A go to a model process running on
  your own hardware. They never leave your machine for the purpose of
  inference.
- **Cloud backend** (optional, and something *you* configure — TLDR ships
  with no cloud endpoint and no built-in API key): if you set `llm.base_url`
  to a hosted provider (OpenAI, Anthropic, Google Gemini, OpenRouter,
  Together AI, etc.) and supply your own API key, the page text or
  transcript you're processing is sent to that provider, under your
  account, billed to you — the same as if you'd pasted it into that
  provider's own chat interface. This is entirely your choice; the project
  does not make it for you and has no relationship with any provider.

**Either way, fetching the content itself needs the internet.** Even in
local mode, the daemon still has to download the actual page HTML, PDF
bytes, or video/audio from wherever it's hosted (and, per the cookies
section above, may forward your cookies to do so for sites that require
login). That outbound fetch is not telemetry — it's the daemon retrieving
the exact material you asked it to process, from the site you asked about.

### Web search for Q&A

When you ask a follow-up question in the side panel, the daemon
(`daemon/src/llm/qa.py`) first asks the LLM whether the material you
already processed fully answers it. If not — or if that check fails for
any reason, which biases toward searching — the daemon runs a DuckDuckGo
text search (`daemon/src/workers/search.py`, `ddg_search_with_content`)
using a query built from your question, then fetches and extracts text
from a handful of the resulting pages (identifying itself with a
`TLDR-bot` user agent) to enrich the answer. This is default behavior for
Q&A, not something you opt into separately — but it can be turned off:
the `qa.web_search` setting in `tldr.yaml` (surfaced as a "Let follow-up
questions search the web" checkbox on the extension's options page)
defaults to `true`; set it to `false` and this step never runs at all —
no DuckDuckGo query, no page fetch — and the daemon answers only from the
processed material and the model's own training knowledge, stating
plainly when something isn't covered rather than guessing. It means: **a
search-engine query
derived from your question, and requests to whatever pages DuckDuckGo
returns, go out over the internet as part of answering that question** —
distinct from the LLM backend and from the original page/video fetch.
Nothing about your identity is attached beyond what any normal web request
carries (your daemon's outbound IP), and no cookies or extension data are
involved in this path.

You are not left guessing when this happens: while the search runs, the
daemon publishes a `searching` stage that the side panel displays, so a
question answered with help from the web is visibly different from one
answered from the material alone.

## What's stored locally, where, and how to delete it

TLDR stores everything in a single SQLite database, plus a couple of
on-disk caches, under the daemon's data directory (`daemon/src/paths.py`):

- **macOS (native install, the default)**:
  `~/Library/Application Support/tldr/data/tldr.db`, with the daemon's
  config at `~/Library/Application Support/tldr/tldr.yaml`.
- **Docker install**: inside the named `tldr` Docker volume, mounted at
  `/data` in the container.
- (Linux/Windows native installs use the platform-conventional config/data
  directories — see `daemon/src/paths.py`.)

What's in there: processed jobs (URL, title, extracted/transcribed text,
generated summary, transcript source), chat/Q&A message history per job,
cached translations, and — for video jobs where the Q&A flow looked at a
frame — small cached video-frame JPEGs and cached audio files under the
same data directory (`daemon/src/workers/frames.py`, `daemon/src/storage/repo.py`).
The daemon's own log file also lives here (see "Logs" below).

**Retention**: `storage.retention_days` in `tldr.yaml` (default 365,
`daemon/src/config.py`) is a real, active setting — a background sweep
(`daemon/src/workers/retention.py`) runs every 6 hours and deletes any job
(plus its cached audio/frames) older than that many days, counted from
when the job landed on your machine. Setting it to `0` disables the sweep
entirely; there is no separate "off" default — it just means old jobs are
kept indefinitely until you delete them yourself.

**Deleting your data**:
- Delete an individual job from the Library UI, or delete the whole
  database by removing the data directory above.
- Docker installs: `task reset` runs `docker compose down -v`, which
  destroys the SQLite volume entirely (this is intentionally destructive
  and asks for confirmation first).
- Native installs: stop the daemon and delete the
  `~/Library/Application Support/tldr/data` folder (or the equivalent
  platform data directory) yourself — there's presently no one-command
  reset outside Docker.

## No telemetry — and one thing worth flagging clearly

The extension and daemon code were checked for analytics SDKs, crash/error
reporting services (e.g. Sentry-style tools), and update-check pings.
None were found. The only outbound network activity the code makes is:

1. Requests to the LLM/Whisper backend you configured (local or cloud, per
   above).
2. Fetching the actual page, PDF, or video/audio you asked TLDR to
   process (including, for authenticated sites, the cookie forwarding
   described above).
3. yt-dlp's own self-update mechanism (it checks for and downloads newer
   extractor definitions so YouTube/site changes don't break extraction —
   this is yt-dlp's standard behavior, not something TLDR adds).
4. **The DuckDuckGo web search described above** — this is the one item
   that doesn't fit neatly into "your configured backend" or "the content
   you asked about," which is why it gets its own section rather than
   being folded into a blanket "no telemetry" claim.

No user accounts, no sign-up, no persistent identifiers sent anywhere, no
usage analytics.

## Diagnostics export

The options page has a "Build diagnostics report" button
(`extension/src/options/options.js`) that calls the daemon's
`GET /diagnostics` endpoint (`daemon/src/api/diagnostics.py`). This is
entirely manual and entirely local:

- Nothing is generated or sent unless you click the button.
- The result is an HTTP response your own browser fetches from your own
  daemon, then displays for you to copy or save as a file — it is never
  transmitted anywhere by the daemon itself. What you do with the
  downloaded file afterwards (e.g. paste it into a bug report) is up to
  you.
- Before anything is returned, the daemon redacts it: API keys (matched
  against the exact resolved key value for both the LLM and Whisper
  sections) are replaced, your home directory path is replaced with `~`,
  and any URL that isn't a loopback address (`127.0.0.1`/`localhost`/`::1`)
  is replaced with a placeholder — so the page or video URLs you've
  actually processed are never included. Page/transcript content, job
  titles, and job URLs are never put into the report in the first place
  (not just redacted after the fact).

## Logs

The daemon writes its own rotating log file
(`daemon/src/logging_setup.py`) under the data directory (`logs/daemon.log`,
capped at 5 MB × 4 backups = 20 MB), used only for local troubleshooting.
Two things worth noting:

- Web request logging (`uvicorn.access`) has a filter that strips
  everything after `?` from logged request paths, specifically because
  query strings on this daemon routinely contain the page/video URL being
  processed — so that log line doesn't become a browsing-history record.
- These logs stay on your disk. They are read by the diagnostics export
  above (redacted, and only the last ~300 lines) if you choose to build
  one, and are never transmitted anywhere on their own.

## Questions

This is a one-person, no-company project. If you have a question about
any of the above, or think something here doesn't match what the code
does, please open an issue:
[github.com/melnikaite/tldr-free/issues](https://github.com/melnikaite/tldr-free/issues).
