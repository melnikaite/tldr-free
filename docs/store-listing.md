# Chrome Web Store listing — copy-paste source

This file exists so the developer can copy the right text into the Chrome
Web Store dashboard without re-deriving it each time. Every claim below is
checked against the current code (`extension/manifest.json`,
`extension/src/lib/cookies.js`, `daemon/src/api/schemas.py`, `PRIVACY.md`,
`README.md`) as of 2026-08-18. Nothing here is invented; where the dashboard
asks something the code doesn't answer, that's called out instead of guessed.

Related existing docs, not duplicated here: [`../PRIVACY.md`](../PRIVACY.md)
(full privacy policy — link this from the store's "Privacy policy URL"
field), [`permissions-justification.md`](permissions-justification.md) (a
prior, more code-forensic pass over the same permissions), and
[`privacy-policy.md`](privacy-policy.md) (an earlier, shorter privacy note —
`PRIVACY.md` at the repo root is the current, more detailed one; prefer it).

---

## 1. Permission justifications

Source list: `extension/permissions` and `extension/host_permissions` in
`extension/manifest.json`.

**sidePanel**
Renders TLDR's summary, transcript and Q&A interface in Chrome's side panel
— this is the extension's entire UI surface.

**tabs**
Lets the side panel follow the active tab (detect tab switches and URL
changes) so it shows the summary for whatever page you're currently on,
and lets a clicked `[MM:SS]` timecode find and seek the right video tab.

**activeTab**
Grants temporary access to the page you're on at the moment you click the
toolbar button, so the content script can extract that page's text without
requesting standing access to every site up front.

**scripting**
Injects a content script into the current page, on your request only, to
extract its readable article text (or detect playable video/audio) before
sending it to the local daemon for summarization — this is the mechanism
behind "summarize this page."

**storage**
Stores the extension's own settings locally in the browser (daemon URL,
output language, and similar preferences) — no account, no sync to any
server.

**cookies**
Reads the cookies for the specific site or media URL you asked TLDR to
process, and forwards them to the local daemon on `127.0.0.1` only, so it
can download content you're already authorized to see — for example a
members-only or age-restricted YouTube video, or a PDF that sits behind a
login. Two call sites, both scoped to the content being processed, never
the whole cookie jar: `getCookiesForDomain(".youtube.com")` for YouTube
jobs, and `getCookiesForUrl(url)` (exactly the cookies a real browser
request to that URL would send) for a generic media URL or an http(s) PDF.
Cookies never leave the user's machine — they go from the browser to the
user's own daemon process and from there to the site itself, exactly as a
normal authenticated request would.

**Host permission: `http://localhost/*`, `http://127.0.0.1/*`**
Lets the extension talk to the daemon, which runs on the user's own
machine on these addresses. This is the extension's only backend.

**Host permission: `<all_urls>`**
TLDR is a general-purpose "summarize whatever I'm looking at" tool — the
user can invoke it on any page, PDF, video, or podcast embed, and which
site that will be is not known ahead of time. Broad host access is also
what lets the side-panel button (not just the toolbar icon) trigger
extraction, and what lets `chrome.cookies.getAll({url})` and the PDF fetch
path reach arbitrary CDN/media domains that differ from the page's own
origin.

---

## 2. Single purpose statement

> TLDR generates summaries, transcripts, translations, and answers to
> follow-up questions for the web page, PDF, or audio/video the user is
> currently viewing.

Everything the extension does — summary, transcript, translation, Q&A,
video-frame lookups — is a view onto that one material the user is
currently looking at. That's the one purpose the rest follows from.

---

## 3. Data use disclosure

Based on `PRIVACY.md` (repo root), mapped onto the categories the Chrome
Web Store dashboard's data-disclosure form asks about.

**Does the item collect or use any of the following data types?**

| Category | Collected? | Notes |
|---|---|---|
| Personally identifiable information | No | No accounts, no sign-up, no identifiers sent anywhere. |
| Health information | No | — |
| Financial and payment information | No | — |
| Authentication information | Handled, not collected | Session cookies for the site/media being processed are read and forwarded — see below. Not passwords; not stored or transmitted anywhere but the user's own daemon and the site itself. |
| Personal communications | No | — |
| Location | No | — |
| Web history | Handled, not collected by us | The extension reads the URL/title of the active tab to know what to process, and sends it to the user's own local daemon so the daemon can fetch that page. This never reaches the developer or any third party by default. |
| User activity | No | No analytics, no usage tracking, no telemetry, no crash reporting. |
| Website content | Handled, not collected by us | Page text, PDF content, and video/audio are sent to the user's own local daemon for processing. See "Where does it go" below — this is the core function, not incidental collection. |

**Certifications the dashboard requires:**
- Not sold to third parties: true — there is no server to sell it from, and no third-party relationship in the code.
- Not used for purposes unrelated to the item's single purpose: true.
- Not used to determine creditworthiness or for lending: true.

**Where does the data go (the nuance to get right in the form's free-text
box):**

The extension sends page/video/PDF content to a daemon running locally on
the *same machine as the browser* — `http://127.0.0.1:8765` or
`localhost`, not a server operated by the developer. That is the default
and only behavior of the extension itself.

From there, whether anything leaves the machine depends entirely on how
the user configured their own daemon:

- **Local model (default)**: the daemon hands the content to a model
  process also running on the user's machine (Ollama, LM Studio,
  mlx-openai-server, etc.). Nothing leaves the machine for inference.
- **Cloud model (opt-in, user-configured)**: if the user points
  `llm.base_url` at a hosted provider and supplies their own API key, the
  page text or transcript is sent to *that provider*, under the user's own
  account — not to the developer, and not to any endpoint the developer
  operates or has access to.

Separately, when answering a follow-up question, the daemon may run a
DuckDuckGo web search and fetch a few resulting pages to help answer the
question — this is on by default and controlled by the `qa.web_search`
setting (exposed in the extension's options page as "Let follow-up
questions search the web"); turning it off stops that network activity
entirely for Q&A.

**Privacy policy URL to submit**: the repo's
[`PRIVACY.md`](../PRIVACY.md) — link to its raw or rendered GitHub URL,
e.g. `https://github.com/melnikaite/tldr-free/blob/main/PRIVACY.md`.

---

## 4. Reviewer notes / test instructions

**The daemon is required in all cases.** A cloud API key changes which
model answers the request — it does not remove the need for the local
daemon. There is no hosted version of TLDR the reviewer can hit without
installing anything.

**1. Install the daemon** (native, no Docker — from `README.md`, "Install —
native, no Docker"):

```bash
curl -fsSL https://raw.githubusercontent.com/melnikaite/tldr-free/main/scripts/install-uv.sh | sh
```

This installs `uv` if missing, installs the daemon as a uv tool, writes a
default config, and registers a user-level autostart service (launchd on
macOS, systemd user unit on Linux), then waits for `/health`. The daemon
listens on `127.0.0.1:8765`.

**2. Point it at a model.** The installer's default config expects a local
OpenAI-compatible backend. To review with a cloud model instead, edit the
daemon's `tldr.yaml` (or use the extension's options page) and set:

```yaml
llm:
  base_url: <PLACEHOLDER — e.g. https://api.openai.com/v1>
  api_key: <PLACEHOLDER — do not commit or reuse this key beyond review>
  model: <PLACEHOLDER — e.g. gpt-5-mini>
```

*(Owner note, not for the reviewer: issue a review key with the lowest
practical spend cap on the provider's dashboard, and revoke it once the
listing is approved.)*

**3. Verify it works:**
- Load the unpacked extension (`chrome://extensions` → Developer mode →
  Load unpacked → `extension/` directory).
- Open any article page or a YouTube video.
- Click the TLDR toolbar icon — the side panel opens and a summary streams
  in with clickable `[MM:SS]` timecodes (timecodes only apply to
  video/audio jobs).
- Ask a follow-up question in the chat box at the bottom of the panel and
  confirm an answer streams back.
- If backend setup needs checking: the extension's options page (right-click
  the toolbar icon → Options) has a **Test setup** button under the LLM
  section that probes the configured backend directly and reports/suggests
  corrected settings — useful if the reviewer isn't sure the base
  URL/model/key combination is right.

---

## 5. Listing copy

### Short description

Copied verbatim from `description` in `extension/manifest.json` (identical
in `extension/manifest.firefox.json`):

> Summaries, subtitles, translation and Q&A for pages, PDFs, audio and
> video. Free and 100% private on your own machine.

### Detailed description

> **Requires a small local program (the "daemon") running on your own
> computer. TLDR is not a cloud service** — there is no TLDR server
> anywhere. One command installs it; see the Setup section below.
>
> TLDR is a Chrome side-panel extension that summarizes whatever you're
> looking at — an article, a PDF, a YouTube video, a podcast embed, or any
> other audio/video yt-dlp can extract — and lets you ask follow-up
> questions about it. Click the toolbar button and a streaming summary
> appears in the side panel, with clickable `[MM:SS]` timecodes that seek
> the video straight to that moment.
>
> **What it does:**
> - Summarizes pages, PDFs, YouTube videos and other audio/video, with
>   clickable timecodes in the summary, in Q&A answers, and in the full
>   transcript.
> - Full transcripts, translatable on demand into any language you choose.
> - A chat box for follow-up questions about the material — persisted per
>   item, so it's still there after a restart.
> - Q&A can look at the actual video frame when a question is about
>   something the speaker points at on screen, not just the transcript
>   text.
> - A persistent local library: everything you process is saved to a
>   SQLite database on your own disk, searchable and exportable, and never
>   expires unless you set a retention window.
> - Works with any OpenAI-compatible LLM/Whisper backend — Ollama, LM
>   Studio, mlx-openai-server, llama.cpp, vLLM, or a cloud provider of your
>   choice with your own API key.
>
> **Why TLDR instead of a built-in or cloud summarizer:** it handles audio
> and video, not just text pages; it uses whatever context window your own
> model supports, so a two-hour podcast gets summarized in one pass instead
> of chopped into snippets; and your summaries, transcripts and chat
> history live in a local library that survives restarts, instead of
> disappearing after one session.
>
> **On privacy**: in local mode, the page or video content you process is
> sent to a model running on your own machine — it does not reach the
> developer or any third party. If you choose to configure a cloud model
> yourself, your content goes to that provider, under your own account,
> the same as pasting it into that provider's chat interface — that's your
> choice to make, not something TLDR does by default. This does not cover
> the network requests inherent to fetching the material itself: TLDR
> still has to download the page, PDF, or video/audio from wherever it's
> hosted, and, for content requiring login, forward the relevant cookies to
> do so. Full details: [PRIVACY.md](https://github.com/melnikaite/tldr-free/blob/main/PRIVACY.md).

---

## 6. Submission checklist

- [ ] **Developer account** — one-time $5 registration fee on the Chrome
      Web Store dashboard (not yet done as of this writing; nothing in the
      repo tracks this).
- [x] **Icons** — all three sizes the manifest declares are present:
      `extension/public/icons/icon16.png`, `icon48.png`, `icon128.png`
      (plus a source `icon.svg`). Nothing further needed here.
- [x] **Screenshots** — `docs/screenshots/` covers summary/timecodes, PDF,
      podcast/Whisper, video-frame Q&A and the library, and now also the
      first-run states (`sidepanel-welcome.png`, `sidepanel-welcome-model.png`)
      so the daemon-required nature is visible in the images and not only in
      the text. Note the size rule: store screenshots must be 1280×800 or
      640×400, which a 400px-wide panel capture is not — the listing-ready
      composite is `docs/store-assets/screenshot-welcome-1280x800.png`, with
      the real panel document rendered inside it.
- [ ] **Privacy policy** — `PRIVACY.md` (repo root) exists and is current
      (dated 2026-08-18); use its GitHub URL as the store's privacy policy
      link.
- [x] **Manifest version** — `extension/manifest.json` and
      `extension/manifest.firefox.json` both declare `"version": "1.0.0"`,
      and `DAEMON_VERSION` matches so a diagnostics report doesn't show two
      different numbers for one release. `DAEMON_API_VERSION` is deliberately
      NOT tied to this — it tracks schema compatibility, not the release.
      Bump before each submission per the store's version-monotonicity rule.
- [x] **Promo tiles** — `docs/store-assets/promo-small-440x280.png` (the
      card image) and `promo-marquee-1400x560.png` (featured placements
      only). Brand-only by design — no fabricated UI, no screenshot
      collage — in the palette of `docs/logo-banner.svg`. Regenerable from
      the harnesses in that directory; see its README.
- [ ] **Still to do by hand in the dashboard**: category selection, the
      developer account and its one-time $5 fee, and the reviewer API key
      (issue it with a small spend cap, paste it into the reviewer notes,
      revoke it after approval — it must not live in this repository).

---

## Discrepancies between the task brief and the code

- The brief's guess about `<all_urls>` as "the user clicks a button on an
  arbitrary page" is correct but incomplete — `permissions-justification.md`
  documents a second reason: the side-panel button (as opposed to the
  toolbar icon) doesn't qualify for an `activeTab` grant, so `<all_urls>`
  is also what makes the in-panel "process this page" action work. Folded
  into the justification above.
- The brief names the LLM section's probe button "Test setup" — confirmed
  exactly right: `extension/src/options/index.html` line 131 has
  `id="test-connection"` with visible label **"Test setup"** for the LLM
  section specifically. The Whisper section's equivalent button is
  labeled **"Test connection"**, a different string — the reviewer
  instructions above only cite the LLM one ("Test setup"), matching what
  the brief asked for.
- `docs/permissions-justification.md` and `docs/privacy-policy.md` already
  existed before this task, covering similar ground at a more technical
  level (audited 2026-06-11). I did not modify either — `store-listing.md`
  cross-references them and defers to `PRIVACY.md` at the repo root (dated
  2026-08-18, more detailed) as the canonical privacy source per the task
  brief.
- No `identity` permission and no OAuth flow anywhere in the manifest or
  code, consistent with the brief — nothing written about it.
- One thing the brief didn't ask about but is worth the owner knowing: the
  Chrome Web Store dashboard also wants promotional tile images at fixed
  pixel dimensions, distinct from listing screenshots. Nothing like that
  exists in `docs/screenshots/` or anywhere else in the repo; flagged in
  the checklist above rather than fabricated.
