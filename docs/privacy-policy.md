# Privacy Policy — TLDR free

**Effective date: 2026-06-11**

## The short version

Everything stays on your machine. The extension talks only to a daemon
running on your own computer (`127.0.0.1`). We operate no servers, collect
no data, and embed no analytics. There is nothing for us to sell, leak,
or subpoena — we never see your data in the first place.

## What the extension processes, and where it goes

When you ask TLDR to summarise a page, video, or PDF:

- **Page text / video transcripts / PDF content** are extracted in your
  browser and sent to **your local daemon** at `http://127.0.0.1:8765`
  (configurable, local by default). Nothing is sent to the extension's
  authors or any third-party service by the extension.
- **Summaries, transcripts, translations and chat history** are stored in
  a **local SQLite database** on your machine. Retention is configurable
  (`storage.retention_days`); you can delete any item from the Library at
  any time, and the uninstall script can wipe the whole database.
- **Cookies**: when you process a video, embedded media, or an
  access-protected PDF, the extension reads the cookies for that site and
  passes them to **your local daemon only**, so that the daemon can download
  content you already have access to (e.g. age-gated or membership videos).
  Cookies are never sent anywhere except `127.0.0.1` and are not stored by
  the extension.

## Network connections made by the daemon

The daemon (which runs on your machine, under your control) connects to:

1. **The site hosting the content** (e.g. YouTube, the page's media CDN) —
   to download transcripts, audio, or PDFs you asked it to process.
2. **Your configured AI backend** — local by default (Ollama, LM Studio,
   LocalAI, mlx, …). If you deliberately point `config/tldr.yaml` at a
   remote backend, your content goes to that backend; that is your
   configuration choice, not a default.
3. **PyPI** on startup — to keep the `yt-dlp` downloader up to date. No
   user data is included in that request.

## What we collect

Nothing. No analytics, no telemetry, no crash reporting, no accounts,
no unique identifiers. The project has no backend infrastructure.

## Browser permissions

The extension asks for broad host access and the `cookies` permission
solely to support "summarise anything on any site" and cookie-authorised
media downloads, as described above. A per-permission breakdown lives in
[permissions-justification.md](permissions-justification.md).

## Changes

Changes to this policy are published in this file's git history — every
edit is publicly auditable.

## Contact

Open an issue: <https://github.com/melnikaite/tldr-free/issues>
