<p align="center">
  <img src="docs/logo-banner.svg" alt="TLDR free — local summaries and Q&A" width="600" />
</p>

<p align="center">
  <strong>Local-first summaries, transcripts and Q&amp;A for web pages, PDFs, YouTube —
  and any audio or video your browser can see.</strong><br/>
  Clickable timecodes. Persistent library. Open source. Local by default —
  bring your own cloud model if you want one.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python 3.11">
  <img src="https://img.shields.io/badge/Chrome-MV3%20side%20panel-ffce44.svg" alt="Chrome MV3 side panel">
  <a href="CLAUDE.md"><img src="https://img.shields.io/badge/AI%20agent-ready%20docs-8A2BE2.svg" alt="AI-agent-ready docs"></a>
</p>

---

TLDR is a Chrome side-panel extension plus a small FastAPI daemon. Click the
toolbar button on any page, PDF, YouTube video or podcast embed and you get a
streaming summary with clickable `[MM:SS]` timecodes, plus a chat box to ask
follow-up questions about the same material. Everything you process lands in
a local library (SQLite on your disk) you can come back to any time. The
daemon talks to an LLM/Whisper backend over the **OpenAI-compatible HTTP
API** — pick whatever runner you like.

<table>
  <tr>
    <td><img src="docs/screenshots/sidepanel-youtube.png" alt="YouTube video summary with clickable timecodes" /></td>
    <td><img src="docs/screenshots/sidepanel-pdf.png" alt="PDF paper summary" /></td>
    <td><img src="docs/screenshots/sidepanel-podcast.png" alt="Podcast audio summary via local Whisper" /></td>
  </tr>
  <tr align="center">
    <td>YouTube video</td>
    <td>PDF paper</td>
    <td>Podcast audio (local Whisper)</td>
  </tr>
</table>

<p align="center">
  <img src="docs/screenshots/library.png" alt="Local library of processed pages, videos and podcasts" width="820" />
</p>

<!-- TODO: 30-second demo GIF here (toolbar click → streaming summary →
     timecode click seeks the video → Q&A). See distribution-plan.md §4. -->

## Why TLDR, not yet another summarizer?

Browsers are growing built-in page summaries, and cloud summarizer extensions
are a dime a dozen. TLDR aims at what those don't do:

- **Any audio or video, not just pages.** If yt-dlp can extract it — a
  YouTube video, a podcast embed, a raw `<video>` tag — TLDR gets a
  transcript (official captions → auto-captions → local Whisper) and
  summarises it.
- **Clickable `[MM:SS]` timecodes** in the summary, in Q&A answers and in
  the full transcript. Click one and the player seeks right there.
- **A persistent local library.** Summaries, transcripts, translations and
  per-item chat history live in SQLite on your machine, survive restarts
  and never expire unless you say so.
- **Your model, your context window.** Any OpenAI-compatible backend —
  a local 128K-context model or a cloud model with a much bigger window —
  a two-hour podcast summarised in one pass, not snippets fed to a tiny
  built-in model.
- **Transcripts are first-class.** Read the full text, translate it into
  your language on demand, navigate by timecode.

If all you need is "shorten this article", built-in browser AI is fine. TLDR
is for *"I have 40 tabs, three lectures and a podcast backlog — condense all
of it, keep it, and keep it private."*

## Features

- **Side panel that follows the active tab.** Switch tabs and you see the
  cached summary (or "no summary yet"). Click a `[MM:SS]` timecode and the
  panel doesn't reset — same canonical URL.
- **Streaming everywhere.** Watch tokens appear live for both the summary and
  the Q&A.
- **Two paths for YouTube transcripts.** First the official transcript API,
  then yt-dlp's auto-captions, then Whisper as a last resort. Timecodes
  preserved on the first two paths.
- **Beyond YouTube: any media on the page.** Native `<video>`/`<audio>` tags
  and whitelisted embeds are detected and transcribed through the same chain;
  if several candidates are found you pick which one to process.
- **Transcript tab with translation.** The full transcript lives next to the
  summary, translated on demand into any language, navigable by timecode.
- **PDFs work too.** http(s) or local `file://` PDFs are parsed in the
  side panel via pdf.js and summarised like any other page. (Image-only
  scans need OCR first — not built in.)
- **Persistent chat per job.** Q&A history is stored in SQLite, survives tab
  switches and browser restarts.
- **Pause/resume all background ML** when you need the machine for foreground
  work. The in-flight step finishes; the next step parks at a checkpoint
  until you click Resume. Q&A stays responsive throughout.
- **Auto retry of failed jobs** — keeps the cached audio file so the slow
  yt-dlp step is skipped on retry.
- **No build step for the extension.** Vanilla JS + ES modules. Edit a file,
  click the reload icon.

## Quick start

TLDR needs two OpenAI-compatible endpoints: one for the LLM (`llm.base_url`)
and one for Whisper transcription (`whisper.base_url`). They can be the same
server or different ones — configure them independently in `config/tldr.yaml`.

### LLM backend (required)

Any OpenAI-compatible server works — local or cloud. Local is the default
and the point of the project; here are the popular local choices first,
cloud backends further down.

| Backend | Platform | LLM | Whisper | Notes |
|---|---|---|---|---|
| [**Ollama**](https://ollama.com/) | Any OS, CPU / GPU | ✅ | ❌ | [Download](https://ollama.com/download), then `ollama pull gemma4:e4b` |
| [**LM Studio**](https://lmstudio.ai/) | macOS / Windows | ✅ | ❌ | GUI; enable local server on port 1234 |
| [**mlx-openai-server**](https://pypi.org/project/mlx-openai-server/) | macOS Apple Silicon | ✅ | ✅ | Fastest local; `task install:mlx` |
| [**llama-server**](https://github.com/ggml-org/llama.cpp) | Any OS | ✅ | ❌ | `brew install llama.cpp` |
| vLLM, openai-edge, … | Any OS | ✅ | ❌ | Any OpenAI-compat endpoint |

> **Context window — expand it or long pages get silently truncated.**
> Gemma 4 E4B supports 128K but both Ollama and LM Studio default to a much smaller window.
>
> **Ollama** — create a custom variant with the full context:
> ```bash
> printf 'FROM gemma4:e4b\nPARAMETER num_ctx 131072\n' > Modelfile
> ollama create gemma4:e4b-128k -f Modelfile
> ```
> Then set `model: gemma4:e4b-128k` and `context_length: 131072` in `config/tldr.yaml`.
>
> **LM Studio** — after loading the model, open its settings and set **Context Length** to `131072`.

### Cloud backends (optional)

`llm.base_url` can point at any OpenAI-compatible **cloud** endpoint just as
well — same daemon, same pipeline, no code changes. Ready-made blocks for
the usual suspects are in `config/tldr.yaml.example`; here's the gist:

**OpenAI**
```yaml
llm:
  base_url: https://api.openai.com/v1
  api_key_file: ~/.config/tldr/openai.key   # see "API key storage" below
  model: gpt-5                              # or gpt-5-mini, o4-mini, ...
  context_length: 400000                    # check your model's window — not gemma's 128K
  single_pass_token_limit: 240000           # ~60% of context_length
  max_concurrent_calls: 3                   # hosted backends tolerate more parallelism than a laptop GPU
```

**Anthropic** (via its [OpenAI-compatible endpoint](https://docs.anthropic.com/en/api/openai-sdk))
```yaml
llm:
  base_url: https://api.anthropic.com/v1
  api_key_file: ~/.config/tldr/anthropic.key
  model: claude-sonnet-4-5
  context_length: 200000
  single_pass_token_limit: 120000
  max_concurrent_calls: 3
```

**Google Gemini** (via its [OpenAI-compatible endpoint](https://ai.google.dev/gemini-api/docs/openai))
```yaml
llm:
  base_url: https://generativelanguage.googleapis.com/v1beta/openai/
  api_key_file: ~/.config/tldr/gemini.key
  model: gemini-2.5-flash
  context_length: 1000000
  single_pass_token_limit: 600000
  max_concurrent_calls: 3
```

**OpenRouter** (one key, routes to almost any hosted model)
```yaml
llm:
  base_url: https://openrouter.ai/api/v1
  api_key_file: ~/.config/tldr/openrouter.key
  model: openai/gpt-5                       # provider/model — pick anything OpenRouter hosts
  context_length: 400000                    # match whichever model you route to
  single_pass_token_limit: 240000
  max_concurrent_calls: 3
```

Whichever provider you pick, set `context_length` / `single_pass_token_limit`
to *that model's* window, not the 128K figure the local Gemma blocks use —
otherwise you're leaving most of a paid context window unused (or, the other
way, tripping the backend's real limit).

Reasoning models (GPT-5/o-series, and "thinking" models generally) spend
part of their output budget on hidden reasoning before the visible answer
starts. `reasoning_headroom_tokens` (default `4000`) reserves room for that
so the answer doesn't get cut off partway. `token_param` (default `auto`)
and `send_temperature` are escape hatches for when the daemon's automatic
backend-dialect detection guesses wrong — normally you don't need to set
them; `auto` probes the backend's dialect from its first response (and
retries once on an HTTP 400), so GPT-5/o-series work out of the box.

#### API key storage

Four ways to give the daemon a key, in priority order (first match wins):

1. **`TLDR__LLM__API_KEY` environment variable** — overrides everything
   below. Good for CI, or when the key already lives in your service's
   environment.
2. **OS keychain** — `api_key_keychain` (service name) +
   `api_key_keychain_account` (account name), backed by macOS Keychain or
   the Linux Secret Service. Requires the optional `keychain` extra:
   `uv tool install --force './daemon[keychain]'` natively, or
   `tldr-daemon[keychain]` in the Docker image. Store the secret once —
   ```bash
   security add-generic-password -s tldr-daemon -a openai -w '<your-api-key>'   # macOS
   ```
   — then reference it:
   ```yaml
   api_key_keychain: tldr-daemon
   api_key_keychain_account: openai
   ```
   Note: `uv tool install --force` rebuilds the daemon binary, so macOS will
   ask you to re-approve Keychain access again the first time it runs after
   an upgrade — that's expected, not a bug.
3. **`api_key_file`** (recommended for real cloud keys) — a path (`~`
   expands) to a file holding just the key, locked to `0600`:
   ```bash
   install -m 600 /dev/null ~/.config/tldr/openai.key
   printf '%s' 'sk-...' > ~/.config/tldr/openai.key
   ```
   (or `umask 077` before creating the file by hand.)
4. **`api_key`** inline in `tldr.yaml` — fine for local backends that ignore
   the value (`ollama`, `dummy`, `lm-studio`, …). Avoid it for real cloud
   keys: `tldr.yaml` is created `0600`, but a plaintext key in a config file
   you might `cat`, screen-share, or back up is still a plaintext key.

For systemd (native Linux install), an alternative to all of the above is an
`EnvironmentFile` on the `tldr-daemon` unit setting `TLDR__LLM__API_KEY`,
kept outside the repo with its own restrictive permissions.

#### Privacy and cost, with a cloud backend

Point `llm.base_url` at a cloud provider and the page text or transcript
you process leaves your machine and goes to that provider — same as pasting
it into their chat UI. The "nothing leaves your machine" story only holds
for a local backend; going cloud is an explicit trade you're opting into.
Cloud inference is billed by the provider per token, and cloud Whisper
transcription (e.g. OpenAI's `whisper-1`) is billed per minute — both are
the provider's cost, not TLDR's.

### Whisper backend (optional — only for YouTube without captions)

Required only when `youtube-transcript-api` and yt-dlp captions both fail.
If you skip it, those videos will error instead of transcribing via Whisper.

| Backend | Platform | Notes |
|---|---|---|
| **mlx-openai-server** | macOS Apple Silicon | Already included if you use it for LLM |
| [**faster-whisper-server**](https://github.com/fedirz/faster-whisper-server) | Any OS, CPU / GPU | `docker run -p 8000:8000 fedirz/faster-whisper-server` |
| [**whisper.cpp server**](https://github.com/ggml-org/whisper.cpp) | Any OS | `brew install whisper-cpp`; start with `whisper-server` |
| **OpenAI Whisper API** | Cloud | `base_url: https://api.openai.com/v1`, `model: whisper-1` — pay-per-minute |

### Install — native, no Docker (recommended)

One command; works on macOS and Linux (Windows is experimental):

```bash
curl -fsSL https://raw.githubusercontent.com/melnikaite/tldr-free/main/scripts/install-uv.sh | sh
# or from a checkout: task install:uv
```

The script installs [uv](https://docs.astral.sh/uv/) if missing, installs the
daemon as a uv tool, creates the config from the packaged template, registers
a user-level autostart service (launchd LaunchAgent on macOS, systemd user
unit on Linux) and waits for `/health`.

Lifecycle after that:

```bash
tldr-daemon service status      # unit present? /health ok?
tldr-daemon service uninstall   # stop + remove autostart
tldr-daemon service install     # register + start again (= restart)
tldr-daemon                     # or run in the foreground, no service
task uninstall:uv               # remove everything (keeps your data)
```

Config and data live in the platform-conventional dirs —
`~/Library/Application Support/tldr/` on macOS,
`$XDG_CONFIG_HOME/tldr` + `$XDG_DATA_HOME/tldr` on Linux. Edit
`tldr.yaml` there (backend URLs point at `127.0.0.1`, and it's created
`0600`), then restart the service. Switching `llm.base_url` to a cloud
provider works the same way here as in Docker — see
[Cloud backends](#cloud-backends-optional) and
[API key storage](#api-key-storage); on Linux, an `EnvironmentFile` on the
`tldr-daemon` systemd unit is a good place for `TLDR__LLM__API_KEY` instead
of putting the key in `tldr.yaml` at all.

To **update**: `uv tool install --force git+https://github.com/melnikaite/tldr-free#subdirectory=daemon`
(or `--force ./daemon` from a checkout), then restart the service. yt-dlp and
youtube-transcript-api self-update on every daemon start, so YouTube breakage
usually fixes itself with a restart.

`ffmpeg` on PATH is needed for the Whisper fallback
(`brew install ffmpeg` / `apt install ffmpeg`).

### Install — Docker

```bash
task install            # config + daemon image + extension vendor libs
# Edit config/tldr.yaml — set llm.base_url (and whisper.base_url if needed)
# Ready-made blocks for Ollama, LM Studio, mlx, llama-server, and cloud
# providers (OpenAI, Anthropic, Gemini, OpenRouter) are in the file
task up                 # starts daemon (and mlx-server if you ran task install:mlx)
task status             # health check
```

If you use `task install:mlx`, the live mlx-server config lives at
`~/.mlx-server/config.yaml` — outside this repo so you can share it with
other tools. Edit that file, `task down && task up`, done.

Load the extension once:

1. Open `chrome://extensions`, enable Developer mode.
2. Click "Load unpacked", select the `extension/` directory.
3. After source changes, hit the reload icon — no rebuild step.

## Daily commands

Native mode: `tldr-daemon service status|install|uninstall` (see above).
Docker mode:

```
task up          # start
task down        # stop (sqlite volume preserved)
task status      # health check
task logs        # tail daemon logs (mlx logs are in ~/.mlx-server/logs/server.{out,err}.log)
task reset       # destructive: wipes the database volume (asks for confirmation)
task test        # ruff + mypy + pytest inside the daemon container
```

## Configuration

`config/tldr.yaml` (created from `tldr.yaml.example` on `task install`, or
from the packaged template on first native run — see below) holds the
backend URLs, API keys, output language, retry behaviour, retention window,
and concurrency caps. It's created with `0600` permissions so only your user
account can read it.

`llm.base_url` and `whisper.base_url` are **independent** — point them at the
same server or different ones:

```yaml
# Example: LM Studio for LLM, mlx-server for Whisper
llm:
  base_url: http://host.docker.internal:1234/v1    # LM Studio
  model: google/gemma-4-e4b                        # model ID shown by LM Studio
  context_length: 131072                           # must match what the backend loaded
  single_pass_token_limit: 80000                   # ~60% of context_length
  max_concurrent_calls: 1

whisper:
  base_url: http://host.docker.internal:18000/v1   # mlx-openai-server
  model: whisper

output:
  language: en                                     # ISO 639-1 or full name

youtube:
  subtitle_lang_preferences: ["en", "ru"]

storage:
  retention_days: 365                              # 0 disables auto-cleanup
```

**`context_length` must match what the backend actually loaded** — a mismatch
causes "n_keep >= n_ctx" errors. Check with `lms ps` (LM Studio) or look at
the `context_length` field in `~/.mlx-server/config.yaml` (mlx-server).
`single_pass_token_limit` caps the input before map-reduce kicks in; keep it
at ~60–70% of `context_length` to leave room for the system prompt and output.

**Editing settings from the extension** (backend/model/API key/output
language) is also possible without touching YAML by hand: open it via
`chrome://extensions` → TLDR → Details → Extension options (or right-click
the toolbar icon → Options). The page's **Test connection** button is what
answers "is my key even valid?" — it calls `POST /config/test` below. The
daemon exposes `GET /config`, `PATCH /config`, and `POST /config/test` (probes
credentials — reachability + a minimal completion — without saving). Partial
`PATCH` writes land in `tldr.local.yaml`, a second file created next to
`tldr.yaml` and deep-merged on top of it at load time (env var overrides
still win over both); `tldr.yaml` itself is never rewritten, so its comments
and backend examples stay intact. Both files are `0600`. `GET`/`PATCH`
responses never include the API key itself — only `api_key_set` (bool),
`api_key_hint` (last 4 chars), and `api_key_source` (`env` / `keychain` /
`file` / `inline` / `none`). Picking `api_key_storage: file` (the default)
or `keychain` via `PATCH` keeps the key out of both YAML files entirely.
Changing `llm.max_concurrent_calls` needs a daemon restart to take effect —
the response's `restart_required` flag says so.

`tldr.yaml.example` has ready-made blocks for each backend combination:
mlx-openai-server (LLM+Whisper), LM Studio+mlx, Ollama, llama-server+whisper.cpp,
LLM-only (no Whisper), and the cloud providers from
[Cloud backends](#cloud-backends-optional) above. For a cloud `llm.base_url`,
set `context_length` / `single_pass_token_limit` to that model's context
window, not the 128K figure the local Gemma examples use, and prefer
`api_key_file` (or the keychain fields) over inline `api_key` — see
[API key storage](#api-key-storage).

To free the machine for foreground work, click the **Pause processing**
button in the Library page (top-right). It pauses everything: the Whisper
queue stops picking up new transcriptions, and any new page/YouTube job
parks before the LLM call. In-flight work finishes; QA stays unblocked.
The same gate from the API:

```bash
curl -X POST http://localhost:8765/workers/pause
curl -X POST http://localhost:8765/workers/resume
curl       http://localhost:8765/workers           # status
```

State is in-memory and resets on daemon restart. To space jobs out without
fully pausing, set `workers.cooldown_seconds` in `config/tldr.yaml` — the
worker waits that many seconds between consecutive jobs.

## Architecture

```
┌─ Host ────────────────────────────────────────────────────────┐
│                                                               │
│  Any OpenAI-compatible LLM/Whisper backend                    │
│  (Ollama / LM Studio / mlx-openai-server / vLLM / ...)        │
│                                                               │
│  ┌─ Docker: daemon (port 8765) ─────────────────────────────┐ │
│  │  FastAPI                                                 │ │
│  │  Async POST /jobs → background pipeline                  │ │
│  │  Per-job event broker fans out stage / delta / done      │ │
│  │  /ai/stream — single SSE endpoint for summary + Q&A      │ │
│  │  Whisper queue with pause/resume                         │ │
│  │  Retry endpoint reuses cached audio                      │ │
│  │  yt-dlp + auto-captions + Whisper fallback chain         │ │
│  │  SQLite in named volume `tldr-data`                      │ │
│  └──────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
                            ▲
                            │ http://localhost:8765
                            │
        ┌─ Chrome extension (MV3, vanilla JS) ─────┐
        │  Side panel follows the active tab        │
        │  Live timeline + streaming markdown       │
        │  Library page with retry / delete / pause │
        └───────────────────────────────────────────┘
```

More detail in [`.claude/architecture.md`](.claude/architecture.md), plus
topic-specific docs under [`.claude/`](.claude/) — see
[`CLAUDE.md`](CLAUDE.md) for the full map.

## Repository layout

```
.
├── README.md
├── CLAUDE.md                     # orientation for code agents (links to .claude/*.md)
├── .claude/                      # topic-named contributor docs (see CLAUDE.md for the map)
├── Taskfile.yml                  # all dev commands
├── docker-compose.yml
├── scripts/
│   ├── install.sh                # core install (config + daemon image + vendor libs)
│   └── mlx.sh                    # optional Apple Silicon backend: install + start/stop/status
├── config/
│   ├── mlx-server.yaml.example   # template; on `task install:mlx` copied to ~/.mlx-server/config.yaml
│   └── tldr.yaml.example         # template; on `task install` copied to config/tldr.yaml
├── docs/
│   └── logo-banner.svg
├── daemon/                       # FastAPI service in Docker
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── src/
└── extension/                    # Chrome MV3 extension (vanilla JS, no build)
    ├── manifest.json
    ├── public/icons/             # icon.svg → icon{16,48,128}.png
    ├── src/
    └── vendor/                   # marked, DOMPurify, Readability (downloaded by installer)
```

## Requirements

- **Daemon**: Docker (OrbStack or Docker Desktop). Anything with Python
  works — the container is `python:3.11-slim`. No host Python needed.
- **A backend**: see Quick start. Anything OpenAI-compatible works.
- **Chrome 116+** (Manifest V3 side panel).
- **Apple Silicon, optional**: only if you want the bundled mlx setup (`task install:mlx`).
  ~6 GB disk for Gemma 4 E4B (4-bit) + Whisper large-v3 weights.

## Roadmap

Near-term, roughly in order:

- [ ] Chrome Web Store listing (signed, auto-updating install)
- [ ] Daemon install without Docker (`pipx install` / single binary)
- [ ] Zero-config pairing with an already-running Ollama
- [ ] Full-text search across the library
- [ ] Firefox port
- [ ] Export to Markdown / Obsidian

Opinions and PRs welcome — open an issue.

## Contributing

The codebase ships orientation docs for humans and AI agents alike: start at
[CLAUDE.md](CLAUDE.md), which maps the topic docs in [.claude/](.claude/) —
architecture, event model, worker invariants, dev runbook. `task test` runs
ruff + mypy + pytest in the daemon container; the extension has no build
step at all.

## License

MIT.
