# Runbook — setup, daily ops, troubleshooting

Operational reference. When something breaks, find the symptom heading. For
day-to-day commands see the top section. README has the user-facing version
of setup; this one is for the contributor / maintainer view.

## Contents

- [First-time setup](#first-time-setup)
- [Daily commands](#daily-commands)
- [Reload matrix — what to do when X changed](#reload-matrix)
- [Logs & diagnostics](#logs--diagnostics)
- [Troubleshooting](#troubleshooting)
- [Updating components](#updating-components)

## First-time setup

You need three things running: a backend (LLM, optionally Whisper), the
daemon, and the Chrome extension loaded.

1. **Clone + install deps.** `task install` — generates `config/tldr.yaml`
   from the example, builds the daemon Docker image, downloads extension
   vendor libs (marked, DOMPurify, Readability) into `extension/vendor/`.

2. **Pick + start a backend.** Any OpenAI-compatible LLM endpoint works.
   `config/tldr.yaml.example` has copy-paste blocks for the common ones —
   uncomment one in `config/tldr.yaml` and adjust:

   | Backend | Platform | Notes |
   |---|---|---|
   | Ollama | any | `ollama pull qwen3-vl:8b`, then a Modelfile to bump ctx to 65536 (Gemma 4 alternative: `gemma4:e4b` + Modelfile to 131072) |
   | LM Studio | macOS / Windows | GUI; enable local server on port 1234; set ctx 65536 in model settings (131072 for Gemma 4) |
   | mlx-openai-server | macOS arm64 | Bundled — `task install:mlx` sets up `~/.mlx-server/config.yaml` and pulls Qwen3-VL 8B + Whisper turbo (~7-8 GB) |
   | llama-server | any | `brew install llama.cpp`, launch with `-c 65536` (or `-c 131072` for Gemma 4) |

   Whisper is only used as a last-resort fallback for YouTube videos with
   no captions — you can skip it; those specific videos will error.

3. **Start the daemon.** `task up` — runs the daemon container plus
   mlx-server if installed. `task status` should show `daemon healthy`.

4. **Load the extension.** `chrome://extensions` → enable Developer mode →
   Load unpacked → select `extension/`. The TLDR card should appear.

5. **Smoke test.** Open any web page, click the TLDR toolbar button. Side
   panel opens, you should see "extracting" → "summarizing" → streaming
   tokens. If the panel says daemon unreachable, jump to
   [Troubleshooting](#troubleshooting).

## Daily commands

| Action | Command |
|---|---|
| Start everything | `task up` |
| Stop everything | `task down` (SQLite volume preserved) |
| Health check | `task status` |
| Tail daemon logs | `task logs` |
| Run tests + lint + typecheck | `task test` |
| Wipe the database | `task reset` (destructive, prompts) |

mlx-server logs (if installed) are at `~/.mlx-server/logs/server.{out,err}.log`
— not visible to `task logs`, which only tails the daemon container.

Native (uv) installs don't have a container to tail — see
[Logs & diagnostics](#logs--diagnostics) for where those logs live instead.

## Reload matrix

Chrome and Docker don't auto-pick-up changes the same way. After editing:

| What changed | What to run | Why |
|---|---|---|
| `daemon/src/*.py` | `docker compose restart daemon` | uvicorn runs without `--reload` (would orphan jobs in the Whisper worker) |
| `daemon/pyproject.toml` | `task install` (or `docker compose build daemon`) | image rebuild |
| `config/tldr.yaml` | `task down && task up` | config is read once at startup |
| `~/.mlx-server/config.yaml` | `task down && task up` | mlx-server reads it on launch |
| `extension/src/**` (JS/HTML/CSS) | reload icon in `chrome://extensions` | Chrome does not watch unpacked extensions |
| `extension/manifest.json` | sometimes Remove + Load unpacked | depends on which keys changed |
| `extension/vendor/` (rerun `task install`) | reload icon | same as JS |

If you're not sure, `task down && task up` followed by the extension reload
icon is the universal hammer.

## Logs & diagnostics

**Where logs live**: `<data_dir>/logs/daemon.log` (+ rotated
`daemon.log.1` … `daemon.log.4`), written by `src/logging_setup.py`. Both
the app's own log lines and uvicorn's access/error logging go through this
one `RotatingFileHandler` — 5 MB per file, 4 backups, ~20 MB total ceiling
(`LOG_MAX_BYTES`/`LOG_BACKUP_COUNT` in that module).

- **Docker**: this is purely additive — `task logs` / `docker compose
  logs` keep working exactly as before (stdout, rotated by docker's own log
  driver); the file just gives `GET /diagnostics` something to tail inside
  the container's data volume too.
- **In both environments**: `uvicorn.access` never writes a request's query
  string, only its path — `GET /jobs?url=<page>&limit=1` is logged as
  `GET /jobs`. This isn't part of the Docker/native split above; it's
  unconditional, because the query string here routinely IS the page/video
  URL the user is looking at (or, worse, arbitrary text passed as another
  parameter) — an access log with it left in is a browsing-history log,
  not a diagnostic one.
- **Native (uv/launchd) install**: the rotating file REPLACES stdout/stderr
  for anything that goes through Python's `logging` module. launchd's
  `daemon.out.log`/`daemon.err.log` (set by the LaunchAgent's
  `StandardOutPath`/`StandardErrorPath` — see `src/service.py`) used to
  receive everything logged and grow unbounded (no rotation possible: the
  file is opened once by launchd itself, so renaming it out from under that
  fd doesn't stop it from writing to the old inode). After this change
  those two files only catch what bypasses `logging` entirely — an
  interpreter traceback on a hard crash — so they should stay small from
  here on.
- **Existing installs**: the first time `tldr-daemon` (the native CLI
  entrypoint, `src/cli.py`) runs with this code, it truncates
  `<data_dir>/daemon.out.log`/`daemon.err.log` in place
  (`os.truncate(..., 0)`, never delete — deleting would orphan launchd's
  open, `O_APPEND` fd on an invisible, still-growing inode). This is a
  ONE-TIME migration, guarded by `<data_dir>/logs/.legacy_logs_truncated`,
  so a later crash-loop restart never wipes out a fresh traceback you
  haven't had a chance to read yet. No backup copy is kept — those files
  had accumulated full processed-page/video URLs with zero rotation, which
  is precisely the problem being fixed, not something worth relocating
  elsewhere. This truncation deliberately lives ONLY in the native CLI
  entrypoint, not in the FastAPI app's startup — see [ops.md](ops.md#logging)
  for why that separation matters (it's the fix for a real incident, not
  tidiness).

**Diagnostics report**: `GET /diagnostics` (add `?job_id=<id>` for one
job's metadata — 404s if it doesn't exist), or the "Diagnostics" section on
the extension's options page (builds the report into a read-only textarea,
then Copy to clipboard / Save to file — nothing is ever sent anywhere
automatically). Contains: daemon version + Python/platform info, the full
`/health` response, the daemon config with secrets redacted (`api_key_set`/
`api_key_source` — NOT `api_key_hint`, which `GET /config` has but
diagnostics deliberately drops, see below), the last 300 lines of the
rotating log, and a count of jobs by status.

Everything text-shaped is scrubbed by `src/api/diagnostics.py` before it's
returned, by construction rather than by review:

- either API key is redacted even as a fragment (matched against the exact
  resolved value, same as `POST /config/test`'s existing redaction) —
  including `api_key_hint` itself, which is dropped from the config block
  entirely rather than merely redacted: it carries zero diagnostic value
  beyond what `api_key_set` already reports, while being 4 real characters
  of the configured key;
- the home directory is replaced with `~` — including in
  `config.config_path`/`config.overrides_path`, which are absolute paths on
  the reporter's own machine and otherwise leak the account name;
- any URL that ISN'T a loopback address (`127.0.0.1`/`localhost`) is
  replaced with a placeholder — local-backend addresses
  (`http://127.0.0.1:1240/...`) are kept since they're exactly what a
  diagnosis needs to see. This matches a URL whether it's written literally
  OR percent-encoded (`https%3A%2F%2F...`) — the log tail can contain
  either, since `uvicorn.access` used to write a request's full query
  string (including any page URL passed as `?url=`) percent-encoded before
  the fix described below, and older, already-rotated log files can still
  have that on disk regardless. `config.llm.base_url`/`config.whisper.base_url`
  are the one URL-shaped exception: a configured backend address (local or
  cloud) is diagnostic-relevant config, not browsed content, so it's kept
  even when non-loopback (only checked for an embedded API key, same as
  everything else).

Page/video titles, URLs, and transcript/summary content are never put into
the response in the first place (not scrubbed — just never read) — see
`DiagnosticsJobInfo` in `src/api/schemas.py`. Separately, `uvicorn.access`
itself no longer writes a query string to `logs/daemon.log` at all (in
Docker or native) — `src/logging_setup.py` strips it at the source, since
data that's never written can't leak from a copied log file either; the
diagnostics scrubbing above is the safety net for logs that predate that
fix, not the primary defense.

## Troubleshooting

### YouTube videos error out / "video unavailable"

Google ships breaking changes to YouTube's internals every few weeks.
yt-dlp + youtube-transcript-api are **auto-upgraded on every container
start** by `daemon/docker-entrypoint.sh`, so the standard fix is just:

```
task down && task up
```

Watch `task logs` — the entrypoint prints the pip upgrade line. If the
issue persists after a fresh upgrade, the bug is genuinely upstream and not
yet fixed in a released wheel. Try the pre-release:

```
docker compose run --rm daemon pip install --upgrade --pre yt-dlp
```

Persistent fix once upstream releases: bump the version pin in
`daemon/pyproject.toml`, then `task install`.

If `task logs` shows the upgrade succeeded but transcription still fails:
check the actual error message in the Library row's failed-state. A
`transcript_blocked` code means cookies are needed — log into YouTube in
the same Chrome profile that has the extension; the extension forwards
`.youtube.com` cookies on submit.

### Summary is mysteriously short / video truncated halfway

Backend context window is too small. The daemon **silently truncates** if
your input exceeds `llm.context_length`. Symptoms: a 60-minute video gets
summarised through minute 30; a long article ends mid-section.

- **Ollama**: defaults to 2048(!). Make a custom Modelfile:
  `printf 'FROM qwen3-vl:8b\nPARAMETER num_ctx 65536\n' > Modelfile`
  then `ollama create qwen3-vl:8b-64k -f Modelfile`. Update
  `model: qwen3-vl:8b-64k` in `config/tldr.yaml`. (Gemma 4: same recipe with
  `gemma4:e4b`, `num_ctx 131072`, `gemma4:e4b-128k`.)
- **LM Studio**: open the loaded model's settings, set Context Length to
  65536 (131072 for Gemma 4). Verify with `lms ps` (CONTEXT column).
- **llama-server**: launch with `-c 65536` (`-c 131072` for Gemma 4).

`config.llm.context_length` MUST match what the backend actually loaded.
A mismatch on the high side causes `n_keep >= n_ctx` errors; on the low
side, silent truncation. After fixing: `task down && task up`.

### Whisper transcription dies partway through a long video

mlx-server v1.8.1 known bug — the idle-unload timer can fire mid-stream
during continuous batches. Defences already in place:

- Long `on_demand_idle_timeout` in `~/.mlx-server/config.yaml` (llm model
  (Qwen3-VL or Gemma 4): 1800s, whisper: 3600s).
- Per-chunk timeout `llm.stream_chunk_timeout_seconds` (default 60s).

If still flaky: raise the whisper `on_demand_idle_timeout` to 7200 in
`~/.mlx-server/config.yaml`, then `task down && task up`. Or set
`on_demand: false` for whisper to keep it always loaded (uses ~3 GB RAM
constantly). See [llm.md](llm.md) for the full invariant.

### Side panel shows old summary / stale state after minimise

The side panel has built-in defences against this (`visibilitychange`
listener + proactive cache writes on SSE done/error). If you see staleness
that doesn't recover on Chrome window focus:

1. Open DevTools on the side panel itself (right-click inside → Inspect)
2. `chrome.storage.session.clear()`
3. Close + reopen the panel

Reproducible? File an issue with steps. See [extension.md](extension.md) →
"When the side panel may go stale".

### Daemon unreachable / port 8765 not responding

```
docker compose ps                     # daemon container running?
task logs                             # startup errors?
curl http://localhost:8765/health     # what does it say?
```

Common cases:

- **Backend unreachable from container.** On Linux, `host.docker.internal`
  doesn't resolve by default. Add to `docker-compose.yml` under the daemon
  service: `extra_hosts: ["host.docker.internal:host-gateway"]`.
- **Port conflict.** Something else owns 8765. Check `lsof -i :8765`.
- **Volume corruption.** Rare but possible after a crash. `task reset`
  wipes the SQLite volume.

### "Could not read the local PDF" on a `file://` URL

Chrome blocks extensions from reading local files by default. Fix:
`chrome://extensions` → TLDR → Details → enable **"Allow access to file
URLs"**. Reload the PDF tab, click the toolbar button again. http(s) PDFs
work without this toggle — the daemon fetches them directly.

### "Local PDF is N MB — over the 50 MB upload cap"

`file://` PDFs are read in the extension's MV3 service worker and
base64-encoded for upload. The cap exists because the SW heap is bounded
and a 200 MB encoded payload would OOM-kill the entire extension.

Workarounds (pick one):
- Serve the file locally — `cd /path/to/dir && python -m http.server` then
  open `http://localhost:8000/file.pdf`. The daemon fetches it directly
  with no extension memory cost; the cap doesn't apply.
- Pre-OCR with `ocrmypdf` and the cap-on-output is irrelevant — the
  daemon's text-first path reads the result instantly.
- Split the PDF (`pdftk` or `qpdf`) and summarise sections separately.

### PDF takes forever / "OCR page N/M" in the timeline

The PDF triggered the vision OCR fallback (pypdf returned ~no text, so
the daemon assumes it's scanned). Vision OCR sends each page to the
multimodal LLM separately — 10-60 seconds per page, measured on a local
Apple Silicon Gemma 4 E4B (not re-measured against the current Qwen3-VL 8B
default). Up to 100 pages by default; longer PDFs error out.

Faster alternatives:
- Pre-OCR with `ocrmypdf input.pdf output.pdf`, then summarise the
  output — daemon's text-first path will pick it up instantly.
- Use a beefier backend (a remote vision model, or local Gemma 4 27B
  if your machine fits it) and raise `llm.max_concurrent_calls`.

### Transcript tab shows one `[00:00]` block for a whole hour-long video

Whisper segments + auto-detected language aren't reaching the daemon.
Upstream `mlx-openai-server` (v1.8.1 at least) drops both in its HTTP
response even though `mlx_whisper.transcribe()` produces them
internally. Fix: apply our patch.

```
bash scripts/mlx.sh patch                          # idempotent
bash scripts/mlx.sh stop && bash scripts/mlx.sh start
```

Verify on a fresh job: `Job.transcript_language` should be filled, and
the Transcript tab shows one line per ~5 seconds rather than one giant
block. After upgrading `mlx-openai-server` the patch needs to be
re-applied (the upgrade overwrites the venv files); `task install:mlx`
does this automatically on the next install/upgrade run. See
`scripts/mlx-patches/README.md`.

### Translation chip stuck on "running X%" after browser restart

Should self-heal — `re_enqueue_running_on_startup` (in `main.lifespan`)
re-spawns the translator for any row left in `running` and the
sidepanel picks up live updates via `/events`. If it stays stuck:

```
task logs | grep translator                # check the worker error
docker compose restart daemon              # forces lifespan to re-run
```

Failed translations show with a red chip; click "Retry failed" in the
sidepanel's language bar to re-queue all of them at once.

### Library shows jobs in "queued" forever after restart

Expected for MEDIA jobs — they can't be resumed (`media_url` not
persisted). Re-submit from the extension by clicking the toolbar button on
the source page again. YouTube jobs in queued/running are re-enqueued on
startup automatically. Translations are also recoverable (see above).
See [workers.md](workers.md).

## Native (uv) mode

Alternative to Docker: the daemon installed as a uv tool, running directly
on the host with a user-level autostart service.

- **Install / uninstall**: `task install:uv` / `task uninstall:uv` (routers
  to `scripts/install-uv.sh` / `scripts/uninstall-uv.sh`). Manual:
  `uv tool install ./daemon`, then `tldr-daemon service install`.
- **Run by hand**: `tldr-daemon` (uvicorn on 127.0.0.1:8765). One-off:
  `uvx --from ./daemon tldr-daemon`.
- **Config**: auto-created from the packaged template on first run at
  `~/Library/Application Support/tldr/tldr.yaml` (macOS) /
  `$XDG_CONFIG_HOME/tldr/tldr.yaml` (Linux), created with `0600`
  permissions. `TLDR_CONFIG` overrides as usual. Backend URLs are rewritten
  to `127.0.0.1` (no `host.docker.internal` natively).
- **Cloud API key**: three ways in, on top of inline `api_key` in
  `tldr.yaml` — `api_key_keychain` + `api_key_keychain_account` (macOS
  Keychain / Linux Secret Service; recommended and the default `PATCH
  /config` picks when available — `keyring` is a base dependency, no
  extra install step, and the daemon reads its own previously-written
  entry back with zero prompts, including after `uv tool install
  --force`), `api_key_file` (path, `~` expands, must be `0600`; the right
  choice for Docker installs), or the `TLDR__LLM__API_KEY` env var, which
  for a systemd user unit is best supplied via an `EnvironmentFile=` line
  pointing at a `0600` file outside the repo. See [llm.md](llm.md) for the
  full resolution order and the `token_param` / `reasoning_headroom_tokens`
  escape hatches for reasoning models.
- **Data**: `~/Library/Application Support/tldr/data` (macOS) /
  `$XDG_DATA_HOME/tldr` (Linux). A configured `data_dir: /data` is
  auto-substituted when `/data` doesn't exist (i.e. outside the container).
- **Code reload**: no `--reload` here either — restart the process
  (`tldr-daemon service uninstall && tldr-daemon service install`, or
  Ctrl-C + rerun for a foreground run). Code changes also need
  `uv tool install --force ./daemon` since the tool venv has its own copy.
- **Service commands**: `tldr-daemon service install|uninstall|status` —
  launchd LaunchAgent on macOS, hardened systemd user unit on Linux,
  experimental schtasks logon task on Windows. `status` reports unit
  presence + `/health`.
- **yt-dlp self-heal**: on every server start the CLI upgrades yt-dlp +
  youtube-transcript-api in its own venv (`src/selfupdate.py`; uv if
  available, else pip; 60s timeout, non-fatal offline). Skip with
  `TLDR_SKIP_PKG_UPDATE=1` — the Docker entrypoint sets this since it
  upgrades on its own. uv is located via PATH and, failing that, well-known
  install dirs (`~/.local/bin`, `~/.cargo/bin`, Homebrew) — a launchd/systemd
  service inherits a thin PATH that omits them, and a uv-tool venv has no pip
  to fall back on.
- **ffmpeg**: needed by yt-dlp's audio postprocessing (and absent from the
  thin service PATH). Resolved by `src/workers/ffmpeg.py`: a system
  ffmpeg+ffprobe wins (PATH or known dirs like `/opt/homebrew/bin`),
  otherwise a static build is fetched once via the `static-ffmpeg` dependency
  and cached under `<data_dir>/ffmpeg` (cross-platform, no brew/apt; warmed in
  a background thread at startup). The path is passed to yt-dlp via
  `ffmpeg_location`, so the daemon's PATH is irrelevant.
- **deno** (JS runtime for YouTube's "n"/sig challenge — needed to download
  audio for caption-less videos): `src/workers/jsruntime.py`, same pattern as
  ffmpeg. System `deno` wins (PATH / known dirs / venv scripts dir), else a
  static binary (deno ≥ 2.3.0) is downloaded once from GitHub releases and
  cached under `<data_dir>/deno` (pin with `TLDR_DENO_VERSION`, else latest).
  Passed to yt-dlp as `js_runtimes: ["deno:<path>"]`. The solver code itself
  ships in the `yt-dlp-ejs` dependency (no live GitHub fetch). Captioned videos
  don't need any of this — they go through the youtube-transcript-api fast
  path. No prebuilt deno for Windows-arm64 → that one platform degrades to
  yt-dlp's JS-less clients.

## Updating components

### yt-dlp / youtube-transcript-api

Auto-handled — every `task up` runs `pip install --upgrade` for both. No
manual action unless you need the pre-release (see Troubleshooting).

### Daemon Python dependencies

Edit `daemon/pyproject.toml`, then `task install` (or `docker compose
build daemon`), then `task up`.

### mlx-openai-server (if installed via `task install:mlx`)

```
bash scripts/mlx.sh stop
# upgrade the host-side install — depends how you installed; usually:
pip install --upgrade mlx-openai-server
task up                              # mlx.sh start runs again
```

### The extension itself

Pull latest, run `task install` (refreshes `extension/vendor/` in case any
vendor lib was bumped), then click the reload icon in `chrome://extensions`.
Manifest changes occasionally need a full Remove + Load unpacked.
