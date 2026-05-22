# CLAUDE.md — context for code agents

TLDR is a local Chrome extension + Python daemon that summarises web pages
(HTML and PDF), YouTube videos, and any other audio/video found on the
page (whatever yt-dlp can extract), and answers follow-up questions about
the processed material. Single-user, runs on whoever's machine.

The daemon talks to any **OpenAI-compatible** LLM/Whisper backend over HTTP
— Ollama, LM Studio, mlx-openai-server, vLLM, llama.cpp, etc. The bundled
mlx setup (`task install:mlx`) is just one option for Apple Silicon users
who want the fastest local path; nothing in the daemon assumes it.

## Read first

Pick the file that matches what you're doing — don't read them all at once.

**Layout & data flow** (start here for "how is this structured"):

- [architecture.md](.claude/architecture.md) — components, why mlx is on the
  host and the rest is in Docker, request lifecycles per `JobKind`.
- [daemon.md](.claude/daemon.md) — lifespan startup order, how a POST /jobs
  flows through, where to add new things.
- [extension.md](.claude/extension.md) — surfaces × daemon connections,
  side panel lifecycle, tab tracking, side-panel staleness defences.

**Invariants** (don't break these without good reason):

- [contract.md](.claude/contract.md) — schemas mirror, URL normalization.
- [events.md](.claude/events.md) — three SSE surfaces, broker model, repo
  auto-publish.
- [workers.md](.claude/workers.md) — async POST, soft pause, restart-safety
  (incl. media-job ephemerality).
- [llm.md](.claude/llm.md) — single semaphore, single timecode formatter,
  `output_language` threading.

**Dev loop & operations**:

- [runbook.md](.claude/runbook.md) — first-time setup, daily commands,
  reload matrix per change-type, troubleshooting (YouTube broken, context
  truncation, mlx idle-unload, daemon unreachable, …), how to update yt-dlp /
  Python deps / mlx-server / extension.
- [ops.md](.claude/ops.md) — invariants behind the runbook: why uvicorn has
  no `--reload`, why yt-dlp auto-upgrades, Taskfile-as-router policy,
  testing philosophy.

Code is the source of truth for details. These docs orient you fast; they
don't try to mirror every line.

## Quick command reference

```bash
task install              # one-time: config + daemon image + extension vendor libs
task install:mlx          # OPTIONAL: macOS arm64 mlx-openai-server + Gemma 4 + Whisper (~6 GB)
task up                   # start daemon (+ mlx-server if installed)
task down                 # stop (sqlite volume preserved)
task test                 # ruff + mypy + pytest inside the daemon container
task status               # health + container status
task logs                 # tail daemon logs
task reset                # destructive: wipe sqlite volume (prompts)
```

After editing code: daemon → `docker compose restart daemon`; extension →
reload icon in `chrome://extensions`. Full matrix in
[runbook.md](.claude/runbook.md#reload-matrix).

## Adding features — 30-second tour

1. **New API endpoint** → Pydantic model in `daemon/src/api/schemas.py` AND
   mirror in `extension/src/lib/api-types.js` (same commit). Route in
   `daemon/src/api/<file>.py`. Prefer extending `/ai/stream` body for new AI
   modes over a new endpoint.
2. **New SQLite column** → edit the v1 migration in
   `daemon/src/storage/migrations.py` (we wipe DB during dev — `task reset`)
   + field on the SQLModel in `daemon/src/storage/db.py` + helper in `repo.py`.
3. **New worker / external integration** → file under `daemon/src/workers/`.
   Errors typed under `workers/errors.py` with a `code` matching `DeferredReason`.
   Publish progress to `workers.broker.get_broker()` keyed by `job_id` so
   `/ai/stream` AND `/events` subscribers both see it (per-job broker
   mirrors into the global one).
4. **New UI surface** → file under `extension/src/{sidepanel,library,options}/`.
   Use `daemon-client.js` for HTTP, `markdown.js` for rendering,
   `event-stream.js` to react to daemon state without polling.
5. Always: `task test` before considering it done. Update the relevant
   `.claude/*.md` if you changed an invariant.
