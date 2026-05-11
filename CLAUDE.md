# CLAUDE.md — context for code agents

TLDR is a local Chrome extension + Python daemon that summarises web pages
and YouTube videos and answers follow-up questions about the processed
material. Single-user, runs on whoever's machine.

The daemon talks to any **OpenAI-compatible** LLM/Whisper backend over HTTP
— Ollama, LM Studio, mlx-openai-server, vLLM, llama.cpp, etc. The bundled
mlx setup (`task install:mlx`) is just one option for Apple Silicon users
who want the fastest local path; nothing in the daemon assumes it.

## Read first

Pick the file that matches what you're doing — don't read them all at once.

- [.claude/architecture.md](.claude/architecture.md) — components, data flow,
  why mlx is on the host and the rest is in Docker, the request lifecycles.
- [.claude/daemon.md](.claude/daemon.md) — Python source layout, where to add
  endpoints / migrations / workers / prompts.
- [.claude/extension.md](.claude/extension.md) — JS extension layout (MV3,
  vanilla, no build step), tab tracking, side panel + library wiring.
- [.claude/conventions.md](.claude/conventions.md) — invariants you must
  respect (API contract mirroring, single timecode formatter, output_language
  threading, URL normalization, hot-reload rules).

Code is the source of truth for details. These docs orient you fast; they
don't try to mirror every line.

## Working

```bash
task install              # one-time: config + daemon image + extension vendor libs
task install:mlx          # OPTIONAL: macOS arm64 mlx-openai-server + Gemma 4 E4B + Whisper weights (~6 GB)
task up                   # daemon (docker) + mlx-server if installed
task down                 # stop (sqlite volume preserved)
task test                 # ruff + mypy + pytest inside the daemon container
task status               # health + container status
task logs                 # tail daemon logs
task reset                # destructive: wipe sqlite volume (asks for confirmation)
```

## Reload after changes

- **Daemon code** (`daemon/src/*`): mounted in the container but uvicorn does
  not auto-reload. Run `docker compose restart daemon`.
- **Daemon dependencies** (`pyproject.toml`): `task install` (or
  `docker compose build daemon`).
- **YouTube libs** (`yt-dlp`, `youtube-transcript-api`): auto-upgraded on
  every container start by `daemon/docker-entrypoint.sh`. So `task down &&
  task up` is the universal "fix it" reflex when YouTube breaks.
- **Extension code**: hit the reload icon on the TLDR card in
  `chrome://extensions`. Chrome does NOT auto-reload unpacked extensions on
  file change. Manifest changes sometimes need a full Remove + Load unpacked.

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
