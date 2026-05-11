# Daemon source layout

FastAPI app on port 8765, single SQLite database, plus three long-running
background tasks: the Whisper queue worker, the retention sweeper, and
ad-hoc `pipeline.run_pipeline` tasks per POST /jobs. All blocking I/O
(yt-dlp, trafilatura, mlx multipart upload) runs inside `asyncio.to_thread`.

## Top-level

```
daemon/
├─ Dockerfile
├─ docker-entrypoint.sh        # auto-upgrades yt-dlp + youtube-transcript-api on start
├─ pyproject.toml              # runtime + dev deps; ruff/mypy/pytest config
├─ src/
│  ├─ main.py                   # FastAPI app + lifespan (DB, migrations, workers start/stop)
│  ├─ config.py                 # YAML loader + env overrides; get_config() is the only accessor
│  ├─ api/
│  │  ├─ schemas.py            # Pydantic contract — READ-ONLY for non-API streams
│  │  ├─ jobs.py               # /jobs CRUD + POST /jobs/{id}/retry + GET /jobs/{id}/messages
│  │  ├─ ai.py                  # POST /ai/stream (summary + QA, per-job SSE)
│  │  ├─ events.py              # GET /events (global SSE, type-filterable)
│  │  ├─ workers.py             # GET/POST /workers/{pause,resume}
│  │  └─ health.py             # GET /health
│  ├─ storage/
│  │  ├─ db.py                  # SQLModel engine + pragmas + ORM (Job, Message)
│  │  ├─ migrations.py          # idempotent runner + v1 schema (tables + FTS5 + triggers)
│  │  ├─ repo.py                # CRUD; auto-publishes job_event on every write
│  │  └─ cookies.py             # Cookie[] → requests.Session  /  Netscape file (yt-dlp)
│  ├─ llm/
│  │  ├─ client.py              # AsyncOpenAI + global semaphore (config.llm.max_concurrent_calls)
│  │  ├─ tokens.py              # tiktoken cl100k_base proxy
│  │  ├─ chunking.py            # split_for_summary — paragraph + sentence-aware, [MM:SS]-safe
│  │  ├─ summary.py             # stream_summarize() = single-pass OR sequential map-reduce
│  │  └─ qa.py                  # stream_answer() = pick context, format prompt, stream tokens
│  ├─ workers/
│  │  ├─ broker.py              # JobEventBroker (per-job) + EventBroker (global); per-job mirrors into global
│  │  ├─ control.py             # WorkerControl pause/resume; publishes workers_event itself
│  │  ├─ pipeline.py            # run_pipeline() — extract + yt-dlp captions fallback + summary
│  │  ├─ retention.py           # retention_worker() — periodic sweep of old jobs
│  │  ├─ runner.py              # whisper_worker — consumes queue, reuses cached audio on retry
│  │  ├─ queue.py               # WhisperQueue snapshot + restart-safe re_enqueue_pending
│  │  ├─ errors.py              # TranscriptError hierarchy with `code` matching DeferredReason
│  │  ├─ timecodes.py           # build_marked_text — SINGLE source of truth for [MM:SS]
│  │  ├─ youtube.py             # extract_video_id, fetch_transcript, download_audio + subtitles + metadata
│  │  ├─ page.py                # trafilatura fetch+extract fallback
│  │  └─ transcribe.py          # streaming multipart upload to /v1/audio/transcriptions
│  └─ prompts/
│     ├─ summary_single.txt    # all use {output_language}, {title}, ...
│     ├─ summary_chunk.txt
│     ├─ summary_reduce.txt
│     └─ qa.txt
└─ tests/
   ├─ conftest.py              # bootstraps TLDR_CONFIG so get_config() resolves outside docker
   ├─ test_api_jobs.py         # async POST flow + /ai/stream summary + QA + messages
   ├─ test_api_jobs_race.py    # parallel POST /jobs same URL must dedupe
   ├─ test_repo.py             # CRUD round-trip
   ├─ test_repo_emit.py        # every repo write publishes the right job_event
   ├─ test_migrations.py
   ├─ llm/                     # chunking/summary/qa/prompts
   └─ workers/
      ├─ test_broker.py        # fan-out, drop-on-full, per-job→global mirror
      ├─ test_control.py       # pause/resume flag + workers_event publish
      └─ ...                   # timecodes/errors/youtube/queue/runner
```

## Where to add things

| Change | Where |
|---|---|
| New endpoint | `src/api/<file>.py` route + `src/api/schemas.py` model + mirror in `extension/src/lib/api-types.js` |
| New SQLite column | edit v1 migration in `src/storage/migrations.py` (we wipe DB, no v2 yet) + field on model in `src/storage/db.py` + helper in `repo.py`. If the column is user-visible, add it to `repo.job_summary_dict` so /events carries it. |
| New repo write function | follow the auto-emit pattern: call `_emit_updated(job_id)` (or created/deleted) at the end. See conventions.md. |
| New external integration | file under `src/workers/`. Typed errors in `workers/errors.py` |
| New AI mode | extend `POST /ai/stream` in `api/ai.py` — keep the same event shapes; add a request field if needed |
| New global state to broadcast | publish via `get_event_broker().publish(workers_event(...))` from the owner module; do NOT add a parallel SSE endpoint |
| LLM behavior change | `src/llm/` + prompts in `src/prompts/`. Always pass `output_language` from config |
| New CLI / task | `Taskfile.yml` one-liner that delegates to a script in `scripts/` |

## Background tasks lifecycle

`main.lifespan` starts three classes of background work:

1. **Whisper worker** (`workers.runner.whisper_worker`) — single coroutine,
   sequential, consumes the WhisperQueue. Cancellation on shutdown is
   clean: in-flight job stays in `running`, `find_pending_for_restart`
   re-enqueues on next startup.
2. **Retention worker** (`workers.retention.retention_worker`) — sleeps
   6h between sweeps; safe to cancel anytime.
3. **Per-job pipelines** (`workers.pipeline.run_pipeline`) — spawned by
   each POST /jobs. Held in `_BACKGROUND_TASKS` set in `api.jobs` to
   prevent garbage collection killing the task. Each one runs to
   completion (or pushes to the whisper queue if YouTube transcript fails).

All three publish events into the same `JobEventBroker` keyed by job_id,
so `/ai/stream` subscribers see a unified event flow regardless of which
task is producing. The per-job broker mirrors every event into the
global `EventBroker` with `job_id` attached, so `/events` subscribers
(Library, Side panel) see the same stream without per-job subscribe.

## Tests

`task test` runs ruff + mypy + pytest inside the container. Tests directory
is volume-mounted (`docker-compose.yml`) so edits hot-reload. External
services (mlx, youtube_transcript_api, yt-dlp, trafilatura, the Whisper
worker AND the retention worker) are mocked at module-load time:

- `llm.summary.stream_summarize` → returns a deterministic async iterator
- `llm.qa.stream_answer` → returns a deterministic async iterator
- `workers.youtube.fetch_transcript_with_retry` → raises `PermanentTranscriptError`
- `workers.page.extract_with_trafilatura` → returns deterministic text
- `workers.runner.whisper_worker` + `workers.retention.retention_worker`
  → no-op coroutines so the lifespan can start/stop cleanly

See `tests/test_api_jobs.py` `client` fixture for the canonical pattern,
including `_wait_until_done` that polls `GET /jobs/{id}` (since POST is
async).

## Hot reload caveat

`uvicorn` runs without `--reload` (the Whisper worker would orphan jobs on
each reload). Code changes require `docker compose restart daemon`. Test
changes don't — pytest re-collects on each invocation.
