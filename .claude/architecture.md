# Architecture

```
┌─ Host ────────────────────────────────────────────────────────┐
│                                                               │
│  Any OpenAI-compatible LLM / Whisper backend                  │
│   /v1/chat/completions      ←─ summary + QA                   │
│   /v1/audio/transcriptions  ←─ Whisper fallback               │
│  Examples: mlx-openai-server (Apple Silicon), Ollama,         │
│   LM Studio, vLLM, llama.cpp, openai-edge, ...                │
│                                                               │
│  ┌─ Docker: daemon (port 8765) ─────────────────────────────┐ │
│  │  FastAPI                                                 │ │
│  │  Async POST /jobs → background pipeline (no inline wait) │ │
│  │  Per-job event broker → /ai/stream (summary + Q&A)       │ │
│  │  Global event broker  → /events (Library + Side panel)   │ │
│  │  Whisper queue (asyncio, restart-safe, globally pausable)│ │
│  │  Periodic retention sweep (storage.retention_days)       │ │
│  │  yt-dlp auto-captions + Whisper fallback chain           │ │
│  │  /workers {pause,resume} throttles all background ML     │ │
│  │  Reaches host backend via host.docker.internal           │ │
│  │  SQLite in named volume `tldr-data`                      │ │
│  └──────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
                            ▲
                            │ http://localhost:8765
                            │
        ┌─ Chrome extension (MV3, vanilla JS) ─────┐
        │  Toolbar action → side panel              │
        │  Side panel follows the active tab        │
        │  Single /events SSE per surface — no poll │
        │  Library page with retry / delete / pause │
        └───────────────────────────────────────────┘
```

## Why these splits

- **Backend on the host.** Any OpenAI-compatible runner does. The bundled
  mlx-openai-server option (`task install:mlx`) is fastest on Apple Silicon
  because MLX needs direct Metal / Neural Engine access — inside Docker
  (Linux VM on macOS) it falls back to CPU and runs 50–100× slower. Other
  backends (Ollama, LM Studio, vLLM, …) are equally first-class via config.
- **Daemon runs in Docker.** Clean control over ffmpeg / yt-dlp / Python
  versions. SQLite lives in a named volume so `docker compose down`
  preserves the user's library. All HTTP is OpenAI-compat — daemon doesn't
  care which backend is on the other end.
- **Extension is vanilla JS, no build step.** Edit a file, click reload,
  done. Saves the entire bundler / TypeScript / sourcemap stack at the cost
  of mirrors-by-hand for shared types.

## API model — three SSE-bearing surfaces

The daemon exposes a small surface around three ideas:

1. **/jobs** — CRUD + chat history. POST is **always async**: it persists
   the row, kicks off `workers.pipeline.run_pipeline` as a background task,
   and returns 202 with the new id. The client never blocks waiting for a
   summary. Subscribe via `/ai/stream` (the specific job) or `/events`
   (the global firehose) to follow progress.

2. **/ai/stream** — single SSE endpoint for ALL streaming AI responses.
   Two modes selected by the request body:

   - `{job_id}` (no `question`) → SUMMARY mode. Subscribes to the job's
     per-job channel of `JobEventBroker`. If the job is still running,
     events come live; if already done, `summary_md` is replayed as one
     delta + one done.
   - `{job_id, question}` → QA mode. Triggers a fresh QA call, persists
     user + assistant messages, streams the answer.

3. **/events** — single global SSE for Library + Side panel. Subscribes
   the global `EventBroker`. The per-job broker mirrors every event into
   the global broker with `job_id` attached, so a `/events` subscriber
   sees stage / delta / done / error AND `job_event` (created / updated
   / deleted) AND `workers_event` (paused / queue size) on one connection.
   Clients filter with `?types=job,workers,done,error` to skip the
   high-volume `delta` chatter when they only need status badges.

AI-stream events use the same shapes regardless of mode (`AIStageEvent`,
`AIDeltaEvent`, `AIDoneEvent`, `AIErrorEvent`). Job-list events use
`job_event(action, job)`; workers state uses `workers_event(state)`.

The chain that matters: a state-changing call in `src/storage/repo.py`
publishes `job_event(...)` itself. So `repo.mark_done(...)` flips the
DB row AND tells the Library to re-render the row, atomically as far as
callers are concerned. See conventions.md for the invariant.

## Request flows

### Page (async, ~10–15s)

```
Toolbar click → Readability → POST /jobs (page_text included) → 202 {id}
   ↓
   pipeline.run_pipeline (background task):
     extracting → ready → summarizing → publish deltas → mark_done → done event
   ↓
Side panel POST /ai/stream {id}: subscribes; sees stage("ready"),
stage("summarizing"), deltas streaming live, done with summary_md.
```

### YouTube with captions (async, ~15–30s)

```
POST /jobs → pipeline:
  extract_video_id → fetch_transcript_with_retry → build_marked_text
  → publish stage("ready") → stream_summarize → publish deltas → mark_done
```

### YouTube without captions (deferred, 3–5min)

```
POST /jobs → pipeline:
  fetch_transcript fails → enqueue WhisperTask → publish stage("queued")
  → return (pipeline coroutine ends)

(asynchronously)
workers.runner.whisper_worker picks task →
  publish stage("transcribing", "downloading") → yt-dlp
  → publish stage("transcribing") → mlx /v1/audio/transcriptions
  → publish stage("ready") → publish stage("summarizing") → stream tokens
  → mark_done → publish done

Subscribers to /ai/stream see the ENTIRE arc from a single SSE — they don't
need to know whether the path was fast or deferred.

Restart-safe: repo.find_pending_for_restart() re-enqueues queued/running
rows on daemon startup.
```

### Q&A (any time after extraction)

```
POST /ai/stream {job_id, question}:
  insert Message(role="user", content=question) →
  publish stage("thinking") → llm.qa.stream_answer (streaming) →
  insert Message(role="assistant", content=full) →
  emit done with message_id
```

`GET /jobs/{id}/messages` returns the full chat history. The side panel
loads it on every job switch so chats persist across tab changes.

## Storage

Single SQLite database under the named docker volume `tldr-data`, mounted
at `/data/tldr.db`. Tables: `Job`, `Message`, `_migrations`, plus the
FTS5 virtual table `job_fts` and its AI/AD/AU triggers. FK cascade
deletes `Message` rows when a Job is removed.

Pragmas (per connection): `journal_mode=WAL`, `synchronous=NORMAL`,
`cache_size=-64000`, `mmap_size=268435456`, `temp_store=MEMORY`,
`foreign_keys=ON`.

`Job.raw_text` carries inline `[MM:SS]` markers for YouTube — no separate
column. Single source of truth: `workers.timecodes.build_marked_text`.

`list_jobs()` defers `raw_text`, `summary_md`, `error` columns —
multi-megabyte text would otherwise dominate every Library refresh. The
detail GET pulls those via `get_job(id)`.

A periodic background coroutine (`workers.retention.retention_worker`)
deletes jobs older than `config.storage.retention_days` (default 365)
every 6 hours. Set to 0 to disable.
