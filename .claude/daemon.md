# Daemon

FastAPI on port 8765, single SQLite DB in a Docker volume, three classes of
background work. The interesting questions: **what runs when**, **what
talks to what**, and **where to put new logic**.

For the tree itself: `ls daemon/src/`. The names are descriptive; this doc
only covers what isn't obvious.

## Lifespan startup order

`main.lifespan` runs on FastAPI startup in this exact sequence (and reverse
on shutdown):

1. Resolve config (`config.get_config()` — loads YAML + env overrides once,
   cached for the process lifetime).
2. Create DB engine, apply migrations.
3. Construct singletons: `workers.broker.get_event_broker()`,
   `workers.broker.get_job_broker()`, `workers.queue.get_queue()`,
   `workers.control.get_control()`.
4. Re-enqueue persisted state: `repo.find_pending_for_restart()` pushes any
   YouTube job left in queued/running back onto the Whisper queue. Media
   jobs from a prior run get marked failed (their `media_url` wasn't
   persisted — see [workers.md](workers.md)).
5. Spawn long-running coroutines: `whisper_worker` (single, sequential) and
   `retention_worker` (sleeps 6h between sweeps).
6. Mount API routers: `api/{jobs,ai,events,workers,health}.py`.

In-flight Whisper task on shutdown: the coroutine is cancelled, but the
DB row stays in `running`. Next startup, step 4 picks it up. Idempotent.

## How a POST /jobs flows through the daemon

```
HTTP POST /jobs (api/jobs.py)
   ↓ persist Job row (status=running)  — repo.create_job
   ↓ auto-emits job_event(created) via broker side-effect
   ↓ spawn run_pipeline(job_id) into _BACKGROUND_TASKS set  ← held to dodge GC
   ↓ return 202 immediately
   
run_pipeline (workers/pipeline.py)
   ↓ dispatch on JobKind:
   │   PAGE     → _run_page     → extract → summarize → mark_done
   │   YOUTUBE  → _run_youtube  → fetch_transcript → (fallback caps) → summarize
   │   │                          ↳ on TranscriptError → enqueue WhisperTask
   │   MEDIA    → _run_media    → enqueue WhisperTask
   ↓ at every step: publish stage_event / delta_event / done_event via broker
   ↓ checkpoint_pause between steps (see workers.md soft-pause)
   ↓ repo.mark_done / mark_failed (auto-emits the corresponding job_event)

WhisperTask consumed by workers/runner.py whisper_worker:
   ↓ yt-dlp download → transcribe via /v1/audio/transcriptions → summarize
```

The broker side of this is unified — every step talks to the same per-job
channel of `JobEventBroker`, which mirrors into the global `EventBroker`.
Subscribers (`/ai/stream` per-job, `/events` global) don't care which task
produced an event. See [events.md](events.md).

## What runs blocking-ish

All third-party I/O that doesn't expose an async API runs inside
`asyncio.to_thread`: yt-dlp (`workers/youtube.py`), trafilatura
(`workers/page.py`), the streaming multipart upload to whisper
(`workers/transcribe.py`). If you add a new external integration that
isn't natively async, follow the same pattern — don't block the event loop.

## Where to add things

| Change | Where |
|---|---|
| New HTTP endpoint | `src/api/<file>.py` route + `src/api/schemas.py` model + mirror in `extension/src/lib/api-types.js` (same commit). See [contract.md](contract.md). |
| New AI mode | Extend `POST /ai/stream` body in `api/ai.py` — keep the same event shapes. Add a new endpoint only if the response semantics are genuinely different. |
| New SQLite column | Edit the v1 migration in `src/storage/migrations.py` (we wipe DB pre-1.0) + field on model in `src/storage/db.py` + helper in `repo.py`. If user-visible, add to `repo.job_summary_dict` so `/events` carries it. |
| New repo write function | Follow the auto-emit pattern: call the `_publish_*` helper at the end so callers don't need explicit broker calls. See [events.md](events.md). |
| New external integration | File under `src/workers/`. Typed errors in `workers/errors.py` with a `code` matching `DeferredReason`. Run blocking calls through `asyncio.to_thread`. |
| New global state to broadcast | Publish via `get_event_broker().publish(workers_event(...))` from the owner module; do NOT add a parallel SSE endpoint. |
| LLM behavior change | `src/llm/` + prompts in `src/prompts/`. Always thread `output_language` from config — see [llm.md](llm.md). |
| New CLI / task | `Taskfile.yml` one-liner that delegates to a script in `scripts/`. See [ops.md](ops.md). |

## Gotchas

- **`_BACKGROUND_TASKS` set in `api/jobs.py`.** Python's GC will kill a
  bare `asyncio.create_task(...)` whose handle nobody holds. The set keeps
  references alive; tasks remove themselves on completion via `add_done_callback`.
- **Broker mirror direction.** Publish to the per-job broker; it mirrors
  into the global broker with `job_id` attached. Publishing directly to
  the global broker from a job context skips the mirror's tagging step.
- **Config singleton.** `get_config()` caches on first call. Tests fake it
  via `conftest.py` (sets `TLDR_CONFIG` env var before any import).
- **`set_audio` doesn't emit.** Intentional — `audio_path` is internal
  plumbing the UI doesn't render. Don't grep for "why no event here".
