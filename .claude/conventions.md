# Invariants — don't break these

These are choices the code depends on and that aren't immediately obvious
from reading individual files.

## API contract is mirrored, not generated

`daemon/src/api/schemas.py` ↔ `extension/src/lib/api-types.js`. Manual
sync. When you change one, change the other in the same commit. Bump
`DAEMON_API_VERSION` in `daemon/src/config.py` for breaking shape changes.

The extension's `daemon-client.js` is the only place that issues HTTP. If
you're adding an endpoint, add it to `daemon-client.js` and JSDoc-annotate
its return type with the api-types alias.

## Three SSE endpoints, one event broker

The daemon produces SSE streams from three places, all wired through the
same in-memory pub/sub:

- **POST /ai/stream** — summary mode and QA mode. Body shape decides:
  no `question` → summary; `question` set → QA. Subscribes the per-job
  channel of `workers.broker.JobEventBroker`.
- **GET /events** — single global stream for the Library + Side panel.
  Subscribes the global `workers.broker.EventBroker`. Clients narrow the
  firehose with `?types=job,workers,done,error` (server-side filter).
- The whisper worker and the per-job pipeline publish into the per-job
  broker; the per-job broker mirrors every event into the global broker
  with `job_id` attached. So `/events` sees stage/delta/done/error AND
  job/workers state changes — one connection per UI surface.

Event shapes for AI streams are uniform: `AIStageEvent`, `AIDeltaEvent`,
`AIDoneEvent`, `AIErrorEvent`. Job-list events use `job_event(action,
job)` — `action` ∈ {created, updated, deleted}, `job` is the same shape
as `JobSummary`. Workers state uses `workers_event({paused, queue_size,
running})`.

If you add a new AI capability, prefer extending the request body of
`/ai/stream` over creating a new endpoint. If you add a new app-wide
state change, publish a `job_event` / `workers_event` to the global
broker rather than inventing a sibling endpoint.

## State-changing repo functions auto-publish

Every function in `src/storage/repo.py` that mutates a Job row publishes
the corresponding `job_event` to the global event broker as a side
effect — `create_job` → `created`, `update_status / mark_done /
mark_failed / set_extracted / reset_for_retry` → `updated`, `delete_job
/ delete_jobs_older_than` → `deleted`. `set_audio` is intentionally
silent (audio_path is internal plumbing the UI doesn't render).

Why: keeps "DB write happened" and "UI told" inseparable. Three real
bugs we hit (title-not-updating, dedup race, pause-broken) all came from
a state change that forgot its event publish. Callers in `api/jobs.py`,
`workers/pipeline.py`, `workers/runner.py` no longer need (and don't
have) explicit emit calls.

Likewise `workers.control.WorkerControl.pause/resume` publish
`workers_event` themselves — `api/workers.py` is just a thin endpoint.

If you add a new repo write function, follow the same pattern: late
import `from src.workers.broker import …` inside a `_publish_*` helper,
wrap the publish in `contextlib.suppress(Exception)` so a broker hiccup
never rolls back the user's write.

## POST /jobs is async, always

It returns 202 with `{id, kind, status}` immediately and
spawns a background pipeline. Clients NEVER block on summarization in the
HTTP request; they subscribe via `POST /ai/stream {job_id}` to follow.

The pipeline coroutine is tracked in a module-level `_BACKGROUND_TASKS`
set in `api.jobs` so Python's GC doesn't kill it.

## Output language comes from config, never hardcoded

All LLM calls thread `config.output.language_name` into prompts as
`{output_language}`. The user sets `output.language` in `config/tldr.yaml`
to an ISO 639-1 code (`en`, `ru`, `de`, …) and the `language_name`
property expands it to the human-readable name the LLM follows reliably.
Anything that isn't a known code (a full name, or e.g. `"Brazilian
Portuguese"`) is passed through verbatim. Don't hardcode a language
anywhere — in code OR in prompts.

## LLM concurrency goes through one semaphore

`llm.client._llm_lock()` (sized by `config.llm.max_concurrent_calls`,
default 1) gates every `complete()` and `stream_complete()` call. New LLM
work — summary chunks, QA, anything else — must go through these two
functions; never bypass with raw HTTP. This keeps a single-user laptop
from thrashing the GPU when multiple jobs land at once.

The lock is **pause-aware**: acquire waits for both the semaphore AND
the global pause flag, and re-checks pause AFTER acquire so a flip that
landed while the caller was queued still holds them off. Q&A passes
`respect_pause=False` to bypass the gate (the user is actively waiting
on the answer).

`stream_complete` enforces a per-chunk timeout
(`config.llm.stream_chunk_timeout_seconds`, default 60 s). If the backend
stops sending tokens for that long we raise `TimeoutError` so the LLM
semaphore is released and the queue keeps moving — without it a single
hung backend stream would lock the queue forever.

This is defence-in-depth against an mlx-server v1.8.1 on-demand quirk:
the idle-unload timer is scheduled when the *first* request after a load
completes, and subsequent requests on the same loaded model don't reset
it (their resolve path short-circuits via `get_handler` before
`ensure_on_demand_loaded`). So a batch of streaming requests within the
idle window can unload the model mid-stream of the last request.

We do NOT patch upstream — the bug only bites continuous batches, and
sparse usage (one job, gap, another job) reloads the model fresh each
time. Mitigation is in `~/.mlx-server/config.yaml` (seeded from
`config/mlx-server.yaml.example` by `task install:mlx`): pick idle_timeouts long
enough that any realistic batch fits inside a single window
(gemma4: 1800 s, whisper: 3600 s). The per-chunk timeout above catches
the case where someone runs a huge batch and the timer still fires.

## Background work is globally pausable — soft pause between steps

A single `workers.control.get_control()` flag — flipped via
`POST /workers/{pause,resume}` — gates *all* background ML steps. The
contract is **soft pause**: the in-flight step (whatever it is — a
yt-dlp download, a Whisper transcription, an LLM summary stream)
**finishes normally**; the next step parks at a checkpoint until the
user clicks Resume, then carries on from where it stopped. Jobs are
never failed by pause and never restarted from scratch.

Checkpoints are placed at every step boundary:

- `whisper_worker` waits before pulling the next task from the queue.
- `pipeline._checkpoint_pause` is called between page extraction,
  YouTube transcript fetch, yt-dlp captions fallback, metadata probe,
  and the summary call.
- `runner._checkpoint_pause` is called between yt-dlp download, mlx
  transcription, and the summary call.
- `llm.client._acquire_llm_slot` re-checks at the lock — this catches
  callers that were queued behind the semaphore when pause flipped.

When a checkpoint blocks it sets `progress_stage="paused"` and publishes
`stage_event("paused")`, so the Library row shows "Paused". On resume
the stage is restored to whatever was about to run (e.g. "Transcribing")
so the row picks up where it was. Q&A never goes through this gate.
State is in-memory and resets on daemon restart.

`config.workers.cooldown_seconds` (default 0) inserts a sleep between
consecutive background jobs in both the Whisper worker and pipeline tasks
— useful when running a big backlog overnight on a fanless laptop.

## Timecodes are formatted in ONE place

`daemon/src/workers/timecodes.build_marked_text` is the single source of
truth for the `[MM:SS]` / `[HH:MM:SS]` format. The YouTube fast path
(`youtube-transcript-api` segments) and the Whisper path
(`/v1/audio/transcriptions verbose_json`) both feed segments through it.
The format is then opaque inside `Job.raw_text` — no separate column,
no parallel structures, prompts treat the markers as plain text.

The extension's `markdown.js` post-processes those markers into clickable
YouTube `?t=Ns` links (DOM-walk, skipping text inside `<a>`, `<code>`,
`<pre>`).

## URL normalization

The extension normalizes every URL through `lib/url.js#normalizeUrl` before
sending to the daemon (both create and lookup). Implications:

- Clicking a `[12:34]` timecode (which opens `?v=X&t=754s` in a new tab)
  doesn't look like a different page from the original `?v=X` job.
- Same article visited via `?utm_source=tw` and direct link map to the
  same canonical URL.
- For YouTube, identity is the video id alone — `/shorts/`, `/embed/`,
  `youtu.be/...`, `&list=...` all collapse to
  `https://www.youtube.com/watch?v=<id>`.
- Daemon stores whatever the extension sends; lookup happens on the same
  canonical form.

## Migrations: edit v1 in place during active dev

We're still pre-1.0. Schema changes go into the existing v1 migration —
the user is expected to `task reset` (wipes the SQLite volume) when the
schema changes incompatibly. We'll switch to additive migrations once
real users have data they care about.

## Daemon hot reload

`uvicorn` runs without `--reload` because the Whisper worker would orphan
jobs on every reload. Code changes are picked up only via
`docker compose restart daemon` (or `task down && task up`). Tests don't
require a restart — `tests/` is volume-mounted and pytest re-collects.

`pyproject.toml` changes need `task install` (or `docker compose build daemon`).

## YouTube libs auto-upgrade on container start

`daemon/docker-entrypoint.sh` runs `pip install --upgrade yt-dlp
youtube-transcript-api` whenever the daemon starts via `uvicorn`. Skipped
for `task test` and other ad-hoc commands so they stay fast. If pip can't
reach the network, the entrypoint falls back to whatever the image bundled.

This is the answer to "Google broke YouTube again" — `task down && task up`,
restart pulls the latest fix, no rebuild.

## Taskfile is a router, not a shell

`Taskfile.yml` keeps every `cmd:` block as a one-liner that delegates to a
script in `scripts/` (`install.sh`, `mlx.sh`). Task uses
the `mvdan.cc/sh` interpreter which mishandles `$!`, `$(cat ...)`, `kill -0`,
and other bashisms — putting that logic in real bash files avoids the trap
entirely. Add new lifecycle/install logic to a script, not inline in the YAML.

## Tests

`task test` runs ruff + mypy + pytest inside the daemon container. Always
run before declaring work done. External services (mlx, youtube_transcript_api,
yt-dlp, trafilatura, the Whisper worker, the retention worker) are mocked
so the test suite stays hermetic and fast (~5s for the full 120+ tests).

POST /jobs is async — tests that need the final state poll
`GET /jobs/{id}` until status transitions (see `_wait_until_done` in
`tests/test_api_jobs.py`). For repo / runner async tests, prefer the
condition-poll helper `_wait_until(predicate, timeout=...)` over a
fixed `for _ in range(N): await asyncio.sleep(...)` loop.

Concurrency / event tests live next to the contracts they protect:

- `tests/test_api_jobs_race.py` — parallel POST /jobs same URL must dedupe.
- `tests/test_repo_emit.py` — every write function publishes the right
  `job_event` to the global broker.
- `tests/workers/test_control.py` — pause/resume flips flag AND publishes.
- `tests/workers/test_broker.py` — fan-out, drop-on-full, unsubscribe,
  per-job→global mirror.

The extension has no test framework; ad-hoc `node --check` for syntax and
`node --eval` for logic that doesn't touch the chrome.* APIs.
