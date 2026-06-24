# Events and SSE

## Three SSE surfaces, one broker

The daemon produces SSE streams from three places, all wired through the
same in-memory pub/sub (`workers.broker`):

- **POST /ai/stream** — body shape decides:
  - `{job_id}` → SUMMARY mode: subscribes the per-job channel.
  - `{job_id, question}` → QA mode: triggers a fresh QA call, persists
    user + assistant messages, streams the answer.
- **GET /events** — single global stream for Library + Side panel.
  Subscribes the global broker. Clients narrow the firehose with
  `?types=job,workers,done,error` (server-side filter, so the high-volume
  per-token `delta` chatter never reaches surfaces that only need status).

The per-job broker (`JobEventBroker`) mirrors every event into the global
broker (`EventBroker`) with `job_id` attached. So `/events` sees stage /
delta / done / error AND job/workers state changes on **one connection per
UI surface**. This matters because Chrome's 6-per-origin HTTP/1.1 cap will
stall fetches once you open per-job SSEs in parallel.

## Event shapes (uniform regardless of mode)

- AI streams — `AIStageEvent | AIDeltaEvent | AIDoneEvent | AIErrorEvent`
- Job-list — `job_event(action, job)` where `action ∈ {created, updated, deleted}`
- Workers state — `workers_event({paused, queue_size, running})`

## When adding new capabilities

Prefer extending the request body of `/ai/stream` over creating a new
endpoint. For new app-wide state, publish via the global broker; don't
invent a sibling SSE endpoint.

## State-changing repo functions auto-publish

Every function in `src/storage/repo.py` that mutates a Job row publishes
the matching `job_event` itself as a side effect: `create_job` → created,
`update_status` / `mark_done` / `mark_failed` / `set_extracted` /
`reset_for_retry` → updated, `delete_job` / `delete_jobs_older_than` →
deleted. `set_audio` is intentionally silent (internal plumbing the UI
doesn't render).

The invariant: "DB write happened" and "UI told" are inseparable. Callers
in `api/jobs.py`, `workers/pipeline.py`, `workers/runner.py` no longer
need (and don't have) explicit emit calls. Three real bugs we hit
(title-not-updating, dedup race, pause-broken) all came from a state
change that forgot its event publish — the auto-emit closes the gap.

New write functions follow the same pattern: late import the broker inside
a `_publish_*` helper, wrap publish in `contextlib.suppress(Exception)` so
broker hiccups never roll back the user's write.

Likewise `workers.control.WorkerControl.pause/resume` publishes
`workers_event` itself — `api/workers.py` is just a thin endpoint.
