# Contracts between layers

The agreements that survive any internal refactor. Break one, the other
side stops working in a way the type checker can't catch.

## API contract is mirrored, not generated

`daemon/src/api/schemas.py` ↔ `extension/src/lib/api-types.js`. Manual sync
— when you change one, change the other in the same commit. Bump
`DAEMON_API_VERSION` in `daemon/src/config.py` for breaking shape changes
so old extensions detect the mismatch instead of mis-parsing payloads.

`extension/src/lib/daemon-client.js` is the only place that issues HTTP to
the daemon. New endpoints go there with a JSDoc return-type annotation
against the api-types alias.

`FrameRef` (the QA video-frame thumbnail — seconds, timecode, phrase,
frame_url) is one instance of this rule, not an exception to it:
`schemas.py`'s `FrameRef`/`AIFramesEvent` and `api-types.js`'s matching
JSDoc typedefs are kept in step the same manual way as everything else
mirrored here.

`Job.queued_reason` (`storage/db.py`, migration v9) ↔
`JobDetails.queued_reason` (`schemas.py`) ↔ `JobDetails.queued_reason`
JSDoc (`api-types.js`) is another instance. It uses the same `_UNSET`
sentinel trick as `transcript_missing_seconds` in `storage/repo.py`
(`update_status`'s `queued_reason` kwarg) because it has to be able to
regress from set back to unset: `workers/pipeline.py` sets it when a
YouTube job's transcript fast path defers to Whisper,
`workers/runner.py`'s `_process_one` clears it back to `None` the moment
that job actually resumes.

`Job.diagnostics_json` (`storage/db.py`, migration v10) ↔
`JobDiagnosticsResponse`/`JobDiagnosticsChunking`/
`JobDiagnosticsCoverageRecheck` (`schemas.py`) ↔ their JSDoc twins
(`api-types.js`) is a fourth instance — `GET /jobs/{id}/diagnostics`
(`api/jobs.py`), distinct from the daemon-wide `GET /diagnostics`. The
stored JSON shape is produced by `workers/transcribe.TranscribeDiagnostics`
(`dataclasses.asdict`, written by `runner.py`) and read back into the
Pydantic models field-by-field in the route handler — keep the two shapes
in step the same manual way as everything else here. Hard privacy
constraint on this one: never add cookies or transcript text to it (see
the dataclass's own docstring for the full audit) — the whole point is
that the resulting JSON is safe to hand to someone else.

`GET /jobs`'s `q` param (`api/jobs.py` → `storage/repo.list_jobs`) is a case-
insensitive `LIKE '%...%'` search across `Job.title`, `Job.summary_md`,
`Job.raw_text`, and `transcript_translation.text` for every language cached
for that job, combined with `status`/`kind`/`since`/`url` via AND. No schema
change was needed (additive query param on an existing list endpoint), but
the translation half of it IS an invariant worth protecting: it's a single
correlated `EXISTS (SELECT 1 FROM transcript_translation WHERE job_id =
job.id AND text LIKE :q)`, not filtered by translation `status`. Do not
replace it with a per-language column, a language list in code/config, or a
separate query per language — the whole point is that it falls out of the
`(job_id, language_code)` schema and needs no maintenance as languages are
added. `%`/`_` in `q` are escaped (`repo._escape_like`) so user text matches
literally. Plain `LIKE` (not FTS5) is a deliberate scale call — see the
docstring on `list_jobs` for when to revisit.

## The export bundle is a second, slower-moving contract

`storage/bundle.py`'s zip format has its own `version` in `manifest.json`,
independent of `DAEMON_API_VERSION`. It has to be: an API version says what
two live processes agree on right now, while a bundle is read by a daemon
that may be months newer or older than the one that wrote it, on another
machine.

So `job.json`'s shape is not free to drift with the ORM. Adding a field is
safe — the importer ignores what it doesn't know, and an older bundle
simply arrives without it. Renaming or repurposing one is not: bump
`BUNDLE_VERSION`, and keep reading the old shape unless you're willing to
tell people their exports are now unreadable. The importer rejects a
`version` above its own precisely so that a newer bundle fails loudly
instead of importing half of itself.

The member-name patterns in that module are a security boundary, not
formatting: they're what stops a hostile zip from writing outside a job's
own frame directory. Widen them only with the containment check in
`_copy_frames` in mind.

## URL normalization

The extension normalizes every URL through `lib/url.js#normalizeUrl` before
sending to the daemon (both create and lookup). Implications:

- Same article via `?utm_source=tw` and direct link map to the same canonical URL.
- Clicking `[12:34]` (opens `?v=X&t=754s`) stays the same job as the original `?v=X`.
- For YouTube, identity is the video id alone — `/shorts/`, `/embed/`,
  `youtu.be/...`, `&list=...` all collapse to
  `https://www.youtube.com/watch?v=<id>`.
- Daemon stores whatever it gets and looks up on the same canonical form.

If you add a new URL family, extend `normalizeUrl` — never special-case on
the daemon side.

## Settings API writes to an overrides file, never to the template

`GET /config` / `PATCH /config` / `POST /config/test`
(`daemon/src/api/config.py`) let the extension edit backend/model/API
key/output language without hand-editing YAML. `tldr.yaml` is a hand-edited,
comment-heavy template — writing to it with `yaml.safe_dump` would destroy
those comments. `PATCH` instead writes `tldr.local.yaml` (a sibling file,
`src/config.py#overrides_path`), which `get_config()` deep-merges on top of
the template before env-var overrides are applied (env still wins over
both). Every `PATCH` is validated (`config.validate_full_config`) before
anything is written. API keys are never echoed back by `GET`/`PATCH` — only
`api_key_set` / `api_key_hint` (last 4 chars) / `api_key_source`.
