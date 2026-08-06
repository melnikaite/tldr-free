# Workers, concurrency, restart-safety

## POST /jobs is async, always

Returns 202 with `{id, kind, status}` immediately and spawns a background
pipeline. Clients NEVER block on summarization in the HTTP request — they
subscribe via `POST /ai/stream {job_id}` to follow.

The pipeline coroutine is held in a module-level `_BACKGROUND_TASKS` set
in `api.jobs` so Python's GC doesn't kill it.

## Background work is globally pausable — soft pause between steps

A single `workers.control.get_control()` flag (flipped via
`POST /workers/{pause,resume}`) gates **all** background ML steps. The
contract is **soft pause**: the in-flight step (yt-dlp download, Whisper
transcription, LLM summary stream) finishes normally; the next step parks
at a checkpoint until Resume, then carries on from where it stopped. Jobs
are never failed by pause and never restarted from scratch.

Checkpoints sit at every step boundary:

- `whisper_worker` — before pulling the next task from the queue.
- `pipeline._checkpoint_pause` — between page/transcript extract, captions
  fallback, metadata probe, summary call.
- `runner._checkpoint_pause` — between yt-dlp download, transcribe, summary.
- `llm.client._acquire_llm_slot` — re-checks at the lock, catches anyone
  queued behind the semaphore when pause flipped.

When a checkpoint blocks it sets `progress_stage="paused"` and publishes
`stage_event("paused")` — Library row shows "Paused". On resume the stage
restores to whatever was about to run, so the row picks up where it was.
Q&A bypasses this gate (the user is actively waiting). State is in-memory;
resets on daemon restart.

`config.workers.cooldown_seconds` (default 0) inserts a sleep between
consecutive background jobs in both Whisper worker and pipeline tasks —
useful when running a big backlog overnight on a fanless laptop.

## Transcript translation is its own background task

`workers/translator.py` runs independently of the main pipeline. Each
`(job_id, language_code)` pair gets a coroutine spawned by `enqueue_translation`
and held in a module-level `_BACKGROUND_TASKS` set so GC doesn't kill
it. The HTTP endpoint returns 202 immediately; the worker carries on
across SSE-subscriber comings-and-goings and across browser restarts.

Restart behaviour is **continue, don't fail**: `re_enqueue_running_on_startup`
(called from `main.lifespan`) picks up any row stuck in `running` /
`queued` and respawns the worker. Unlike MEDIA / PDF jobs, translations
are fully recoverable — `raw_text` is in the DB and the target language
is on the row, nothing external is needed. Cost: restart-continued
translations start from chunk 0 (no partial-chunk checkpointing).

Dedup: a second POST for an in-flight `(job_id, lang)` is a no-op —
the existing row's status is returned. UI binds Enter on the language
input directly to this endpoint without debouncing.

## Media and PDF jobs are ephemeral on restart

YouTube jobs are recoverable: `Job.url` is the canonical URL and
`re_enqueue_pending` resubmits them on daemon startup. Two kinds are NOT:

- **MEDIA**: the actual `media_url` (a `<video src>` or iframe yt-dlp
  will fetch) lives only in the in-flight `WhisperTask` and is never
  written to the row.
- **PDF** with `file://` URL: the PDF bytes came in the POST body
  (`pdf_bytes_b64`) and aren't persisted. http(s) PDFs could in
  principle re-fetch on restart, but for simplicity restart goes
  through the user clicking summarize again, not the queue.

On restart these are marked `failed` with an explanatory error; the user
resubmits from the extension.

Don't persist `media_url` or `pdf_bytes` to "fix" this. CDN URLs are
signed and expire; PDF bytes can be megabytes per row. Keeping the
inputs fresh at submit-time is the whole point.

## Video frames: on-demand, budget-capped, ephemeral

Two workers back the QA LOOK step (see llm.md): `workers/deixis.py` decides
WHERE to look, `workers/frames.py` fetches WHAT to look at.

`workers/deixis.py` is pure text analysis — no network, no LLM — over a
job's already-persisted `Job.raw_segments_json`. It finds moments where the
transcript's speech points at the video's picture and classifies each
`ACTION` (a demonstrated action, worth several consecutive frames),
`OBJECT` (a shown object/label, worth one good frame), or `EXTERNAL` (an
explicit defer-to-elsewhere reference — "the link in the description" —
where no frame helps at all; the point is to detect that the material
defers OUTSIDE itself). Marker tables are per language (en/ru/de);
extending to a new language means appending one marker list, nothing else
in the module changes. Bare demonstratives ("this"/"это"/"das") are
deliberately rejected as too common on their own — only a phrase pairing a
deictic word with a visual/imperative cue ("this WAY", "вот ТАК", "hier
SEHT ihr") counts as a candidate.

`workers/frames.py` turns a chosen candidate into JPEGs on disk. It
downloads only a short `yt-dlp --download-sections` window around the
timestamp — never the whole video (measured: a 3-second 480p section cost
152 KB / 4.1s wall against 71 MB for the full 720p video) — then extracts
a handful of frames with ffmpeg. Resolution is picked by the caller per
`DeixisCategory`: `SECTION_MAX_HEIGHT_PX` (480p) when the point is just to
SEE something (a gesture doesn't need to be legible), or
`SECTION_MAX_HEIGHT_READABLE_PX` (720p) when the point is to READ
something (a product label, on-screen text needs to actually be legible).
Two caps bound the cost regardless: `MAX_FRAMES_PER_CALL` limits one call,
`MAX_FRAMES_PER_JOB` (24) is a hard budget shared across every call made
for the same job over its whole lifetime — derived by counting JPEGs
already on disk rather than tracked separately, so it survives daemon
restarts with no schema change. Past the job cap, `fetch_frames` logs a
warning and returns `[]`, not an error — the same "degrade, don't break"
spirit as the rest of the QA flow.

`ensure_ffmpeg_on_path` (`workers/ffmpeg.py`) exists specifically for this
module's section downloads: yt-dlp's `download_ranges` precheck resolves
ffmpeg from PATH only and aborts before it ever consults
`ffmpeg_location` — measured directly, a thin launchd/systemd PATH makes
the section download fail with "ffmpeg is not installed" on a machine
where ffmpeg is plainly there via `ffmpeg_location` alone. Every section
download calls it first.

Section downloads are transient-flaky: measured directly against one real
YouTube video, three attempts on each of two ranges, one attempt in six
failed with `ffmpeg exited with code 8`, and which range failed moved
between runs — a CDN-side hiccup, not a property of any particular
section. `_download_video_section` retries the section download up to
`SECTION_DOWNLOAD_MAX_ATTEMPTS` (3) times, `SECTION_DOWNLOAD_RETRY_DELAY_SECONDS`
(1.5s) apart, before giving up — recovering most of these for free, well
before the caller pays for a full-video download. Only the section path
retries; the opt-in full-download fallback does not.

Frames are ephemeral like MEDIA/PDF jobs above, but for a different
reason — not "can't be recovered after a restart," but "not worth
persisting at all." They live at
`<data_dir>/frames/<job_id>/t<second>/frame_NN.jpg`, no DB column (same
shape as `media_url` for MEDIA jobs). `repo.delete_job` and
`repo.delete_jobs_older_than` each call `workers.frames.delete_job_frames`
alongside the existing cached-audio cleanup, so a job's frame directory
shares the audio file's lifecycle exactly — covered whether the user
deletes the job from the Library or it ages out under
`storage.retention_days`. The section/full clip yt-dlp downloads on the
way to the frames never lands in that directory at all: it lives in a
`tempfile.TemporaryDirectory` under `frames_scratch/` for the one
`fetch_frames` call that needs it and self-cleans before the call
returns, success or failure — no retention hook needed for that part.

## Large-PDF memory shape

PDF uploads (`file://` only — http(s) PDFs bypass this entirely) flow:

```
extension SW: fetch(file://) → ArrayBuffer → base64 string → POST body
              │                  N bytes        4/3 N            ≈ 4/3 N
daemon:       JSON body → JobCreateRequest.pdf_bytes_b64 → b64decode
              4/3 N         4/3 N (string)                  N (bytes)
pipeline:     run_pipeline holds pdf_bytes; process_pdf:
              text path:   pypdf reads from BytesIO → text → drop bytes
              vision path: per-page render+OCR, one PNG in flight at a time
```

Three invariants keep this bounded:

1. **Extension caps `file://` PDF size at 50 MB** (`MAX_LOCAL_PDF_BYTES`
   in background.js). Larger and the SW heap OOM-kills the whole
   extension. Users get a friendly error telling them to host locally
   over http or pre-OCR.
2. **Base64 encoding in the SW is chunked** so peak overhead above the
   input is ~4/3 ×, not 3 × (the naive `String.fromCharCode(...all)`
   form). See `_arrayBufferToBase64` in background.js.
3. **Vision OCR renders + OCRs one page at a time** in
   `workers/pdf._vision_ocr`, never holding the full set of PNGs in
   memory. The pipeline also clears `pdf_bytes` in a `finally` after
   extraction so the bytes don't sit through the long summary phase.
