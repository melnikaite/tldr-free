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
