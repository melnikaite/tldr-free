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

## Media jobs: duration probe + page-text fallback

The extension's `<video>`/`<audio>` detection filters by duration, not
visibility (see [extension.md](extension.md) — the audio path exists for
hidden, script-driven players, so filtering by visibility would break real
podcast/lecture support). That filter can still be wrong — the DOM's
`el.duration` may be unknown client-side, or the server-side probe below may
disagree — so `kind=media` jobs get a second, independent check before the
(slow) yt-dlp download ever starts, plus a fallback if Whisper still comes
back empty.

Establishing "is this clip long enough to plausibly contain speech" turned
out to need THREE tiers, not one — the first tier alone silently failed to
fire for the exact case it was built for (see below), which is why a
duration below threshold can now be discovered at three different points
in the pipeline, and why the transcript-quality check downstream (further
below) had to be tightened independently.

**Tier 1 — yt-dlp metadata probe (free, pre-download).**
`runner._process_one`, gated to `kind == "media"` AND no cached audio to
reuse (a retry that already has downloaded audio skips straight to
transcribing, same as before), calls `youtube.fetch_video_metadata` — the
same `extract_info(url, download=False)` probe already used
post-transcription for YouTube title/language backfill — BEFORE the
download step. When the early probe DID run, its result is reused for the
post-transcription title/language backfill instead of calling
`fetch_video_metadata` a second time.

**This tier alone is not sufficient.** yt-dlp's *generic* extractor — the
one that handles a plain static-asset URL like `/assets/notification.mp3`
served by a real web app, as opposed to a media-platform URL yt-dlp has a
dedicated extractor for — does NOT report a `duration` via
`extract_info(download=False)` for that shape of URL. Confirmed live: both
a 3s and a 40s static file came back `duration: None`. Since "a UI
notification ding on a plain static URL" is exactly the case this whole
mechanism exists to catch, tier 1 alone routinely never fires for it, and
the job would proceed straight to download + Whisper every time without
the tiers below.

**Tier 2 — URL-direct ffprobe (optional, best-effort, pre-download).** If
tier 1 came back with no known duration, `runner._process_one` tries
`transcribe.probe_url_duration(url)`: ffprobe reading the duration
directly off the URL, no download. ffmpeg's http(s) protocol supports
byte-range requests, so for a normal seekable static file this is
typically 1-2 small requests rather than a full download. Guarded by a
hard wall-clock `subprocess.run(..., timeout=...)` (protocol-agnostic — it
kills the subprocess regardless of what it's doing internally, so a slow/
non-seekable/hostile server can never hang the single Whisper worker) and
swallows every failure mode (missing ffprobe, non-zero exit, timeout,
unparseable output) to `None`, falling through with zero behavior change.
Does not forward cookies — an authenticated URL simply fails this probe (a
disclosed limitation, not a reliability regression) and falls through to
the normal cookie-aware download.

**Whichever of tier 1/2 first produces a known duration below
`runner.MEDIA_MIN_DURATION_SECONDS` (12.0, mirrors the extension's
`MIN_MEDIA_DURATION_SECONDS`) skips the download + Whisper transcribe
entirely** and the job falls back to page text (below). Still unknown
after both -> falls through to the normal download path, unmodified.

**Tier 3 — local ffprobe on the downloaded file (required, post-download,
pre-Whisper).** This is the tier that actually catches the static-asset
case: after `youtube.download_audio` returns, if duration is STILL
unknown, `runner._process_one` calls `transcribe.probe_duration(audio_
path)` — a thin async wrapper around the same `_probe_duration`/`ffprobe`
helper `transcribe.py` already uses internally for chunked transcription —
on the real, now-local file. This is authoritative: the file exists, so
ffprobe either reads its real duration or genuinely can't (ffmpeg
unavailable/corrupt file), in which case behavior is unchanged (proceeds to
Whisper as before). If it reveals a duration below threshold, Whisper is
never called — the job goes straight to the page-text fallback.

Because this reject happens AFTER a real download (unlike tiers 1-2), the
downloaded file needs the same "delete, don't keep for retry" cleanup a
normal success gets, even though `transcribe_audio` was never called. The
`finally` block's cleanup branch keys off `transcribe_done` to choose
delete-vs-keep; this path sets `transcribe_done = True` before returning to
reach that same branch — see the comment at that assignment and at the
`finally` block for why `transcribe_done` no longer literally means
"Whisper ran".

**The transcript-usability fallback.** Separately, if Whisper actually runs
(all three duration tiers were inconclusive, or confirmed the clip is long
enough) but its output carries no usable content, the same fallback
triggers rather than summarizing garbage into a fabricated-looking result.
This used to check only `not raw_text.strip()` (literal emptiness) — too
weak: Whisper fed a few seconds of a real "ding" hallucinated a short,
plausible-looking, NON-empty transcript (e.g. `"[chime]"`), which sailed
through that check and produced a real fabricated summary in production.
The check is now `transcribe.transcript_is_unusable(whisper_result.
segments)` — reusing (not reimplementing) `_is_confirmed_silence` (empty/
punctuation-only/annotation-only) plus a degenerate-repeated-run check via
`timecodes.collapse_repeated_segments`. Checked against the raw segment
list, not the `[MM:SS]`-marked `raw_text` string — the marker brackets
would make the annotation-only check misfire on a real transcript. This is
purely a `kind=media` thing — the youtube fast/deferred paths are untouched.

**The fallback itself** (`runner._fallback_to_page_text`, shared by all of
the triggers above) summarizes `WhisperTask.page_text` — extension-supplied text,
extracted alongside the media candidate by `extract.js`'s
`extractPageText()` (see extension.md) — instead of audio.
`from_audio_transcript=False` (it isn't a transcript), and it persists via
`set_extracted`/`mark_done` with **`transcript_source=TranscriptSource.
PAGE_EXTRACT`** — reusing the same enum value the `kind=page` Readability
path already uses, deliberately, rather than adding a new one. The
combination of `kind=media` + `transcript_source=page_extract` on the row IS
the signal that page text, not audio, was summarized; also logged at
`log.info`. If `page_text` is empty/missing (e.g. an old extension build
that predates this field) or the resulting summary is empty, it raises
`RuntimeError` with a specific message — `_process_one`'s existing "raises
on any failure, caller (`whisper_worker`) calls `mark_failed`" contract
handles it the same as any other failure; the fallback never calls
`mark_failed` itself.

`WhisperTask.page_text` follows the exact same ephemerality convention as
`media_url` (see "Media and PDF jobs are ephemeral on restart" below): it
rides along only inside the in-flight task, not a DB column, and is lost on
daemon restart — a restart already marks in-flight media jobs failed via
`re_enqueue_pending`, so this adds no new gap.

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
