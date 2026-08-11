# LLM calls and prompts

## Output language comes from config, never hardcoded

All LLM calls thread `config.output.language_name` into prompts as
`{output_language}`. The user sets `output.language` in `config/tldr.yaml`
to an ISO 639-1 code (`en`, `ru`, `de`, …) and the `language_name`
property expands it to the human-readable name that the LLM follows
reliably. Anything that isn't a known code (a full name, or
e.g. `"Brazilian Portuguese"`) is passed through verbatim.

Don't hardcode a language anywhere — in code OR in prompts.

## All LLM calls go through one semaphore

`llm.client._llm_lock()` (sized by `config.llm.max_concurrent_calls`,
default 1) gates every `complete()`, `stream_complete()`,
`complete_with_messages()`, and `stream_with_messages()` call. New LLM
work — summary chunks, QA, multimodal PDF OCR, anything else — must go
through these four functions; never bypass with raw HTTP. This keeps a
single-user laptop from thrashing the GPU when multiple jobs land at once.

Multimodal calls (image_url content) use the same primitives:
`workers/pdf.py` builds a messages list with text + image_url parts and
calls `complete_with_messages` — no special LLM API path. `llm/qa.py`'s
LOOK step (below) is the other multimodal caller.

The lock is **pause-aware**: acquire waits for both the semaphore AND
the global pause flag, and re-checks pause AFTER acquire so a flip that
landed while the caller was queued still holds them off. Q&A passes
`respect_pause=False` to bypass the gate (the user is actively waiting).

`stream_complete` enforces a per-chunk timeout
(`config.llm.stream_chunk_timeout_seconds`, default 60s). Without it a
hung backend stream would lock the queue forever.

This is defence-in-depth against a known mlx-server v1.8.1 on-demand quirk
(the idle-unload timer can fire mid-stream during continuous batches). We
do NOT patch upstream; mitigation is long idle_timeouts in
`~/.mlx-server/config.yaml` (seeded from `config/mlx-server.yaml.example`)
plus this per-chunk timeout for outliers.

## The LOOK step: forced tool calls for video-picture Q&A

For a job with a timestamped transcript, `llm/qa.py`'s plan → search →
synthesis Q&A flow gains an extra step between plan and search: LOOK,
which lets the model actually see a moment of the video instead of only
reading what was said. `workers/deixis.py` (pure text analysis, no
LLM/network) finds moments where the speech points at the picture ("watch
this", "вот так", "hier seht ihr") and classifies each `ACTION` / `OBJECT`
/ `EXTERNAL`. When candidates exist, the PLAN tool call gains an extra
`look_at_indices` field and the model names at most `_MAX_LOOK_AT_MOMENTS`
(2) worth fetching a frame for — with no candidates the PLAN tool/prompt
stay byte-identical to before this feature existed, so a page/PDF job (or
a video whose transcript has no deixis moments) takes exactly the old
path.

For each chosen moment, `workers/frames.fetch_frames` downloads a short
section around that timestamp (see workers.md) and every resulting JPEG
goes to the multimodal LLM in ONE call, not one call per frame: all the
frames are a ~1fps sample of the SAME few seconds, so cross-frame
reasoning ("the hand moves from A to B across these frames") only works
if the model sees them together — deliberately unlike `workers/pdf.py`'s
one-image-per-call OCR, where each page is an independent document. The
call is a FORCED `report_frame_findings` tool call (same technique as the
PLAN tool) returning a structured `VisionResult(finding, relevant,
best_frame_index)`, never free prose, so `stream_answer` decides whether a
frame actually contributed without any regex/keyword guessing over the
answer text. A `relevant=True` result with a valid `best_frame_index` is
what produces a `FrameRef` (a thumbnail the side panel renders under the
answer); `finding` text goes into the synthesis prompt's VISUAL FINDINGS
block either way — "we checked and there was nothing to see" is still
useful context. EXTERNAL candidates (a defer-to-elsewhere reference: "the
link in the description") are never fetched — the plan prompt tells the
model not to pick one, but the real guarantee is `stream_answer`'s LOOK
loop skipping an EXTERNAL index outright, independent of the model's
compliance.

Two prompts, two distinct anti-fabrication defenses:

- `qa_frames.txt` (the vision call) forbids treating the QUESTION as
  evidence about the picture — the question only says where to look, and
  the model must name a thing only as specifically as the picture
  supports.
- `qa.txt` (synthesis) forbids describing footage nobody looked at — the
  material is a transcript of SOUND, and VISUAL FINDINGS is the only
  evidence about what was shown; if it's empty or silent on the moment
  asked about, the model must say the picture wasn't examined rather than
  reconstruct a plausible scene.

### Measured: leading questions make the vision model fabricate

A leading question makes the vision model invent an answer that fits the
question's premise instead of the picture. Asking "what is he doing with
the BRUSH on the SCREEN" about frames of a girl drawing with a marker
produced a confident "tablet with a stylus, stylus tip in contact with the
tablet" — the same frames with a neutral question were described
accurately. That's why `qa_frames.txt` treats the question as a hint about
where to look, never as evidence, and why it tells the model to name
things only as specifically as the picture itself supports.

Structuring the call as a forced tool call made this WORSE before an
explicit guard was added to both the prompt and the tool-schema field
description: with the JSON tool-call format and no guard, a question
presupposing a laptop produced "На столе лежит черный ноутбук" ("there's a
black laptop on the table") where none existed — the model was more
willing to accept the question's premise inside structured tool-call
output than it was inside free prose. The guard now lives in both places:
`qa_frames.txt` itself and the `finding` field's description in
`_VISION_TOOL` (`llm/qa.py`).

Vision quality is model-dependent, and the prompt is honest about the
limit rather than pretending otherwise: at the readable resolution (OBJECT
moments), the vision model transcribed an on-screen overlay exactly
("EVERYDAY 14 / NORMAL 10"); at the lower resolution (ACTION moments) it
often could only say the action wasn't determinable from blurry frames.
Measured on `qwen3-vl-8b-instruct` — gemma is weaker at this (see the
README's Quick start section).

## Timecodes are formatted in ONE place

`daemon/src/workers/timecodes.build_marked_text` is the single source of
truth for the `[MM:SS]` / `[HH:MM:SS]` format. Both the YouTube fast path
(`youtube-transcript-api` segments) and the Whisper path
(`/v1/audio/transcriptions verbose_json`) feed segments through it. The
format is then opaque inside `Job.raw_text` — no separate column, no
parallel structures, prompts treat the markers as plain text.

Whisper segments require the patched mlx-openai-server. Upstream's handler
drops the `segments` + `language` fields that `mlx_whisper.transcribe()`
already produces — `scripts/mlx-patches/apply.py` puts them back. If the
patch isn't applied, `transcribe.transcribe_audio` falls back to a single
all-encompassing segment so the pipeline doesn't crash, but `[MM:SS]`
granularity collapses to one marker per video. See
`scripts/mlx-patches/README.md`.

The extension's `markdown.js` post-processes those markers into clickable
links (DOM-walk, skipping text inside `<a>`, `<code>`, `<pre>`). Two
flavours depending on the job:

- **YouTube** → `youtube.com/watch?v=ID&t=Ns`. Side panel click handler
  finds the open YouTube tab and seeks `<video>.currentTime` directly.
- **Generic media** (`kind=media`) → `<page-url>#t=Ns` (the page that
  contained the embedded media — `media_url` itself isn't persisted).
  Click handler tries to focus that page's tab and seek the first
  `<video>/<audio>` via `executeScript`; falls back to opening the URL.
  Iframe-embedded players (Vimeo etc.) won't seek — they're a separate
  document scope.

Page (HTML) and PDF jobs don't get `[MM:SS]` markers from the LLM in the
first place, so the post-processing short-circuits naturally for them.

## Whisper repetition-loop collapse

Whisper-family models occasionally decode-loop over noisy/silent audio,
emitting the same sentence (or a slowly-drifting near-copy) as dozens to
hundreds of consecutive segments — measured on real jobs: a 291-segment run
of an identical sentence, a 57-segment run of "Ja.". YouTube caption
segments never show this (measured 0-5% duplicate lines, longest run 1-2),
so it's a Whisper-only failure mode.

Note the boundary: this defect and its fix are independent of which chat
model is configured — it happens in the transcription step and is a property
of the transcript, so it needs no per-model verification (unlike the marker
cap below, where behaviour genuinely varies by model). It was verified by
running `collapse_repeated_segments` over every stored `raw_segments_json`
in the live DB: the two polluted Whisper transcripts collapse 595→95 and
656→387 segments, while all seven YouTube-sourced transcripts lose only
byte-identical duplicates.

`workers/timecodes.collapse_repeated_segments(segments)` collapses
CONSECUTIVE exact-or-near-duplicate runs down to their first occurrence
(kept as-is; the rest of the run is dropped). Near-duplicates are caught via
`difflib.SequenceMatcher` ratio ≥ `_NEAR_DUP_RATIO` (0.82) on top of
exact-match, since hallucination loops often drift slightly each repeat
rather than repeating byte-for-byte. The collapse threshold scales with
segment length — `_SHORT_SEGMENT_RUN_THRESHOLD` (6) for short segments
(≤ `_SHORT_SEGMENT_MAX_CHARS`, 12 chars) so a legitimate short dialogue
repeat ("Ja." ×3-5) survives, vs. `_LONG_SEGMENT_RUN_THRESHOLD` (1) for
longer segments, since a full sentence repeating even twice back-to-back is
essentially never real speech. Only consecutive runs collapse — this is
never a global dedup; the same sentence recurring later, separated by other
content, is untouched.

Hook point: `workers/transcribe.transcribe_audio()`, applied to the final
`segments` list right before returning `TranscribeResult` — after chunked
transcription has already merged all chunks back together, so a loop
spanning a chunk boundary is still caught. This is upstream of every
consumer of Whisper segments (`build_marked_text` for the summary, the
persisted `raw_segments_json` for the Transcript tab, `workers/translator.py`
for translation source) so they all see clean segments automatically — no
changes needed in `runner.py` or elsewhere. Intentionally NOT applied to the
YouTube caption fast path (`pipeline.py`), which builds segments directly
from `youtube-transcript-api`/yt-dlp and is measured clean.

## Marker-per-line cap

The LLM sometimes attaches every timecode it saw near a point to one
summary bullet (e.g. 9 markers on one line) instead of picking one, making
it unclear which link to click. Prompt wording is off the table — see
"Why not the prompt" below for what was tried and how each attempt failed
on both models — so this is enforced deterministically in code:
`workers/timecodes.cap_markers_per_line(text, max_markers=1)` keeps only the
first (earliest, leftmost) `max_markers` markers per line and strips the
rest, reusing `_tidy_after_bracket_removal` for whitespace cleanup. Default
is 1 — even 2 leaves ambiguity for the reported worst case, and the earliest
marker is always the most defensible seek target. No-op by construction on
marker-less text and on lines at-or-under the cap.

Streaming/stored consistency: both summary streaming call sites
(`pipeline._summarize_and_finish` and `runner`'s inline whisper-worker loop)
wrap the raw LLM delta stream in `workers/timecodes.cap_markers_in_stream`
instead of consuming `llm_summary.stream_summarize(...)` directly. This
guarantees what's PUBLISHED as a delta and what's ACCUMULATED into `parts`
(→ stored `summary_md`) are the literal same capped text at every point in
the stream — not just once at the end — so there's no visible "snap" from
an uncapped live view down to a capped final view when `done` fires.

Buffering granularity is per-MARKER, not per-line. `_MarkerCapState` (a
small char-fed state machine) holds text back only while it might still be
a `[MM:SS]`/`[HH:MM:SS]`-shaped bracket — bounded to ~10-11 chars — and lets
everything else through immediately, unbuffered. A first version buffered
whole LINES instead (simpler, and `cap_markers_per_line` — the non-streaming
primitive, used as-is elsewhere — could be reused directly on each completed
line), but that regressed real streaming latency badly: measured real
summaries have their single LONGEST line as the very FIRST thing the model
generates (the "## Обзор"/Overview paragraph, 613-706 chars measured) —
with
line-level buffering the side panel sat empty for 3-5s (gemma, ~50 tok/s) to
7-8s (qwen3-vl, ~21 tok/s) at the most-watched moment of the whole UX, then
dumped a wall of text at once. Marker-granularity holdback fixes that:
ordinary text streams char-by-char exactly as before, and only a
bracket-shaped run of characters is ever delayed, resolving a handful of
characters later either as a genuine marker or (via one extra lookahead
character, mirroring `_TIMECODE_MARKER`'s `(?!\(` guard) as an untouched,
uncounted markdown link.

Invariant, not a trade-off: `cap_markers_in_stream` and `cap_markers_per_line`
produce byte-identical output for the same input, checked directly by a
parametrized test in `test_timecodes.py` (several representative shapes ×
chunk sizes 1/3/7). In particular, when a marker beyond the cap is dropped,
`_MarkerCapState` also drops the whitespace run immediately preceding it
(held in `ws_hold`, itself bounded — it can only ever precede a `[`, never
grow line-length) — it isn't just cosmetic to skip this: two-or-more spaces
immediately before a newline is a Markdown hard line break (`<br>`), and a
bullet ending in several markers (the single most common shape of the
problem this cap exists to fix) is exactly where the dropped marker's
leading space would otherwise land right before the trailing `\n`.

This was kept server-side (not a `markdown.js` mirror) for the same reason
as before: the extension has no test runner at all (no way to verify JS
changes via `task test`), and the project already centralizes timecode
logic in Python (`build_marked_text` is the other example) — the "one
place" principle above extends naturally to capping.

### Why not the prompt — measured on two models

Everything below was measured against two locally-served backends through
LocalAI's llama.cpp backend: `gemma-4-e4b-it-qat-q4_0` (QAT Q4_0) and
`qwen3-vl-8b-instruct` (post-training Q4_K_M). Both were driven through the
production prompts and the production call shape, on real material from the
live SQLite DB rather than synthetic text.

Marker density is not a stable property of the prompt — it is a property of
the model, and the two disagree. With the production prompt unchanged, gemma
sat at 1-2 markers per bullet across runs, while qwen produced a maximum of
1, 3 and 9 on the *same* material in three separate runs. So no observation
from a single model generalises, and "it looked fine when I tried it" is not
evidence.

Three prompt rewrites were tried on the timestamp rule in
`prompts/summary_single.txt`. Each moved the failure somewhere else, and the
two models failed the same wording in *opposite* directions:

- Appending a cap to the end of the existing conditional rule ("attach at
  most one … never three or more"): no effect on qwen (still 3 per bullet);
  gemma unchanged, since it was already compliant.
- Leading with an unconditional imperative ("EXACTLY ONE timestamp per key
  point"): fixed density on both models (max 1, four runs of four) but made
  them FABRICATE markers on sources that contain none — gemma emitted 8 on
  the "Attention Is All You Need" PDF, qwen 10 on the same PDF in a separate
  run. Two different models, same wording, same new defect: the leading
  imperative overrides the exception for untimed sources further down.
- Leading with the condition instead ("first decide whether the source
  contains markers …"): fixed the document case on both, but then qwen
  emitted ZERO markers on a video that does have them (26 bullets, no
  markers at all) — strictly worse, because jump-to-moment is the feature.

Hence the deterministic cap. Known limitation to keep in mind: the cap
bounds markers *per line*, so it does not fix fabrication. On untimed
sources with the production prompt (three runs each), qwen emitted no
markers in 6 of 6 runs, but gemma emitted 9 on a Wikipedia page in 1 of 6 —
and a capped version of that is 1 spurious marker per line, not none.
Removing markers from material that never had timestamps is a separate
concern; `strip_all_timecodes` already exists for the QA path and is the
natural tool if this needs closing on the summary path too.

Scope caveat: two models, both quantised, both served by the same llama.cpp
build. A different backend or a cloud model may behave differently — treat
the numbers above as evidence that model-level behaviour varies, not as a
characterisation of any model in general.

## Transcript translation

`workers/translator.py` translates a job's transcript into a target
language, caching the result in `transcript_translation` (PK
`job_id+language_code`). Triggered by
`POST /jobs/{id}/transcript/translate {lang}`; dedup is at the row level —
a second POST for an in-flight `(job_id, lang)` returns the existing
status without spawning a duplicate worker.

**Never trust the model's line alignment — verify it deterministically
and repair by narrowing the window.** Losing input is structurally
impossible; the worst case is a line left in the source language and an
honest `partial` status, never a silently dropped chunk.

1. **Line-aware packing, not `split_for_summary`.** The translation
   prompt (`prompts/transcript_translate.txt`) is a strict one-input-line
   → one-output-line contract with a `[MM:SS]` marker copied verbatim.
   `llm.chunking.split_for_summary` cannot honor that: it splits on blank
   lines first, and a marked transcript has none, so it degrades to
   splitting the whole transcript by SENTENCE — tearing multi-sentence
   lines in half and detaching the marker from the second half before the
   model ever sees it. The translator instead uses
   `llm.chunking.pack_lines`, which packs whole lines into token-budgeted
   groups and never splits one.
2. **Marker-based verification, not blind trust.** `_align_translation`
   walks the output with a forward-only cursor, matching each input
   line's marker to the first not-yet-consumed output line carrying the
   same marker BY VALUE (`[1:02]` matches `[01:02]` — tolerates a model
   that mangles the marker's formatting while copying it; the rebuilt
   line always uses the INPUT's own marker text, never the model's copy).
   Forward-only search makes duplicate markers align in encounter order,
   and unmarked output lines that follow a match get absorbed into it
   (recovers a line the model split in two). Any input marker not found
   fails verification — as does a repetition-loop catch mirroring
   `api.ai._DEGEN_TAIL_RE` (line-granular here since output is
   line-shaped by contract): a run of more than `_MAX_REPEATED_LINES` (6)
   identical output lines, EXCEPT the threshold floats up to match
   whatever run-length the INPUT of that call already contains
   (`_degenerate_run_threshold`) — a faithful translation of a
   genuinely-repetitive source (measured live: 28 consecutive
   `[05:28] Ja.` Whisper segments) is just as repetitive, and the
   translator must not assume `timecodes.collapse_repeated_segments` has
   already cleaned the input.
3. **Bisection with a bounded budget, leaf fallback.** A group that fails
   verification is split in half and each half retried independently, up
   to `_MAX_BISECT_DEPTH` (8); a single line that still fails gets one
   more retry before it's kept verbatim from the source. Depth has to be
   deep enough to actually REACH single-line granularity on a real group
   or that retry path is dead code: `pack_lines` produces much bigger
   groups than the old sentence-shredding chunker did (measured live: a
   656-line transcript packed into 4 groups of 145-177 lines, not 6
   ragged ones), so a shallower cap bottoms out well above 1 line — depth
   4 measured stopping at ~11 lines on a 170-line group, turning one bad
   line into an 11-line fallback block. A per-job `_CallBudget`
   (`max(_CALL_BUDGET_FLOOR, groups + 2 × (2×_MAX_BISECT_DEPTH + 1))`
   calls — the derivation assumes two fully-isolated bad regions each
   reaching max depth, plus one call per group) caps how far a
   pathological model can make this fan out — budget or depth exhausted
   falls the whole remaining (sub-)group back to source text. Raising
   the depth constant means re-deriving the budget one, not just the
   depth.
4. **`max_tokens` is dynamic, not fixed.** Russian output measured ~1.5×
   the source token count on the same transcript (11573 RU vs 7691 DE) —
   a fixed 4000 was thin against a 2000-token chunk plus repair overhead.
   `count_tokens(text) * 2.5`, floor 512 / ceiling 8000.
5. **`status="partial"` is the honest outcome of imperfect repair.** If
   any line fell back to source, the row is `partial` (not `done`), with
   `text` fully populated and `error` a plain count
   ("N of M lines could not be translated…"). The sidepanel treats
   `partial` like `done` for selection/export — it has real text — but
   flags it visually and offers it to "Retry all" alongside `failed`.
6. **Markerless sources** (the `raw_text` fallback for PDF/HTML/legacy
   jobs — no `[MM:SS]` to verify against): when fewer than 90% of a
   group's lines carry a marker, `_align_translation` drops to an
   emptiness/degeneration check only. Bisection and the `partial` outcome
   still apply on top of that weaker check.
7. **Pause-aware**: between groups (`_checkpoint_pause_translation`) and
   inside `stream_complete(respect_pause=True)`. Same pause flag as the
   summary path.
8. **Restart-safe**: rows left in `running` at daemon startup get
   re-enqueued by `re_enqueue_running_on_startup` (called from
   `main.lifespan`) — the source text is in the DB and the language code
   is on the row, nothing external is needed. Restart-continued
   translations start from group 0 (no partial-group checkpointing); the
   user "loses" any progress percent shown before the restart but the
   result is correct.

`en→en` (or whatever the source is) short-circuits — no LLM run, the
endpoint returns `is_source=true` and the sidepanel switches to display
the original via `GET /transcript` directly.

Language codes are normalised by `llm.languages.normalize_lang`. Accepts
ISO-639-1, ISO-639-2, English names, autonyms, and aliases; rejects
anything unknown with a helpful 400 listing supported codes.

## Cloud backends: API key resolution, dialect auto-detect, reasoning headroom

`llm.base_url` (and independently `whisper.base_url`) may point at a cloud
OpenAI-compatible endpoint instead of a local one — nothing else in the
pipeline changes. Both sections get the exact same key-storage machinery,
implemented once in `_ApiKeyConfigMixin` (`src/config.py`) and inherited by
both `LLMConfig` and `WhisperConfig` — not two copies that could diverge.
Config fields, per section: `api_key`, `api_key_file`, `api_key_keychain` +
`api_key_keychain_account`, and its own env var (`TLDR__LLM__API_KEY` /
`TLDR__WHISPER__API_KEY`).

- **Resolution order (first match wins): env → keychain → file → inline.**
  This lets an operator override a committed `tldr.yaml` at deploy time
  (env, also handy for Docker/foreground runs), keep the key out of any
  file at all (keychain — recommended and the default `PATCH /config`
  picks when available; `keyring` is a base dependency, no extra install
  step), or at least keep it out of the YAML (`api_key_file`, path expands
  `~`, must be `0600` — the right choice for Docker installs, which have
  neither macOS Keychain nor a Secret Service) without touching `api_key`
  in plain text. Whichever source wins, resolution happens once at config
  load (`effective_api_key`, read once per process lifetime via the
  `lru_cache`d client — not per request) — changing the keychain entry or
  key file needs a daemon restart like any other config change.
  `llm` and `whisper` are fully independent: separate keychain service
  (`tldr-daemon-llm` / `tldr-daemon-whisper`, see
  `src/api/config.py#_KEYCHAIN_SERVICES`), separate managed key file
  (`llm.key` / `whisper.key`, see `config.api_key_file_path(section)`),
  separate env var — patching one section's key via `PATCH /config` never
  touches the other's storage.
  Headless operation (no logged-in user at the console) is explicitly
  unsupported — the extension needs a human at the browser regardless —
  so keychain access dialogs are never a problem in practice: the daemon
  writes its own key (via the options page / `PATCH /config`), the
  creator of a Keychain item is auto-added to its own trusted-app ACL, and
  that ACL is persistent (survives reboot, and a `uv tool install --force`
  reinstall — the venv is rebuilt but the underlying macOS Python binary
  and the `keyring` code aren't). `config.keychain_backend_available()`
  (cached for the process lifetime) is the single source of truth for
  "is keychain actually usable right now" — shared by both sections (it's
  a machine-level fact, not per-backend) — it backs `GET /config`'s
  `keychain_available` field, the `PATCH /config` default-storage choice,
  and the options-page UI that disables the keychain option when false
  (Linux without a Secret Service in the session).
- **Backend dialect auto-detection is cached per process.** Some
  OpenAI-compatible backends want `max_tokens`, others (reasoning models —
  gpt-5, o-series) require `max_completion_tokens` and reject `temperature`.
  `token_param: auto` (default) probes this from the live backend: on the
  first call it uses its best guess, and on an HTTP 400 it retries once
  with the other dialect and remembers the result for the rest of the
  process. `send_temperature` follows the same auto/cache pattern. Set
  either explicitly only if a backend's 400 response is ambiguous enough
  to fool the detector.
- **`reasoning_headroom_tokens`** (default 4000) is added on top of the
  requested output size for reasoning models, since their hidden
  chain-of-thought consumes part of the same output-token budget before
  the visible answer starts — without headroom, long answers get cut off
  mid-stream on gpt-5/o-series. `max_output_tokens` is an optional hard
  cap on the visible output, independent of headroom.
- `tldr.yaml` is created with `0600` permissions (both the Docker
  `task install` path and the native packaged-template path) since it may
  contain a plaintext cloud key via `api_key`.
