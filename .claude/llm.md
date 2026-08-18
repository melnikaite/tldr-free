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

## SEARCH has an on/off switch: `config.qa.web_search`

The plan → look → search → synthesis flow's SEARCH step (DuckDuckGo query
+ page fetch via `workers/search.py`) is gated by `config.qa.web_search`
(`QaConfig`, default `True`) in addition to PLAN's `sufficient` verdict —
`stream_answer` only calls `_search.ddg_search_with_content` when BOTH
`not sufficient` AND `web_search` is true. With it `False`, the step never
runs at all: no query built into a DDG call, no page fetched, regardless
of what PLAN decided. PLAN itself still runs unconditionally either way —
turning search off doesn't change PLAN's tool/prompt, it just makes the
verdict a no-op for step 3. Exposed via `GET`/`PATCH /config` (`qa.web_search`,
mirrored in the extension's `api-types.js`) and a checkbox on the options
page; documented in `tldr.yaml.example`.

Turning search off removes the one source `qa.txt`'s "never refuse with
'the material doesn't say'" permission assumes is available to catch a bad
guess. Rather than editing `qa.txt` itself (which would also touch the
`web_search=True` path, where behavior must stay identical to before this
setting existed), `_answer_messages` appends an extra
`_NO_WEB_SEARCH_RULE` block to the prompt ONLY when `web_search` is
`False` — same "append, don't interpolate" technique `_plan_messages`
already uses for the LOOK step's candidates block. The appended rule keeps
the "general knowledge to explain/contextualize" permission intact but
tells the model plainly to say "the material doesn't cover this" instead
of inventing a current/external specific it has no way to have looked up.

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

## Transcript coverage: a "done" job can still be silently short

Measured live TWICE on the same ~21.5 min video, two different failures:

- Job `3IXBfawKZrj7` (chunked upload): Whisper decode-looped 205 s into a
  648 s chunk, repeating an already-said sentence for the rest of the
  chunk. The collapse above correctly folded that run down to one segment
  near the START of the gap — nothing checked that the run then continued
  to the chunk's own end, so ~6 minutes of real speech afterward vanished.
- Job `Y7odGFeN7agb` (same video, re-run on an earlier version of this
  fix, single-request upload — the file fit under the cap, no chunking at
  all): Whisper produced normal speech to 728.9 s, then nothing until one
  stray one-word segment ("2025.") at 1290.9 s of a 1291.6 s file — 9m22s
  missing from the MIDDLE, with one trailing word dragging the last
  segment's `end` to within a second of the real duration.

The first version of this check compared only the LAST kept segment's
`end` against the known duration, and the second case is exactly why
that's wrong: that shape reads as ~100% covered by an end-of-last-segment
test while 52% of the audio's actual runtime (669 s of 1292 s) produced
nothing. The hole isn't reliably at the tail, and chunking isn't the
cause either — the two failures happened at different timestamps on
different upload paths against the exact same file, so this has to work
identically whether or not the file got chunked.

`workers/transcribe.py` checks this deterministically instead of
trusting the model — but, as of this rewrite, with **no size-based
correctness threshold at all**. An earlier version flagged a unit as
short only when its worst gap exceeded a fixed 90s, on the theory that
short gaps are "probably fine". That conflates two different questions:
how much a false positive costs (one extra, bounded Whisper call —
cheap) versus whether content actually went missing, which no size
threshold can answer, only guess. A real measured loss of ~40s — well
under that old 90s cutoff — would have passed through silently; it's
exactly the shape the rewrite below closes.

**Every suspicious interval gets re-transcribed and re-checked; the
question "was this actually speech?" is answered by what comes back, not
by how big the hole was.** Two sources feed the suspicious-interval list,
merged (`_merge_intervals`, so an overlap between the two isn't
double-counted):

- `timecodes.collapse_repeated_segments` now returns `(kept_segments,
  discarded_runs)` — `discarded_runs` is a list of `DiscardedRun(start,
  end)`, the EXACT span of segments a collapsed run threw away (the run's
  first occurrence is kept; `start`/`end` bound everything from the
  second occurrence's own start through the last occurrence's own end).
  This is known precisely at the moment collapse discards it — no
  arithmetic guessing required downstream.
- `_find_gaps` remains as a safety net for the shape a discarded run
  can't cover: Whisper returning nothing at all for a span, with no
  repeat-run for collapse to have discarded anything from (the measured
  `Y7odGFeN7agb` case — a stray trailing segment, no repeat loop
  involved).

The only threshold left, `_MIN_RECHECK_SECONDS` (~5s), is a **cost
regulator, not a correctness gate**: an interval below it isn't
re-transcribed at all (an ordinary breath/pause isn't worth a Whisper
call), but there is no size above it that's treated as "probably fine" —
every interval that clears it gets checked, regardless of how far above
5s it sits. Getting this constant slightly wrong costs a few extra cheap
Whisper calls in one direction, or checking a few more short-but-real
pauses in the other — never a silently-accepted content loss the way the
old 90s threshold could.

**Deciding "was that actually speech?" — and a regression from getting
this distinction wrong.** A first version of this rewrite folded a
degenerate repeated run (a hallucination loop reproducing on the recheck
itself) into the same "not speech" verdict as confirmed silence. That
shipped, ran on the same measured video, and made things WORSE than
before this whole feature existed: a 562s hole that provably contained
live dialogue (manually verified: 20s hand-transcribed at the same
offset came back with real conversation) got reported as "not lost
content" with `missing_seconds` silently going to 0/`None` — coverage
dropped from 88% to 52% while the job LOOKED clean. The reasoning error:
"a loop means it's not real content" is backwards. A loop means the
recheck reproduced the SAME kind of failure the original transcription
had — we still don't know what's there, which is a different, weaker
claim than "confirmed no speech". Silence and "still unknown" must never
share a verdict.

So the classification is now explicitly THREE-way, split across two
independent checks:

- `_is_confirmed_silence` — the ONLY verdict allowed to exclude a window
  from `missing_seconds` and log "not lost content". Structural, in
  order: (1) empty/whitespace-only → confirmed silence; (2) only
  punctuation/dash characters (a bare `-`) → confirmed silence; (3) the
  text as a WHOLE is made of bracket/asterisk/paren annotations
  (`*Dramatic music*`, `[Musik]`, `(laughs)`) → confirmed silence (real
  dialogue with an incidental parenthetical aside does not match — the
  whole text has to be annotations, not merely contain one). Matching
  structure rather than known phrases matters here too — measured live,
  one backend produced 172 CONSECUTIVE `*Musik*` segments over a musical
  stretch where a different backend said `*Dramatic music*` once for the
  same kind of audio.
- A degenerate repeated run, checked SEPARATELY by `_ensure_coverage`
  itself via `collapse_repeated_segments` directly on the recheck's own
  segments (not folded into `_is_confirmed_silence`): NEVER added to
  `missing_seconds`'s exclusion list, NEVER logged as "not lost". What
  DOES survive is `collapse_repeated_segments`'s own first-occurrence
  rule — usually the real content recognized before the loop took over —
  spliced in exactly like real speech, rather than discarding the whole
  slice. The remainder past that point stays exactly as suspicious as
  before: either re-sliced-and-retried within budget, or, honestly,
  still counted as missing.
- Otherwise (real speech): spliced in in full.

A window confirmed silent is left untouched (whatever was there — the
collapsed representative segment, or nothing — stays). A window that
can't even be re-checked (ffmpeg unavailable to cut it, or the budget
below ran out before reaching it) conservatively counts as missing too —
not knowing what's there is not the same as confirming it's fine.

**The recheck's own ASK SIZE matters, not just whether one happens.** The
same live incident above traced back to a second cause: `_ensure_coverage`
was cutting and re-sending the ENTIRE suspicious interval in one request
— 562s in that case — which reproduces the exact conditions Whisper
already failed under (a long, hard-to-track request is what triggers a
decode loop in the first place), so of course it fails the same way
again. The manual 20s probe worked precisely because it was short.
`_MAX_RECHECK_SLICE_SECONDS` (90s) caps how much audio ANY single recheck
request may cover; a suspicious interval longer than that is split into
consecutive slices (`_split_into_slices`) and rechecked slice by slice,
never as one oversized ask. 90s sits comfortably above the 20s probe
(real margin, not a bare minimum) while staying far below both measured
failure sizes — roughly 2x clear of the 205s chunk-internal decode loop
and 6x clear of the 562s regression — and divides that worst measured
hole into a manageable ~7 slices. Only the FIRST slice of an oversized
interval is eligible for the leading-edge `_PREFIX_DISTRUST_SECONDS`
widening below — an internal split point between two slices of the same
interval is an artificial chop point we introduced, not a genuine gap
edge Whisper actually drifted before, so widening there would just
re-transcribe extra already-good audio for no reason.

**Rechecks are budgeted and memoized, not merely counted per-gap.**
`_MAX_COVERAGE_RECHECKS` (12) bounds the number of Whisper calls per unit
of work (the single request, or one chunk) — ONE shared counter across
every slice of every suspicious interval in the unit, deliberately: it's
what stops one oversized hole's slicing from silently consuming the
entire budget and starving every other suspicious interval in the same
unit, since once an interval's own slices are all checked it stops
competing for budget. A slice already rechecked this call is never asked
again (a `checked` set keyed by rounded `(start, end)`) — the same
"don't ask a deterministic decoder the same question twice" discipline
`workers/translator.py` uses for its bisection retries. Together these
guarantee the loop terminates regardless of how many distinct suspicious
intervals — or how large any one of them is — a pathological transcript
produces. 12 comes directly from the slice size: the worst measured hole
(562s) needs `ceil(562 / 90) = 7` slices to fully re-cover; 12 leaves
headroom for a second hole in the same unit, or for a slice that only
partially resolves (a degenerate repeat needing a follow-up on its own
remainder) — while staying a small, fixed, auditable number.

**Why `collapse_repeated_segments` still collapses to the run's FIRST
segment, not its last** (a "more honest timeline" alternative was
considered and rejected): on the FIRST measured failure, Whisper's raw,
PRE-collapse segment timestamps kept climbing in step with real audio
time even while hallucinating — the repeated sentence's `end` reached the
chunk's true end regardless, so a check over the raw segments would have
seen full nominal coverage and missed the bug entirely. Collapsing to the
run's first occurrence (current behaviour, unchanged) is what
manufactures the interval this check reads; extending the collapsed
segment's `end` to the run's last occurrence would silently erase it. On
the SECOND failure this particular argument doesn't even apply — there
was no repeat-run to collapse, Whisper's own raw output already had the
gap — but the conclusion holds either way: check the collapsed segments,
every suspicious interval, not just the worst one.

When a suspicious interval is confirmed as speech, `_ensure_coverage`
splices in ONLY that interval's own recheck (never the whole unit, and
never "everything after some point"). Each recheck re-cuts the audio
`_RETRY_BACKOFF_SECONDS` (5s) past BOTH edges of the interval, rather than
resending identical bytes: a deterministic decoder asked the exact same
question twice has no reason to answer differently, and a cut boundary
landing mid-phrase is a known trigger for this kind of failure, so
shifting both edges is the one lever actually likely to change the
outcome. Which CONFIRMED segments survive the splice is keyed on the
interval's own boundaries, not the backoff-extended cut window — a
segment that ends at/before the interval's own start or begins at/after
its own end survives whole; splitting on the cut window instead would
drop an entire long confirmed segment (possibly minutes) just because
its last few seconds fall inside the backoff margin.

**The retry's OWN output also has to be clipped to those same gap
boundaries before splicing** — `_clip_to_gap`, added after a live re-run
of the fix above surfaced the seam it missed: the backoff context is
there purely for the decoder, but an unclipped retry re-transcribes that
few-second margin too, so it comes back TWICE (once from the confirmed
neighbor, once from the retry) — measured live as two duplicated long
lines at the splice seam. Worse, a segment the retry places inside that
margin can start BEFORE an already-confirmed segment, which breaks the
non-decreasing-`start` order every downstream consumer assumes:
`build_marked_text`'s timecodes go backward, and `workers/translator.py`'s
`_align_translation` — which walks the output with a forward-only cursor
matching markers by value — desyncs on a marker that appears earlier than
where its cursor already is, producing a spurious `partial` translation
that has nothing to do with the model. `_clip_to_gap` drops anything the
retry produced entirely inside the backoff margin and clamps anything
straddling an edge to it, then sorts what's left by `start` (the backend
is generally well-ordered, but nothing guarantees it, and letting a
single out-of-order pair through would reintroduce the same defect).
`_ensure_coverage` asserts the final list is non-decreasing by `start` on
every return path as a cheap backstop — by construction it always holds
given a well-ordered input, so a failure means a bug in this splice logic,
not bad transcript data.

**The prefix immediately before a gap can't be trusted just because it has
no gap of its own** — `_PREFIX_DISTRUST_SECONDS` (30s), added after a
second live re-run of the fix above surfaced a seamless duplicate: the
same dialogue appeared twice, once from the original transcription and
once from the retry, with no gap and no overlap-with-a-confirmed-neighbor
to catch it. The tell was in the ORIGINAL segments' timing: the five
segments right before the gap were each marked EXACTLY 1.000s long — real
Whisper output is never that round — while a re-cut of that same span
produced the same dialogue with normal, live-sounding durations instead.
Whisper doesn't only drop content once it falls into a hallucination
loop; it can drift for a while BEFORE the loop starts, misattributing
real speech to earlier, wrong timestamps as it loses sync. So the retry
target's leading edge is `gap_start - _PREFIX_DISTRUST_SECONDS`, not
`gap_start` itself — deliberately asymmetric with the trailing edge's
`_RETRY_BACKOFF_SECONDS` (5s), since nothing measured shows the same
drift happening on the way OUT of a gap. 30s is 3x the ~10s over which
the drift was observably visible via those suspiciously-round durations,
because that 10s is just where the symptom happened to become visible,
not necessarily where the drift actually began — a false positive here
only costs a bounded extra retranscription of already-fine audio, never
lost content (the retry's cut still covers that same audio, just
re-reads it), so the margin errs generous. Both `_clip_to_gap` and the
prefix/suffix split above key off this same widened boundary, not the
gap's own `gap_start` — everything else about the splice (bounded
retries, monotonicity assertion, cost of a false positive) is unchanged.

If rechecks don't resolve every suspicious interval as either spliced-in
speech or confirmed non-speech, the job still completes normally (no
special status, restart-safety/soft-pause untouched) but
`workers/runner.py` persists the summed length of everything still open
above the cost cutoff on `Job.transcript_missing_seconds` (migration v8)
and logs it — mirrored to the API as `JobSummary.transcript_missing_seconds`
/ `extension/src/lib/api-types.js`. `None` means either "not a Whisper
job" or "checked, nothing suspicious survived above the cost cutoff"; a
positive value means this "done" job is known to be missing that much
SPEECH somewhere (never confirmed music/noise/silence — see
`_is_confirmed_silence` above; a degenerate repeated run that never got
resolved counts here too, deliberately, since it's still unknown, not
confirmed silent), so the Library/Transcript tab can say so instead of
looking exactly like a complete one.

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

## No request may exceed the model's context — checked, not assumed

`llm/summary.py` used to reason about size in two places and trust the
answer in neither. Both cost a real job (a 4770-line YouTube transcript)
its summary: one request of 72196 tokens went out against a 32768-token
context, three retries, three identical failures.

1. **`split_for_summary` now guarantees its budget.** It previously
   *aimed* for `target_tokens` with no floor under the aim, and on a
   marked transcript it returned the whole 65k-token input as ONE chunk.
   Same root cause the translator hit (see below): `build_marked_text`
   emits one line per sentence with `[MM:SS]` at the start and no blank
   lines, so `_split_into_paragraphs` sees a single giant paragraph, and
   `_SENTENCE_RE`'s lookahead — which excludes `[` precisely so a split
   never orphans a marker — finds zero breakpoints because the character
   after every `.` IS `[`. `_segments_for` now runs a final waterfall over
   any still-oversized segment: by LINE via `pack_lines` (line-atomic, a
   marker can't be torn off), then by WORD for a single line that alone
   busts the budget. The only chunk that may still exceed `target_tokens`
   is one unsplittable "word".
2. **`stream_summarize` checks sizes instead of inferring them.** A lone
   chunk is not proof it fits — single-pass is taken only when
   `count_tokens(chunk) < threshold`. The map-phase chunk budget is
   derived as `min(4000, single_pass_token_limit - 1)`, so an operator
   configuring a threshold below the chunk target can't produce chunks
   too big to send either.
3. **The reduce phase has a budget too.** Correctly bounded chunks still
   produce N partials whose concatenation is unbounded in N (17 on the
   job above; longer videos, proportionally more). `_fold_partials`
   measures the join and, if it doesn't fit, groups partials with
   `pack_lines` and folds each group through an intermediate reduce
   (reusing `prompts/summary_reduce.txt`), capped at 6 rounds.

`config.llm.context_length` is a claim, not a measurement — the daemon
believes it and builds prompts to fit. Declaring more than the backend
actually serves doesn't create room, it just moves the failure into the
backend. It also feeds `qa._select_context`, which uses
`context_length - 4000` to decide whether to hand QA the full transcript
instead of the summary.

## `POST /config/test` (target="llm"): backend settings are probed, not hand-set

`context_length`, `single_pass_token_limit`, and `reasoning_effort` used to
be pure guesswork for the person editing `tldr.yaml`/the options page — get
`context_length` wrong and the failure above is what you get; leave
`reasoning_effort` unset on a thinking backend and the summary/translation
silently degrades instead of erroring (measured: the same model translating
139/139 lines with `reasoning_effort: "none"` vs 25/139 without it). The
options page now has a "Test setup" button (`daemon/src/api/config.py`'s
`_test_llm`) that determines all three by actually exercising the CANDIDATE
backend (the fields the user just typed, not necessarily saved yet — see
`ConfigTestLLMOverrides`), never by declaration:

1. **reachable** / **models** — `GET {base_url}/models` split into two
   steps: got an HTTP response at all vs. the model list itself (200).
2. **completion** — a real chat completion against the target model,
   through `llm_client.call_with_dialect_adaptation` exactly like the real
   call path (same 400-adaptation logic), just against a throwaway
   client/model/dialect instead of the cached prod ones. Trivial prompt
   ("reply with the single word ok") — this step exists only to prove the
   model answers at all, nothing more; see why thinking detection
   deliberately does NOT reuse this call, next.
3. **thinking** + **translation**, from ONE real call
   (`_probe_translation_and_thinking`) — thinking detection does NOT get
   its own trivial-prompt call. It used to, and that was a measured false
   negative: Gemma 4's thinking is ADAPTIVE, triggered by a prompt with
   actual rules to reason about, not by "reply with one word." Live
   measurement on the same model: 0 reasoning chars on the trivial prompt,
   1789 chars of `reasoning` plus truncated content (3 of 4 lines) on the
   REAL translation prompt with `reasoning_effort` unset — a detector
   watching only the trivial call would report "no thinking" and be wrong
   on exactly the model this whole feature was built for. So thinking
   detection instead inspects the response of the translation probe below
   (non-empty `reasoning`/`reasoning_content`, or empty `content`), which
   already sends a rule-heavy, contract-shaped prompt. Detected → retried
   once with `reasoning_effort: "none"`; suggested, and carried forward as
   the dialect for every later call (translation retries, context), ONLY
   if that retry actually comes back with real content — never
   speculatively. The translation half verifies a fixed 4-line probe
   through the REAL `prompts/transcript_translate.txt` prompt, checked with
   the production translator's own `workers.translator._align_translation`
   (which embeds `_group_is_echo`) — reused unmodified, not re-derived, so
   "the test passed" means what it says. See "Transcript translation"
   below for what that verification actually checks, including its one
   documented gap (echo at single-line bisection granularity — the probe
   inherits it rather than pretending to be stricter than production).
4. **context** — runs LAST, deliberately, on its OWN dedicated time budget
   (`_CONTEXT_PROBE_OWN_BUDGET_SECONDS`) separate from the overall
   deadline. This ordering is a fix, not the original design: on a live
   run against gemma-4-e4b on LocalAI, the context probe (then step 2 of
   3, before translation) consumed the ENTIRE 90s overall budget — 4
   attempts at ~20s each — and the translation step that ran after it
   never executed, leaving the one decisive "will my model actually work"
   answer unattempted while the slow, diagnostic step took everything.
   Context can now blow its own budget without touching anything else.
   One deliberately oversized request (`_CONTEXT_PROBE_HUGE_TOKENS`, built
   via `llm.tokens.make_filler_text`); on a rejection, the real ceiling is
   parsed straight out of the backend's own error text (both observed
   phrasings put the ceiling as the SMALLER of two "<N> tokens" mentions).
   Falls back to bisection, bounded to `_CONTEXT_PROBE_MAX_ATTEMPTS` (6)
   calls and its own budget, only when parsing fails. `single_pass_token_limit`
   is suggested as ~60% of whatever `context_length` comes out to — the
   same ratio every backend example in tldr.yaml.example already documents
   in a comment.

   **A failure is not automatically "over context" — and neither is the
   HTTP status code that carries it.** The same live run reported
   "approximately 35250 tokens" for a model whose LocalAI `context_size`
   was actually 131072 — the bisection was treating ANY failed attempt
   (including a plain timeout on a huge, slow-to-prefill request) as proof
   of hitting the wall. It isn't: a slow CPU-bound backend just taking a
   long time looks identical to "rejected" if you only check "did it
   raise." Each probe attempt now classifies by MESSAGE CONTENT
   (`_looks_like_context_overflow` — mentions `context`/`tokens` together
   with an overflow word: exceeds/too many/maximum/limit) into `"ok"` /
   `"rejected"` / `"inconclusive"` (nothing recognizable — a timeout, a
   dropped connection, an unrelated 5xx). Only `"rejected"` ever narrows
   the bisection window; `"inconclusive"` stops it outright.

   Content, not status code, because status code turned out to be
   gateway-dependent: a first fix keyed "rejected" off `status_code == 400`
   specifically — and broke the ONE backend (qwen3-vl-8b-instruct on
   LocalAI) where the signal had been perfect before. LocalAI relayed the
   exact same kind of error — `rpc error: code = Internal desc = request
   (40008 tokens) exceeds the available context size (32768 tokens)` — as
   an HTTP **500**, not 400. The number was sitting right there in the
   text; the status-code classifier threw it away as "inconclusive" purely
   because of the wrapping code. Message content is where the real
   evidence lives; status code is not a reliable proxy for it.

   If even the initial huge probe is inconclusive, the whole step reports
   `ok: null` with an explanation instead of a number — the project has
   already paid for a wrong-but-declared `context_length` silently
   corrupting a real job (see the section above); a fabricated
   *discovered* one would be the same mistake with better PR. That
   explanation must itself be legible: `asyncio.wait_for`'s own client-side
   `TimeoutError()` carries NO message at all (`str(TimeoutError())` is
   `""` by construction, not a bug losing content along the way) —
   `_describe_probe_failure` substitutes a readable fallback sentence
   rather than reporting a blank after a colon.

Every step's report is returned even when an earlier one failed — later
steps read `ok: null` ("not attempted"), not a truncated list — so a
partial failure is legible instead of silently swallowed. The whole probe
never writes to config; the options page's "Apply" button does that
afterward, via an ordinary `PATCH /config`.

**Per-call `reasoning_effort` override, added for step 3 above:**
`llm.client.call_with_dialect_adaptation` already accepted a throwaway
`client`/`model`/`dialect` for exactly this kind of one-off probe (see its
docstring) — everything except `reasoning_effort`, which `_extra_body` used
to read straight from `get_config().llm.reasoning_effort` on every call.
That field now lives on `_Dialect` itself (populated from config by
`_new_dialect()`, exactly as before, for every REAL call path); a caller
that builds its own `_Dialect` — only the config-test probe does — can pin
a candidate value for one call without writing anything to saved config
first, and without touching the semaphore (still gated by the same rule
as any other cached-client bypass: only acceptable for a probe against a
possibly-different, not-yet-saved backend, never as a shortcut for real
work).

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
7. **Structural verification alone can't tell a translation from a
   copy — closed with a GROUP-level echo check.** Everything above
   (markers match, no repetition loop, no degenerate run) is satisfied by
   a model that just echoes its input back. Measured live on job
   `cWiAdufn-6j8` (139-line English transcript → Russian): `qwen3-1.7b`
   returned 139/139 lines byte-identical to the source and the old check
   accepted it — status `done`, `error=null`, user sees "translated" text
   that never was. `gemma-4-e4b-it-qat-q4_0` did it partially: 45 lines
   identical, only 12 honestly flagged, the other 33 slipped through.
   `_group_is_echo` compares each line's text AFTER the marker (never the
   marker itself — that's contractually required to match, not evidence)
   against the source, `.strip()`-normalized, and flags the group ONLY
   when the WHOLE group matches — never per-line, since a single line
   legitimately matching the source (a number, a proper noun, "OK",
   "2024") is unremarkable and must not trip this. An echo verdict is fed
   back through `_align_translation` as `None`, i.e. it is not a new
   failure mode — it goes down the exact same
   retry/bisection/leaf-fallback path as a missing marker or a
   degenerate run. Two guards on top of the whole-group rule: (1) groups
   of size 1 are never flagged — once bisection has narrowed a mismatch
   down to a single line, that line matching the source is the NORMAL
   case, and flagging it would stop bisection from ever converging; (2)
   skipped when the job's `transcript_language` is known and equals the
   target — a same-language "translation" is supposed to come back
   identical (though `enqueue_translation` already short-circuits that
   case before any LLM call happens). If `transcript_language` is
   unknown (`None`), the check runs anyway rather than guessing via an
   alphabet heuristic — `source_lang` is threaded through explicitly from
   `Job.transcript_language` down through every bisection/retry call
   rather than inferred.

   Known residual gap, and it's a direct consequence of the size-1
   exemption above, not an oversight: a model that echoes UNCONDITIONALLY
   regardless of how small the ask is will, once bisection reaches
   single-line granularity, have those lines accepted — matching a lone
   line is indistinguishable from the legitimate case. This only bites
   when bisection depth/budget is enough to actually reach size-1 on the
   affected span; if it isn't (depth or the per-job call budget runs out
   first), the whole remaining unresolved span still falls back honestly.
   A mid-batch echo embedded inside an otherwise-successfully-verified
   larger call (some lines echo, most translate, and the call as a whole
   verifies fine) is also not caught by design — rule is whole-group-only,
   deliberately, to avoid false positives on legitimate short matches.
8. **Pause-aware**: between groups (`_checkpoint_pause_translation`) and
   inside `stream_complete(respect_pause=True)`. Same pause flag as the
   summary path.
9. **Restart-safe**: rows left in `running` at daemon startup get
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
