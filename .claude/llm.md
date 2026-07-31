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
calls `complete_with_messages` — no special LLM API path.

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

## Transcript translation

`workers/translator.py` translates `Job.raw_text` into a target language,
caching the result in `transcript_translation` (PK `job_id+language_code`).
Triggered by `POST /jobs/{id}/transcript/translate {lang}`; dedup is at
the row level — a second POST for an in-flight `(job_id, lang)` returns
the existing status without spawning a duplicate worker.

Three invariants:

1. **Chunked**: long transcripts go through `llm.chunking.split_for_summary`
   so each LLM call fits the context. The translation prompt
   (`prompts/transcript_translate.txt`) instructs Gemma to keep
   `[MM:SS]` markers verbatim per line — without that the
   transcript-tab's binary-search highlight breaks.
2. **Pause-aware**: between chunks (`_checkpoint_pause_translation`) and
   inside `stream_complete(respect_pause=True)`. Same pause flag as the
   summary path.
3. **Restart-safe**: rows left in `running` at daemon startup get
   re-enqueued by `re_enqueue_running_on_startup` (called from
   `main.lifespan`). raw_text is in the DB and the language code is on
   the row — nothing external is needed. Restart-continued translations
   start from chunk 0 (no partial-chunk checkpointing); the user "loses"
   any progress percent shown before the restart but the result is
   correct.

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
