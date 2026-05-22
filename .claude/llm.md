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

The extension's `markdown.js` post-processes those markers into clickable
YouTube `?t=Ns` links (DOM-walk, skipping text inside `<a>`, `<code>`,
`<pre>`).
