# mlx-openai-server patches

We patch the user's local `mlx-openai-server` install to surface
**`segments` + `language`** from Whisper transcriptions (the underlying
`mlx_whisper.transcribe()` already returns them, but the server's handler
silently drops both fields).

Daemon needs both:
- `segments` → per-line `[MM:SS]` markers for the transcript-tab UI, plus
  source-language captions when injecting `<track>` into a `<video>`.
- `language` → autodetected ISO-639-1 code, persisted as
  `Job.transcript_language`, used as the source label in the language
  switcher.

## Tested against

- `mlx-openai-server==1.8.1` (the version `task install:mlx` currently
  installs). Different versions: the apply script checks installed
  version and refuses to run on unknown versions to avoid silent breakage.

## How to apply

Automatic — `task install:mlx` runs the apply step at the end. After every
`pip install --upgrade mlx-openai-server` you must re-apply:

```
bash scripts/mlx.sh patch
```

Manual:

```
python3 scripts/mlx-patches/apply.py --venv ~/.venvs/mlx-server
```

The script is idempotent — re-running on an already-patched file is a no-op.

## How it works

`apply.py` does anchor-string find/replace on:
- `app/schemas/openai.py` — adds `VERBOSE_JSON` to the response-format
  enum, adds a `TranscriptionSegment` model, extends `TranscriptionResponse`
  with optional `language` / `segments` / `duration`, extends
  `TranscriptionResponseStreamChoice` with the same.
- `app/handler/mlx_whisper.py` — wires `mlx_whisper.transcribe()`'s
  `segments` and `language` into the response when `response_format=
  verbose_json`. Stream path also surfaces per-chunk segments with
  `chunk_start` offset added so timings are absolute.

Anchor-string approach (not unified diff) so the patch survives minor
upstream whitespace/comment changes — only fails if the anchor string
disappears (which means the surrounding code restructured and we should
revisit the patch).

## Upstream

PR open: [cubist38/mlx-openai-server#310](https://github.com/cubist38/mlx-openai-server/pull/310)

The PR covers the same changes as `apply.py` but is a clean code edit
against the current upstream `main` (post-1.8.1), which also patches the
two IPC-proxy code paths (`transcribe_from_data`, `transcribe_stream_from_data`)
that were added after 1.8.1.

Once the PR is merged and a new PyPI release ships, `task install:mlx` can
`pip install mlx-openai-server` at the release version and skip `apply.py`
entirely. Until then the patch script remains the install path.
