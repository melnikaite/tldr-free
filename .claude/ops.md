# Operations — dev loop, reloads, tests

## Migrations: edit v1 in place during active dev

We're still pre-1.0. Schema changes go into the existing v1 migration —
the user is expected to `task reset` (wipes the SQLite volume) when the
schema changes incompatibly. Switch to additive migrations once real users
have data they care about.

## Daemon hot reload

`uvicorn` runs without `--reload` (the Whisper worker would orphan jobs on
every reload). Code changes need `docker compose restart daemon` (or
`task down && task up`). Tests don't — `tests/` is volume-mounted and
pytest re-collects on each invocation.

`pyproject.toml` changes need `task install` (or `docker compose build daemon`).

## YouTube libs auto-upgrade on container start

`daemon/docker-entrypoint.sh` runs `pip install --upgrade yt-dlp
youtube-transcript-api` whenever the daemon starts via uvicorn. Skipped
for `task test` and other ad-hoc commands so they stay fast. If pip can't
reach the network, the entrypoint falls back to whatever the image bundled.

This is the answer to "Google broke YouTube again" — `task down && task up`,
restart pulls the latest fix, no rebuild.

## Native (uv) mode invariants

The daemon also installs as a uv tool (`task install:uv` →
`scripts/install-uv.sh`; entrypoint `tldr-daemon` in `daemon/src/cli.py`).
Invariants the native path must keep:

- **Docker behavior unchanged.** Container detection is structural: if
  `/app/config/tldr.yaml` / `/data` exist (compose mounts), they win.
  Platform paths (`daemon/src/paths.py`) only apply when they don't.
- **Config auto-create only at the default path.** An explicit `TLDR_CONFIG`
  pointing at a missing file still raises; the packaged template
  (`daemon/src/assets/tldr.yaml.example`, kept byte-identical to
  `config/tldr.yaml.example` — edit one, copy to the other) is only used
  for the platform default path, with `host.docker.internal` rewritten to
  `127.0.0.1`. Either way `tldr.yaml` is written with `0600` permissions,
  since it may hold a plaintext cloud API key (`llm.api_key`). Prefer
  `api_key_keychain` (recommended default when available) or `api_key_file`
  (Docker installs) for real cloud keys — see [llm.md](llm.md).
- **One upgrade per start.** `src/selfupdate.py` refreshes yt-dlp +
  youtube-transcript-api at CLI startup; `docker-entrypoint.sh` exports
  `TLDR_SKIP_PKG_UPDATE=1` because it already upgrades itself. Pytest is
  always skipped.
- **Logging is Docker-vs-native, structurally, same as above.** See
  `src/logging_setup.py` / [runbook.md](runbook.md#logs--diagnostics): the
  rotating file handler is always added, but only in native mode does it
  ALSO strip the stdout/stderr handlers `logging.basicConfig`/uvicorn
  installed. Getting this backwards would either leave launchd's
  `daemon.{out,err}.log` growing unbounded again (native) or silence
  `docker compose logs` (container) — both regressions this module exists
  to prevent. Detection reuses `paths.CONTAINER_DATA.is_dir()`, not a new
  heuristic.
- **Service registration shells out via `service._run`.** Tests monkeypatch
  it and assert on generated plist/systemd content — never run
  launchctl/systemctl in tests. systemd unit keeps the hardening block
  (NoNewPrivileges, ProtectSystem=strict, ProtectHome=read-only,
  ReadWritePaths=<data dir>, PrivateTmp).

## Logging

App logging AND uvicorn's own access/error logging are funneled into one
`RotatingFileHandler` at `<data_dir>/logs/daemon.log` (`src/logging_setup.py`,
constants `LOG_MAX_BYTES`/`LOG_BACKUP_COUNT`) — see the invariant above for
the Docker/native split, and [runbook.md](runbook.md#logs--diagnostics) for
the user-facing story (where the file lives). `configure_logging()` runs
from `src.main`'s FastAPI `lifespan` and is safe to call more than once per
process — it replaces its own previously-installed handler rather than
stacking a new one — because `with TestClient(app)` re-runs the lifespan
(and therefore this function) once per test file that uses it; see
`tests/test_logging_setup.py`'s idempotency test.

**The one-time legacy-log truncation is NOT in the lifespan.**
`truncate_legacy_launchd_logs_once()` (same module) is called ONLY from
`cli.py`'s `_serve()` — the real `tldr-daemon` native entrypoint — never
from `src.main.lifespan`. This is a deliberate, load-bearing separation, not
a style choice: a lifespan hook runs every time *anything* constructs the
app, including `with TestClient(app)` in the test suite, and the test suite
is RIGHT to exercise it. Reaching outside the app's own data at that layer
— truncating whatever `daemon.{out,err}.log` happens to sit at
`storage.data_dir` — is a "real native daemon starting on this machine for
the first time" operation, not a "an ASGI app object exists" one. Getting
this wrong once already had a real consequence: an earlier version DID call
truncate() from the lifespan, and running `pytest` on a real dev machine
truncated that developer's actual, in-use
`~/Library/Application Support/tldr/data/daemon.{out,err}.log` (555 KB /
8.1 MB of live diagnostic history, gone) — because most test files never
override `storage.data_dir`, so it resolved through
`StorageConfig._resolve_data_dir()`'s native fallback to the real platform
data directory. Two things now guard against this regressing, at different
levels:

- `tests/conftest.py`'s `_isolate_native_data_dir` (autouse) patches the
  bare, no-argument call `paths.platform_data_dir()` — the exact form
  `_resolve_data_dir()` invokes — to a fresh pytest tmp directory for every
  test, so even a test that forgets to set `storage.data_dir` explicitly
  can never resolve to the real one. Calls WITH explicit
  platform/home/env arguments (`tests/test_native_install.py`'s own unit
  tests of the cross-platform logic) pass through untouched.
- `tests/test_main_lifespan.py` pins down the behavioral guarantee
  directly: it seeds fake `daemon.{out,err}.log` files, runs the full
  lifespan via `TestClient(app)`, and asserts they're byte-for-byte
  untouched and no truncation sentinel was written. `tests/test_cli_serve.py`
  covers the positive case — `_serve()` DOES call the truncation, mocking
  out `uvicorn.run` so nothing actually binds a port.

**`uvicorn.access` never gets to write a query string, in ANY
environment.** `configure_logging()` also installs a `logging.Filter`
(`_AccessLogDropQueryStringFilter`) on the `uvicorn.access` logger that
rewrites its request-target argument to drop everything from `?` onward,
mutating `record.args` before any handler formats it — this is
unconditional (not gated on Docker vs native), because it's a
data-minimization policy, not a rotation concern: `GET /jobs?url=<page>`
IS the page/video URL the user is looking at, so an access log with query
strings intact is a browsing-history log, not a diagnostic one, and no
diagnostic question here needs more than path + status code. This was a
real incident too, one level down from the launchd-truncation one above:
an early version of the rotating-log fix DID write query strings into
`logs/daemon.log` (and, worse, an already-percent-encoded query string can
carry arbitrary user content — not just a page address — e.g. a translated
message passed as a query parameter). See
`tests/test_logging_setup.py::test_configure_logging_drops_query_string_from_access_log`.

`GET /diagnostics` (`src/api/diagnostics.py`) reads the tail of this same
rotating log file and scrubs it (API keys, home directory, non-loopback
URLs) before returning it — see that module's docstring for the exact
redaction rules, and `tests/test_diagnostics.py` for the tests that pin
them down. This is the SECOND, independent layer, for logs that already
exist on disk without the filter above having ever run on them: `_redact_urls`
matches a URL whether it's literal (`https://...`) OR percent-encoded
(`https%3A%2F%2F...`, any case) — `unquote()`-ing the match before reading
its host is what makes one code path handle both, and anything that fails
to parse as a URL after decoding is treated as suspicious (redacted), never
passed through on the assumption it "probably wasn't a URL". Any new log
line added anywhere in the daemon should assume it MIGHT end up in a
user-pasted bug report and avoid putting a raw, non-loopback URL or a key
in the message — the scrubbing is a safety net, not a substitute for not
logging secrets in the first place (the project's existing rule — see the
module docstring of `src/api/config.py`), and the access-log filter above
is a safety net's safety net: preferred over relying on scrubbing alone
because data that's never written can't leak from a copied-and-pasted log
file either.

## Taskfile is a router, not a shell

`Taskfile.yml` keeps every `cmd:` block as a one-liner that delegates to a
script in `scripts/`. Task uses the `mvdan.cc/sh` interpreter which
mishandles `$!`, `$(cat ...)`, `kill -0`, and other bashisms — putting
that logic in real bash files avoids the trap entirely. Add new
lifecycle/install logic to a script, not inline in the YAML.

## Tests

`task test` runs ruff + mypy + pytest inside the daemon container. Always
run before declaring work done. External services (mlx, youtube-transcript-api,
yt-dlp, trafilatura, the Whisper worker, the retention worker) are mocked
at module-load time so the suite stays hermetic and fast (~5s for 120+ tests).

**No test may touch the real native data directory or a shared `/tmp`
path.** See the "Logging" section above for why this is a hard rule, not a
preference — it already caused real data loss once. `storage.data_dir` in
a test config should always be that test's own `tmp_path`; `tests/conftest.py`'s
autouse `_isolate_native_data_dir` is a safety net for the tests that don't,
not a substitute for setting it explicitly.

POST /jobs is async — tests that need the final state poll
`GET /jobs/{id}` until status transitions (see `_wait_until_done` in
`tests/test_api_jobs.py`). For repo / runner async tests, prefer the
condition-poll helper `_wait_until(predicate, timeout=...)` over a fixed
`for _ in range(N): await asyncio.sleep(...)` loop.

Concurrency / event tests live next to the contracts they protect:

- `tests/test_api_jobs_race.py` — parallel POST /jobs same URL must dedupe.
- `tests/test_repo_emit.py` — every write function publishes the right
  `job_event` to the global broker.
- `tests/workers/test_control.py` — pause/resume flips flag AND publishes.
- `tests/workers/test_broker.py` — fan-out, drop-on-full, unsubscribe,
  per-job → global mirror.

The extension has no test framework; ad-hoc `node --check` for syntax and
`node --eval` for logic that doesn't touch the chrome.* APIs.
