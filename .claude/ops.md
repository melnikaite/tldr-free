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
- **Service registration shells out via `service._run`.** Tests monkeypatch
  it and assert on generated plist/systemd content — never run
  launchctl/systemctl in tests. systemd unit keeps the hardening block
  (NoNewPrivileges, ProtectSystem=strict, ProtectHome=read-only,
  ReadWritePaths=<data dir>, PrivateTmp).

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
