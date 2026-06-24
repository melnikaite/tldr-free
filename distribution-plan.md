# Distribution plan v2 — from 0 stars to findable

> **Status:** ACTIVE. Monetization explicitly out of scope (owner decision 2026-06-09).
> v2 (2026-06-10) reworks v1 after the backend-landscape research: one-command
> installer becomes the core of Tier A, LocalAI becomes the recommended
> non-Apple backend, Docker is demoted to "only where it already exists".
>
> Positioning agreed: lead with what competitors don't have — the media
> pipeline (any audio/video → transcript → timecoded summary), transcripts
> with translation, and the persistent local library.

## Principles

1. **Detect-existing-first** (the ".NET kit" pattern): never install what the
   user already has. Probe known backends, reuse, only then install.
2. **One command end-to-end**: runtime → models → autostart → verify. Every
   step idempotent — re-running the installer is a safe no-op/repair.
3. **Docker only where it already exists** (Windows dockerists, self-hosted
   catalogs). Mac/Linux first-run path has no Docker in it.
4. A launch (Show HN) is a one-shot — phases before it exist to not waste it.

## Key facts the plan rests on (verified 2026-06-10)

- **No consumer LLM app does STT**: Ollama, LM Studio, Jan, GPT4All, Foundry
  Local — none expose `/v1/audio/transcriptions`. LocalAI does.
- **LocalAI** (46.8k★): one server = LLM + Whisper + vision. Transcription
  schema confirmed in source (`core/schema/transcription.go`): verbose_json
  with float-seconds segments + language — exactly what `runner.py` eats.
  Gallery API installs models programmatically. Since 3.5.0: MLX backends
  (Mac), Purego whisper + VAD. **Drop-in Ollama API emulation since 4.2.0.**
- **LocalAI install story (current)**: Docker (recommended), macOS DMG,
  Linux binary, `brew install localai` (4.3.6, homebrew-core, **no service
  block** — `brew services` won't work). The old `curl install.sh` with
  systemd registration is **gone** (404; file removed from master). Launcher
  app (alpha, Mac/Linux) is unsigned on macOS. **No native Windows path**
  (Docker/WSL only). ⇒ Autostart of LocalAI is OUR job on every platform.
- **Ollama** self-registers at login today, but a Feb 2026 proposal disables
  that for new users — onboarding must probe, not assume.
- **Pinokio** (7.5k★): MIT but 1 contributor, zero external PRs merged ever,
  no OS-level autostart. Channel (recipe), not foundation. Forks: best has 2★.
- **llmfit** (27.7k★): our PR #603 (Whisper/ASR entries) **merged 2026-06-10**.
  AudioFit phase unblocked. PRs from us demonstrably welcome.
- **Win11**: no Python out of the box (Store stub only); winget IS preinstalled.
  uv bootstraps Python itself: `winget install astral-sh.uv && uvx tldr-daemon`.
- Benchmark: Page Assist 8k★ (one-click CWS install), Lumos 1.5k★ (dead since
  2025 — stars without distribution+maintenance die too).

---

## Phase 0 — repo hygiene (now, hours)

- [x] **0.1 README repositioning** — media-first tagline, "Why TLDR", badges,
  Roadmap + Contributing. *Done 2026-06-09.*
- [ ] **0.2 Fix `licenseInfo: null`** — LICENSE exists only on
  `qa-search-and-fixes`; GitHub scans the default branch. Merge to `main`.
  DoD: `gh api repos/melnikaite/tldr-free/license` returns MIT.
- [ ] **0.3 Repo metadata** — topics (`gh repo edit` one-liner: chrome-extension,
  local-llm, ollama, localai, whisper, yt-dlp, summarization, youtube, privacy,
  self-hosted, fastapi), description, social preview 1280×640.
- [ ] **0.4 CONTRIBUTING.md + issue/PR templates** — link to CLAUDE.md map,
  `task test`, reload matrix. Light.
- [ ] **0.5 Seed 5–8 good-first-issues** from the roadmap with file pointers:
  backend probe module, Obsidian export, library full-text search, OCR for
  scans, Firefox research, ps1 installer.

## Phase 1 — visible quality: CI + releases (days)

- [ ] **1.1 GitHub Actions**: ruff + mypy + pytest directly on ubuntu-latest
  (no Docker for the test job) + a separate image-build job. Badge in README.
- [ ] **1.2 GHCR image** `ghcr.io/melnikaite/tldr-daemon` on tag.
- [ ] **1.3 Release v0.1.0**: changelog + `extension/` zip asset (unpacked
  install without cloning until CWS is live).

## Phase 2 — Tier A core: the one-command install (the heart of the plan)

> Goal UX: `curl -fsSL https://…/install.sh | sh` → working TLDR after one
> command, both fresh machines and machines that already run an LLM app.

- [x] **2.0 SPIKE — DONE 2026-06-12 (LocalAI 4.4.2, macOS arm64). Verdict: GO
  for the Whisper role everywhere + LLM role on Linux/Windows; NOT yet for
  the LLM role on Apple Silicon.** Findings:
  - ✅ `/v1/audio/transcriptions` verbose_json: float-second segments, fast
    (whisper-large-v3-q5: 9.5s audio ≈ 4.4s warm). Model id: `whisper-large-q5_0`.
  - ✅ Gallery API (`POST /models/apply` + job polling) is the right wizard
    integration point. The `local-ai models install` CLI is NOT — it uses
    CWD-relative paths and ignores the running server.
  - ✅ llama-cpp backend (`gemma-4-e4b-it-qat-q4_0`): correct output, vision
    via mmproj; ~20 tok/s on M-series (vs 45 tok/s native mlx — 2.3×; on an
    8k-token prompt 39s vs 12s end-to-end). Fine where llama.cpp is the only
    option (Linux/Win). CAVEAT (measured in the sibling voiceassistant
    project, 6-run samples): Gemma-4 thinking under the gallery GGUF jinja
    template is ADAPTIVE — fires nondeterministically, eats small max_tokens
    budgets. `reasoning_effort: "none"` fully suppresses it (0/6, faster
    median); "low" does not (1/6); `enable_thinking: false` unreliable (1/6).
    Wizard must set "none" when configuring LocalAI/llama.cpp + Gemma.
    Responses include `"reasoning": null` — avoid strict schema validation.
  - ❌ mlx backend + Gemma-4 thinking: leaks `<|channel|>` tokens into
    content / returns empty answers — immature; re-test on future releases.
  - ⚠️ Upstream bugs found (contribution candidates): (1) backend `run.sh`
    breaks when backends-path contains spaces (unquoted `$(realpath $0)`) —
    bites macOS `Application Support` paths; (2) `language` is always null
    in transcription responses (runner.py degrades gracefully); (3) stray
    files (CACHEDIR.TAG) listed as models in /v1/models.
  - Wizard note: a LocalAI instance may be shared with other local apps
    (here: a voice assistant's launchd unit on :1240) — detect-existing-first
    applies to LocalAI itself; never assume we own the process.
- [ ] **2.1 PyPI package** — `tldr-daemon` entrypoint via `uv tool install`;
  port the entrypoint's yt-dlp auto-upgrade into daemon startup (uv mode must
  keep the "Google broke YouTube" self-heal).
- [ ] **2.2 Backend wizard in the daemon** — on start without a valid backend:
  probe 11434 (Ollama or LocalAI-emulated), 1234 (LM Studio), 1337 (Jan),
  4891 (GPT4All), 8080 (LocalAI), 1237/18000 (mlx); `GET /v1/models` each.
  Expose `GET /setup/candidates` + `POST /setup/apply` so the side panel
  renders "Found LM Studio with qwen3-4b — use it?". Config write + hot
  re-init, no restart.
- [ ] **2.3 Model auto-install with progress** — simple built-in hardware
  heuristic (RAM/GPU → model tier) → LocalAI gallery API / `ollama pull` →
  progress streamed through our existing broker/SSE into the side panel.
  When configuring LocalAI/llama.cpp + Gemma, the wizard MUST set
  `reasoning_effort: "none"` (see spike findings).
- [ ] **2.4 `tldr-daemon service install [--with-localai]`** — writes and loads
  launchd plists (macOS) / systemd --user units (Linux) / Task Scheduler
  entry (Windows) for the daemon AND, when we installed it, LocalAI.
  KeepAlive/Restart=on-failure → crash supervision like docker's
  `restart: unless-stopped` gives us today.
  Security note (uv mode runs as the user, no container boundary): generate
  Linux units with hardening flags (`NoNewPrivileges`, `ProtectSystem=strict`,
  `ProtectHome=read-only`, `ReadWritePaths=<data dir>`, `PrivateTmp`) — free
  half-sandbox around ffmpeg/pdf/yt-dlp parsing of untrusted content. Ship
  uv.lock with hashes; keep the Docker path documented as the most-isolated
  option. macOS launchd has no equivalent — risk accepted there.
- [ ] **2.5 install.sh orchestrator** — detect OS/arch/RAM/GPU → probe (2.2
  logic, CLI mode) → install backend if none: macOS `brew install localai`,
  Linux binary from GitHub releases → ensure uv → `uv tool install
  tldr-daemon` → models (2.3) → services (2.4) → verify `/health` + micro
  transcription → print extension link. Idempotent. Hosted at repo raw URL
  (custom domain optional, later).
- [ ] **2.6 docker-compose profile `localai`** — one extra service block;
  `COMPOSE_PROFILES=localai task up` = daemon + LocalAI wired
  (`http://localai:8080/v1` for both llm and whisper). Covers Windows
  dockerists (Docker Desktop = WSL2 under the hood) and self-hosted. CPU
  image default, GPU variant documented (their Win11 GPU-in-docker issues
  noted in #4331).
- [ ] **2.7 README install matrix** — first screen, 90-second path per
  platform: Mac → one command (brew localai or mlx) · Linux → one command ·
  Windows → Ollama/Foundry (LLM-only; subtitles cover most YouTube; Whisper
  arrives with Tier B) or compose profile · self-hosted → compose/GHCR.

## Phase 3 — Chrome Web Store (review runs in parallel with Phase 2)

- [ ] **3.1 DECIDE NAME FIRST** *(owner; blocks listing + PyPI name)* —
  `tldr-free` is unfindable under tldr-pages/tldr.tech/TLDR This.
- [ ] **3.2 Permission audit** — `<all_urls>` + `cookies` triggers slow review.
  Try activeTab+scripting for click-driven extraction; make `cookies`
  optional/at-use (`permissions.request`); tighten `web_accessible_resources`.
  If broad perms must stay — write the justification up front.
- [ ] **3.3 Privacy policy** — one honest page on GitHub Pages: everything
  local, extension talks only to 127.0.0.1, no analytics.
- [ ] **3.4 Assets** — 5 screenshots 1280×800 (summary+timecodes, transcript
  +translation, library, Q&A, wizard), small tile, description from README
  "Why". $5 dev account.
- [ ] **3.5 Submit + version-bump discipline** (CWS auto-update replaces
  "click reload"). Expect 1–3 weeks.
- [ ] **3.6 (second wave) Firefox port** — sidebar_action; privacy audience
  lives there. Good-first-project, not a launch blocker.

## Phase 4 — demo assets (one afternoon)

- [ ] **4.1 The 30-second GIF** — terminal: one command runs → browser:
  YouTube lecture → toolbar click → summary streams → click `[12:34]` →
  player seeks → Q&A answer cites a timecode. Replaces the TODO comment in
  README. Highest-leverage single asset.
- [ ] **4.2 Screenshots** (double as 3.4). **4.3 Social preview** (0.3).

## Phase 5 — launch (gate: one-command works on Mac+Linux, CI green, GIF live)

- [ ] **5.1 Show HN** Tue–Thu 8–10am ET: "Show HN: Local summaries with
  clickable timecodes for any video/audio — one command, your own models".
  First comment: architecture, honest limitations, why local.
- [ ] **5.2 Community posts ~1 week apart, different write-ups**:
  r/LocalLLaMA (works with whatever you run — Ollama/LM Studio/LocalAI/mlx),
  r/selfhosted (compose profile, Umbrel/CasaOS candidacy), lobste.rs.
- [ ] **5.3 Pinokio recipe** → pinokiofactory (community-accepted channel).
  Recipe CAN bundle llama.cpp+model fully; honest caveat: no OS autostart,
  lifecycle tied to Pinokio. Reaches local-AI hobbyists.
- [ ] **5.4 Umbrel / CasaOS app-store submissions** — reuse compose profile.
- [ ] **5.5 Prepared FAQ** — why not Chrome built-in AI (context, media,
  library), why a daemon, Windows-Whisper status, Firefox when.
- [ ] **5.6 Post-launch SLA** — every issue answered <24h for a month.

## Cut from the plan (owner decisions, 2026-06-12)

- **Tier B consumer tray app — CANCELLED.** The uv path is the chosen
  distribution; the no-terminal audience is better served by LocalAI's own
  consumer trajectory (Launcher, macOS DMG, eventual native Windows build)
  than by our signed Electron/Tauri shell (+$99/yr Apple, +$10/mo Windows,
  updater, CI matrix). Honest consequence: the "Whisper on Windows without
  Docker" gap now waits on upstream LocalAI, not on us. Re-open only if
  real non-technical demand shows up in issues.
- **llmfit track — REMOVED from TLDR's roadmap.** After the full switch to
  LocalAI, model installation is the gallery's job; "what fits this hardware"
  for our narrow case is a ~10-line built-in RAM heuristic in the wizard
  (2.3), not a dependency. AudioFit/llmfit contributions remain the owner's
  personal upstream track, decoupled from TLDR distribution.

## Owner decisions needed

| # | Decision | Blocks |
|---|---|---|
| 1 | Project name (keep `tldr-free` vs rename) | 3.x, 2.1 (PyPI), 5.1 |
| 2 | PyPI package name | 2.1 |
| 3 | GitHub Pages for privacy policy — ok? | 3.3 |
| 4 | Custom domain for `curl …/install.sh` one-liner (optional; raw URL works) | 2.5 polish |
