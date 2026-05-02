#!/usr/bin/env bash
#
# MLX backend — install + lifecycle for mlx-openai-server (Apple Silicon).
#
# This is one of several OpenAI-compatible backends the daemon can talk to.
# Use it on macOS arm64 for the fastest local path; on other platforms point
# config/tldr.yaml at Ollama / LM Studio / llama.cpp / vLLM instead.
#
# Subcommands:
#   install               Install mlx-openai-server in ~/.venvs/mlx-server +
#                         download Qwen + Whisper weights (~10 GB)
#   start                 Start mlx-server in background, write PID to .mlx.pid
#   start-if-present      Start only if installed; no-op otherwise
#                         (used by `task up` so non-mlx backends "just work")
#   stop                  Stop mlx-server (SIGTERM, clean up PID file)
#   status                Print PID + /v1/models reachability
#
# Flags (for `install`):
#   --skip-models         Don't pre-download model weights (fetched on first request)
#   --yes | -y            Don't prompt; assume yes

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PID_FILE=".mlx.pid"
LOG_DIR="data/logs"
VENV="$HOME/.venvs/mlx-server"
BIN="$VENV/bin/mlx-openai-server"
CONFIG="config/mlx-server.yaml"
CONFIG_EXAMPLE="config/mlx-server.yaml.example"
HEALTH_URL="http://localhost:18000/v1/models"

if [ -t 1 ]; then
  C_OK="\033[32m"; C_SKIP="\033[2m"; C_WARN="\033[33m"; C_ERR="\033[31m"; C_HDR="\033[1;36m"; C_END="\033[0m"
else
  C_OK=""; C_SKIP=""; C_WARN=""; C_ERR=""; C_HDR=""; C_END=""
fi
ok()   { printf "${C_OK}✓${C_END} %s\n" "$1"; }
skip() { printf "${C_SKIP}↷ %s${C_END}\n" "$1"; }
warn() { printf "${C_WARN}⚠${C_END} %s\n" "$1"; }
err()  { printf "${C_ERR}✗${C_END} %s\n" "$1" >&2; exit 1; }
hdr()  { printf "\n${C_HDR}==> %s${C_END}\n" "$1"; }

usage() {
  sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
}

# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

cmd_install() {
  local download_models=1
  local assume_yes=0
  for arg in "$@"; do
    case "$arg" in
      --skip-models) download_models=0 ;;
      --yes|-y)      assume_yes=1 ;;
      *) err "Unknown flag for 'install': $arg" ;;
    esac
  done

  confirm() {
    if [ "$assume_yes" = 1 ]; then return 0; fi
    printf "%s [y/N] " "$1"
    read -r ans || return 1
    [[ "$ans" =~ ^[yY]$ ]]
  }

  hdr "Validate platform"
  [ "$(uname -s)" = "Darwin" ] || err "macOS required for mlx-openai-server (uname=$(uname -s))"
  [ "$(uname -m)" = "arm64" ] || err "Apple Silicon required (MLX needs Metal). uname -m=$(uname -m)"
  ok "macOS Apple Silicon"

  hdr "Find Python 3.11 or 3.12 (upstream constraint of mlx-openai-server)"
  # Daemon runs in Docker on python:3.11-slim regardless. The host needs 3.11
  # or 3.12 only because mlx-openai-server pins Requires-Python <3.13.
  local python=""
  for candidate in python3.12 python3.11 \
                   /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 \
                   /usr/local/bin/python3.12 /usr/local/bin/python3.11; do
    if command -v "$candidate" > /dev/null 2>&1; then
      local minor
      minor=$("$candidate" -c 'import sys; print(sys.version_info[1])')
      if [ "$minor" = "11" ] || [ "$minor" = "12" ]; then
        python="$candidate"
        break
      fi
    fi
  done
  if [ -z "$python" ]; then
    warn "No Python 3.11 or 3.12 found"
    if command -v brew > /dev/null 2>&1; then
      if confirm "Install python@3.12 via Homebrew? (~50 MB, coexists with any newer Python)"; then
        brew install python@3.12
        python="$(brew --prefix)/opt/python@3.12/bin/python3.12"
      else
        err "Python 3.11 or 3.12 required. Install one and re-run."
      fi
    else
      err "Homebrew not found. Install Homebrew or Python 3.11/3.12 manually, then re-run."
    fi
  fi
  ok "Python: $python ($("$python" --version))"

  hdr "Create venv at $VENV"
  if [ -d "$VENV" ]; then
    ok "venv exists at $VENV"
  else
    mkdir -p "$HOME/.venvs"
    "$python" -m venv "$VENV"
    ok "venv created at $VENV"
  fi
  "$VENV/bin/pip" install --upgrade pip --quiet
  ok "pip upgraded"

  if "$BIN" --help > /dev/null 2>&1; then
    skip "mlx-openai-server already installed"
  else
    "$VENV/bin/pip" install --quiet mlx-openai-server
    ok "mlx-openai-server installed"
  fi

  hdr "Copy mlx-server.yaml"
  [ -f "$CONFIG_EXAMPLE" ] || err "$CONFIG_EXAMPLE not found"
  if [ -f "$CONFIG" ]; then
    skip "$CONFIG already exists"
  else
    cp "$CONFIG_EXAMPLE" "$CONFIG"
    ok "$CONFIG copied from example"
  fi

  if [ "$download_models" = 1 ]; then
    hdr "Download Qwen + Whisper weights (~10 GB, may take 10–30 min)"
    "$VENV/bin/pip" install --quiet huggingface-hub
    for repo in mlx-community/Qwen2.5-14B-Instruct-4bit mlx-community/whisper-large-v3-mlx-4bit; do
      local cache_dir="$HOME/.cache/huggingface/hub/models--${repo//\//--}"
      if [ -d "$cache_dir/blobs" ] && [ "$(du -sm "$cache_dir/blobs" 2>/dev/null | awk '{print $1}')" -gt 1 ]; then
        skip "$repo already cached"
      else
        echo "  Downloading $repo ..."
        "$VENV/bin/python" -c "from huggingface_hub import snapshot_download; snapshot_download('$repo')"
        ok "$repo cached"
      fi
    done
  else
    skip "model download skipped (--skip-models). They'll be fetched on first request."
  fi

  echo
  ok "MLX install complete. Next: 'task up' to start the daemon + mlx-server."
}

# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------

cmd_start() {
  mkdir -p "$LOG_DIR"
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    skip "mlx-server already running (PID=$(cat "$PID_FILE"))"
    return 0
  fi
  if [ ! -x "$BIN" ]; then
    err "mlx-openai-server not found at $BIN. Run 'task install:mlx' or point config/tldr.yaml at another OpenAI-compatible backend."
  fi
  if [ ! -f "$CONFIG" ]; then
    err "$CONFIG not found. Run 'task install:mlx' to copy from example."
  fi
  # IMPORTANT: stdin redirected from /dev/null. Without this, the multiprocessing
  # workers mlx-openai-server forks inherit our (closed/bad) stdin fd and die
  # with "Fatal Python error: init_sys_streams: can't initialize sys standard streams".
  nohup "$BIN" launch --config "$CONFIG" \
    </dev/null \
    >"$LOG_DIR/mlx.out.log" 2>"$LOG_DIR/mlx.err.log" &
  echo $! > "$PID_FILE"
  ok "mlx-server started (PID=$(cat "$PID_FILE"))"
}

cmd_start_if_present() {
  if [ -x "$BIN" ]; then
    cmd_start
  else
    skip "mlx-openai-server not installed (skipping; using whatever backend is in config/tldr.yaml). Install with: task install:mlx"
  fi
}

cmd_stop() {
  if [ ! -f "$PID_FILE" ]; then
    skip "mlx-server not running"
    return 0
  fi
  local pid
  pid=$(cat "$PID_FILE")
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    ok "mlx-server stopped (PID=$pid)"
  else
    skip "mlx-server not running (stale PID file)"
  fi
  rm -f "$PID_FILE"
}

cmd_status() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
      ok "mlx-server running (PID=$(cat "$PID_FILE"), /v1/models OK)"
    else
      warn "mlx-server up (PID=$(cat "$PID_FILE")) but /v1/models not responding (loading?)"
    fi
  else
    printf "${C_ERR}✗${C_END} mlx-server not running\n"
  fi
}

# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

sub="${1:-}"
shift || true
case "$sub" in
  install)          cmd_install "$@" ;;
  start)            cmd_start ;;
  start-if-present) cmd_start_if_present ;;
  stop)             cmd_stop ;;
  status)           cmd_status ;;
  ""|-h|--help)     usage ;;
  *)
    echo "Unknown subcommand: $sub" >&2
    echo "Run '$0 --help' for usage." >&2
    exit 2
    ;;
esac
