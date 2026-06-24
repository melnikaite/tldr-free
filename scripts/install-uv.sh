#!/bin/sh
#
# TLDR — native (uv) installer. No Docker required.
# POSIX sh, macOS + Linux. Idempotent: safe to re-run.
#
# Steps:
#   1. Detect OS/arch; warn if ffmpeg is missing (needed for Whisper fallback).
#   2. Install uv via the official installer if absent.
#   3. `uv tool install` the daemon — from this checkout when run inside the
#      repo, otherwise from GitHub.
#   4. First-run config init (auto-created from the packaged template).
#   5. Register the user-level autostart service + wait for /health.
#
# Usage: sh scripts/install-uv.sh
#        curl -fsSL https://raw.githubusercontent.com/melnikaite/tldr-free/main/scripts/install-uv.sh | sh

set -eu

HEALTH_URL="http://127.0.0.1:8765/health"
REPO_SPEC="git+https://github.com/melnikaite/tldr-free#subdirectory=daemon"

ok()   { printf '\033[32m+\033[0m %s\n' "$1"; }
warn() { printf '\033[33m!\033[0m %s\n' "$1"; }
err()  { printf '\033[31mx\033[0m %s\n' "$1" >&2; exit 1; }
hdr()  { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }

# --- 1. platform checks ------------------------------------------------------
hdr "Platform"
os=$(uname -s)
arch=$(uname -m)
case "$os" in
  Darwin|Linux) ok "$os/$arch" ;;
  *) err "Unsupported OS: $os (use the Docker install, or Windows is experimental via 'tldr-daemon service install')" ;;
esac

if command -v ffmpeg >/dev/null 2>&1; then
  ok "ffmpeg found"
else
  if [ "$os" = "Darwin" ]; then
    warn "ffmpeg not found — Whisper fallback for caption-less videos won't work. Install: brew install ffmpeg"
  else
    warn "ffmpeg not found — Whisper fallback for caption-less videos won't work. Install: sudo apt install ffmpeg"
  fi
fi

# --- 2. uv -------------------------------------------------------------------
hdr "uv"
if command -v uv >/dev/null 2>&1; then
  ok "uv already installed ($(uv --version))"
else
  curl -fsSL https://astral.sh/uv/install.sh | sh || err "uv install failed"
  # The installer drops uv into ~/.local/bin; pick it up for this session.
  PATH="$HOME/.local/bin:$PATH"
  export PATH
  command -v uv >/dev/null 2>&1 || err "uv not on PATH after install; open a new shell and re-run"
  ok "uv installed"
fi

# --- 3. install the daemon ---------------------------------------------------
hdr "Install tldr-daemon"
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -f "$script_dir/../daemon/pyproject.toml" ]; then
  ok "installing from local checkout"
  uv tool install --force "$script_dir/../daemon"
else
  ok "installing from $REPO_SPEC"
  uv tool install --force "$REPO_SPEC"
fi
# uv tool shims live in ~/.local/bin
PATH="$HOME/.local/bin:$PATH"
export PATH
command -v tldr-daemon >/dev/null 2>&1 || err "tldr-daemon not on PATH; run 'uv tool update-shell' and re-run"
ok "tldr-daemon installed: $(command -v tldr-daemon)"

# --- 4. config ---------------------------------------------------------------
hdr "Config"
if [ "$os" = "Darwin" ]; then
  cfg="$HOME/Library/Application Support/tldr/tldr.yaml"
else
  cfg="${XDG_CONFIG_HOME:-$HOME/.config}/tldr/tldr.yaml"
fi
if [ -f "$cfg" ]; then
  ok "config exists: $cfg"
else
  warn "config will be auto-created from the packaged template on first start: $cfg"
fi

# --- 5. service + health -----------------------------------------------------
hdr "Autostart service"
tldr-daemon service install || err "service install failed"

hdr "Waiting for the daemon"
i=0
while [ "$i" -lt 30 ]; do
  if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
    ok "daemon healthy ($HEALTH_URL)"
    break
  fi
  i=$((i + 1))
  sleep 2
done
if [ "$i" -ge 30 ]; then
  warn "daemon did not become healthy in 60s — check: tldr-daemon service status"
fi

# --- backend probe (informational only) ---------------------------------------
hdr "LLM backend probe"
backend=""
curl -sf -m 2 http://127.0.0.1:11434/v1/models >/dev/null 2>&1 && backend="Ollama (port 11434)"
[ -z "$backend" ] && curl -sf -m 2 http://127.0.0.1:1234/v1/models >/dev/null 2>&1 && backend="LM Studio (port 1234)"
[ -z "$backend" ] && curl -sf -m 2 http://127.0.0.1:8080/v1/models >/dev/null 2>&1 && backend="llama-server (port 8080)"
if [ -n "$backend" ]; then
  ok "found a backend: $backend — make sure $cfg points at it"
else
  warn "no OpenAI-compatible backend detected on ports 11434/1234/8080"
  warn "install one (e.g. https://ollama.com/download) and point $cfg at it"
fi

cat <<EOF

Done. Next:
  1) Edit the config if needed: $cfg
     (restart after edits: tldr-daemon service uninstall && tldr-daemon service install)
  2) Load the Chrome extension: chrome://extensions -> Developer mode ->
     "Load unpacked" -> select the extension/ directory of the repo.

Status:    tldr-daemon service status
Uninstall: sh scripts/uninstall-uv.sh
EOF
