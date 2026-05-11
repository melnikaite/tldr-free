#!/usr/bin/env bash
#
# TLDR — core installer.
# Idempotent: safe to re-run after partial failure.
#
# Installs config files, the daemon Docker image, and the extension's vendored
# libs. The daemon talks to any OpenAI-compatible backend (Ollama, LM Studio,
# llama.cpp, vLLM, mlx-openai-server, etc.) — pick one and point
# ``config/tldr.yaml`` at it.
#
# To also install Apple Silicon's mlx-openai-server + Gemma 4 + Whisper weights,
# run ``bash scripts/mlx.sh install`` (or ``task install:mlx``) afterwards.
#
# Flags:
#   --skip-extension     Skip downloading the extension's vendored libs
#   --yes | -y           Don't prompt; assume yes
#   -h | --help          Print usage

set -euo pipefail

SKIP_EXTENSION=0
ASSUME_YES=0

for arg in "$@"; do
  case "$arg" in
    --skip-extension)  SKIP_EXTENSION=1 ;;
    --yes|-y)          ASSUME_YES=1 ;;
    -h|--help)
      sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

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

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

hdr "Copy config files"
for f in tldr.yaml; do
  src="config/${f}.example"
  dst="config/${f}"
  [ -f "$src" ] || err "$src not found"
  if [ -f "$dst" ]; then
    skip "$dst already exists"
  else
    cp "$src" "$dst"
    ok "$dst copied from example"
  fi
done

hdr "Build daemon Docker image"
if ! command -v docker > /dev/null 2>&1; then
  err "docker CLI not found. Install OrbStack or Docker Desktop."
fi
if ! docker info > /dev/null 2>&1; then
  err "Docker daemon not reachable. Start OrbStack/Docker Desktop and re-run."
fi
docker compose build daemon
ok "daemon image built"

hdr "Vendor extension libraries (marked, DOMPurify, Readability)"
if [ "$SKIP_EXTENSION" = 1 ]; then
  skip "extension vendor skipped (--skip-extension)"
else
  mkdir -p extension/vendor
  vendor_specs=(
    "marked.min.js|https://cdn.jsdelivr.net/npm/marked/marked.min.js"
    "purify.min.js|https://cdn.jsdelivr.net/npm/dompurify/dist/purify.min.js"
    "readability.js|https://raw.githubusercontent.com/mozilla/readability/main/Readability.js"
  )
  for spec in "${vendor_specs[@]}"; do
    name="${spec%%|*}"
    url="${spec##*|}"
    dst="extension/vendor/$name"
    if [ -s "$dst" ]; then
      skip "vendor/$name already present"
    else
      curl -sfLo "$dst" "$url" && ok "vendor/$name downloaded" || err "Failed to download $url"
    fi
  done
fi

echo
ok "Core install complete."

cat <<EOF

Next:
  1) Point config/tldr.yaml at your OpenAI-compatible LLM/Whisper backend.
     Examples (Ollama, LM Studio, llama.cpp) are in config/tldr.yaml.example.
     On Apple Silicon you can also run: task install:mlx
  2) task up
  3) Load the unpacked Chrome extension from the extension/ directory:
     - chrome://extensions, enable Developer mode
     - "Load unpacked" → select the extension/ directory
     - After source changes, hit the reload icon — no rebuild step.

Stop everything:  task down  (volume preserved)
Wipe data:        task reset  (destructive, asks for confirmation)
EOF
