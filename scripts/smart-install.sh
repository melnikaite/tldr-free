#!/usr/bin/env bash
#
# TLDR — smart installer.
#
# Uses llmfit to detect your hardware and already-running backends, then
# picks the best LLM backend + model (preferring Qwen3-VL 8B, falling back to
# Gemma 4 on smaller machines) and Whisper backend automatically. Writes
# config/tldr.yaml with the right endpoints, model names, and context_length.
#
# Replaces the manual "which backend do I use?" step — you no longer need to
# choose between task install and task install:mlx.
#
# Usage:
#   bash scripts/smart-install.sh [--yes|-y] [--skip-extension] [--dry-run]
#
# Flags:
#   --yes | -y          Don't prompt; accept all detected recommendations.
#   --skip-extension    Skip downloading the extension's vendored libs.
#   --dry-run           Print detected plan but don't install or write config.
#   -h | --help         Print this help.
#
# Decision tree:
#   1. Check existing running services (LM Studio :1234, Ollama :11434,
#      mlx-server :18000, llama.cpp :8080). If a suitable model is already
#      loaded → use it, skip new installs for that slot.
#   2. Run `llmfit --json system` for hardware (backend, VRAM/RAM, CPU).
#   3. Pick LLM backend + model from hardware:
#        Metal  + ≥16 GB  → mlx-server + Qwen3-VL-8B-Instruct-4bit (65 536 ctx,
#                            no reasoning_effort — not a thinking model)
#        Metal  + 8–15 GB → mlx-server + gemma-4-e2b-it-4bit  (131 072 ctx,
#                            reasoning_effort low). Unchanged from Gemma:
#                            the frame-understanding step (video-picture QA)
#                            was only measured on Qwen3-VL 8B, so smaller
#                            machines keep Gemma and get weaker
#                            video-picture answers.
#        CUDA   + ≥12 GB  → Ollama + qwen3-vl:8b + Modelfile   (65 536 ctx,
#                            no reasoning_effort)
#        CUDA   + 6–11 GB → Ollama + gemma4:e2b + Modelfile   (131 072 ctx)
#        ROCm/Vulkan       → Ollama + gemma4:e2b (best effort, ROCm support varies)
#        CPU only          → Ollama + gemma4:e2b (warn: slow)
#   4. Pick Whisper backend from hardware:
#        Metal        → mlx-server whisper-large-v3-turbo (port 18000)
#        CUDA ≥8 GB   → faster-whisper-server Docker image  (port 8000)
#        CUDA 4–7 GB  → faster-whisper-server, medium model (port 8000)
#        CPU / other  → whisper.cpp (brew / apt) on port 8178
#   5. Install chosen backends, configure context window, write config/tldr.yaml.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

ASSUME_YES=0
SKIP_EXTENSION=0
DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --yes|-y)           ASSUME_YES=1 ;;
    --skip-extension)   SKIP_EXTENSION=1 ;;
    --dry-run)          DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,44p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) printf "Unknown flag: %s\n" "$arg" >&2; exit 2 ;;
  esac
done

if [ -t 1 ]; then
  C_OK="\033[32m"; C_SKIP="\033[2m"; C_WARN="\033[33m"; C_ERR="\033[31m"
  C_HDR="\033[1;36m"; C_BOLD="\033[1m"; C_END="\033[0m"
else
  C_OK=""; C_SKIP=""; C_WARN=""; C_ERR=""; C_HDR=""; C_BOLD=""; C_END=""
fi
ok()    { printf "${C_OK}✓${C_END} %s\n" "$1"; }
skip()  { printf "${C_SKIP}↷ %s${C_END}\n" "$1"; }
warn()  { printf "${C_WARN}⚠${C_END} %s\n" "$1"; }
err()   { printf "${C_ERR}✗${C_END} %s\n" "$1" >&2; exit 1; }
hdr()   { printf "\n${C_HDR}==> %s${C_END}\n" "$1"; }
info()  { printf "  %s\n" "$1"; }
bold()  { printf "${C_BOLD}%s${C_END}\n" "$1"; }

ask() {
  # ask "message" → returns 0 (yes) or 1 (no).  ASSUME_YES always returns 0.
  local msg="$1"
  if [ "$ASSUME_YES" = 1 ]; then
    printf "  %s [y] (auto-confirmed)\n" "$msg"
    return 0
  fi
  printf "  %s [Y/n] " "$msg"
  read -r ans
  case "$ans" in
    [nN]*) return 1 ;;
    *) return 0 ;;
  esac
}

# ---------------------------------------------------------------------------
# 1. Ensure llmfit is installed
# ---------------------------------------------------------------------------
hdr "llmfit — hardware profiler"

_install_llmfit() {
  if [[ "$(uname)" == "Darwin" ]]; then
    if command -v brew &>/dev/null; then
      info "Installing via Homebrew…"
      brew install AlexsJones/llmfit/llmfit
    else
      info "Installing via shell script…"
      curl -fsSL https://llmfit.axjns.dev/install.sh | sh
    fi
  elif [[ "$(uname)" == "Linux" ]]; then
    info "Installing via shell script…"
    curl -fsSL https://llmfit.axjns.dev/install.sh | sh
  else
    warn "Windows detected — install llmfit via: scoop install llmfit"
    warn "Then re-run this script."
    exit 1
  fi
}

if ! command -v llmfit &>/dev/null; then
  warn "llmfit not found."
  if ask "Install llmfit now? (used for hardware detection — no data sent anywhere)"; then
    _install_llmfit
    ok "llmfit installed"
  else
    err "llmfit is required. Install manually: brew install AlexsJones/llmfit/llmfit"
  fi
else
  ok "llmfit $(llmfit --version 2>/dev/null | head -1)"
fi

# ---------------------------------------------------------------------------
# 2. Read hardware via llmfit --json system
# ---------------------------------------------------------------------------
hdr "Detecting hardware"

_SYSTEM_JSON="$(llmfit --json system 2>/dev/null)" || _SYSTEM_JSON="{}"

_jq() {
  # Lightweight JSON field extractor — avoids jq dependency.
  python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
keys = '$1'.split('.')
for k in keys:
    if isinstance(d, list): k = int(k)
    d = d.get(k) if isinstance(d, dict) else (d[int(k)] if isinstance(d, list) else None)
    if d is None: break
print(d if d is not None else '')
" <<< "$_SYSTEM_JSON"
}

BACKEND="$(_jq backend)"          # Metal | Cuda | ROCm | Vulkan | CpuArm | CpuX86 | …
TOTAL_RAM="$(_jq total_ram_gb)"   # float
VRAM="$(_jq total_gpu_vram_gb)"   # float or empty (null on unified-memory systems)
UNIFIED="$(_jq unified_memory)"   # True/False
GPU_NAME="$(_jq gpu_name)"        # "Apple M3 Pro" | "NVIDIA GeForce RTX 4080" | …
CPU_NAME="$(_jq cpu_name)"

# For Apple Silicon: total_ram_gb IS the unified GPU pool.
EFFECTIVE_GPU_MEM=""
if [ "$UNIFIED" = "True" ]; then
  EFFECTIVE_GPU_MEM="$TOTAL_RAM"
elif [ -n "$VRAM" ] && [ "$VRAM" != "None" ] && [ "$VRAM" != "" ]; then
  EFFECTIVE_GPU_MEM="$VRAM"
fi

info "CPU:     ${CPU_NAME:-unknown}"
info "GPU:     ${GPU_NAME:-none detected}"
info "Backend: ${BACKEND:-unknown}"
info "RAM:     ${TOTAL_RAM:-?} GB"
[ -n "$EFFECTIVE_GPU_MEM" ] && info "GPU mem: ${EFFECTIVE_GPU_MEM} GB ($([ "$UNIFIED" = "True" ] && echo 'unified' || echo 'VRAM'))"

# ---------------------------------------------------------------------------
# 3. Detect already-running backends
# ---------------------------------------------------------------------------
hdr "Scanning for running backends"

_probe() {
  # _probe <url> — returns 0 if responds with HTTP 200
  curl -sf --max-time 2 "$1" > /dev/null 2>&1
}

_models_at() {
  # _models_at <base_url> — returns first model id from /v1/models, or ""
  curl -sf --max-time 2 "$1/v1/models" 2>/dev/null \
    | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    models=d.get('data',[]) or d.get('models',[]) or []
    if models: print(models[0].get('id','') or models[0].get('name',''))
except: pass
" 2>/dev/null || true
}

# LM Studio
LMSTUDIO_RUNNING=0
LMSTUDIO_MODEL=""
if _probe "http://127.0.0.1:1234/v1/models"; then
  LMSTUDIO_RUNNING=1
  LMSTUDIO_MODEL="$(_models_at 'http://127.0.0.1:1234')"
  ok "LM Studio running on :1234 (model: ${LMSTUDIO_MODEL:-unknown})"
else
  skip "LM Studio not running on :1234"
fi

# Ollama
OLLAMA_RUNNING=0
OLLAMA_MODEL=""
if _probe "http://localhost:11434/api/tags"; then
  OLLAMA_RUNNING=1
  OLLAMA_MODEL="$(curl -sf --max-time 2 http://localhost:11434/api/tags 2>/dev/null \
    | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    models=d.get('models',[])
    # Prefer qwen3-vl variants (the new default), then gemma4, then first model
    for m in models:
        name=m.get('name','')
        if 'qwen3-vl' in name or 'qwen3vl' in name: print(name); exit()
    for m in models:
        name=m.get('name','')
        if 'gemma4' in name or 'gemma-4' in name: print(name); exit()
    if models: print(models[0].get('name',''))
except: pass
" 2>/dev/null || true)"
  ok "Ollama running on :11434 (model: ${OLLAMA_MODEL:-none pulled yet})"
else
  skip "Ollama not running on :11434"
fi

# mlx-server (TLDR default port)
MLX_RUNNING=0
MLX_MODEL=""
if _probe "http://localhost:18000/v1/models"; then
  MLX_RUNNING=1
  MLX_MODEL="$(_models_at 'http://localhost:18000')"
  ok "mlx-openai-server running on :18000 (model: ${MLX_MODEL:-unknown})"
else
  skip "mlx-openai-server not running on :18000"
fi

# llama.cpp server
LLAMACPP_RUNNING=0
if _probe "http://localhost:8080/health"; then
  LLAMACPP_RUNNING=1
  ok "llama.cpp server running on :8080"
else
  skip "llama.cpp not running on :8080"
fi

# faster-whisper server
FASTER_WHISPER_RUNNING=0
if _probe "http://localhost:8000/health"; then
  FASTER_WHISPER_RUNNING=1
  ok "faster-whisper server running on :8000"
else
  skip "faster-whisper not running on :8000"
fi

# whisper.cpp server
WHISPERCPP_RUNNING=0
if _probe "http://localhost:8178/health" || _probe "http://localhost:8178"; then
  WHISPERCPP_RUNNING=1
  ok "whisper.cpp server running on :8178"
else
  skip "whisper.cpp not running on :8178"
fi

# ---------------------------------------------------------------------------
# 4. Decide LLM backend + model
# ---------------------------------------------------------------------------
hdr "Choosing LLM backend"

LLM_BACKEND=""    # "mlx" | "ollama" | "lmstudio" | "llamacpp"
LLM_BASE_URL=""
LLM_MODEL=""
LLM_CTX=131072
LLM_SINGLE_PASS_LIMIT=80000   # ~60% of LLM_CTX; overridden alongside LLM_CTX below
LLM_CONCURRENT=1
LLM_REASONING_EFFORT=""
LLM_OLLAMA_CTX_SUFFIX="-128k"  # tag suffix Ollama Modelfile creation strips/adds; matches LLM_CTX
LLM_NEW_INSTALL=0  # 1 if we need to install something new

# Helper: compare floats (returns 0 if $1 >= $2)
_gte() { python3 -c "import sys; sys.exit(0 if float('${1:-0}') >= float('$2') else 1)" 2>/dev/null; }

# Priority 1: LM Studio already running with a capable model
if [ "$LMSTUDIO_RUNNING" = 1 ] && [ -n "$LMSTUDIO_MODEL" ]; then
  LLM_BACKEND="lmstudio"
  LLM_BASE_URL="http://host.docker.internal:1234/v1"
  LLM_MODEL="$LMSTUDIO_MODEL"

  # Query LM Studio's own REST API (distinct from the OpenAI-compat
  # /v1/models used above just to get a model id) for the model's ACTUAL
  # loaded context length and whether it exposes a reasoning/thinking
  # capability at all — real, per-model signals instead of assuming
  # "131072 + reasoning_effort low" for whatever happens to be loaded.
  # Per lmstudio.ai/docs/developer/rest, GET /api/v1/models returns
  # loaded_instances[].config.context_length and capabilities.reasoning.
  # Best-effort: older LM Studio versions may not have this endpoint, in
  # which case we fall back below instead of asserting a number/flag we
  # have no way to verify.
  _LMS_INFO="$(curl -sf --max-time 3 "http://127.0.0.1:1234/api/v1/models" 2>/dev/null \
    | LMS_TARGET="$LMSTUDIO_MODEL" python3 -c "
import json, os, sys
target = os.environ.get('LMS_TARGET', '')
try:
    d = json.load(sys.stdin)
    for m in d.get('models', []):
        if m.get('key') == target or m.get('id') == target:
            ctx = None
            insts = m.get('loaded_instances') or []
            if insts:
                ctx = (insts[0].get('config') or {}).get('context_length')
            if ctx is None:
                ctx = m.get('max_context_length')
            reasoning = 'reasoning' in (m.get('capabilities') or {})
            if ctx is not None:
                print(f'{int(ctx)}|{1 if reasoning else 0}')
            break
except Exception:
    pass
" 2>/dev/null)"

  if [ -n "$_LMS_INFO" ]; then
    LLM_CTX="${_LMS_INFO%%|*}"
    _LMS_REASONING="${_LMS_INFO##*|}"
    case "$LLM_CTX" in
      65536) LLM_SINGLE_PASS_LIMIT=40000 ;;
      131072) LLM_SINGLE_PASS_LIMIT=80000 ;;
      *) LLM_SINGLE_PASS_LIMIT=$(( LLM_CTX * 3 / 5 )) ;;
    esac
    if [ "$_LMS_REASONING" = "1" ]; then
      LLM_REASONING_EFFORT="low"
      info "'$LMSTUDIO_MODEL' exposes a reasoning capability; setting reasoning_effort: low."
    fi
    ok "Read context_length=$LLM_CTX for '$LMSTUDIO_MODEL' from LM Studio's REST API."
  else
    # Couldn't confirm from LM Studio (older version without
    # /api/v1/models, or no matching entry) — default to this project's own
    # shipped Qwen3-VL numbers rather than assert the old Gemma-era
    # "131072 + reasoning_effort low" for a model we haven't identified.
    # reasoning_effort is left UNSET here, not "low": whether an
    # unsupported field is harmless on LM Studio's OpenAI-compat layer is
    # unverified. The daemon's own call_with_dialect_adaptation
    # (daemon/src/llm/client.py) auto-retries without reasoning_effort on
    # an HTTP 400 that blames the field BY NAME — but if LM Studio ever
    # 400s without naming it, that retry can't fire and every call breaks
    # instead of just costing one wasted round-trip. Not worth risking on
    # an unidentified model.
    LLM_CTX=65536
    LLM_SINGLE_PASS_LIMIT=40000
    warn "Could not read '$LMSTUDIO_MODEL''s context/reasoning capability from LM Studio's REST API (older LM Studio version, or no match)."
    warn "Verify its actual context in LM Studio (Settings -> Context Length, or 'lms ps') and correct config/tldr.yaml's context_length/single_pass_token_limit if it isn't 65536. If it's a thinking model, add reasoning_effort yourself (\"low\" for most; check its docs)."
  fi
# Priority 2: Ollama already running
elif [ "$OLLAMA_RUNNING" = 1 ] && [ -n "$OLLAMA_MODEL" ]; then
  LLM_BACKEND="ollama"
  LLM_BASE_URL="http://host.docker.internal:11434/v1"

  # Ollama's /api/show echoes back any num_ctx already baked into this exact
  # tag via a prior Modelfile (its "parameters" field is a text blob like
  # "num_ctx 65536\ntemperature 0.7"). Check that BEFORE assuming this tag
  # needs a fresh context-expanded Modelfile — this is the real, queryable
  # truth for a tag that already exists, not a guess.
  _OLLAMA_NUM_CTX="$(curl -sf --max-time 3 -X POST http://localhost:11434/api/show \
      -d "{\"model\":\"$OLLAMA_MODEL\"}" 2>/dev/null \
    | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    for line in (d.get('parameters') or '').splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == 'num_ctx':
            print(parts[1])
            break
except Exception:
    pass
" 2>/dev/null)"

  if [ -n "$_OLLAMA_NUM_CTX" ]; then
    # This tag already has num_ctx configured (e.g. a previous smart-install
    # run's Modelfile tag) — reuse as-is, no new Modelfile needed.
    LLM_MODEL="$OLLAMA_MODEL"
    LLM_CTX="$_OLLAMA_NUM_CTX"
    case "$LLM_CTX" in
      65536) LLM_SINGLE_PASS_LIMIT=40000 ;;
      131072) LLM_SINGLE_PASS_LIMIT=80000 ;;
      *) LLM_SINGLE_PASS_LIMIT=$(( LLM_CTX * 3 / 5 )) ;;
    esac
    ok "Ollama model '$OLLAMA_MODEL' already has num_ctx=$LLM_CTX configured; reusing as-is."
  else
    # No num_ctx on this tag — it needs a context-expanded Modelfile tag.
    # Pick the target context by model family, matching the sniffer's own
    # qwen3-vl-first / gemma4-second preference above; default to this
    # project's own Qwen3-VL numbers (not the old Gemma ones) for anything
    # unrecognised, since guessing Gemma's 131072 for an unknown model would
    # be an equally baseless assumption in the other direction.
    case "$OLLAMA_MODEL" in
      *qwen3-vl*|*qwen3vl*)
        LLM_CTX=65536; LLM_SINGLE_PASS_LIMIT=40000; LLM_OLLAMA_CTX_SUFFIX="-64k" ;;
      *gemma4*|*gemma-4*)
        LLM_CTX=131072; LLM_SINGLE_PASS_LIMIT=80000; LLM_OLLAMA_CTX_SUFFIX="-128k" ;;
      *)
        LLM_CTX=65536; LLM_SINGLE_PASS_LIMIT=40000; LLM_OLLAMA_CTX_SUFFIX="-64k"
        warn "Ollama model '$OLLAMA_MODEL' doesn't match a known qwen3-vl/gemma4 naming convention; assuming this project's 65536 default context. Check with: ollama show $OLLAMA_MODEL"
        ;;
    esac
    LLM_MODEL="${OLLAMA_MODEL%$LLM_OLLAMA_CTX_SUFFIX}${LLM_OLLAMA_CTX_SUFFIX}"
    # Ollama itself is already running, but this tag still needs the
    # Modelfile step below to actually exist at the expanded context —
    # both `ollama pull` and `ollama create -f` are idempotent, so running
    # them even though Ollama is "already running" is safe.
    LLM_NEW_INSTALL=1
    info "Will create/update Ollama Modelfile '$LLM_MODEL' for $LLM_CTX context."
  fi
# Priority 3: mlx-server already running
elif [ "$MLX_RUNNING" = 1 ] && [ -n "$MLX_MODEL" ]; then
  LLM_BACKEND="mlx"
  LLM_BASE_URL="http://host.docker.internal:18000/v1"
  LLM_MODEL="$MLX_MODEL"

  # mlx-openai-server's /v1/models does NOT expose context_length (confirmed
  # from its v1.8.1 source: app/core/model_registry.py::list_models() only
  # returns id/object/created/owned_by), so we can't query the running
  # server for it over HTTP. Read the local seed file this project itself
  # writes instead (~/.mlx-server/config.yaml, via scripts/mlx.sh) and look
  # up the entry whose served_model_name matches what's actually running —
  # the real config the server was launched with, not a guess.
  _MLX_CFG_CTX=""
  if [ -f "$HOME/.mlx-server/config.yaml" ]; then
    _MLX_CFG_CTX="$(awk -v target="$MLX_MODEL" '
      function check() { if (name == target && ctx != "" && !printed) { print ctx; printed = 1 } }
      /^  - model_path:/ { check(); name = ""; ctx = "" }
      /served_model_name:/ {
        line = $0; sub(/^[^:]*:[ \t]*/, "", line); sub(/[ \t]*#.*/, "", line)
        gsub(/^[ \t]+|[ \t]+$/, "", line); name = line
      }
      /context_length:/ {
        line = $0; sub(/^[^:]*:[ \t]*/, "", line); sub(/[ \t]*#.*/, "", line)
        gsub(/^[ \t]+|[ \t]+$/, "", line); ctx = line
      }
      END { check() }
    ' "$HOME/.mlx-server/config.yaml" 2>/dev/null)"
  fi

  if [ -n "$_MLX_CFG_CTX" ]; then
    LLM_CTX="$_MLX_CFG_CTX"
    case "$LLM_CTX" in
      65536) LLM_SINGLE_PASS_LIMIT=40000 ;;
      131072) LLM_SINGLE_PASS_LIMIT=80000 ;;
      *) LLM_SINGLE_PASS_LIMIT=$(( LLM_CTX * 3 / 5 )) ;;
    esac
    ok "Read context_length=$LLM_CTX for '$MLX_MODEL' from ~/.mlx-server/config.yaml"
  else
    # No local config match (server started some other way, or file missing)
    # — fall back to a name-based guess, and failing that, default to this
    # project's own shipped Qwen3-VL numbers rather than the old Gemma ones.
    case "$MLX_MODEL" in
      *qwen3-vl*|*qwen3vl*|*Qwen3-VL*)
        LLM_CTX=65536; LLM_SINGLE_PASS_LIMIT=40000
        warn "Could not read context_length from ~/.mlx-server/config.yaml; inferred 65536 from model name '$MLX_MODEL'. Verify against ~/.mlx-server/config.yaml."
        ;;
      *gemma*|*Gemma*)
        LLM_CTX=131072; LLM_SINGLE_PASS_LIMIT=80000
        warn "Could not read context_length from ~/.mlx-server/config.yaml; inferred 131072 from model name '$MLX_MODEL'. Verify against ~/.mlx-server/config.yaml."
        ;;
      *)
        LLM_CTX=65536; LLM_SINGLE_PASS_LIMIT=40000
        warn "Could not determine context_length for mlx model '$MLX_MODEL' (no ~/.mlx-server/config.yaml match, name doesn't match a known convention). Defaulting to this project's shipped 65536 — check ~/.mlx-server/config.yaml and correct config/tldr.yaml's context_length/single_pass_token_limit by hand if that's wrong for your model."
        ;;
    esac
  fi
# Priority 4: Install based on hardware
elif [ "$BACKEND" = "Metal" ]; then
  LLM_BACKEND="mlx"
  LLM_BASE_URL="http://host.docker.internal:18000/v1"
  LLM_NEW_INSTALL=1
  if _gte "${EFFECTIVE_GPU_MEM:-0}" "16"; then
    LLM_MODEL="qwen3-vl"  # served_model_name in mlx-server config
    LLM_CTX=65536
    LLM_SINGLE_PASS_LIMIT=40000
    info "Apple Silicon ≥16 GB → mlx-server + Qwen3-VL-8B-Instruct-4bit"
  else
    LLM_MODEL="gemma4"
    LLM_REASONING_EFFORT="low"
    # The video-frame-understanding QA step was only measured on Qwen3-VL
    # 8B (needs ≥16 GB), so this tier keeps Gemma 4 and gets weaker
    # video-picture answers.
    info "Apple Silicon 8–15 GB → mlx-server + gemma-4-e2b-it-4bit"
  fi
elif [ "$BACKEND" = "Cuda" ]; then
  LLM_BACKEND="ollama"
  LLM_BASE_URL="http://host.docker.internal:11434/v1"
  LLM_NEW_INSTALL=1
  if _gte "${EFFECTIVE_GPU_MEM:-0}" "12"; then
    LLM_CTX=65536
    LLM_SINGLE_PASS_LIMIT=40000
    LLM_OLLAMA_CTX_SUFFIX="-64k"
    LLM_MODEL="qwen3-vl:8b${LLM_OLLAMA_CTX_SUFFIX}"
    info "NVIDIA ≥12 GB → Ollama + qwen3-vl:8b + 65536 context Modelfile"
  else
    LLM_MODEL="gemma4:e2b-128k"
    LLM_REASONING_EFFORT="low"
    info "NVIDIA 6–11 GB → Ollama + gemma4:e2b + 131072 context Modelfile"
  fi
elif [ "$BACKEND" = "ROCm" ] || [ "$BACKEND" = "Vulkan" ]; then
  LLM_BACKEND="ollama"
  LLM_BASE_URL="http://host.docker.internal:11434/v1"
  LLM_MODEL="gemma4:e2b-128k"
  LLM_REASONING_EFFORT="low"
  LLM_NEW_INSTALL=1
  warn "AMD GPU detected — Ollama ROCm support varies by card. Falling back to gemma4:e2b."
else
  LLM_BACKEND="ollama"
  LLM_BASE_URL="http://host.docker.internal:11434/v1"
  LLM_MODEL="gemma4:e2b-128k"
  LLM_REASONING_EFFORT="low"
  LLM_NEW_INSTALL=1
  warn "CPU-only mode — inference will be slow. gemma4:e2b (~4 GB RAM)."
fi

ok "LLM: $LLM_BACKEND — model=$LLM_MODEL  ctx=$LLM_CTX"

# ---------------------------------------------------------------------------
# 5. Decide Whisper backend
# ---------------------------------------------------------------------------
hdr "Choosing Whisper backend"

WHISPER_BACKEND=""   # "mlx" | "faster-whisper" | "whispercpp" | "skip"
WHISPER_BASE_URL=""
WHISPER_MODEL="whisper"
WHISPER_NEW_INSTALL=0

if [ "$FASTER_WHISPER_RUNNING" = 1 ]; then
  WHISPER_BACKEND="faster-whisper"
  WHISPER_BASE_URL="http://host.docker.internal:8000/v1"
  ok "Whisper: existing faster-whisper server on :8000"
elif [ "$WHISPERCPP_RUNNING" = 1 ]; then
  WHISPER_BACKEND="whispercpp"
  WHISPER_BASE_URL="http://host.docker.internal:8178/v1"
  ok "Whisper: existing whisper.cpp server on :8178"
elif [ "$MLX_RUNNING" = 1 ] || ([ "$LLM_BACKEND" = "mlx" ] && [ "$LLM_NEW_INSTALL" = 1 ]); then
  WHISPER_BACKEND="mlx"
  WHISPER_BASE_URL="http://host.docker.internal:18000/v1"
  WHISPER_MODEL="whisper"
  ok "Whisper: mlx-openai-server (shares port 18000 with LLM)"
elif [ "$BACKEND" = "Metal" ]; then
  WHISPER_BACKEND="mlx"
  WHISPER_BASE_URL="http://host.docker.internal:18000/v1"
  WHISPER_MODEL="whisper"
  WHISPER_NEW_INSTALL=1
elif [ "$BACKEND" = "Cuda" ]; then
  if command -v docker &>/dev/null && docker info &>/dev/null; then
    WHISPER_BACKEND="faster-whisper"
    WHISPER_BASE_URL="http://host.docker.internal:8000/v1"
    WHISPER_MODEL="Systran/faster-whisper-large-v3"
    WHISPER_NEW_INSTALL=1
    if ! _gte "${EFFECTIVE_GPU_MEM:-0}" "8"; then
      WHISPER_MODEL="Systran/faster-whisper-medium"
      warn "NVIDIA VRAM < 8 GB — using faster-whisper-medium to save memory"
    fi
    info "NVIDIA → faster-whisper-server (Docker) on :8000"
  else
    WHISPER_BACKEND="skip"
    warn "Docker not available — skipping Whisper (YouTube audio fallback disabled)."
  fi
else
  WHISPER_BACKEND="skip"
  warn "No GPU — skipping Whisper (YouTube audio transcription will be unavailable)."
  info "To enable later: brew install whisper-cpp && whisper-server -m ggml-large-v3.bin -p 8178"
fi

# ---------------------------------------------------------------------------
# 6. Show plan and confirm
# ---------------------------------------------------------------------------
hdr "Installation plan"

bold "LLM backend:    $LLM_BACKEND"
info "  base_url:     $LLM_BASE_URL"
info "  model:        $LLM_MODEL"
info "  context:      $LLM_CTX tokens"

bold "Whisper backend: ${WHISPER_BACKEND:-none}"
[ "$WHISPER_BACKEND" != "skip" ] && [ -n "$WHISPER_BASE_URL" ] && info "  base_url: $WHISPER_BASE_URL"
[ "$WHISPER_BACKEND" != "skip" ] && info "  model:    $WHISPER_MODEL"

echo

if [ "$DRY_RUN" = 1 ]; then
  warn "Dry-run mode — nothing installed or written."
  exit 0
fi

ask "Proceed with this plan?" || { warn "Aborted."; exit 1; }

# ---------------------------------------------------------------------------
# 7. Run core install (Docker image + vendors)
# ---------------------------------------------------------------------------
hdr "Core install (Docker image + extension vendors)"
CORE_FLAGS=""
[ "$SKIP_EXTENSION" = 1 ] && CORE_FLAGS="--skip-extension"
[ "$ASSUME_YES" = 1 ]     && CORE_FLAGS="$CORE_FLAGS --yes"
bash "$REPO_ROOT/scripts/install.sh" $CORE_FLAGS

# ---------------------------------------------------------------------------
# 8. Install backends
# ---------------------------------------------------------------------------

# --- MLX install ---
if [ "$LLM_BACKEND" = "mlx" ] && [ "$LLM_NEW_INSTALL" = 1 ]; then
  hdr "Installing mlx-openai-server"
  _MLX_FLAGS="--yes"
  # mlx.sh honours these three env vars to pick which model it downloads AND
  # seeds into ~/.mlx-server/config.yaml. All three MUST move together —
  # LLM_MODEL/LLM_CTX are exactly what we're about to write into
  # config/tldr.yaml below, so the seeded server config and the daemon
  # config end up describing the same model at the same context length
  # instead of silently drifting apart.
  if _gte "${EFFECTIVE_GPU_MEM:-0}" "16"; then
    export MLX_LLM_MODEL="mlx-community/Qwen3-VL-8B-Instruct-4bit"
  else
    export MLX_LLM_MODEL="mlx-community/gemma-4-e2b-it-4bit"
  fi
  export MLX_LLM_SERVED_NAME="$LLM_MODEL"
  export MLX_LLM_CONTEXT_LENGTH="$LLM_CTX"
  bash "$REPO_ROOT/scripts/mlx.sh" install $_MLX_FLAGS
  ok "mlx-openai-server installed"
fi

# --- Ollama install ---
if [ "$LLM_BACKEND" = "ollama" ] && [ "$LLM_NEW_INSTALL" = 1 ]; then
  hdr "Setting up Ollama"
  if ! command -v ollama &>/dev/null; then
    info "Ollama not found — installing…"
    if [[ "$(uname)" == "Darwin" ]]; then
      brew install ollama
    else
      curl -fsSL https://ollama.com/install.sh | sh
    fi
  fi
  # Start if not running
  if ! _probe "http://localhost:11434/api/tags"; then
    info "Starting Ollama…"
    ollama serve &>/dev/null &
    sleep 3
  fi
  # Pull the base model (without the context-tag suffix, e.g. -128k / -64k).
  # Ollama's own default context window (often 2048-4096) is far below what
  # either Gemma 4 (131072) or Qwen3-VL (65536) supports, so we always
  # create a context-expanded local tag via Modelfile.
  _BASE_MODEL="${LLM_MODEL%$LLM_OLLAMA_CTX_SUFFIX}"
  info "Pulling $_BASE_MODEL …"
  ollama pull "$_BASE_MODEL"
  _MODELFILE_TAG="$LLM_MODEL"
  info "Creating context-expanded model '$_MODELFILE_TAG' (context_length=$LLM_CTX)…"
  printf 'FROM %s\nPARAMETER num_ctx %s\n' "$_BASE_MODEL" "$LLM_CTX" | \
    ollama create "$_MODELFILE_TAG" -f -
  ok "Ollama model '$_MODELFILE_TAG' ready with $LLM_CTX context"
fi

# --- faster-whisper (Docker) ---
if [ "$WHISPER_BACKEND" = "faster-whisper" ] && [ "$WHISPER_NEW_INSTALL" = 1 ]; then
  hdr "Setting up faster-whisper-server (Docker)"
  if ! docker info &>/dev/null; then
    warn "Docker not running — skipping faster-whisper setup."
  else
    info "Pulling fedirz/faster-whisper-server…"
    docker pull fedirz/faster-whisper-server
    info "Starting faster-whisper-server on :8000 (background)…"
    docker run -d --name tldr-whisper --restart=unless-stopped \
      -p 8000:8000 \
      -e WHISPER__MODEL="$WHISPER_MODEL" \
      fedirz/faster-whisper-server 2>/dev/null || \
    docker start tldr-whisper 2>/dev/null || true
    ok "faster-whisper-server started"
    WHISPER_MODEL="whisper-1"  # faster-whisper-server uses this alias
  fi
fi

# ---------------------------------------------------------------------------
# 9. Write config/tldr.yaml
# ---------------------------------------------------------------------------
hdr "Writing config/tldr.yaml"

_LLM_API_KEY="ollama"
[ "$LLM_BACKEND" = "lmstudio" ] && _LLM_API_KEY="lm-studio"
[ "$LLM_BACKEND" = "mlx" ]      && _LLM_API_KEY="dummy"

_WHISPER_API_KEY="dummy"
_WHISPER_BLOCK=""
if [ "$WHISPER_BACKEND" != "skip" ]; then
  _WHISPER_BLOCK="
whisper:
  base_url: $WHISPER_BASE_URL
  api_key: $_WHISPER_API_KEY
  model: $WHISPER_MODEL"
else
  _WHISPER_BLOCK="
# whisper: (not configured — no suitable GPU detected)
# To add later: see config/tldr.yaml.example for backend options."
fi

_REASONING=""
[ -n "$LLM_REASONING_EFFORT" ] && _REASONING="  reasoning_effort: \"$LLM_REASONING_EFFORT\""

_CONFIG="# TLDR daemon configuration — auto-generated by scripts/smart-install.sh
# Backend: $LLM_BACKEND  |  GPU: ${GPU_NAME:-none}  |  ${EFFECTIVE_GPU_MEM:-?} GB
# Regenerate: bash scripts/smart-install.sh
#
# To switch backends later, edit this file and run: task down && task up

llm:
  base_url: $LLM_BASE_URL
  api_key: $_LLM_API_KEY
  model: $LLM_MODEL
  context_length: $LLM_CTX
  single_pass_token_limit: $LLM_SINGLE_PASS_LIMIT
  max_concurrent_calls: $LLM_CONCURRENT
$([ -n "$_REASONING" ] && printf '%s\n' "$_REASONING")
$_WHISPER_BLOCK
"

CONFIG_DST="$REPO_ROOT/config/tldr.yaml"
if [ -f "$CONFIG_DST" ] && ! ask "Overwrite existing config/tldr.yaml?"; then
  warn "config/tldr.yaml left unchanged."
else
  # Write atomically
  _TMP="$CONFIG_DST.tmp.$$"
  printf '%s\n' "$_CONFIG" > "$_TMP"
  mv "$_TMP" "$CONFIG_DST"
  ok "config/tldr.yaml written"
fi

# ---------------------------------------------------------------------------
# 10. Done
# ---------------------------------------------------------------------------
echo
ok "Smart install complete."
printf "\n${C_HDR}Next steps:${C_END}\n"
printf "  task up           — start the daemon\n"
printf "  task status       — verify everything is healthy\n"
printf "  chrome://extensions → load unpacked → extension/ directory\n"
[ "$LLM_BACKEND" = "lmstudio" ] && \
  printf "\n${C_WARN}⚠${C_END} LM Studio: set Context Length to 131072 in model settings (Settings → Context)\n"
[ "$LLM_BACKEND" = "ollama" ] && \
  printf "\n  Verify context: ollama show $LLM_MODEL | grep context\n"
printf "\n"
