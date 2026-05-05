#!/bin/bash
# =============================================================================
# Entrypoint for the GGUF server (Salad / Vast.ai / VPS / bare-metal).
#
# What this does:
#   1. Resolve model configuration from environment variables.
#   2. Download the GGUF model from Hugging Face if not already cached.
#   3. Start llama-server on 127.0.0.1:LLAMA_INTERNAL_PORT (loopback only).
#   4. Wait until llama-server is ready.
#   5. Start the FastAPI auth proxy on [::]:PUBLIC_PORT (IPv4 + IPv6).
#   6. Wait for either process to exit; clean up on shutdown.
#
# Performance notes:
#   - For MoE models (Qwen3.x-A3B, GPT-OSS, etc.) on small GPUs, set
#     OFFLOAD_MODE=cmoe — it pushes all expert tensors to CPU, leaves
#     attention on GPU, and runs ~2-3x faster than naive layer offload.
#   - BATCH_SIZE/UBATCH_SIZE default to 4096 for ~5x faster prefill at the
#     cost of ~3 GB extra VRAM. Drop to 2048 if VRAM is tight.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (all optional; sensible defaults provided)
# -----------------------------------------------------------------------------

# --- Model selection ---
MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.6-35B-A3B-GGUF}"
MODEL_PATTERN="${MODEL_PATTERN:-*UD-Q4_K_XL*}"
MMPROJ_PATTERN="${MMPROJ_PATTERN:-}"
MODEL_ALIAS="${MODEL_ALIAS:-$(basename "${MODEL_REPO}")}"
MODEL_DIR="${MODEL_DIR:-/data/models}"

# --- llama-server runtime ---
LLAMA_INTERNAL_PORT="${LLAMA_INTERNAL_PORT:-8081}"
CTX_SIZE="${CTX_SIZE:-16384}"

# Offload strategy:
#   auto   - llama.cpp picks ngl/cmoe/ncmoe automatically (recommended)
#   ngl    - classic --n-gpu-layers (best for Dense models that fit)
#   cmoe   - all MoE experts to CPU, attention on GPU (small GPU + MoE)
#   ncmoe  - move N MoE layers to CPU; needs OFFLOAD_VALUE
#   manual - skip auto-offload; rely on EXTRA_ARGS
OFFLOAD_MODE="${OFFLOAD_MODE:-auto}"
OFFLOAD_VALUE="${OFFLOAD_VALUE:-}"
FIT_TARGET_MB="${FIT_TARGET_MB:-1024}"

# Batch sizes for prompt processing.
BATCH_SIZE="${BATCH_SIZE:-4096}"
UBATCH_SIZE="${UBATCH_SIZE:-4096}"

PARALLEL="${PARALLEL:-1}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
FLASH_ATTN="${FLASH_ATTN:-on}"
REASONING_FORMAT="${REASONING_FORMAT:-deepseek}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# --- Public proxy ---
PUBLIC_PORT="${PUBLIC_PORT:-8080}"
API_KEY="${API_KEY:-}"
API_KEYS="${API_KEYS:-}"
RATE_LIMIT_PER_MINUTE="${RATE_LIMIT_PER_MINUTE:-60}"
TRUST_FORWARDED_FOR="${TRUST_FORWARDED_FOR:-false}"

# Fall back from /data/models to /models if /data is not mounted.
if [ ! -d "$(dirname "$MODEL_DIR")" ]; then
    echo "[start.sh] WARNING: parent of MODEL_DIR ($MODEL_DIR) does not exist."
    echo "[start.sh] Falling back to /models inside the container."
    echo "[start.sh] Models will be re-downloaded on every container restart."
    echo "[start.sh] Mount a persistent volume at /data to avoid this."
    MODEL_DIR="/models"
fi

TARGET_DIR="${MODEL_DIR}/${MODEL_REPO//\//_}"
mkdir -p "$TARGET_DIR"

# Determine auth status for the banner.
if [ -n "$API_KEYS" ]; then
    AUTH_STATUS="<set: $(echo "$API_KEYS" | tr ',' '\n' | wc -l) keys>"
elif [ -n "$API_KEY" ]; then
    AUTH_STATUS="<set: 1 key>"
else
    AUTH_STATUS="<NOT SET — public access!>"
fi

cat <<EOF
===============================================================
  GGUF Server (Salad / Vast.ai / VPS)
===============================================================
  MODEL_REPO         = ${MODEL_REPO}
  MODEL_PATTERN      = ${MODEL_PATTERN}
  MMPROJ_PATTERN     = ${MMPROJ_PATTERN:-<none>}
  MODEL_ALIAS        = ${MODEL_ALIAS}
  TARGET_DIR         = ${TARGET_DIR}
  CTX_SIZE           = ${CTX_SIZE}
  OFFLOAD_MODE       = ${OFFLOAD_MODE}${OFFLOAD_VALUE:+ ($OFFLOAD_VALUE)}
  BATCH/UBATCH       = ${BATCH_SIZE}/${UBATCH_SIZE}
  PARALLEL           = ${PARALLEL}
  PUBLIC_PORT        = ${PUBLIC_PORT}
  AUTH               = ${AUTH_STATUS}
  TRUST_FWD_FOR      = ${TRUST_FORWARDED_FOR}
  RATE_LIMIT/min     = ${RATE_LIMIT_PER_MINUTE}
  HF_TOKEN           = $([ -n "${HF_TOKEN:-}" ] && echo '<set>' || echo '<not set — downloads will be slower>')
===============================================================
EOF

# -----------------------------------------------------------------------------
# Step 1: Download the model if needed.
# -----------------------------------------------------------------------------

find_main_gguf() {
    find "$TARGET_DIR" -maxdepth 2 -type f -name "*.gguf" \
        ! -name "mmproj*" ! -name "*mmproj*" 2>/dev/null | head -n 1
}

find_mmproj_gguf() {
    find "$TARGET_DIR" -maxdepth 2 -type f -name "mmproj*.gguf" 2>/dev/null | head -n 1
}

MODEL_FILE="$(find_main_gguf || true)"

if [ -z "$MODEL_FILE" ]; then
    echo "[start.sh] Model not found in cache. Downloading from ${MODEL_REPO}..."

    download_args=(
        "$MODEL_REPO"
        --local-dir "$TARGET_DIR"
        --include "$MODEL_PATTERN"
    )
    if [ -n "$MMPROJ_PATTERN" ]; then
        download_args+=(--include "$MMPROJ_PATTERN")
    fi

    hf download "${download_args[@]}"

    MODEL_FILE="$(find_main_gguf || true)"
    if [ -z "$MODEL_FILE" ]; then
        echo "[start.sh] ERROR: no GGUF file matched MODEL_PATTERN='${MODEL_PATTERN}'."
        exit 1
    fi
else
    echo "[start.sh] Using cached model: ${MODEL_FILE}"
fi

MMPROJ_FILE="$(find_mmproj_gguf || true)"
if [ -n "$MMPROJ_FILE" ]; then
    echo "[start.sh] Found multimodal projector: ${MMPROJ_FILE}"
fi

# -----------------------------------------------------------------------------
# Step 2: Build llama-server arguments.
# -----------------------------------------------------------------------------

llama_args=(
    --model "$MODEL_FILE"
    --alias "$MODEL_ALIAS"
    --host 127.0.0.1
    --port "$LLAMA_INTERNAL_PORT"
    --ctx-size "$CTX_SIZE"
    --batch-size "$BATCH_SIZE"
    --ubatch-size "$UBATCH_SIZE"
    --parallel "$PARALLEL"
    --cache-type-k "$CACHE_TYPE_K"
    --cache-type-v "$CACHE_TYPE_V"
    --jinja
    --no-context-shift
)

case "$OFFLOAD_MODE" in
    auto)
        llama_args+=(--fit-target "$FIT_TARGET_MB")
        echo "[start.sh] Offload: auto-fit (recommended)."
        ;;
    ngl)
        ngl_value="${OFFLOAD_VALUE:-999}"
        llama_args+=(--n-gpu-layers "$ngl_value")
        echo "[start.sh] Offload: classic ngl=${ngl_value}."
        ;;
    cmoe)
        llama_args+=(--cpu-moe)
        echo "[start.sh] Offload: cmoe (all MoE experts on CPU)."
        ;;
    ncmoe)
        if [ -z "$OFFLOAD_VALUE" ]; then
            echo "[start.sh] ERROR: OFFLOAD_MODE=ncmoe requires OFFLOAD_VALUE."
            exit 1
        fi
        llama_args+=(--n-cpu-moe "$OFFLOAD_VALUE")
        echo "[start.sh] Offload: ncmoe=${OFFLOAD_VALUE}."
        ;;
    manual)
        echo "[start.sh] Offload: manual (rely on EXTRA_ARGS)."
        ;;
    *)
        echo "[start.sh] ERROR: unknown OFFLOAD_MODE='${OFFLOAD_MODE}'."
        echo "[start.sh] Valid: auto, ngl, cmoe, ncmoe, manual."
        exit 1
        ;;
esac

if [ "$FLASH_ATTN" = "on" ]; then
    llama_args+=(--flash-attn on)
fi

if [ -n "$MMPROJ_FILE" ]; then
    llama_args+=(--mmproj "$MMPROJ_FILE")
fi

if [ -n "$REASONING_FORMAT" ]; then
    llama_args+=(--reasoning-format "$REASONING_FORMAT")
fi

if [ -n "$EXTRA_ARGS" ]; then
    # shellcheck disable=SC2206
    extra=($EXTRA_ARGS)
    llama_args+=("${extra[@]}")
fi

# -----------------------------------------------------------------------------
# Step 3: Start llama-server in the background.
# -----------------------------------------------------------------------------

mkdir -p /var/log
echo "[start.sh] Launching llama-server on 127.0.0.1:${LLAMA_INTERNAL_PORT}..."
echo "[start.sh] Full command: llama-server ${llama_args[*]}"

/app/llama.cpp/llama-server "${llama_args[@]}" > /var/log/llama.log 2>&1 &
LLAMA_PID=$!

# Cleanup on exit.
cleanup() {
    echo "[start.sh] Shutting down..."
    if kill -0 "$LLAMA_PID" 2>/dev/null; then
        kill "$LLAMA_PID" 2>/dev/null || true
    fi
    if [ -n "${PROXY_PID:-}" ] && kill -0 "$PROXY_PID" 2>/dev/null; then
        kill "$PROXY_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[start.sh] Waiting for llama-server (up to 10 minutes)..."
READY=0
for i in $(seq 1 600); do
    if curl -sf "http://127.0.0.1:${LLAMA_INTERNAL_PORT}/health" > /dev/null 2>&1; then
        READY=1
        echo "[start.sh] llama-server ready after ${i}s."
        break
    fi
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
        echo "[start.sh] ERROR: llama-server died. Last 80 log lines:"
        tail -80 /var/log/llama.log
        exit 1
    fi
    sleep 1
done

if [ "$READY" -eq 0 ]; then
    echo "[start.sh] ERROR: llama-server didn't become ready within 600s."
    tail -80 /var/log/llama.log
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 4: Export config for the proxy and start it.
# -----------------------------------------------------------------------------

export UPSTREAM_URL="http://127.0.0.1:${LLAMA_INTERNAL_PORT}"
export PUBLIC_PORT
export API_KEY
export API_KEYS
export RATE_LIMIT_PER_MINUTE
export MODEL_ALIAS
export TRUST_FORWARDED_FOR

echo "[start.sh] Starting auth proxy on [::]:${PUBLIC_PORT} (IPv4+IPv6)..."

# IMPORTANT: --host :: makes the socket dual-stack (IPv4 + IPv6). Salad's
# Container Gateway routes incoming traffic over IPv6, so a process bound
# to 0.0.0.0 (IPv4-only) is unreachable from outside and Salad returns 503
# for every request. `::` works on every platform we target, so we always
# use it — IPv4-only platforms (Vast, RunPod, generic VPS) still accept
# IPv4 connections via the dual-stack mapping.
uvicorn proxy:app \
    --host :: \
    --port "$PUBLIC_PORT" \
    --workers 1 \
    --log-level info \
    --timeout-keep-alive 75 &
PROXY_PID=$!

# Exit when either process dies so the orchestrator can restart us.
wait -n "$LLAMA_PID" "$PROXY_PID"
EXIT_CODE=$?
echo "[start.sh] One of the processes exited with code ${EXIT_CODE}. Shutting down."
exit "$EXIT_CODE"
