#!/bin/bash
# =============================================================================
# Entrypoint for the GGUF server (Salad / Vast.ai / VPS / bare-metal).
#
# Responsibilities:
#   1. Resolve model configuration from environment variables.
#   2. Download the GGUF model from Hugging Face if not already cached.
#   3. Start llama-server on 127.0.0.1:8081 (internal only).
#   4. Start the FastAPI auth proxy on 0.0.0.0:8080 (public).
#   5. Wait for either process and exit if any dies.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (all optional; sensible defaults provided)
# -----------------------------------------------------------------------------

# Hugging Face repo with GGUF files.
MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.6-35B-A3B-GGUF}"

# Glob pattern for the main model file.
MODEL_PATTERN="${MODEL_PATTERN:-*UD-Q4_K_XL*}"

# Glob for multimodal projector (leave empty for text-only models).
MMPROJ_PATTERN="${MMPROJ_PATTERN:-}"

# Model alias exposed via the OpenAI API.
MODEL_ALIAS="${MODEL_ALIAS:-$(basename "${MODEL_REPO}")}"

# Where to cache models. Mount a persistent volume here on platforms
# that support it (Salad container storage, Vast.ai mounted volumes).
MODEL_DIR="${MODEL_DIR:-/data/models}"

# llama-server runtime parameters
LLAMA_INTERNAL_PORT="${LLAMA_INTERNAL_PORT:-8081}"
CTX_SIZE="${CTX_SIZE:-16384}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
PARALLEL="${PARALLEL:-1}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
FLASH_ATTN="${FLASH_ATTN:-on}"

# Reasoning format for hybrid-thinking models (set empty to disable).
REASONING_FORMAT="${REASONING_FORMAT:-deepseek}"

# Extra raw arguments appended to llama-server (e.g. sampling overrides).
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Public proxy parameters
PUBLIC_PORT="${PUBLIC_PORT:-8080}"

# API key for the public endpoint. If unset, the proxy runs WITHOUT auth
# (DANGEROUS for public deployments — only use this on trusted networks).
API_KEY="${API_KEY:-}"

# Per-IP rate limit (requests / minute). 0 disables rate limiting.
RATE_LIMIT_PER_MINUTE="${RATE_LIMIT_PER_MINUTE:-60}"

# Make sure the model cache directory exists. Falls back to /models inside
# the container if /data is not mounted.
if [ ! -d "$(dirname "$MODEL_DIR")" ]; then
    echo "[start.sh] WARNING: parent of MODEL_DIR ($MODEL_DIR) does not exist."
    echo "[start.sh] Falling back to /models inside the container."
    echo "[start.sh] Models will be re-downloaded on every container restart."
    echo "[start.sh] Mount a persistent volume at /data to avoid this."
    MODEL_DIR="/models"
fi

TARGET_DIR="${MODEL_DIR}/${MODEL_REPO//\//_}"
mkdir -p "$TARGET_DIR"

echo "==============================================================="
echo "  GGUF Server (Salad / Vast.ai / VPS)"
echo "==============================================================="
echo "  MODEL_REPO       = ${MODEL_REPO}"
echo "  MODEL_PATTERN    = ${MODEL_PATTERN}"
echo "  MMPROJ_PATTERN   = ${MMPROJ_PATTERN:-<none>}"
echo "  MODEL_ALIAS      = ${MODEL_ALIAS}"
echo "  TARGET_DIR       = ${TARGET_DIR}"
echo "  CTX_SIZE         = ${CTX_SIZE}"
echo "  PARALLEL         = ${PARALLEL}"
echo "  PUBLIC_PORT      = ${PUBLIC_PORT}"
echo "  API_KEY          = $([ -n "$API_KEY" ] && echo '<set>' || echo '<NOT SET — public access!>')"
echo "  RATE_LIMIT/min   = ${RATE_LIMIT_PER_MINUTE}"
echo "==============================================================="

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
    --n-gpu-layers "$N_GPU_LAYERS"
    --parallel "$PARALLEL"
    --cache-type-k "$CACHE_TYPE_K"
    --cache-type-v "$CACHE_TYPE_V"
    --jinja
    --no-context-shift
)

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
/app/llama.cpp/llama-server "${llama_args[@]}" > /var/log/llama.log 2>&1 &
LLAMA_PID=$!

# Cleanup on exit
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

echo "[start.sh] Waiting for llama-server to become ready (up to 10 minutes)..."
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
# Step 4: Export config for the proxy and start it in the foreground.
# -----------------------------------------------------------------------------

export UPSTREAM_URL="http://127.0.0.1:${LLAMA_INTERNAL_PORT}"
export PUBLIC_PORT
export API_KEY
export RATE_LIMIT_PER_MINUTE
export MODEL_ALIAS

echo "[start.sh] Starting auth proxy on 0.0.0.0:${PUBLIC_PORT}..."

# Run proxy in background and wait on either process — if llama-server
# crashes mid-flight we want to exit so the orchestrator restarts us.
uvicorn proxy:app \
    --host 0.0.0.0 \
    --port "$PUBLIC_PORT" \
    --workers 1 \
    --log-level info \
    --timeout-keep-alive 75 &
PROXY_PID=$!

# Wait until either process exits, then exit with that code.
wait -n "$LLAMA_PID" "$PROXY_PID"
EXIT_CODE=$?
echo "[start.sh] One of the processes exited with code ${EXIT_CODE}. Shutting down."
exit "$EXIT_CODE"
