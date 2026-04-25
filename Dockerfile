# syntax=docker/dockerfile:1.6
# Universal GGUF server for non-serverless platforms (Salad, Vast.ai,
# Hetzner GPU, bare-metal). Exposes an OpenAI-compatible HTTP API on
# port 8080 with optional API-key authentication.
#
# Unlike the RunPod variant, this image has no RunPod SDK — it's a
# self-contained server that just runs and serves requests.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# -----------------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        git \
        ca-certificates \
        libcurl4-openssl-dev \
        python3 \
        python3-pip \
        python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# -----------------------------------------------------------------------------
# Build llama.cpp with CUDA support for common GPU architectures.
#   86 = Ampere (RTX 3090 / A5000 / A40)
#   89 = Ada    (RTX 4090 / L40S)
#   90 = Hopper (H100 / H200)
# -----------------------------------------------------------------------------
ARG LLAMA_CPP_REF=master
RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} \
        https://github.com/ggml-org/llama.cpp /tmp/llama.cpp \
    && cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="86;89;90" \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_CURL=ON \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/llama.cpp/build \
        --config Release \
        -j $(nproc) \
        --target llama-server llama-cli llama-gguf-split \
    && mkdir -p /app/llama.cpp \
    && cp /tmp/llama.cpp/build/bin/llama-* /app/llama.cpp/ \
    && rm -rf /tmp/llama.cpp

# -----------------------------------------------------------------------------
# Python dependencies for the auth proxy
# -----------------------------------------------------------------------------
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# -----------------------------------------------------------------------------
# Application code
# -----------------------------------------------------------------------------
COPY proxy.py /app/
COPY start.sh /app/
RUN chmod +x /app/start.sh

# Expose the public port. The internal llama-server stays on 127.0.0.1.
EXPOSE 8080

# Healthcheck for orchestrators (Salad/Vast.ai don't use HEALTHCHECK directly,
# but it helps with `docker run` and image testing).
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["/app/start.sh"]
