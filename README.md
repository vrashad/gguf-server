# gguf-server

A **model-agnostic**, OpenAI-compatible inference server for any GGUF model. Designed for non-serverless GPU platforms — [Salad](https://salad.com), [Vast.ai](https://vast.ai), [Hetzner GPU](https://www.hetzner.com), or any bare-metal box with an NVIDIA GPU.

Built on top of [llama.cpp](https://github.com/ggml-org/llama.cpp). Drop-in compatible with the official `openai` Python SDK.


## Features

- **OpenAI-compatible API** at the URL level. Point any OpenAI client at it unchanged — `client.chat.completions.create(...)` just works.
- **Model-agnostic image.** The Docker image has no model weights baked in; you choose the model at container start time via environment variables.
- **API-key authentication** out of the box. Bearer-token gating with optional per-IP rate limiting.
- **Streaming responses** (Server-Sent Events) for real-time chat.
- **CUDA builds** for Ampere (SM 8.6), Ada (SM 8.9), and Hopper (SM 9.0) — works on RTX 3090, 4090, L40S, H100.
- **Persistent model cache** when a volume is mounted at `/data`.
- **Hybrid-thinking models** supported (Qwen3.x, DeepSeek-R1).

## Architecture

```
                     +-----------------------------------+
                     |  Container                         |
                     |                                    |
   Internet ──┐      |   :8080  FastAPI proxy             |
              └────► |          (auth + rate limit)       |
                     |             │                      |
                     |             ▼ 127.0.0.1:8081       |
                     |          llama-server              |
                     |          (loads GGUF into VRAM)    |
                     |             │                      |
                     |             ▼                      |
                     |           NVIDIA GPU               |
                     +-----------------------------------+
                                    │
                            /data/models (mount)
                            persistent across restarts
```

`llama-server` only listens on localhost; the proxy is the single public entry point.

## Repository layout

```
.
├── Dockerfile          # Builds llama.cpp + installs the proxy.
├── start.sh            # Resolves env vars, downloads model, starts server + proxy.
├── proxy.py            # FastAPI auth proxy (Bearer + per-IP rate limit).
├── requirements.txt    # Python dependencies.
├── examples/
│   ├── openai_client.py    # Use the official OpenAI SDK against this server.
│   └── curl_examples.sh    # Plain curl examples.
├── .dockerignore
├── .gitignore
└── README.md
```

## Quick start (using the prebuilt image)

A prebuilt image is available on Docker Hub:

```
vrashad/gguf-server:latest
```

> If you forked this project, replace the image name with your own.

### Run locally for testing

Requires Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

```bash
docker run --rm --gpus all \
    -p 8080:8080 \
    -v "$PWD/.cache:/data" \
    -e MODEL_REPO=unsloth/Qwen3.6-35B-A3B-GGUF \
    -e MODEL_PATTERN='*UD-Q4_K_XL*' \
    -e API_KEY="$(openssl rand -hex 32)" \
    -e CTX_SIZE=16384 \
    vrashad/gguf-server:latest
```

The first start takes ~5 minutes (model download). Subsequent starts are ~30 seconds because the model is cached in `./.cache`.

Once the logs show `llama-server ready`, the API is available at `http://localhost:8080`. Test it:

```bash
curl http://localhost:8080/health
# {"proxy":"ok","upstream":"ok"}
```

## Configuration

All configuration is done through environment variables.

### Model selection

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_REPO` | `unsloth/Qwen3.6-35B-A3B-GGUF` | Hugging Face repo containing GGUF files. |
| `MODEL_PATTERN` | `*UD-Q4_K_XL*` | Glob for the main model file. Quote it in shells: `'*Q4_K_M*'`. |
| `MMPROJ_PATTERN` | *(empty)* | Glob for the multimodal projector. Leave empty for text-only models. |
| `MODEL_ALIAS` | basename of `MODEL_REPO` | Model name reported via `/v1/models`. |
| `MODEL_DIR` | `/data/models` | Where to cache models. Mount a volume here for persistence. |

### llama-server runtime

| Variable | Default | Description |
| --- | --- | --- |
| `CTX_SIZE` | `16384` | Context window. Higher values consume more VRAM for KV cache. |
| `N_GPU_LAYERS` | `999` | Layers offloaded to GPU. `999` = all. |
| `PARALLEL` | `1` | Concurrent slots in llama-server. Increase for higher throughput at the cost of per-request context. |
| `CACHE_TYPE_K` / `CACHE_TYPE_V` | `q8_0` | KV cache quantization. `q8_0` saves ~50% VRAM with negligible quality loss. |
| `FLASH_ATTN` | `on` | Set to anything else to disable. |
| `REASONING_FORMAT` | `deepseek` | Format for thinking-mode models. Empty to disable. |
| `EXTRA_ARGS` | *(empty)* | Raw extra arguments appended to `llama-server`. |

### Public proxy

| Variable | Default | Description |
| --- | --- | --- |
| `PUBLIC_PORT` | `8080` | Port the proxy listens on (publicly exposed). |
| `API_KEY` | *(empty)* | Bearer token required for all `/v1/*` endpoints. **If empty, the API is unauthenticated.** |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-IP request budget over a sliding 60-second window. `0` disables rate limiting. |

> **Always set `API_KEY` for public deployments.** Without it, anyone who can reach the port can use your GPU.

### Example configurations

**Qwen3.6-35B-A3B (MoE, fits on RTX 3090)**

```
MODEL_REPO=unsloth/Qwen3.6-35B-A3B-GGUF
MODEL_PATTERN=*UD-Q4_K_XL*
MMPROJ_PATTERN=*mmproj-F16*
CTX_SIZE=32768
EXTRA_ARGS=--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00 --presence-penalty 1.5
```

**Qwen3.6-27B (dense)**

```
MODEL_REPO=unsloth/Qwen3.6-27B-GGUF
MODEL_PATTERN=*UD-Q4_K_XL*
CTX_SIZE=16384
```

**Smaller text-only model**

```
MODEL_REPO=unsloth/Qwen2.5-14B-Instruct-GGUF
MODEL_PATTERN=*Q4_K_M*
CTX_SIZE=8192
REASONING_FORMAT=
```

## Deploying on Salad

Salad runs containers on distributed consumer GPUs — cheap but with variable reliability. Best for batch / non-critical workloads.

1. **Salad Portal → Container Groups → Create**.
2. **Image source:** `vrashad/gguf-server:latest` (public).
3. **Replica count:** `1` (or more if you want HA).
4. **Resources:**
   - GPU: filter by 24 GB VRAM (RTX 3090 / 4090).
   - CPU: 4 cores minimum.
   - RAM: 16 GB minimum (more if your model is large).
5. **Networking → Container gateway:** enable, set port to `8080`. This gives you a public URL like `https://<your-id>.salad.cloud`.
6. **Container storage:** add at least 50 GB. Mount it at `/data` so the model survives container restarts. (Salad's persistent storage is per-container-group.)
7. **Environment variables:** add `MODEL_REPO`, `MODEL_PATTERN`, `API_KEY`, etc. as listed above.
8. Deploy.

After the first start (~5 minutes for model download) the API is reachable at the gateway URL Salad shows.

> **Salad reliability note.** Workloads can be reallocated to a different node at any time. If your container moves, the local cache is gone — set up your `MODEL_DIR` on Salad's persistent storage to avoid redownloading 20 GB on every reallocation.

## Deploying on Vast.ai

Vast.ai gives you classic SSH access to a rented GPU machine. Two ways to deploy:

### Option 1: As a Docker template

1. Create an instance with the **`vrashad/gguf-server:latest`** image (Vast.ai → Templates → Create New Template, point at this image).
2. **Docker options:** request port mapping `-p 8080:8080`.
3. **Disk space:** 50 GB minimum (image + one model).
4. **Environment variables:** set `MODEL_REPO`, `API_KEY`, etc. through Vast.ai's env var fields.
5. **On-start script:** none needed — `start.sh` is the container's CMD.
6. Rent a machine with at least 24 GB VRAM.
7. After the instance starts, Vast.ai shows you the public IP and mapped port. Test with:
   ```bash
   curl http://<vast-public-ip>:<mapped-port>/health
   ```

### Option 2: SSH in and run docker manually

If you rent a generic Vast.ai instance (e.g. base Ubuntu image), SSH in and run:

```bash
docker run -d --gpus all \
    --restart unless-stopped \
    -p 8080:8080 \
    -v /workspace/cache:/data \
    -e MODEL_REPO=unsloth/Qwen3.6-35B-A3B-GGUF \
    -e MODEL_PATTERN='*UD-Q4_K_XL*' \
    -e API_KEY=your-secret-here \
    --name gguf-server \
    vrashad/gguf-server:latest

docker logs -f gguf-server  # watch model download + server startup
```

## Deploying on a generic VPS / bare-metal

Same as Vast.ai Option 2, plus you'll likely want HTTPS in front. The simplest stack is:

```
Internet ──► Caddy (port 443, automatic TLS) ──► gguf-server (port 8080)
```

Sample `Caddyfile`:

```
your-domain.example.com {
    reverse_proxy localhost:8080
}
```

Caddy auto-provisions a Let's Encrypt cert and proxies to the container. Combine with `API_KEY` for authenticated HTTPS access.

## Using the API

The server speaks the OpenAI HTTP API verbatim. Every standard client works.

### With curl

```bash
curl https://your-server/v1/chat/completions \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "any",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 200
    }'
```

### With the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-server/v1",
    api_key="your-api-key",
)

response = client.chat.completions.create(
    model="any",                    # ignored — only one model is loaded
    messages=[{"role": "user", "content": "Hi"}],
)
print(response.choices[0].message.content)
```

Streaming, function calling, and embeddings all work the same as against the OpenAI API.

### Available endpoints

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/health` | Unauthenticated. Returns proxy + upstream status. |
| `GET` | `/v1/models` | Lists the single loaded model. |
| `POST` | `/v1/chat/completions` | Standard OpenAI chat. Supports `stream: true`. |
| `POST` | `/v1/completions` | Legacy text completion. |
| `POST` | `/v1/embeddings` | Embeddings (only if the loaded model exposes them). |

llama-server admin routes (`/slots`, `/props`, `/metrics`) are intentionally **not** proxied to the public.

## Building from source

```bash
git clone https://github.com/<your-username>/gguf-server.git
cd gguf-server
docker build -t gguf-server:local .
```

The build takes 20–40 minutes (most of it compiling llama.cpp's CUDA kernels). Subsequent builds use Docker's layer cache and are much faster.

## Troubleshooting

**`MODEL_PATTERN matched no files`.**
The glob is passed verbatim to `hf download --include`. Verify the pattern by browsing the repo's files on Hugging Face. Patterns are case-sensitive and require wildcards: `'*UD-Q4_K_XL*'`, not `UD-Q4_K_XL`.

**`CUDA error: out of memory` during model load.**
Either the chosen quant is too large for the GPU, or `CTX_SIZE` is too high. Try a smaller quant (`UD-Q3_K_XL` instead of `UD-Q4_K_XL`), reduce `CTX_SIZE`, or move to a larger GPU.

**Server is reachable but `/v1/chat/completions` hangs.**
Check `docker logs <container>` — usually means the upstream llama-server crashed mid-request. The proxy will report it via `/health`.

**`429 Rate limit exceeded` immediately.**
Set `RATE_LIMIT_PER_MINUTE=0` to disable, or raise it. Note the limit is per-IP — if all your callers come from the same proxy/NAT, they share one bucket.

**Container restarts before model finishes downloading.**
Some platforms have aggressive startup health-checks. Increase the platform's startup grace period to at least 10 minutes, or pre-populate the `MODEL_DIR` volume from a fast machine before deploying.

**Model re-downloads on every container restart.**
You haven't mounted a persistent volume at `/data` (or wherever `MODEL_DIR` points). Without one, the model lives in the ephemeral container layer and is lost on restart.


## License

MIT. llama.cpp and individual GGUF models are governed by their own upstream licenses.
