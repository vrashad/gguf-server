# gguf-server

A **model-agnostic**, OpenAI-compatible inference server for any GGUF model. Designed for non-serverless GPU platforms — [Salad](https://salad.com), [Vast.ai](https://vast.ai), [Hetzner GPU](https://www.hetzner.com), or any bare-metal box with an NVIDIA GPU.

Built on top of [llama.cpp](https://github.com/ggml-org/llama.cpp). Drop-in compatible with the official `openai` Python SDK.

## Features

- **OpenAI-compatible API.** Point any OpenAI client at it unchanged — `client.chat.completions.create(...)` just works.
- **Model-agnostic image.** No weights baked in; choose the model at container start time via env vars.
- **Auto-fit GPU offload.** Uses llama.cpp's `--fit-target` mode by default, which auto-detects MoE vs. Dense and picks the optimal tensor placement. Manual overrides available via `OFFLOAD_MODE`.
- **Tuned prompt processing.** Default `BATCH_SIZE`/`UBATCH_SIZE` of 4096 give ~5x faster prefill than llama.cpp defaults.
- **Authentication.** Single key (`API_KEY`) or rotating multi-key (`API_KEYS`).
- **Per-IP rate limiting** with IPv6 /64-prefix grouping.
- **Streaming responses** (Server-Sent Events).
- **Per-request access logging** (IP, key fingerprint, tokens, latency).
- **Salad-ready out of the box.** Listens on IPv6 dual-stack, which Salad's Container Gateway requires.
- **CUDA builds** for Ampere (SM 8.6), Ada (SM 8.9), and Hopper (SM 9.0) — RTX 3090 / 4090 / L40S / H100.
- **Hybrid-thinking models** supported (Qwen3.x, DeepSeek-R1).

## Architecture

```
                     +-----------------------------------+
                     |  Container                         |
                     |                                    |
   Internet ──┐      |   [::]:8080  FastAPI proxy         |
              └────► |              (auth + rate limit)   |
                     |                  │                 |
                     |                  ▼ 127.0.0.1:8081  |
                     |               llama-server         |
                     |                  │                 |
                     |                  ▼                 |
                     |                NVIDIA GPU          |
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
├── proxy.py            # FastAPI auth proxy.
├── requirements.txt
├── examples/
│   ├── openai_client.py
│   └── curl_examples.sh
├── .dockerignore
├── .gitignore
└── README.md
```

## Quick start (using the prebuilt image)

```
vrashad/gguf-server:latest
```

> If you forked this project, replace the image name with your own.

### Run locally

Requires Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
docker run --rm --gpus all \
    -p 8080:8080 \
    -v "$PWD/.cache:/data" \
    -e MODEL_REPO=unsloth/Qwen3.6-35B-A3B-GGUF \
    -e MODEL_PATTERN='*UD-Q4_K_XL*' \
    -e API_KEY="$(openssl rand -hex 32)" \
    -e HF_TOKEN="$YOUR_HF_TOKEN" \
    vrashad/gguf-server:latest
```

First start: ~5 minutes (model download). Subsequent starts: ~30 seconds (cached).

```bash
curl http://localhost:8080/health
# {"proxy":"ok","upstream":"ok"}
```

## Configuration

All configuration is via environment variables.

### Model selection

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_REPO` | `unsloth/Qwen3.6-35B-A3B-GGUF` | Hugging Face repo containing GGUF files. |
| `MODEL_PATTERN` | `*UD-Q4_K_XL*` | Glob for the main model file. Quote in shells: `'*Q4_K_M*'`. |
| `MMPROJ_PATTERN` | *(empty)* | Glob for the multimodal projector. Empty for text-only. |
| `MODEL_ALIAS` | basename of `MODEL_REPO` | Model name reported via `/v1/models`. |
| `MODEL_DIR` | `/data/models` | Where to cache models. Mount a volume for persistence. |
| `HF_TOKEN` | *(empty)* | Hugging Face read token. **Strongly recommended** — without it downloads are heavily rate-limited and can take hours instead of minutes. |

### Performance tuning

| Variable | Default | Description |
| --- | --- | --- |
| `OFFLOAD_MODE` | `auto` | One of `auto` / `ngl` / `cmoe` / `ncmoe` / `manual`. See below. |
| `OFFLOAD_VALUE` | *(empty)* | Numeric value for `ngl` or `ncmoe` modes. |
| `FIT_TARGET_MB` | `1024` | VRAM (MB) to keep free in `auto` mode. |
| `BATCH_SIZE` | `4096` | Prompt processing batch. Larger = faster prefill, more VRAM. |
| `UBATCH_SIZE` | `4096` | Per-step micro-batch. Match `BATCH_SIZE`. |
| `CTX_SIZE` | `16384` | Context window. Higher = more KV cache memory. |
| `PARALLEL` | `1` | Concurrent inference slots. Each slot gets `CTX_SIZE / PARALLEL`. |
| `CACHE_TYPE_K` / `CACHE_TYPE_V` | `q8_0` | KV cache quantization. `q8_0` saves ~50% VRAM with negligible quality loss. |
| `FLASH_ATTN` | `on` | Set to anything else to disable. |
| `REASONING_FORMAT` | `deepseek` | Format for thinking-mode models. Empty to disable. |
| `EXTRA_ARGS` | *(empty)* | Raw extra arguments appended to `llama-server`. |

### Offload modes

`OFFLOAD_MODE` controls how the model is split between GPU VRAM and CPU RAM. For MoE models on small GPUs, the right choice can mean a 2-3x speedup.

- **`auto`** (recommended). llama.cpp inspects the model and picks `ngl` for Dense models, `ncmoe` for MoE. Reserves `FIT_TARGET_MB` of VRAM as headroom.
- **`cmoe`** — pushes all MoE expert tensors to CPU, keeps attention on GPU. Best when the full model doesn't fit on GPU. Runs Qwen3.6-35B-A3B on as little as 8 GB VRAM.
- **`ncmoe`** — pushes N MoE layers to CPU. Useful for partial offload; needs `OFFLOAD_VALUE`.
- **`ngl`** — classic `--n-gpu-layers`. Best for Dense models that fully fit.
- **`manual`** — pass nothing automatic; you control everything via `EXTRA_ARGS`.

### Authentication

| Variable | Default | Description |
| --- | --- | --- |
| `API_KEY` | *(empty)* | Single Bearer token. |
| `API_KEYS` | *(empty)* | Comma-separated list of valid keys. Overrides `API_KEY` when set. Use to rotate or revoke individual keys without breaking everyone. |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-IP request budget over a sliding 60s window. `0` disables. IPv6 addresses are bucketed per /64 prefix. |
| `TRUST_FORWARDED_FOR` | `false` | Set to `true` only if a real reverse proxy (Caddy, nginx, Cloudflare) sits in front. Otherwise clients can spoof IPs. |

> **Always set `API_KEY` (or `API_KEYS`) for public deployments.** Without it any traffic to the URL can use your GPU.

### Misc

| Variable | Default | Description |
| --- | --- | --- |
| `PUBLIC_PORT` | `8080` | Port the proxy listens on. |
| `LLAMA_INTERNAL_PORT` | `8081` | Internal llama-server port (loopback only). |

## Recipes

### Maximum speed on RTX 3090 (24 GB) — Qwen3.6-35B-A3B

```
MODEL_REPO=unsloth/Qwen3.6-35B-A3B-GGUF
MODEL_PATTERN=*UD-Q4_K_XL*
MMPROJ_PATTERN=*mmproj-F16*
CTX_SIZE=32768
OFFLOAD_MODE=auto
BATCH_SIZE=4096
HF_TOKEN=hf_xxxxx
API_KEY=<random-32-bytes>
```

Expected: ~80-110 tok/s decode, ~2000+ tok/s prefill, model fully in VRAM.

### Cheaper / faster cold start (smaller quant)

```
MODEL_REPO=unsloth/Qwen3.6-35B-A3B-GGUF
MODEL_PATTERN=*UD-Q2_K_XL*
CTX_SIZE=16384
OFFLOAD_MODE=auto
HF_TOKEN=hf_xxxxx
API_KEY=<random-32-bytes>
```

Q2_K_XL ~13 GB downloads in ~5 minutes vs ~10 for Q4. Quality is still excellent thanks to Unsloth Dynamic 2.0.

### Big MoE on small GPU (RTX 3060 / 12 GB)

```
MODEL_REPO=unsloth/Qwen3.6-35B-A3B-GGUF
MODEL_PATTERN=*UD-Q4_K_XL*
CTX_SIZE=16384
OFFLOAD_MODE=cmoe
BATCH_SIZE=2048
UBATCH_SIZE=2048
HF_TOKEN=hf_xxxxx
```

`cmoe` lets the 21 GB model run on 12 GB VRAM. Throughput drops from ~100 to ~25-40 tok/s, but it works.

### Multi-user serving

```
MODEL_REPO=unsloth/Qwen3.6-35B-A3B-GGUF
MODEL_PATTERN=*UD-Q4_K_XL*
CTX_SIZE=32768          # divided across slots
PARALLEL=4              # 4 concurrent users, 8k context each
OFFLOAD_MODE=auto
RATE_LIMIT_PER_MINUTE=120
API_KEYS=key1,key2,key3 # one key per team member
```

## Deploying on Salad

Salad runs containers on distributed consumer GPUs — cheap but with variable reliability. Best for batch / non-critical workloads.

### Steps

1. **Salad Portal → Container Groups → Create**.
2. **Image source:** `vrashad/gguf-server:latest`.
3. **Replica count:** `1`.
4. **Resources:**
   - GPU: filter by 24 GB VRAM (RTX 3090 / 4090).
   - vCPU: 8+, RAM: 16 GB+.
5. **Networking → Container Gateway:** **enable**, **port `8080`**, authentication **disabled** (we handle auth ourselves).
6. **Container storage:** 50 GB. Mount at `/data`.
7. **Environment variables** — pick a recipe above. **Always include `HF_TOKEN` and `API_KEY`.**
8. **Health Probes:** disable Startup, Liveness, Readiness. Salad's defaults can kill the container during the long initial model download.
9. Deploy.

After the first start (~5-15 minutes for model download with `HF_TOKEN`) the API is reachable at the gateway URL Salad provides.

### Salad gotchas

- **IPv6 is mandatory.** Salad's Container Gateway only routes traffic over IPv6. This image already binds the proxy to `[::]:8080` (dual-stack) — that's why this works. Custom images that bind to `0.0.0.0` will get 503 for every external request.
- **`Container Gateway` settings cannot be edited on a running group.** If you put the wrong port in originally, you have to **Duplicate** the group with the right config and delete the old one. Don't waste time fighting the Edit dialog.
- **Persistent storage isn't really persistent on Community plan.** When a workload is reallocated to a new node (which happens often), the local `/data` cache is gone and the model re-downloads. Set `HF_TOKEN` so this doesn't take an hour.
- **CUDA driver 13.2 produces gibberish output** for Qwen3.6 (and likely other recent models). Run `nvidia-smi` from a web terminal session — if you see `CUDA Version: 13.2`, reallocate to get a different node. CUDA 12.4-12.8 is fine.
- **Generous timeouts.** Set the gateway's *Server Response Timeout* high (30000+ ms) so long generations don't get cut off mid-response.

## Deploying on Vast.ai

Vast.ai gives you classic SSH access to a rented GPU. Either:

### Option 1 — Vast Docker template

1. Templates → Create New Template → image `vrashad/gguf-server:latest`.
2. Docker options: `-p 8080:8080`.
3. Disk: 50 GB+.
4. Env vars: `MODEL_REPO`, `MODEL_PATTERN`, `API_KEY`, `HF_TOKEN`.
5. Rent a machine with 24 GB+ VRAM and CUDA driver ≥ 12.4.
6. Once running, Vast shows a public IP + port mapping. Test with:
   ```bash
   curl http://<vast-public-ip>:<mapped-port>/health
   ```

### Option 2 — SSH in and `docker run`

If you rent a generic Vast instance:

```bash
docker run -d --gpus all \
    --restart unless-stopped \
    -p 8080:8080 \
    -v /workspace/cache:/data \
    -e MODEL_REPO=unsloth/Qwen3.6-35B-A3B-GGUF \
    -e MODEL_PATTERN='*UD-Q4_K_XL*' \
    -e API_KEY=your-secret-here \
    -e HF_TOKEN=hf_xxxxx \
    --name gguf-server \
    vrashad/gguf-server:latest

docker logs -f gguf-server
```

## Deploying on a generic VPS / bare-metal

For HTTPS, put Caddy in front:

```
your-domain.example.com {
    reverse_proxy localhost:8080
}
```

Then in your `gguf-server` env: set `TRUST_FORWARDED_FOR=true` so rate limiting uses the real client IP from `X-Forwarded-For` instead of Caddy's localhost address.

## Using the API

The server speaks the OpenAI HTTP API verbatim. Every standard client works.

### curl

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

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-server/v1",
    api_key="your-api-key",
)

response = client.chat.completions.create(
    model="any",
    messages=[{"role": "user", "content": "Hi"}],
)
print(response.choices[0].message.content)
```

Streaming, function calling, embeddings — all work the same as against the OpenAI API.

### Hybrid-thinking models (Qwen3.x)

Qwen3.6 has reasoning enabled by default — it generates a `<think>...</think>` block before the answer. To disable for faster, more direct responses:

```python
response = client.chat.completions.create(
    model="any",
    messages=[{"role": "user", "content": "Hi"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

To read both the thinking trace and the final answer, check `response.choices[0].message.reasoning_content` (when present) alongside `.content`.

### Available endpoints

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/health` | Unauthenticated. Returns proxy + upstream status. |
| `GET` | `/v1/models` | Lists the loaded model. |
| `POST` | `/v1/chat/completions` | Standard OpenAI chat. Supports `stream: true`. |
| `POST` | `/v1/completions` | Legacy text completion. |
| `POST` | `/v1/embeddings` | Embeddings (only if the loaded model exposes them). |

llama-server admin routes (`/slots`, `/props`, `/metrics`) are intentionally **not** proxied.

## Logs

The proxy emits one access-log line per completed request:

```
2026-05-05T12:34:56 [INFO] proxy.access: POST key=a1b2c3d4 ip=2a01:4f8:c2c:abcd:: route=/v1/chat/completions stream=0 status=200 duration=2.481s prompt_tokens=42 completion_tokens=180
```

`key=` is the first 8 characters of a SHA-256 hash of the API key — safe to put in logs and lets you correlate usage per key when running multi-key.

## Building from source

```bash
git clone https://github.com/<your-username>/gguf-server.git
cd gguf-server
docker build -t gguf-server:local .
```

Build takes 20–40 minutes (most of it compiling llama.cpp's CUDA kernels). Subsequent builds use the layer cache.

## Troubleshooting

**External requests return 503; internal `curl http://127.0.0.1:8080/health` works.**
On Salad: Container Gateway requires IPv6. This image's proxy listens on `[::]:8080` already; if you've replaced `start.sh` make sure `uvicorn` runs with `--host ::`, not `--host 0.0.0.0`.

**`MODEL_PATTERN matched no files`.**
The glob is passed verbatim to `hf download --include`. Verify the pattern by browsing the repo's files on Hugging Face. Patterns are case-sensitive and require wildcards: `'*UD-Q4_K_XL*'`, not `UD-Q4_K_XL`.

**`CUDA error: out of memory` during model load.**
Either the chosen quant is too large or `CTX_SIZE` is too high. Try a smaller quant (`UD-Q3_K_XL` or `UD-Q2_K_XL`), reduce `CTX_SIZE`, switch to `OFFLOAD_MODE=cmoe`, or move to a larger GPU.

**Garbage output from the model.**
Run `nvidia-smi` inside the container — if `CUDA Version: 13.2`, that driver has known issues with Qwen3.6. Reallocate to a different node (Salad) or filter your offer by CUDA version (Vast).

**Server is reachable but `/v1/chat/completions` hangs.**
Check `docker logs <container>`. Usually the upstream llama-server crashed mid-request. The proxy will report `upstream: down` via `/health`.

**`429 Rate limit exceeded` immediately.**
Set `RATE_LIMIT_PER_MINUTE=0` to disable, or raise it. The limit is per-IP — if all your callers share a NAT, they share one bucket.

**Container restarts before model finishes downloading.**
Some platforms have aggressive startup health-checks. On Salad, **disable the Startup Probe**. Otherwise, increase the platform's grace period to ≥ 10 minutes.

**Model re-downloads on every container restart.**
You haven't mounted a persistent volume at `/data`. On Salad Community plan, the cache is also lost when a workload is reallocated to a different node — `HF_TOKEN` makes the re-download fast.

**Downloads are very slow (kB/s).**
Set `HF_TOKEN`. Without it, Hugging Face heavily rate-limits anonymous downloads.

## License

MIT. llama.cpp and individual GGUF models are governed by their own upstream licenses.
