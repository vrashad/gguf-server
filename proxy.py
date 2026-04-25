"""
Authentication and rate-limiting proxy for llama-server.

Sits in front of llama-server and adds:

  - Bearer-token authentication (single shared API_KEY).
  - Per-IP token-bucket rate limiting.
  - Streaming response support (SSE passthrough for /v1/chat/completions).
  - Strict route allowlist (admin endpoints like /slots are not exposed).

Configuration (all read from environment):

  UPSTREAM_URL              Base URL of the local llama-server (default
                            http://127.0.0.1:8081).
  PUBLIC_PORT               Port to listen on (default 8080).
  API_KEY                   Shared secret. If empty, auth is disabled.
  RATE_LIMIT_PER_MINUTE     Per-IP request budget. 0 disables rate limiting.
  MODEL_ALIAS               Name reported via /v1/models.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

UPSTREAM_URL = os.environ.get("UPSTREAM_URL", "http://127.0.0.1:8081")
API_KEY = os.environ.get("API_KEY", "")
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "local-model")

# Routes the proxy is willing to forward. Anything else returns 404.
# We deliberately do NOT expose llama-server admin endpoints (/slots,
# /props, /metrics) to the public.
ALLOWED_ROUTES = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/models",
}

AUTH_ENABLED = bool(API_KEY)
if not AUTH_ENABLED:
    print("[proxy] WARNING: API_KEY is not set — running without authentication.")
    print("[proxy] Anyone with network access to this port can use the model.")


# -----------------------------------------------------------------------------
# Rate limiter (per-IP sliding window over the last 60 seconds)
# -----------------------------------------------------------------------------

class SlidingWindowLimiter:
    """Lightweight in-process per-IP rate limiter.

    For a single-instance deployment this is sufficient. For multi-instance
    setups put a real reverse proxy (nginx, Caddy, Traefik) in front instead.
    """

    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self.window: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic()
        cutoff = now - 60.0
        async with self._lock:
            bucket = self.window[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(now)
            return True


limiter = SlidingWindowLimiter(RATE_LIMIT_PER_MINUTE)


# -----------------------------------------------------------------------------
# HTTP client (long-lived)
# -----------------------------------------------------------------------------

# A single shared client is more efficient than recreating one per request.
# Timeouts are generous because individual generations can take minutes.
client = httpx.AsyncClient(
    base_url=UPSTREAM_URL,
    timeout=httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0),
    limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
)


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(title="GGUF Server", docs_url=None, redoc_url=None)


def _client_ip(request: Request) -> str:
    """Best-effort client IP extraction.

    On Salad/Vast.ai there is usually no upstream proxy, so request.client.host
    is the real client. If you put nginx/Cloudflare in front, prefer
    X-Forwarded-For.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_auth(authorization: Optional[str]) -> None:
    if not AUTH_ENABLED:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization[7:].strip()
    # Constant-time-ish compare. For a single-key system this is fine; for
    # multi-key systems, hash and compare against a set instead.
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


async def _check_rate(request: Request) -> None:
    ip = _client_ip(request)
    if not await limiter.check(ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({RATE_LIMIT_PER_MINUTE} req/min per IP)",
        )


# -----------------------------------------------------------------------------
# Public endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, object]:
    """Unauthenticated health probe.

    Reports both proxy liveness and upstream readiness so orchestrators can
    distinguish "container up but llama-server crashed" from a clean state.
    """
    upstream_ok = False
    try:
        r = await client.get("/health", timeout=5.0)
        upstream_ok = r.status_code == 200
    except httpx.HTTPError:
        upstream_ok = False
    status_code = 200 if upstream_ok else 503
    return JSONResponse(
        {"proxy": "ok", "upstream": "ok" if upstream_ok else "down"},
        status_code=status_code,
    )


@app.get("/v1/models")
async def models(authorization: Optional[str] = Header(None)) -> Dict[str, object]:
    _check_auth(authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ALIAS,
                "object": "model",
                "owned_by": "local",
                "created": int(time.time()),
            }
        ],
    }


@app.post("/v1/{path:path}")
async def proxy_post(
    path: str,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Forward POST requests to the relevant llama-server endpoint."""
    route = f"/v1/{path}"
    if route not in ALLOWED_ROUTES:
        raise HTTPException(status_code=404, detail=f"Route {route} not found")

    _check_auth(authorization)
    await _check_rate(request)

    body = await request.body()

    # Detect streaming requests by inspecting the JSON body. We avoid full
    # JSON parsing on the hot path — a substring check on small payloads
    # is good enough for chat completions.
    is_stream = b'"stream":true' in body or b'"stream": true' in body

    # Pass through Content-Type and similar headers, but strip auth/host.
    forwarded_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in {"host", "authorization", "content-length"}
    }

    if is_stream:
        async def relay():
            try:
                async with client.stream(
                    "POST", route, content=body, headers=forwarded_headers
                ) as upstream:
                    if upstream.status_code != 200:
                        err = await upstream.aread()
                        yield f"data: {{\"error\":\"upstream {upstream.status_code}: {err.decode('utf-8', 'replace')}\"}}\n\n".encode()
                        return
                    async for chunk in upstream.aiter_raw():
                        yield chunk
            except httpx.HTTPError as exc:
                yield f"data: {{\"error\":\"upstream connection failed: {exc}\"}}\n\n".encode()

        return StreamingResponse(relay(), media_type="text/event-stream")

    try:
        upstream = await client.post(route, content=body, headers=forwarded_headers)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")

    return JSONResponse(
        content=upstream.json() if upstream.content else {},
        status_code=upstream.status_code,
    )


# -----------------------------------------------------------------------------
# Lifecycle hooks
# -----------------------------------------------------------------------------

@app.on_event("shutdown")
async def _shutdown() -> None:
    await client.aclose()
