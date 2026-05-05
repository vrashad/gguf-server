"""
Authentication and rate-limiting proxy for llama-server.

Sits in front of llama-server and adds:

  - Bearer-token authentication (one or many shared API keys).
  - Per-IP rate limiting with IPv6 /64-prefix grouping.
  - Streaming response support (SSE passthrough for /v1/chat/completions).
  - Strict route allowlist (admin endpoints like /slots are not exposed).
  - Per-request access log with IP, key fingerprint, tokens, and latency.

Configuration (all read from environment):

  UPSTREAM_URL              Base URL of the local llama-server
                            (default http://127.0.0.1:8081).
  PUBLIC_PORT               Port to listen on (default 8080).
  API_KEY                   Single shared secret. Empty disables auth.
  API_KEYS                  Comma-separated list of valid keys (overrides
                            API_KEY when both are set). Lets you rotate
                            individual credentials without touching others.
  RATE_LIMIT_PER_MINUTE     Per-IP request budget. 0 disables rate limiting.
  MODEL_ALIAS               Name reported via /v1/models.
  TRUST_FORWARDED_FOR       If 'true', honour the X-Forwarded-For header
                            for client-IP extraction. Only enable when an
                            actual reverse proxy (Caddy, nginx, Cloudflare)
                            sits in front — otherwise clients can spoof IPs.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import logging
import os
import sys
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Set

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse


# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------

# Plain text format on stdout — Salad / Vast / Docker capture this and
# present it in their UIs. Structured logging can be added later if needed.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("proxy")
access_log = logging.getLogger("proxy.access")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

UPSTREAM_URL = os.environ.get("UPSTREAM_URL", "http://127.0.0.1:8081")
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "local-model")
TRUST_FORWARDED_FOR = os.environ.get("TRUST_FORWARDED_FOR", "").lower() == "true"


def _load_api_keys() -> Set[str]:
    """Collect API keys from API_KEYS (preferred) or API_KEY (legacy)."""
    raw_multi = os.environ.get("API_KEYS", "").strip()
    raw_single = os.environ.get("API_KEY", "").strip()
    keys: Set[str] = set()
    if raw_multi:
        keys.update(k.strip() for k in raw_multi.split(",") if k.strip())
    if raw_single:
        keys.add(raw_single)
    return keys


API_KEYS: Set[str] = _load_api_keys()
AUTH_ENABLED = bool(API_KEYS)

if not AUTH_ENABLED:
    log.warning("API_KEY/API_KEYS is not set — running WITHOUT authentication.")
    log.warning("Anyone with network access to this port can use the model.")
else:
    log.info("Authentication enabled with %d key(s).", len(API_KEYS))


# Routes the proxy is willing to forward. Anything else returns 404.
# We deliberately do NOT expose llama-server admin endpoints (/slots,
# /props, /metrics) to the public.
ALLOWED_ROUTES = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/models",
}


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
    """Best-effort client IP for rate-limit bucketing.

    For IPv6 addresses we collapse to the /64 prefix because most ISPs hand
    out at least a /64 to each customer; without this, a single user's
    rotating IPv6 source addresses each get their own rate-limit budget.

    X-Forwarded-For is only honoured when TRUST_FORWARDED_FOR=true to avoid
    trivial spoofing when the proxy is exposed directly.
    """
    raw = ""
    if TRUST_FORWARDED_FOR:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            raw = forwarded.split(",")[0].strip()
    if not raw:
        raw = request.client.host if request.client else "unknown"

    try:
        addr = ipaddress.ip_address(raw)
    except ValueError:
        return raw

    if isinstance(addr, ipaddress.IPv6Address):
        # Group every IPv6 address in a /64 under one bucket.
        return str(ipaddress.ip_network(f"{raw}/64", strict=False).network_address)
    return raw


def _key_fingerprint(token: str) -> str:
    """Short hash of the API key, safe to put in logs."""
    return hashlib.sha256(token.encode()).hexdigest()[:8]


def _check_auth(authorization: Optional[str]) -> Optional[str]:
    """Validate the Bearer token. Returns the key fingerprint (for logs)."""
    if not AUTH_ENABLED:
        return None
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization[7:].strip()
    if token not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return _key_fingerprint(token)


async def _check_rate(client_ip: str) -> None:
    if not await limiter.check(client_ip):
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

    started = time.monotonic()
    client_ip = _client_ip(request)
    key_fp = _check_auth(authorization)
    await _check_rate(client_ip)

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
            status = 200
            try:
                async with client.stream(
                    "POST", route, content=body, headers=forwarded_headers
                ) as upstream:
                    if upstream.status_code != 200:
                        status = upstream.status_code
                        err = await upstream.aread()
                        yield (
                            f'data: {{"error":"upstream {status}: '
                            f'{err.decode("utf-8", "replace")}"}}\n\n'
                        ).encode()
                        return
                    async for chunk in upstream.aiter_raw():
                        yield chunk
            except httpx.HTTPError as exc:
                status = 502
                yield f'data: {{"error":"upstream connection failed: {exc}"}}\n\n'.encode()
            finally:
                elapsed = time.monotonic() - started
                access_log.info(
                    "%s key=%s ip=%s route=%s stream=1 status=%d duration=%.3fs",
                    request.method, key_fp or "-", client_ip, route, status, elapsed,
                )

        return StreamingResponse(relay(), media_type="text/event-stream")

    # Non-streaming
    try:
        upstream = await client.post(route, content=body, headers=forwarded_headers)
    except httpx.HTTPError as exc:
        elapsed = time.monotonic() - started
        access_log.warning(
            "%s key=%s ip=%s route=%s stream=0 status=502 duration=%.3fs error=%s",
            request.method, key_fp or "-", client_ip, route, elapsed, exc,
        )
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")

    elapsed = time.monotonic() - started
    payload = upstream.json() if upstream.content else {}

    # Capture token usage for logs when llama-server reports it.
    usage = (payload or {}).get("usage") or {}
    pt = usage.get("prompt_tokens", "-")
    ct = usage.get("completion_tokens", "-")
    access_log.info(
        "%s key=%s ip=%s route=%s stream=0 status=%d duration=%.3fs prompt_tokens=%s completion_tokens=%s",
        request.method, key_fp or "-", client_ip, route, upstream.status_code,
        elapsed, pt, ct,
    )

    return JSONResponse(content=payload, status_code=upstream.status_code)


# -----------------------------------------------------------------------------
# Lifecycle hooks
# -----------------------------------------------------------------------------

@app.on_event("shutdown")
async def _shutdown() -> None:
    await client.aclose()
