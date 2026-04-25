"""
Example client using the official OpenAI SDK against gguf-server.

Unlike RunPod, this server is OpenAI-compatible at the URL level, so the
official `openai` Python package works unchanged — you just need to point
its base_url at your container's public URL.

Usage:

    pip install openai
    export GGUF_SERVER_URL=https://your-instance.salad.cloud:8080/v1
    export GGUF_SERVER_KEY=your-api-key

    python examples/openai_client.py
"""

from __future__ import annotations

import os
import sys

from openai import OpenAI


BASE_URL = os.environ.get("GGUF_SERVER_URL", "http://localhost:8080/v1")
API_KEY = os.environ.get("GGUF_SERVER_KEY", "no-key")

if not API_KEY or API_KEY == "no-key":
    print("Note: GGUF_SERVER_KEY is unset. Make sure your server is running")
    print("without authentication, or set the variable to your real key.")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


# -----------------------------------------------------------------------------
# Example 1: list available models
# -----------------------------------------------------------------------------

print("Available models:")
for m in client.models.list().data:
    print(f"  - {m.id}")


# -----------------------------------------------------------------------------
# Example 2: simple chat completion
# -----------------------------------------------------------------------------

print("\nNon-streaming response:")
response = client.chat.completions.create(
    model="any",  # the server only has one model — alias is ignored
    messages=[
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "What is the capital of Azerbaijan?"},
    ],
    temperature=0.7,
    max_tokens=100,
)
print(response.choices[0].message.content)


# -----------------------------------------------------------------------------
# Example 3: streaming
# -----------------------------------------------------------------------------

print("\nStreaming response:")
stream = client.chat.completions.create(
    model="any",
    messages=[{"role": "user", "content": "Count from 1 to 5."}],
    stream=True,
    max_tokens=50,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content if chunk.choices else None
    if delta:
        print(delta, end="", flush=True)
print()
