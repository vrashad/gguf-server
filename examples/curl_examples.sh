#!/bin/bash
# Curl examples for gguf-server.
# Usage: edit the variables below and run `bash examples/curl_examples.sh`.

URL="${GGUF_SERVER_URL:-http://localhost:8080}"
KEY="${GGUF_SERVER_KEY:-}"

AUTH_HEADER=""
if [ -n "$KEY" ]; then
    AUTH_HEADER="-H \"Authorization: Bearer $KEY\""
fi

echo "=== Health check ==="
curl -sS "$URL/health"
echo

echo "=== List models ==="
eval curl -sS $AUTH_HEADER "$URL/v1/models"
echo

echo "=== Chat completion ==="
eval curl -sS $AUTH_HEADER -H \"Content-Type: application/json\" \
    -X POST "$URL/v1/chat/completions" \
    -d \''{
        "model": "any",
        "messages": [{"role": "user", "content": "Say hello in Azerbaijani."}],
        "max_tokens": 50,
        "temperature": 0.7
    }'\'
echo

echo "=== Streaming chat completion ==="
eval curl -sSN $AUTH_HEADER -H \"Content-Type: application/json\" \
    -X POST "$URL/v1/chat/completions" \
    -d \''{
        "model": "any",
        "messages": [{"role": "user", "content": "Count from 1 to 3."}],
        "max_tokens": 50,
        "stream": true
    }'\'
echo
