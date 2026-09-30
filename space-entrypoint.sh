#!/usr/bin/env bash
set -Eeuo pipefail

export WEAVIATE_MODE="${WEAVIATE_MODE:-cloud}"
export EMBEDDING_PROVIDER=ollama
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_BASE_URL="http://${OLLAMA_HOST}"

if [[ "$WEAVIATE_MODE" != "cloud" ]]; then
    echo "Hugging Face Space requires WEAVIATE_MODE=cloud" >&2
    exit 1
fi

: "${WEAVIATE_URL:?Set the Weaviate Cloud URL in Space Variables}"
: "${WEAVIATE_VIEWER_API_KEY:?Set the Weaviate viewer key in Space Secrets}"

ollama serve &
ollama_pid=$!
trap 'kill "$ollama_pid" 2>/dev/null || true' EXIT

ready=false
for _ in {1..60}; do
    if curl --fail --silent "http://${OLLAMA_HOST}/api/tags" >/dev/null; then
        ready=true
        break
    fi
    if ! kill -0 "$ollama_pid" 2>/dev/null; then
        echo "Ollama stopped before its API became ready" >&2
        exit 1
    fi
    sleep 1
done

if [[ "$ready" != true ]]; then
    echo "Ollama API did not become ready within 60 seconds" >&2
    exit 1
fi

python app.py
