FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl zstd \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && python -m pip install --no-cache-dir uv

RUN useradd --create-home --uid 1000 user \
    && mkdir -p /home/user/app \
    && chown user:user /home/user/app

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_BASE_URL=http://localhost:11434 \
    OLLAMA_MODELS=/home/user/.ollama/models

WORKDIR /home/user/app

COPY --chown=user pyproject.toml uv.lock README.md ./
USER user

RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=user src ./src
COPY --chown=user app.py space-entrypoint.sh ./

RUN uv sync --frozen --no-dev

# Bake the query-embedding model into the image: it's small, fixed, and only
# changes when this line changes, so there's no reason to fetch it at
# runtime or keep it in a persistent bucket. Space restarts (which happen on
# every deploy) then need no network round-trip before serving.
RUN mkdir -p "$OLLAMA_MODELS" && \
    (ollama serve &) && \
    for i in $(seq 1 30); do \
        curl --fail --silent http://127.0.0.1:11434/api/tags >/dev/null && break; \
        sleep 1; \
    done && \
    ollama pull nomic-embed-text

ENV PATH=/home/user/app/.venv/bin:$PATH

EXPOSE 7860

ENTRYPOINT ["./space-entrypoint.sh"]
