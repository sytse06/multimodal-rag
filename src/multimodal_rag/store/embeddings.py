"""Text embedding via OpenRouter (OpenAI-compatible API)."""

import logging

from openai import OpenAI

logger = logging.getLogger(__name__)

# OpenRouter embedding endpoint is OpenAI-compatible
_BATCH_SIZE = 100


def embed_texts(
    texts: list[str],
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    model: str = "openai/text-embedding-3-small",
) -> list[list[float]]:
    """Embed a list of texts via OpenRouter, returning vectors.

    Batches requests to stay within API limits.
    """
    if not texts:
        return []

    client = OpenAI(api_key=api_key, base_url=base_url)
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        response = client.embeddings.create(input=batch, model=model)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    logger.info("Embedded %d texts with model %s", len(texts), model)
    return all_embeddings
