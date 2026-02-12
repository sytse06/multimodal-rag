"""Text embedding via LangChain Embeddings interface."""

import logging

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100


def embed_texts(
    texts: list[str],
    embeddings: Embeddings,
) -> list[list[float]]:
    """Embed a list of texts, returning vectors.

    Batches requests to stay within API limits.
    """
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        batch_embeddings = embeddings.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    logger.info("Embedded %d texts", len(texts))
    return all_embeddings
