"""Text embedding via LangChain Embeddings interface."""

import logging

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

_BATCH_SIZE = 5
_MAX_WORDS = 800


def _truncate(text: str, max_words: int = _MAX_WORDS) -> str:
    """Truncate text to max_words to stay within model context limits."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def embed_texts(
    texts: list[str],
    embeddings: Embeddings,
) -> list[list[float]]:
    """Embed a list of texts, returning vectors.

    Batches requests to stay within API limits.
    """
    if not texts:
        return []

    safe_texts = [_truncate(t) for t in texts]
    all_embeddings: list[list[float]] = []
    for i in range(0, len(safe_texts), _BATCH_SIZE):
        batch = safe_texts[i : i + _BATCH_SIZE]
        batch_embeddings = embeddings.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    logger.info("Embedded %d texts", len(texts))
    return all_embeddings
