"""Text embedding via LangChain Embeddings interface."""

import logging

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

_BATCH_SIZE = 5
_MAX_WORDS = 400
_RETRY_MAX_WORDS = 200


def _truncate(text: str, max_words: int = _MAX_WORDS) -> str:
    """Truncate text to max_words to stay within model context limits."""
    words = text.split()
    if len(words) <= max_words:
        return text
    logger.warning("Truncating text from %d to %d words", len(words), max_words)
    return " ".join(words[:max_words])


def embed_texts(
    texts: list[str],
    embeddings: Embeddings,
) -> list[list[float]]:
    """Embed a list of texts, returning vectors.

    Batches requests to stay within API limits.
    On context-length errors, retries the failing batch with aggressive truncation.
    """
    if not texts:
        return []

    safe_texts = [_truncate(t) for t in texts]
    all_embeddings: list[list[float]] = []
    for i in range(0, len(safe_texts), _BATCH_SIZE):
        batch = safe_texts[i : i + _BATCH_SIZE]
        try:
            batch_embeddings = embeddings.embed_documents(batch)
        except Exception as exc:
            if "context length" not in str(exc).lower():
                raise
            logger.warning(
                "Context length error on batch %d-%d, retrying with %d-word limit",
                i,
                i + len(batch),
                _RETRY_MAX_WORDS,
            )
            retry_batch = [_truncate(t, _RETRY_MAX_WORDS) for t in batch]
            batch_embeddings = embeddings.embed_documents(retry_batch)
        all_embeddings.extend(batch_embeddings)

    logger.info("Embedded %d texts", len(texts))
    return all_embeddings
