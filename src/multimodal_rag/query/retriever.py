"""Retrieval: embed query, search Weaviate, return SearchResults."""

import logging

from multimodal_rag.models.chunks import SourceType
from multimodal_rag.models.query import SearchResult
from multimodal_rag.store.weaviate import WeaviateStore

logger = logging.getLogger(__name__)


def _distance_to_score(distance: float | None) -> float:
    """Convert Weaviate cosine distance to a 0-1 relevance score."""
    if distance is None:
        return 0.0
    # Weaviate cosine distance: 0 = identical, 2 = opposite
    return max(0.0, 1.0 - distance)


def retrieve(
    query: str,
    store: WeaviateStore,
    top_k: int = 5,
) -> list[SearchResult]:
    """Embed query, search Weaviate, return ranked SearchResults."""
    raw_results = store.search(query, top_k=top_k)

    results: list[SearchResult] = []
    for hit in raw_results:
        score = _distance_to_score(hit.get("_distance"))
        ts = hit.get("timestamp_seconds")
        results.append(
            SearchResult(
                text=str(hit.get("text", "")),
                source_type=SourceType(str(hit.get("source_type", "web"))),
                source_url=str(hit.get("source_url", "")),
                source_name=str(hit.get("source_name", "")),
                timestamp_seconds=int(ts) if ts is not None else None,
                section_heading=str(hit.get("section_heading") or "")
                or None,
                relevance_score=score,
            )
        )

    logger.info(
        "Retrieved %d results for query: %.60s...",
        len(results),
        query,
    )
    return results


def format_context(results: list[SearchResult]) -> str:
    """Format retrieved results as numbered context for the LLM prompt."""
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        label = r.citation_label
        parts.append(f"[{i}] {label}\n{r.text}")
    return "\n\n".join(parts)
