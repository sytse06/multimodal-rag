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


def _build_results(raw: list[dict]) -> list[SearchResult]:
    results = []
    for hit in raw:
        score = _distance_to_score(hit.get("_distance"))
        ts = hit.get("timestamp_seconds")
        results.append(
            SearchResult(
                text=str(hit.get("text", "")),
                source_type=SourceType(str(hit.get("source_type", "web"))),
                source_url=str(hit.get("source_url", "")),
                source_name=str(hit.get("source_name", "")),
                timestamp_seconds=int(ts) if ts is not None else None,
                section_heading=str(hit.get("section_heading") or "") or None,
                relevance_score=score,
            )
        )
    return results


def retrieve(
    query: str,
    store: WeaviateStore,
    top_k: int = 5,
) -> list[SearchResult]:
    """Embed query, search Weaviate, return ranked SearchResults.

    Fetches a larger candidate pool and guarantees at least half the results
    are video chunks, so video content is not crowded out by web chunks.
    """
    pool = _build_results(store.search(query, top_k=top_k * 4))

    videos = [r for r in pool if r.source_type == SourceType.VIDEO]
    web = [r for r in pool if r.source_type != SourceType.VIDEO]

    n_video = min(len(videos), (top_k + 1) // 2)
    n_web = min(len(web), top_k - n_video)

    selected = sorted(
        videos[:n_video] + web[:n_web],
        key=lambda r: r.relevance_score,
        reverse=True,
    )

    logger.info(
        "Retrieved %d results (%d video, %d web) for query: %.60s...",
        len(selected),
        n_video,
        n_web,
        query,
    )
    return selected


def format_context(results: list[SearchResult]) -> str:
    """Format retrieved results as numbered context for the LLM prompt."""
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        label = r.citation_label
        parts.append(f"[{i}] {label}\n{r.text}")
    return "\n\n".join(parts)
