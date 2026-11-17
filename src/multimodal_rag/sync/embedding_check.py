"""Is the index embedded by the runtime that production queries with?

One look at the embedding runtime (version and weights) and at whether a sample of
stored vectors still equals a fresh embedding of its text. A mismatch means new chunks
would outrank old ones regardless of relevance, which is how a partial apply once failed
the retrieval gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.embeddings import Embeddings

from multimodal_rag.checks import Status
from multimodal_rag.models.config import AppSettings
from multimodal_rag.store.fingerprint import (
    EmbeddingFingerprint,
    embedding_fingerprint,
    pinned_ollama_version,
    runtime_problems,
)
from multimodal_rag.sync.reembed import verify_current

DRIFT_OK = 0.9999  # two runtimes of the same family agree to 0.99999
DRIFT_FAIL = 0.99
SAMPLE = 30


@dataclass(frozen=True)
class EmbeddingStatus:
    fingerprint: EmbeddingFingerprint
    runtime_problems: list[str]
    lowest: float
    mean: float


def inspect_embeddings(
    settings: AppSettings,
    collection: Any,
    embeddings: Embeddings,
    sample: int = SAMPLE,
) -> EmbeddingStatus:
    fingerprint = embedding_fingerprint(
        str(settings.embeddings.base_url), settings.embedding_model
    )
    lowest, mean = verify_current(collection, embeddings, sample)
    return EmbeddingStatus(
        fingerprint=fingerprint,
        runtime_problems=runtime_problems(fingerprint, pinned_ollama_version()),
        lowest=lowest,
        mean=mean,
    )


def drift_status(lowest: float) -> Status:
    if lowest >= DRIFT_OK:
        return Status.OK
    return Status.WARN if lowest >= DRIFT_FAIL else Status.FAIL


def apply_problems(status: EmbeddingStatus) -> list[str]:
    """Reasons an apply must refuse: a wrong runtime, or vectors that have drifted."""
    problems = list(status.runtime_problems)
    if status.lowest < DRIFT_OK:
        problems.append(
            f"stored vectors no longer match their text (lowest cosine"
            f" {status.lowest:.4f}); re-embed first with make sync-reembed"
        )
    return problems
