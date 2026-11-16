"""Retrieval gate: did a content update push known-good sources out of the results?

Each canary question lists the sources that answer it. The gate retrieves for every
question against the hosted collection (the state before the update) and the local one
(the state after) and compares the best rank of an expected source. It calls no LLM, so
a run is deterministic, takes seconds and costs nothing. Answers are read by a person
only when a canary fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from multimodal_rag.models.config import AppSettings
from multimodal_rag.models.llm import create_embeddings
from multimodal_rag.models.query import SearchResult
from multimodal_rag.preflight import Check, Status, format_report
from multimodal_rag.query.retriever import retrieve
from multimodal_rag.store.weaviate import WeaviateStore

CANARIES_PATH = Path("config/canaries.yaml")
RANK_DROP_WARN = 3


class Canary(BaseModel):
    question: str
    expect: list[str] = Field(min_length=1)


class CanaryFile(BaseModel):
    canaries: list[Canary]


def load_canaries(path: Path = CANARIES_PATH) -> list[Canary]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return CanaryFile.model_validate(raw).canaries


def from_changed_pages(pages: list[tuple[str, str]]) -> list[Canary]:
    """Canaries for pages an update changed: each page's title must find the page."""
    return [Canary(question=title, expect=[url]) for title, url in pages]


def best_rank(results: list[SearchResult], expected: list[str]) -> int | None:
    """1-based position of the first expected source, or None if none is returned."""
    for position, result in enumerate(results, 1):
        if result.source_url in expected:
            return position
    return None


def evaluate(
    canary: Canary, before: int | None, after: int | None, top_k: int
) -> Check:
    question = canary.question
    name = question if len(question) <= 60 else question[:57] + "..."
    if after is None:
        history = (
            "and was absent before the update"
            if before is None
            else f"and was rank {before} before the update"
        )
        detail = f"no expected source in the top {top_k} {history}"
        return Check(name, Status.FAIL, detail)
    if before is None:
        return Check(name, Status.OK, f"rank {after} (not in the top {top_k} before)")
    if after - before >= RANK_DROP_WARN:
        return Check(name, Status.WARN, f"rank fell from {before} to {after}")
    return Check(name, Status.OK, f"rank {before} before, {after} after")


def run_gate(
    canaries: list[Canary], before: Any, after: Any, top_k: int
) -> list[Check]:
    """Compare two stores; ``before`` is the baseline, ``after`` the candidate."""
    checks = []
    for canary in canaries:
        ranks = [
            best_rank(retrieve(canary.question, store, top_k=top_k), canary.expect)
            for store in (before, after)
        ]
        checks.append(evaluate(canary, ranks[0], ranks[1], top_k))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=CANARIES_PATH)
    args = parser.parse_args()
    settings = AppSettings()
    embeddings = create_embeddings(settings)
    canaries = load_canaries(args.file)
    with (
        WeaviateStore(
            weaviate_url=settings.weaviate_hosted_url,
            embeddings=embeddings,
            weaviate_mode="cloud",
            weaviate_api_key=settings.weaviate_viewer_api_key.get_secret_value(),
        ) as hosted,
        WeaviateStore(
            weaviate_url=str(settings.weaviate_url), embeddings=embeddings
        ) as local,
    ):
        checks = run_gate(canaries, hosted, local, settings.top_k)
    print(format_report(checks))
    sys.exit(1 if any(c.status is Status.FAIL for c in checks) else 0)


if __name__ == "__main__":
    main()
