"""Recompute every stored vector from its stored text with the pinned runtime.

``make sync-reembed`` replaces only the vector of each local chunk, with the same input
``add_chunks`` uses, and leaves every property alone. ``DRY=1`` measures the drift
without writing. The index must come from one embedding function, the one production
embeds its queries with; a mix makes new chunks outrank old ones regardless of
relevance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import weaviate
from langchain_core.embeddings import Embeddings

from multimodal_rag.models.config import AppSettings
from multimodal_rag.models.llm import create_embeddings
from multimodal_rag.store.embeddings import embed_texts
from multimodal_rag.store.fingerprint import embedding_fingerprint
from multimodal_rag.store.snapshot import export_snapshot
from multimodal_rag.store.weaviate import COLLECTION_NAME

BATCH = 32
SAME_VECTOR = 0.99999  # a vector this close to its re-embedding is current
READ_BACK = 0.999999


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def vector_of(obj: Any) -> list[float]:
    vector = obj.vector
    return list(vector["default"] if isinstance(vector, dict) else vector)


def kind_of(properties: dict[str, Any]) -> str:
    return "video" if properties["source_type"] == "video" else "web"


def embedding_input(properties: dict[str, Any]) -> str:
    return f"{properties['source_name']}: {properties['text']}"


def properties_digest(objects: list[Any]) -> str:
    rows = sorted(
        (str(o.uuid), json.dumps(o.properties, sort_keys=True, default=str))
        for o in objects
    )
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


@dataclass
class KindStats:
    count: int = 0
    already_current: int = 0
    cosines: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float | int]:
        c = sorted(self.cosines)
        return {
            "count": self.count,
            "already_current": self.already_current,
            "mean_cosine": round(sum(c) / len(c), 4) if c else 1.0,
            "min_cosine": round(c[0], 4) if c else 1.0,
        }


@dataclass
class ReembedReport:
    total: int = 0
    written: int = 0
    by_kind: dict[str, KindStats] = field(
        default_factory=lambda: defaultdict(KindStats)
    )
    problems: list[str] = field(default_factory=list)


def reembed(
    collection: Any, embeddings: Embeddings, *, write: bool, batch: int = BATCH
) -> ReembedReport:
    report = ReembedReport()
    before = list(collection.iterator(include_vector=True))
    properties_before = properties_digest(before)
    computed: dict[str, list[float]] = {}
    for start in range(0, len(before), batch):
        chunk = before[start : start + batch]
        fresh = embed_texts([embedding_input(o.properties) for o in chunk], embeddings)
        for obj, vector in zip(chunk, fresh):
            sim = cosine(vector_of(obj), vector)
            stats = report.by_kind[kind_of(obj.properties)]
            stats.count += 1
            stats.cosines.append(sim)
            stats.already_current += sim >= SAME_VECTOR
            report.total += 1
            computed[str(obj.uuid)] = vector
            if write:
                collection.data.update(uuid=obj.uuid, vector={"default": vector})
                report.written += 1
    if write:
        after = list(collection.iterator(include_vector=True))
        if len(after) != len(before):
            report.problems.append(f"object count {len(before)} became {len(after)}")
        if properties_digest(after) != properties_before:
            report.problems.append("properties changed during the re-embed")
        wrong = sum(
            1
            for obj in after
            if cosine(vector_of(obj), computed[str(obj.uuid)]) < READ_BACK
        )
        if wrong:
            report.problems.append(f"{wrong} vectors do not read back as written")
    return report


def verify_current(
    collection: Any, embeddings: Embeddings, sample: int = 40
) -> tuple[float, float]:
    """(lowest, mean) cosine between stored vectors and fresh embeddings of their text.

    The sample is random but fixed-seed, so a run is repeatable.
    """
    objects = list(collection.iterator(include_vector=True))
    objects = random.Random(7).sample(objects, min(sample, len(objects)))
    fresh = embed_texts([embedding_input(o.properties) for o in objects], embeddings)
    sims = [cosine(vector_of(o), v) for o, v in zip(objects, fresh)]
    return min(sims), sum(sims) / len(sims)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-snapshot", action="store_true")
    args = parser.parse_args()
    settings = AppSettings()
    if settings.weaviate_mode != "local":
        print("Error: re-embedding writes to the local collection only")
        sys.exit(1)
    fingerprint = embedding_fingerprint(
        str(settings.embeddings.base_url), settings.embedding_model
    )
    print(f"embedding runtime: {fingerprint.as_dict()}")
    url = urlparse(str(settings.weaviate_url))
    client = weaviate.connect_to_local(
        host=url.hostname or "localhost", port=url.port or 8080
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path("runs") / f"reembed-{stamp}"
    try:
        collection = client.collections.get(COLLECTION_NAME)
        if not args.dry_run and not args.no_snapshot:
            export_snapshot(
                client,
                run_dir / "snapshot-local",
                embedding_provider=settings.embedding_provider,
                embedding_model=settings.embedding_model,
            )
            print(f"snapshot: {run_dir / 'snapshot-local'}")
        report = reembed(
            collection, create_embeddings(settings), write=not args.dry_run
        )
    finally:
        client.close()
    summary = {
        "mode": "dry-run" if args.dry_run else "written",
        "fingerprint": fingerprint.as_dict(),
        "total": report.total,
        "written": report.written,
        "by_kind": {k: v.summary() for k, v in report.by_kind.items()},
        "problems": report.problems,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reembed.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    sys.exit(1 if report.problems else 0)


if __name__ == "__main__":
    main()
