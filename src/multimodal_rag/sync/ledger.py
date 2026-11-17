"""The page ledger: what the index holds per page, and when it was last verified.

The baseline is built from the local collection. Screenshot descriptions are stored with
``source_type="web"`` like page text, so no property tells them apart; a chunk is page
text only if its UUID re-derives as ``uuid5(url|index)``. Everything else is a frozen
chunk that the web lane never changes or deletes.
"""

from __future__ import annotations

import collections
import json
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, Field

from multimodal_rag.gcloud import run_gcloud

LEDGER_OBJECT = "ledger/pages.json"


class PageEntry(BaseModel):
    url: str
    host: str
    text_chunks: int
    frozen_chunks: int
    ingested_at: datetime
    lastmod: datetime | None = None
    # Null for the baseline: the March chunks cannot be compared with a fresh scrape.
    fingerprint: str | None = None


class Ledger(BaseModel):
    version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pages: dict[str, PageEntry] = Field(default_factory=dict)
    # The embedding runtime the index was last verified against (None until the
    # first apply that records it).
    embedding: dict[str, str] | None = None


def text_chunk_id(url: str, index: int) -> UUID:
    return uuid5(NAMESPACE_URL, f"{url}|{index}")


def split_layers(url: str, chunk_ids: list[UUID]) -> tuple[int, int]:
    """Return (page-text chunk count, frozen chunk count) for one page."""
    derivable = {text_chunk_id(url, i) for i in range(len(chunk_ids))}
    text = sum(1 for chunk_id in chunk_ids if chunk_id in derivable)
    return text, len(chunk_ids) - text


def build_baseline(objects: Iterable[Any]) -> Ledger:
    """Build the ledger from collection objects, ignoring video chunks."""
    ids: dict[str, list[UUID]] = collections.defaultdict(list)
    ingested: dict[str, datetime] = {}
    for obj in objects:
        props = obj.properties
        if props["source_type"] == "video":
            continue
        url = props["source_url"]
        ids[url].append(UUID(str(obj.uuid)))
        when = props["ingested_at"]
        if url not in ingested or when > ingested[url]:
            ingested[url] = when
    ledger = Ledger()
    for url, chunk_ids in ids.items():
        text, frozen = split_layers(url, chunk_ids)
        ledger.pages[url] = PageEntry(
            url=url,
            host=urlparse(url).netloc,
            text_chunks=text,
            frozen_chunks=frozen,
            ingested_at=ingested[url],
        )
    return ledger


def ledger_url(bucket: str) -> str:
    return f"gs://{bucket}/{LEDGER_OBJECT}"


def read_ledger(bucket: str, project: str) -> tuple[Ledger, str] | None:
    """Return (ledger, generation) from the bucket, or None if none exists yet."""
    url = ledger_url(bucket)
    try:
        generation = json.loads(
            run_gcloud(
                ["storage", "objects", "describe", url, "--format=json(generation)"],
                project,
            )
        )["generation"]
    except RuntimeError as exc:
        if "404" in str(exc) or "No such object" in str(exc):
            return None
        raise
    with tempfile.TemporaryDirectory(prefix="ledger-") as tmp:
        target = Path(tmp) / "pages.json"
        run_gcloud(["storage", "cp", url, str(target)], project)
        return Ledger.model_validate_json(target.read_text()), str(generation)


def write_ledger(
    ledger: Ledger, bucket: str, project: str, *, if_generation: str
) -> None:
    """Upload the ledger only if the object is still at ``if_generation``.

    ``"0"`` means the object must not exist yet, so the baseline is created once.
    """
    with tempfile.TemporaryDirectory(prefix="ledger-") as tmp:
        source = Path(tmp) / "pages.json"
        source.write_text(ledger.model_dump_json(indent=2))
        run_gcloud(
            [
                "storage",
                "cp",
                str(source),
                ledger_url(bucket),
                f"--if-generation-match={if_generation}",
            ],
            project,
        )
