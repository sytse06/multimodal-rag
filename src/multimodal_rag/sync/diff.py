"""Compare a freshly chunked page with the text chunks stored for it, by position."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from multimodal_rag.ingest.web import FOOTER_MARKER
from multimodal_rag.models.chunks import WebChunk
from multimodal_rag.sync.ledger import text_chunk_id

IDENTICAL = "identical"
TEMPLATE_ONLY = "template-only"  # differs only by the new title or the removed footer
CHANGED = "changed"
ADDED = "added"
REMOVED = "removed"


@dataclass(frozen=True)
class StoredChunk:
    uuid: UUID
    text: str


@dataclass(frozen=True)
class PositionChange:
    index: int
    status: str
    uuid: UUID


@dataclass
class PageDiff:
    url: str
    stored: int
    fresh: int
    changes: list[PositionChange] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(1 for c in self.changes if c.status == status)

    @property
    def unchanged(self) -> bool:
        return all(c.status == IDENTICAL for c in self.changes)

    @property
    def content_changed(self) -> bool:
        """True if anything beyond the template differs."""
        return any(c.status in (CHANGED, ADDED, REMOVED) for c in self.changes)


def stored_from_objects(url: str, objects: Iterable[Any]) -> dict[int, StoredChunk]:
    """Page-text chunks by position. Frozen chunks (screenshots) are left out: a chunk
    counts as page text only if its UUID re-derives from the URL and its position."""
    found = list(objects)
    by_uuid = {text_chunk_id(url, i): i for i in range(len(found))}
    result = {}
    for obj in found:
        uuid = UUID(str(obj.uuid))
        if uuid in by_uuid:
            result[by_uuid[uuid]] = StoredChunk(uuid, obj.properties["text"])
    return result


def _canonical(text: str, title: str | None = None) -> str:
    """Drop what the normaliser changes: a leading title and the footer, and spacing."""
    if title and text.startswith(f"{title}\n\n"):
        text = text[len(title) + 2 :]
    return " ".join(text.split(FOOTER_MARKER)[0].split())


def diff_page(
    url: str,
    stored: dict[int, StoredChunk],
    fresh: list[WebChunk],
    *,
    title: str | None = None,
) -> PageDiff:
    diff = PageDiff(url=url, stored=len(stored), fresh=len(fresh))
    fresh_by_index = {c.chunk_index: c.text for c in fresh if c.chunk_index is not None}
    for index in sorted(set(stored) | set(fresh_by_index)):
        uuid = text_chunk_id(url, index)
        if index not in fresh_by_index:
            diff.changes.append(PositionChange(index, REMOVED, uuid))
        elif index not in stored:
            diff.changes.append(PositionChange(index, ADDED, uuid))
        elif stored[index].text == fresh_by_index[index]:
            diff.changes.append(PositionChange(index, IDENTICAL, uuid))
        elif _canonical(stored[index].text) == _canonical(fresh_by_index[index], title):
            diff.changes.append(PositionChange(index, TEMPLATE_ONLY, uuid))
        else:
            diff.changes.append(PositionChange(index, CHANGED, uuid))
    return diff
