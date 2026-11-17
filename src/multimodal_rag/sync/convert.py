"""Convert the stored chunks to the identity fields of SYNC-001, without touching text.

Pure logic: it reads objects as ``(uuid, properties, vector)`` and returns the property
changes. Text and vectors are never part of an update; ``checksum`` proves that.
"""

import collections
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from multimodal_rag.models import identity

Row = tuple[str, dict[str, Any], list[float]]

FIELDS = ("document_id", "content_hash", "source_lastmod", "embedded_video_ids")


@dataclass
class Conversion:
    updates: dict[str, dict[str, Any]] = field(default_factory=dict)
    total: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    pages: int = 0
    pages_with_lastmod: int = 0
    pages_with_videos: int = 0
    already_current: int = 0


def checksum(rows: list[Row]) -> str:
    """One hash of every object's id, text and vector, to compare before and after."""
    digest = hashlib.sha256()
    for uuid, props, vector in sorted(rows, key=lambda r: r[0]):
        values = ",".join(f"{x:.7g}" for x in vector)
        digest.update(f"{uuid}|{props.get('text', '')}|{values}\n".encode())
    return digest.hexdigest()


def convert(rows: list[Row], lastmods: dict[str, datetime | None]) -> Conversion:
    """Work out the field values every chunk should hold and which differ today.

    ``lastmods`` is the sitemap ``lastmod`` of every page whose stored content is still
    current. A page not in it keeps any ``source_lastmod`` it already holds, because the
    stored content still corresponds to that date; it never gets a guess. A web chunk's
    videos are every video linked in any chunk of its page, together with what it
    already holds, so a rerun never drops one. Video chunks carry an empty list.
    """
    page_videos: dict[str, set[str]] = collections.defaultdict(set)
    for _, props, _ in rows:
        if props["source_type"] == "web":
            videos = page_videos[props["source_url"]]
            videos.update(identity.embedded_video_ids(props["text"]))
            videos.update(props.get("embedded_video_ids") or [])

    result = Conversion(total=len(rows), pages=len(page_videos))
    result.pages_with_videos = sum(1 for v in page_videos.values() if v)
    result.pages_with_lastmod = sum(1 for u in page_videos if lastmods.get(u))
    for uuid, props, _ in rows:
        kind = props["source_type"]
        result.by_type[kind] = result.by_type.get(kind, 0) + 1
        url = props["source_url"]
        target: dict[str, Any] = {
            "document_id": identity.document_id(url),
            "content_hash": identity.content_hash(props["text"]),
            "source_lastmod": (
                (lastmods.get(url) or props.get("source_lastmod"))
                if kind == "web"
                else None
            ),
            "embedded_video_ids": sorted(page_videos[url]) if kind == "web" else [],
        }
        changed = {k: v for k, v in target.items() if props.get(k) != v}
        if changed:
            result.updates[uuid] = changed
        else:
            result.already_current += 1
    return result


def problems(before: list[Row], after: list[Row], expected: Conversion) -> list[str]:
    """Reasons a finished conversion cannot be trusted; empty when it can."""
    found = []
    if len(before) != len(after):
        found.append(f"object count changed: {len(before)} -> {len(after)}")
    if checksum(before) != checksum(after):
        found.append("a text or a vector changed")
    held = {u: p for u, p, _ in after}
    for uuid, changed in expected.updates.items():
        props = held.get(uuid, {})
        wrong = [k for k, v in changed.items() if props.get(k) != v]
        if wrong:
            found.append(f"{uuid}: {', '.join(wrong)} not as planned")
            if len(found) > 10:
                break
    return found
