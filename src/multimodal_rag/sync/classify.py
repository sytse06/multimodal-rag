"""Classify upstream pages against the ledger: modified, new, removed or unchanged."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from urllib.parse import urlparse

from multimodal_rag.sync.inventory import SitemapPage
from multimodal_rag.sync.ledger import Ledger

RENAME_THRESHOLD = 0.5
SLUG_NOISE = {"hydrosym", "documentation", "paro", "software"}


class Kind(str, Enum):
    UNCHANGED = "unchanged"
    MODIFIED = "modified"
    NEW = "new"
    REMOVED = "removed"


@dataclass(frozen=True)
class Item:
    url: str
    kind: Kind
    reason: str = ""
    rename_of: str | None = None  # for a removal: the new page that looks like it
    lastmod: datetime | None = None  # the sitemap's lastmod for this page


def slug_tokens(url: str) -> set[str]:
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"^\d+-", "", slug)
    # A six-letter stem makes "automatic" and "automated" match.
    return {t[:6] for t in slug.split("-") if t and t not in SLUG_NOISE}


def similarity(a: str, b: str) -> float:
    left, right = slug_tokens(a), slug_tokens(b)
    return len(left & right) / len(left | right) if left | right else 0.0


def classify(
    inventory: dict[str, list[SitemapPage]], ledger: Ledger, *, verify: bool = False
) -> list[Item]:
    """Classify every page of the hosts in ``inventory``.

    A host missing from ``inventory`` (its sitemap failed or was empty) is skipped
    entirely, so a failed fetch can never turn into removal candidates.
    """
    items: list[Item] = []
    for host, pages in inventory.items():
        seen = set()
        for page in pages:
            seen.add(page.url)
            entry = ledger.pages.get(page.url)
            if entry is None:
                items.append(
                    Item(page.url, Kind.NEW, "not stored", lastmod=page.lastmod)
                )
                continue
            reference = entry.lastmod or entry.ingested_at
            if verify:
                items.append(
                    Item(page.url, Kind.MODIFIED, "verify run", lastmod=page.lastmod)
                )
            elif page.lastmod is None:
                items.append(
                    Item(page.url, Kind.MODIFIED, "no lastmod in the sitemap")
                )
            elif page.lastmod > reference:
                reason = f"lastmod {page.lastmod.date()}"
                items.append(
                    Item(page.url, Kind.MODIFIED, reason, lastmod=page.lastmod)
                )
            else:
                items.append(Item(page.url, Kind.UNCHANGED, lastmod=page.lastmod))
        new_urls = [
            i.url
            for i in items
            if i.kind is Kind.NEW and urlparse(i.url).netloc == host
        ]
        for url, entry in ledger.pages.items():
            if entry.host != host or url in seen:
                continue
            best = max(new_urls, key=lambda n: similarity(url, n), default=None)
            hint = best if best and similarity(url, best) >= RENAME_THRESHOLD else None
            items.append(Item(url, Kind.REMOVED, "stored, not in the sitemap", hint))
    return items
