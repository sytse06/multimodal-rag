"""Sitemap inventory: which pages exist upstream, and when they last changed.

The registry holds only the root URL of each knowledge base. The pages come from its
sitemap, restricted to the paths the crawler fetches, and matched on the exact host:
hydroman article slugs contain "hydrosym", so a substring match mixes the sites up.
"""

from __future__ import annotations

import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse

SITEMAP_NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
SCOPE_PREFIX = "/article/"  # the crawler's include_paths is /article/*

Fetch = Callable[[str], bytes]


@dataclass(frozen=True)
class SitemapPage:
    url: str
    lastmod: datetime | None


def _http_get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
        return bytes(response.read())


def parse_lastmod(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def parse_sitemap(xml: bytes) -> tuple[list[str], list[SitemapPage]]:
    """Return (child sitemap URLs, pages) from one sitemap or sitemap index."""
    root = ET.fromstring(xml)
    children = [
        loc.text.strip()
        for loc in root.findall("s:sitemap/s:loc", SITEMAP_NS)
        if loc.text
    ]
    pages = []
    for entry in root.findall("s:url", SITEMAP_NS):
        loc = entry.find("s:loc", SITEMAP_NS)
        if loc is None or not loc.text:
            continue
        lastmod = entry.find("s:lastmod", SITEMAP_NS)
        stamp = lastmod.text if lastmod is not None else None
        pages.append(SitemapPage(loc.text.strip(), parse_lastmod(stamp)))
    return children, pages


def in_scope(url: str, host: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc == host and parsed.path.startswith(SCOPE_PREFIX)


def fetch_sitemap(url: str, fetch: Fetch = _http_get) -> list[SitemapPage]:
    children, pages = parse_sitemap(fetch(url))
    for child in children:
        pages.extend(fetch_sitemap(child, fetch))
    return pages


def knowledge_base_pages(
    root_url: str, fetch: Fetch = _http_get
) -> tuple[str, list[SitemapPage], int]:
    """Return (host, in-scope pages, count of the host's pages left out of scope)."""
    parsed = urlparse(root_url)
    host = parsed.netloc
    everything = fetch_sitemap(f"{parsed.scheme}://{host}/sitemap.xml", fetch)
    own = [p for p in everything if urlparse(p.url).netloc == host]
    scoped = [p for p in own if in_scope(p.url, host)]
    return host, scoped, len(own) - len(scoped)
