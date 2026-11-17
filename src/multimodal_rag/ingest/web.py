"""Web knowledge base crawling and chunking."""

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime

from firecrawl import FirecrawlApp
from firecrawl.v2.types import ScrapeOptions

from multimodal_rag.models.chunks import WebChunk
from multimodal_rag.models.identity import embedded_video_ids

logger = logging.getLogger(__name__)

# Bump when a normalising rule changes: it is part of every page fingerprint.
NORMALISER_VERSION = 1
# The byline sits right under the title, so it is only looked for in the page head.
HEADER_REGION = 600
# The real footer is a few hundred characters of fixed wording. A cut that would remove
# more than this is not trusted: the marker phrase probably moved into the article, so
# nothing is cut and the page is flagged for review instead of being truncated. It is
# an absolute size, not a share, because a short article's footer can be most of it.
MAX_FOOTER_CHARS = 1500

# "HydroManUpdated 27 November 20252 min readPrint": product, date, reading time and
# the Print button, joined because they come from separate page elements.
_BYLINE = re.compile(
    r"^\w*Updated \d{1,2} [A-Za-z]+ \d{5,6} min read\s*Print[ \t]*$", re.M
)
# Everything after this line is page furniture: the feedback widget, "Last updated on",
# the search box empty state and the "Still have questions?" call to action.
FOOTER_MARKER = "Did this answer your question?"
_TITLE = re.compile(r"^# (.+?)\s*$", re.M)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_BLANK_RUNS = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class NormalisedPage:
    markdown: str
    title: str | None
    byline_removed: bool
    footer_removed: bool
    footer_suspicious: bool = False


def normalise_page(markdown: str, *, strip_images: bool = False) -> NormalisedPage:
    """Remove the template's header byline and footer from a scraped page.

    The page title is returned separately so the chunker can keep it in the first
    chunk. Images stay by default because the screenshot pipeline reads them from the
    page content; ``strip_images`` is for the text chunks only.
    """
    title_match = _TITLE.search(markdown)
    title = title_match.group(1) if title_match else None
    head, tail = markdown[:HEADER_REGION], markdown[HEADER_REGION:]
    head, byline_hits = _BYLINE.subn("", head, count=1)
    body = head + tail
    cut = body.find(FOOTER_MARKER)
    suspicious = cut >= 0 and len(body) - cut > MAX_FOOTER_CHARS
    if cut >= 0 and not suspicious:
        body = body[:cut]
    if strip_images:
        body = _IMAGE.sub("", body)
    body = _BLANK_RUNS.sub("\n\n", body).strip()
    return NormalisedPage(
        markdown=body,
        title=title,
        byline_removed=byline_hits > 0,
        footer_removed=cut >= 0 and not suspicious,
        footer_suspicious=suspicious,
    )


def page_fingerprint(markdown: str) -> str:
    """SHA-256 of the normalised page, so the same content always hashes the same."""
    page = normalise_page(markdown)
    return hashlib.sha256(
        f"normaliser-v{NORMALISER_VERSION}\n{page.markdown}".encode()
    ).hexdigest()


def chunk_page(
    markdown: str,
    source_url: str,
    source_name: str,
    target_tokens: int = 400,
    *,
    strip_images: bool = False,
    source_lastmod: datetime | None = None,
) -> list[WebChunk]:
    """Normalise a scraped page, chunk it, and keep its title in the first chunk.

    Every chunk carries the page's embedded video ids and the sitemap ``lastmod`` the
    content was fetched at.
    """
    page = normalise_page(markdown, strip_images=strip_images)
    chunks = split_by_sections(
        page.markdown, source_url, source_name, target_tokens=target_tokens
    )
    if chunks and page.title:
        chunks[0] = chunks[0].model_copy(
            update={"text": f"{page.title}\n\n{chunks[0].text}"}
        )
    videos = embedded_video_ids(page.markdown)
    page_values = {"embedded_video_ids": videos, "source_lastmod": source_lastmod}
    return [c.model_copy(update=page_values) for c in chunks]


def crawl_knowledge_base(
    root_url: str,
    api_key: str,
    limit: int = 100,
) -> list[dict[str, str]]:
    """Crawl a knowledge base from root URL, returning page content."""
    app = FirecrawlApp(api_key=api_key)
    result = app.crawl(
        root_url,
        limit=limit,
        max_concurrency=1,
        include_paths=["/article/*"],
        scrape_options=ScrapeOptions(
            formats=["markdown"],
            only_main_content=True,
        ),
    )

    pages: list[dict[str, str]] = []
    for doc in result.data:
        markdown = doc.markdown or ""
        meta = doc.metadata
        url = meta.source_url if meta and meta.source_url else root_url
        title = meta.title if meta else ""
        if markdown.strip():
            pages.append({"url": url, "title": title or "", "content": markdown})

    logger.info("Crawled %s: %d pages with content", root_url, len(pages))
    return pages


def split_by_sections(
    content: str,
    source_url: str,
    source_name: str,
    target_tokens: int = 400,
) -> list[WebChunk]:
    """Split markdown content by headers, with token-based fallback."""
    section_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    matches = list(section_pattern.finditer(content))

    if not matches:
        return _split_by_tokens(
            content, source_url, source_name, None, target_tokens, start_index=0
        )

    chunks: list[WebChunk] = []

    # Content before first header
    pre_header = content[: matches[0].start()].strip()
    if pre_header:
        chunks.extend(
            _split_by_tokens(
                pre_header, source_url, source_name, None, target_tokens,
                start_index=len(chunks),
            )
        )

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section_text = content[start:end].strip()

        if section_text:
            chunks.extend(
                _split_by_tokens(
                    section_text, source_url, source_name, heading, target_tokens,
                    start_index=len(chunks),
                )
            )

    return chunks


def _split_by_tokens(
    text: str,
    source_url: str,
    source_name: str,
    section_heading: str | None,
    target_tokens: int,
    start_index: int = 0,
) -> list[WebChunk]:
    """Split text into chunks of ~target_tokens."""
    words = text.split()
    if not words:
        return []

    # Estimate: 1 word ≈ 1.3 tokens
    words_per_chunk = max(1, int(target_tokens / 1.3))
    chunks: list[WebChunk] = []

    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i : i + words_per_chunk]
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            chunks.append(
                WebChunk(
                    text=chunk_text,
                    source_url=source_url,
                    source_name=source_name,
                    section_heading=section_heading,
                    chunk_index=start_index + len(chunks),
                )
            )

    return chunks


