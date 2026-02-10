"""Web knowledge base crawling and chunking."""

import logging
import re

from firecrawl import FirecrawlApp

from multimodal_rag.models.chunks import WebChunk

logger = logging.getLogger(__name__)


def crawl_knowledge_base(
    root_url: str,
    api_key: str,
    limit: int = 100,
) -> list[dict[str, str]]:
    """Crawl a knowledge base from root URL, returning page content."""
    app = FirecrawlApp(api_key=api_key)
    result = app.crawl_url(
        root_url,
        params={
            "limit": limit,
            "scrapeOptions": {
                "formats": ["markdown"],
                "onlyMainContent": True,
            },
        },
    )

    pages: list[dict[str, str]] = []
    data = result.get("data", []) if isinstance(result, dict) else []
    for page in data:
        markdown = page.get("markdown", "")
        url = page.get("metadata", {}).get("sourceURL", root_url)
        title = page.get("metadata", {}).get("title", "")
        if markdown.strip():
            pages.append({"url": url, "title": title, "content": markdown})

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
        return _split_by_tokens(content, source_url, source_name, None, target_tokens)

    chunks: list[WebChunk] = []

    # Content before first header
    pre_header = content[: matches[0].start()].strip()
    if pre_header:
        chunks.extend(
            _split_by_tokens(pre_header, source_url, source_name, None, target_tokens)
        )

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section_text = content[start:end].strip()

        if section_text:
            chunks.extend(
                _split_by_tokens(
                    section_text, source_url, source_name, heading, target_tokens
                )
            )

    return chunks


def _split_by_tokens(
    text: str,
    source_url: str,
    source_name: str,
    section_heading: str | None,
    target_tokens: int,
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
                )
            )

    return chunks


def fetch_web_chunks(
    root_url: str,
    source_name: str,
    api_key: str,
    target_tokens: int = 400,
    crawl_limit: int = 100,
) -> list[WebChunk]:
    """Crawl a knowledge base and return chunked content.

    Returns empty list if crawling fails.
    """
    try:
        pages = crawl_knowledge_base(root_url, api_key, limit=crawl_limit)
    except Exception:
        logger.exception("Failed to crawl %s", root_url)
        return []

    if not pages:
        logger.warning("No pages found at %s", root_url)
        return []

    all_chunks: list[WebChunk] = []
    for page in pages:
        page_chunks = split_by_sections(
            content=page["content"],
            source_url=page["url"],
            source_name=source_name,
            target_tokens=target_tokens,
        )
        all_chunks.extend(page_chunks)

    logger.info(
        "Processed %s: %d pages → %d chunks",
        source_name,
        len(pages),
        len(all_chunks),
    )
    return all_chunks
