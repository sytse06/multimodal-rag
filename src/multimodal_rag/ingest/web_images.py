"""Web page image extraction and vision LLM description."""

import base64
import logging
import re

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from multimodal_rag.models.chunks import WebChunk

logger = logging.getLogger(__name__)

_IMAGE_URL_RE = re.compile(r"!\[.*?\]\((https?://[^\s)]+)\)")


def extract_image_urls(markdown: str) -> list[str]:
    """Extract absolute image URLs from Firecrawl markdown.

    Returns a deduplicated list of http/https image URLs.
    """
    seen: set[str] = set()
    result: list[str] = []
    for match in _IMAGE_URL_RE.finditer(markdown):
        url = match.group(1)
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def describe_image(image_url: str, llm: BaseChatModel) -> str:
    """Download an image and describe it using a vision LLM.

    Detects content-type from the response header (falls back to image/jpeg).
    Returns the LLM's text description.
    Raises httpx.HTTPStatusError on HTTP errors.
    """
    response = httpx.get(image_url, follow_redirects=True, timeout=30)
    response.raise_for_status()

    raw_ct = response.headers.get("content-type", "image/jpeg")
    content_type = raw_ct.split(";")[0].strip()
    b64 = base64.b64encode(response.content).decode("utf-8")

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Describe what is shown in this screenshot in detail. "
                    "Focus on any UI elements, text, menus, buttons, "
                    "forms, or actions visible. "
                    "Be specific and concise."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{content_type};base64,{b64}"},
            },
        ]
    )
    response_msg = llm.invoke([message])
    return str(response_msg.content)


def fetch_image_chunks(
    page_url: str,
    page_title: str,
    markdown: str,
    llm: BaseChatModel,
) -> list[WebChunk]:
    """Extract images from a web page's markdown and describe each with a vision LLM.

    Returns a list of WebChunks where text is the image description
    and image_url is the source image URL.
    """
    image_urls = extract_image_urls(markdown)
    logger.info("Found %d images on %s", len(image_urls), page_url)

    chunks: list[WebChunk] = []
    for image_url in image_urls:
        try:
            description = describe_image(image_url, llm)
            chunks.append(
                WebChunk(
                    text=description,
                    source_url=page_url,
                    source_name=page_title,
                    image_url=image_url,
                    section_heading=None,
                )
            )
        except Exception:
            logger.warning(
                "Failed to describe image %s on %s, skipping",
                image_url,
                page_url,
                exc_info=True,
            )

    return chunks
