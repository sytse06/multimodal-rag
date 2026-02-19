"""CLI entrypoint for the ingestion pipeline.

Usage: python -m multimodal_rag.ingest
"""

import logging
import sys
import time
from pathlib import Path

import yaml

from multimodal_rag.ingest.web import crawl_knowledge_base, split_by_sections
from multimodal_rag.ingest.youtube import fetch_transcript_chunks
from multimodal_rag.models.chunks import SupportChunk
from multimodal_rag.models.config import AppSettings
from multimodal_rag.models.llm import create_embeddings
from multimodal_rag.models.sources import SourceConfig
from multimodal_rag.store.weaviate import WeaviateStore

logger = logging.getLogger(__name__)

SOURCES_PATH = Path("config/sources.yaml")


def load_sources() -> SourceConfig:
    if not SOURCES_PATH.exists():
        logger.error("Sources file not found: %s", SOURCES_PATH)
        sys.exit(1)
    raw = yaml.safe_load(SOURCES_PATH.read_text()) or {}
    return SourceConfig.model_validate(raw)


def _ingest_chunks(
    store: WeaviateStore,
    chunks: list[SupportChunk],
    label: str,
) -> int:
    """Embed and store chunks for a single unit. Returns count added."""
    if not chunks:
        logger.warning("[%s] No chunks produced, skipping", label)
        return 0
    added = store.add_chunks(chunks)
    logger.info("[%s] Stored %d chunks", label, added)
    return added


def run() -> None:
    settings = AppSettings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    sources = load_sources()
    embeddings = create_embeddings(settings)

    total_added = 0
    total_failed = 0

    with WeaviateStore(
        weaviate_url=settings.weaviate_url,
        embeddings=embeddings,
    ) as store:
        store.ensure_collection()

        # YouTube sources (with rate limiting to avoid IP bans)
        for i, yt in enumerate(sources.youtube):
            if i > 0:
                time.sleep(2)
            label = f"youtube:{yt.name}"
            logger.info("Processing %s", label)
            try:
                tc = fetch_transcript_chunks(
                    video_url=str(yt.url),
                    source_name=yt.name,
                    target_tokens=settings.chunk_size,
                    mistral_api_key=settings.mistral_api_key,
                )
                chunks = [SupportChunk.from_transcript_chunk(c) for c in tc]
                total_added += _ingest_chunks(store, chunks, label)
            except Exception:
                logger.exception("[%s] Failed, skipping", label)
                total_failed += 1

        # Web knowledge base sources
        for i, kb in enumerate(sources.kb_sources):
            if i > 0:
                time.sleep(5)
            kb_label = f"kb:{kb.name}"
            logger.info("Processing %s", kb_label)
            try:
                pages = crawl_knowledge_base(
                    root_url=str(kb.url),
                    api_key=settings.firecrawl_api_key,
                    limit=100,
                )
            except Exception:
                logger.exception("[%s] Crawl failed, skipping", kb_label)
                total_failed += 1
                continue

            for page in pages:
                page_url = page["url"]
                page_label = f"kb:{kb.name}|{page_url}"
                try:
                    web_chunks = split_by_sections(
                        content=page["content"],
                        source_url=page_url,
                        source_name=kb.name,
                        target_tokens=settings.chunk_size,
                    )
                    chunks = [SupportChunk.from_web_chunk(c) for c in web_chunks]
                    total_added += _ingest_chunks(store, chunks, page_label)
                except Exception:
                    logger.exception("[%s] Failed, skipping", page_label)
                    total_failed += 1

        total_in_store = store.count()

    logger.info(
        "Ingestion complete: %d added, %d failed, %d total in store",
        total_added,
        total_failed,
        total_in_store,
    )


if __name__ == "__main__":
    run()
