"""CLI entrypoint for the ingestion pipeline.

Usage: python -m multimodal_rag.ingest
"""

import logging
import sys
from pathlib import Path

import yaml

from multimodal_rag.ingest.web import fetch_web_chunks
from multimodal_rag.ingest.youtube import fetch_transcript_chunks
from multimodal_rag.models.chunks import SupportChunk
from multimodal_rag.models.config import AppSettings
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


def run() -> None:
    settings = AppSettings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    sources = load_sources()
    all_chunks: list[SupportChunk] = []

    # YouTube sources
    for yt in sources.youtube:
        logger.info("Processing YouTube: %s", yt.name)
        tc = fetch_transcript_chunks(
            video_url=str(yt.url),
            source_name=yt.name,
            target_tokens=settings.chunk_size,
        )
        all_chunks.extend(SupportChunk.from_transcript_chunk(c) for c in tc)

    # Web knowledge base sources
    for kb in sources.kb_sources:
        logger.info("Processing KB: %s", kb.name)
        wc = fetch_web_chunks(
            root_url=str(kb.url),
            source_name=kb.name,
            api_key=settings.firecrawl_api_key,
            target_tokens=settings.chunk_size,
        )
        all_chunks.extend(SupportChunk.from_web_chunk(c) for c in wc)

    if not all_chunks:
        logger.warning("No chunks produced. Nothing to ingest.")
        return

    logger.info("Total chunks to ingest: %d", len(all_chunks))

    with WeaviateStore(
        weaviate_url=settings.weaviate_url,
        openrouter_api_key=settings.openrouter_api_key,
        openrouter_base_url=settings.openrouter_base_url,
        embedding_model=settings.embedding_model,
    ) as store:
        store.ensure_collection()
        added = store.add_chunks(all_chunks)
        total = store.count()
        logger.info("Ingestion complete: %d added, %d total in store", added, total)


if __name__ == "__main__":
    run()
