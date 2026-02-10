"""Ingestion pipeline for YouTube transcripts and web content."""

from multimodal_rag.ingest.web import fetch_web_chunks
from multimodal_rag.ingest.youtube import fetch_transcript_chunks

__all__ = [
    "fetch_transcript_chunks",
    "fetch_web_chunks",
]
