"""Chunk models for ingestion and Weaviate storage."""

from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, Field, computed_field

from multimodal_rag.models import identity


class SourceType(StrEnum):
    VIDEO = "video"
    WEB = "web"


class TranscriptChunk(BaseModel):
    """A chunk of video transcript with timestamp metadata."""

    text: str
    source_url: str
    source_name: str
    start_seconds: int
    end_seconds: int
    # A window's narration and its on-screen description are separate chunks, so a
    # query can match either without one diluting the other (SYNC-012).
    kind: Literal["transcript", "visual"] = "transcript"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def timestamp_url(self) -> str:
        return f"{self.source_url}&t={self.start_seconds}s"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def timestamp_display(self) -> str:
        minutes, seconds = divmod(self.start_seconds, 60)
        return f"{minutes:02d}:{seconds:02d}"


class WebChunk(BaseModel):
    """A chunk of web page content with source metadata."""

    text: str
    source_url: str
    source_name: str
    section_heading: str | None = None
    image_url: str | None = None
    chunk_index: int | None = None
    source_lastmod: datetime | None = None
    embedded_video_ids: list[str] = Field(default_factory=list)


class SupportChunk(BaseModel):
    """Unified chunk model for Weaviate storage."""

    chunk_id: UUID | None = None
    text: str
    source_type: SourceType
    source_url: str
    source_name: str
    timestamp_seconds: int | None = None
    section_heading: str | None = None
    url_hash: str = ""
    document_id: str = ""
    content_hash: str = ""
    source_lastmod: datetime | None = None
    embedded_video_ids: list[str] = Field(default_factory=list)
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def model_post_init(self, __context: object) -> None:
        if not self.url_hash:
            self.url_hash = sha256(self.source_url.encode()).hexdigest()
        if not self.document_id:
            self.document_id = identity.document_id(self.source_url)
        if not self.content_hash:
            self.content_hash = identity.content_hash(self.text)
        if self.chunk_id is None:
            self.chunk_id = uuid5(NAMESPACE_URL, self.source_url + "|" + self.text)

    @classmethod
    def from_transcript_chunk(cls, chunk: TranscriptChunk) -> "SupportChunk":
        stable_id = uuid5(
            NAMESPACE_URL,
            chunk.source_url + "|transcript|" + str(chunk.start_seconds),
        )
        return cls(
            chunk_id=stable_id,
            text=chunk.text,
            source_type=SourceType.VIDEO,
            source_url=chunk.source_url,
            source_name=chunk.source_name,
            timestamp_seconds=chunk.start_seconds,
        )

    @classmethod
    def from_web_chunk(cls, chunk: WebChunk) -> "SupportChunk":
        if chunk.chunk_index is not None:
            return cls(
                chunk_id=uuid5(
                    NAMESPACE_URL,
                    chunk.source_url + "|" + str(chunk.chunk_index),
                ),
                text=chunk.text,
                source_type=SourceType.WEB,
                source_url=chunk.source_url,
                source_name=chunk.source_name,
                section_heading=chunk.section_heading,
                source_lastmod=chunk.source_lastmod,
                embedded_video_ids=chunk.embedded_video_ids,
            )
        return cls(
            text=chunk.text,
            source_type=SourceType.WEB,
            source_url=chunk.source_url,
            source_name=chunk.source_name,
            section_heading=chunk.section_heading,
            source_lastmod=chunk.source_lastmod,
            embedded_video_ids=chunk.embedded_video_ids,
        )

    @classmethod
    def from_frame_chunk(cls, chunk: TranscriptChunk) -> "SupportChunk":
        """Create a SupportChunk from a visual frame description.

        chunk_id is stable: keyed on source_url + timestamp, not LLM text.
        """
        stable_id = uuid5(
            NAMESPACE_URL,
            chunk.source_url + "|frame|" + str(chunk.start_seconds),
        )
        return cls(
            chunk_id=stable_id,
            text=chunk.text,
            source_type=SourceType.VIDEO,
            source_url=chunk.source_url,
            source_name=chunk.source_name,
            timestamp_seconds=chunk.start_seconds,
        )

    @classmethod
    def from_fused_chunk(cls, chunk: TranscriptChunk) -> "SupportChunk":
        """Create a SupportChunk from a combined audio+visual fused chunk.

        chunk_id is stable: keyed on source_url + window timestamp.
        """
        stable_id = uuid5(
            NAMESPACE_URL,
            chunk.source_url + "|fused|" + str(chunk.start_seconds),
        )
        return cls(
            chunk_id=stable_id,
            text=chunk.text,
            source_type=SourceType.VIDEO,
            source_url=chunk.source_url,
            source_name=chunk.source_name,
            timestamp_seconds=chunk.start_seconds,
        )

    @classmethod
    def from_visual_chunk(cls, chunk: TranscriptChunk) -> "SupportChunk":
        """Create a SupportChunk from the on-screen description of one video window.

        chunk_id is stable: keyed on source_url + window timestamp, not LLM text. It
        shares source_url and timestamp with the window's narration chunk
        (from_fused_chunk), so a citation points at the same moment.
        """
        stable_id = uuid5(
            NAMESPACE_URL,
            chunk.source_url + "|visual|" + str(chunk.start_seconds),
        )
        return cls(
            chunk_id=stable_id,
            text=chunk.text,
            source_type=SourceType.VIDEO,
            source_url=chunk.source_url,
            source_name=chunk.source_name,
            timestamp_seconds=chunk.start_seconds,
        )

    @classmethod
    def from_screenshot_chunk(cls, chunk: WebChunk) -> "SupportChunk":
        """Create a SupportChunk from a visual screenshot description.

        chunk_id is stable: keyed on image_url, not LLM text.
        chunk.image_url must be non-None.
        """
        if chunk.image_url is None:
            raise ValueError("from_screenshot_chunk requires chunk.image_url to be set")
        stable_id = uuid5(NAMESPACE_URL, chunk.image_url)
        return cls(
            chunk_id=stable_id,
            text=chunk.text,
            source_type=SourceType.WEB,
            source_url=chunk.source_url,
            source_name=chunk.source_name,
            section_heading=chunk.section_heading,
            source_lastmod=chunk.source_lastmod,
            embedded_video_ids=chunk.embedded_video_ids,
        )
