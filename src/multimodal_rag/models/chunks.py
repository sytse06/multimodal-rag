"""Chunk models for ingestion and Weaviate storage."""

from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, Field, computed_field


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


class SupportChunk(BaseModel):
    """Unified chunk model for Weaviate storage."""

    chunk_id: UUID = Field(default=None)  # type: ignore[assignment]
    text: str
    source_type: SourceType
    source_url: str
    source_name: str
    timestamp_seconds: int | None = None
    section_heading: str | None = None
    url_hash: str = ""
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def model_post_init(self, __context: object) -> None:
        if not self.url_hash:
            self.url_hash = sha256(self.source_url.encode()).hexdigest()
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
            )
        return cls(
            text=chunk.text,
            source_type=SourceType.WEB,
            source_url=chunk.source_url,
            source_name=chunk.source_name,
            section_heading=chunk.section_heading,
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
        )
