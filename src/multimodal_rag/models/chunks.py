"""Chunk models for ingestion and Weaviate storage."""

from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
from uuid import UUID, uuid4

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


class SupportChunk(BaseModel):
    """Unified chunk model for Weaviate storage."""

    chunk_id: UUID = Field(default_factory=uuid4)
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

    @classmethod
    def from_transcript_chunk(cls, chunk: TranscriptChunk) -> "SupportChunk":
        return cls(
            text=chunk.text,
            source_type=SourceType.VIDEO,
            source_url=chunk.source_url,
            source_name=chunk.source_name,
            timestamp_seconds=chunk.start_seconds,
        )

    @classmethod
    def from_web_chunk(cls, chunk: WebChunk) -> "SupportChunk":
        return cls(
            text=chunk.text,
            source_type=SourceType.WEB,
            source_url=chunk.source_url,
            source_name=chunk.source_name,
            section_heading=chunk.section_heading,
        )
