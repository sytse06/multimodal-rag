"""Query pipeline response models."""

from pydantic import BaseModel

from multimodal_rag.models.chunks import SourceType


class SearchResult(BaseModel):
    """A retrieved chunk with its relevance score."""

    text: str
    source_type: SourceType
    source_url: str
    source_name: str
    timestamp_seconds: int | None = None
    section_heading: str | None = None
    relevance_score: float

    @property
    def citation_url(self) -> str:
        if self.source_type == SourceType.VIDEO and self.timestamp_seconds is not None:
            return f"{self.source_url}&t={self.timestamp_seconds}s"
        return self.source_url

    @property
    def citation_label(self) -> str:
        if self.source_type == SourceType.VIDEO and self.timestamp_seconds is not None:
            minutes, seconds = divmod(self.timestamp_seconds, 60)
            return f"{self.source_name} @ {minutes:02d}:{seconds:02d}"
        if self.section_heading:
            return f"{self.source_name} — {self.section_heading}"
        return self.source_name

    @property
    def citation_markdown(self) -> str:
        score_pct = round(self.relevance_score * 100)
        return f"[{self.citation_label}]({self.citation_url}) ({score_pct}%)"


class Citation(BaseModel):
    """A single citation in a generated answer."""

    label: str
    url: str
    relevance_score: float
    source_type: SourceType


class CitedAnswer(BaseModel):
    """LLM-generated answer with structured citations."""

    answer: str
    citations: list[Citation]
