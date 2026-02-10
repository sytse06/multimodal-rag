"""Source configuration models for ingestion pipeline."""

from pydantic import BaseModel, HttpUrl


class YouTubeSource(BaseModel):
    url: HttpUrl
    name: str


class KnowledgeBaseSource(BaseModel):
    url: HttpUrl
    name: str


class SourceConfig(BaseModel):
    youtube: list[YouTubeSource] = []
    knowledge_bases: list[KnowledgeBaseSource] | None = None

    @property
    def kb_sources(self) -> list[KnowledgeBaseSource]:
        return self.knowledge_bases or []
