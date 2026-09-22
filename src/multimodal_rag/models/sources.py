"""Source configuration models for ingestion pipeline."""

from pydantic import BaseModel, Field, HttpUrl


class YouTubeSource(BaseModel):
    url: HttpUrl
    name: str
    skip_voxtral: bool = False


class KnowledgeBaseSource(BaseModel):
    url: HttpUrl
    name: str


class SourceConfig(BaseModel):
    youtube: list[YouTubeSource] = Field(default_factory=list)
    knowledge_bases: list[KnowledgeBaseSource] = Field(default_factory=list)

    @property
    def kb_sources(self) -> list[KnowledgeBaseSource]:
        return self.knowledge_bases
