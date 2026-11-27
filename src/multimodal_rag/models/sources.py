"""Source configuration models for ingestion pipeline."""

from pydantic import AliasChoices, BaseModel, Field, HttpUrl


class YouTubeSource(BaseModel):
    url: HttpUrl
    name: str
    # A silent screen recording: no transcript, one frame described every 5 seconds.
    # `skip_voxtral` is the old name, still accepted: bucket registries use it.
    silent: bool = Field(
        default=False, validation_alias=AliasChoices("silent", "skip_voxtral")
    )


class KnowledgeBaseSource(BaseModel):
    url: HttpUrl
    name: str


class SourceConfig(BaseModel):
    youtube: list[YouTubeSource] = Field(default_factory=list)
    knowledge_bases: list[KnowledgeBaseSource] = Field(default_factory=list)

    @property
    def kb_sources(self) -> list[KnowledgeBaseSource]:
        return self.knowledge_bases
