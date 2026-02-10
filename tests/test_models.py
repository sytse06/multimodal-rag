"""Tests for data models."""

import yaml

from multimodal_rag.models import (
    AppSettings,
    Citation,
    CitedAnswer,
    SearchResult,
    SourceConfig,
    SupportChunk,
    TranscriptChunk,
    WebChunk,
)
from multimodal_rag.models.chunks import SourceType


class TestSourceConfig:
    def test_parse_sources_yaml(self) -> None:
        raw = {
            "youtube": [
                {"url": "https://youtube.com/watch?v=abc123", "name": "Test Video"}
            ],
            "knowledge_bases": [
                {"url": "https://docs.example.com", "name": "Docs"}
            ],
        }
        config = SourceConfig(**raw)
        assert len(config.youtube) == 1
        assert len(config.knowledge_bases) == 1
        assert config.youtube[0].name == "Test Video"
        assert config.knowledge_bases[0].name == "Docs"

    def test_empty_sources(self) -> None:
        config = SourceConfig()
        assert config.youtube == []
        assert config.kb_sources == []

    def test_sources_yaml_file(self) -> None:
        with open("config/sources.yaml") as f:
            raw = yaml.safe_load(f)
        config = SourceConfig(**raw)
        assert len(config.youtube) >= 1


class TestTranscriptChunk:
    def test_timestamp_url(self) -> None:
        chunk = TranscriptChunk(
            text="Hello world",
            source_url="https://youtube.com/watch?v=abc123",
            source_name="Test",
            start_seconds=90,
            end_seconds=120,
        )
        assert chunk.timestamp_url == "https://youtube.com/watch?v=abc123&t=90s"

    def test_timestamp_display(self) -> None:
        chunk = TranscriptChunk(
            text="Hello",
            source_url="https://youtube.com/watch?v=abc123",
            source_name="Test",
            start_seconds=135,
            end_seconds=150,
        )
        assert chunk.timestamp_display == "02:15"


class TestWebChunk:
    def test_basic(self) -> None:
        chunk = WebChunk(
            text="Some content",
            source_url="https://docs.example.com/page",
            source_name="Docs",
            section_heading="Getting Started",
        )
        assert chunk.section_heading == "Getting Started"

    def test_no_section(self) -> None:
        chunk = WebChunk(
            text="Content",
            source_url="https://docs.example.com",
            source_name="Docs",
        )
        assert chunk.section_heading is None


class TestSupportChunk:
    def test_from_transcript_chunk(self) -> None:
        tc = TranscriptChunk(
            text="Welcome to the tutorial",
            source_url="https://youtube.com/watch?v=abc123",
            source_name="Tutorial",
            start_seconds=10,
            end_seconds=25,
        )
        sc = SupportChunk.from_transcript_chunk(tc)
        assert sc.source_type == SourceType.VIDEO
        assert sc.timestamp_seconds == 10
        assert sc.url_hash != ""

    def test_from_web_chunk(self) -> None:
        wc = WebChunk(
            text="Documentation content",
            source_url="https://docs.example.com/intro",
            source_name="Docs",
            section_heading="Introduction",
        )
        sc = SupportChunk.from_web_chunk(wc)
        assert sc.source_type == SourceType.WEB
        assert sc.section_heading == "Introduction"
        assert sc.timestamp_seconds is None

    def test_url_hash_deterministic(self) -> None:
        sc1 = SupportChunk(
            text="a",
            source_type=SourceType.WEB,
            source_url="https://example.com",
            source_name="Test",
        )
        sc2 = SupportChunk(
            text="b",
            source_type=SourceType.WEB,
            source_url="https://example.com",
            source_name="Test",
        )
        assert sc1.url_hash == sc2.url_hash

    def test_different_urls_different_hash(self) -> None:
        sc1 = SupportChunk(
            text="a",
            source_type=SourceType.WEB,
            source_url="https://example.com/a",
            source_name="Test",
        )
        sc2 = SupportChunk(
            text="a",
            source_type=SourceType.WEB,
            source_url="https://example.com/b",
            source_name="Test",
        )
        assert sc1.url_hash != sc2.url_hash


class TestSearchResult:
    def test_video_citation_markdown(self) -> None:
        result = SearchResult(
            text="Some content",
            source_type=SourceType.VIDEO,
            source_url="https://youtube.com/watch?v=abc123",
            source_name="Tutorial",
            timestamp_seconds=135,
            relevance_score=0.92,
        )
        md = result.citation_markdown
        assert "Tutorial @ 02:15" in md
        assert "t=135s" in md
        assert "92%" in md

    def test_web_citation_markdown(self) -> None:
        result = SearchResult(
            text="Content",
            source_type=SourceType.WEB,
            source_url="https://docs.example.com/page",
            source_name="Docs",
            section_heading="Setup",
            relevance_score=0.85,
        )
        md = result.citation_markdown
        assert "Docs — Setup" in md
        assert "85%" in md

    def test_web_citation_no_section(self) -> None:
        result = SearchResult(
            text="Content",
            source_type=SourceType.WEB,
            source_url="https://docs.example.com",
            source_name="Docs",
            relevance_score=0.75,
        )
        assert result.citation_label == "Docs"


class TestCitedAnswer:
    def test_structure(self) -> None:
        answer = CitedAnswer(
            answer="Here is the answer.",
            citations=[
                Citation(
                    label="Tutorial @ 01:30",
                    url="https://youtube.com/watch?v=abc&t=90s",
                    relevance_score=0.9,
                    source_type=SourceType.VIDEO,
                ),
            ],
        )
        assert len(answer.citations) == 1
        assert answer.citations[0].source_type == SourceType.VIDEO


class TestAppSettings:
    def test_defaults(self) -> None:
        settings = AppSettings(
            openrouter_api_key="test",
            firecrawl_api_key="test",
        )
        assert settings.llm_model == "openai/gpt-4o-mini"
        assert settings.embedding_model == "openai/text-embedding-3-small"
        assert settings.weaviate_url == "http://localhost:8080"
        assert settings.chunk_size == 400
        assert settings.top_k == 5
