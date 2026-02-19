"""Tests for data models."""

import pytest
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

    def test_chunk_id_deterministic(self) -> None:
        kwargs = dict(
            text="Same text",
            source_type=SourceType.WEB,
            source_url="https://example.com",
            source_name="Test",
        )
        sc1 = SupportChunk(**kwargs)
        sc2 = SupportChunk(**kwargs)
        assert sc1.chunk_id == sc2.chunk_id

    def test_chunk_id_differs_by_text(self) -> None:
        sc1 = SupportChunk(
            text="alpha",
            source_type=SourceType.WEB,
            source_url="https://example.com",
            source_name="Test",
        )
        sc2 = SupportChunk(
            text="beta",
            source_type=SourceType.WEB,
            source_url="https://example.com",
            source_name="Test",
        )
        assert sc1.chunk_id != sc2.chunk_id

    def test_chunk_id_differs_by_url(self) -> None:
        sc1 = SupportChunk(
            text="same",
            source_type=SourceType.WEB,
            source_url="https://example.com/a",
            source_name="Test",
        )
        sc2 = SupportChunk(
            text="same",
            source_type=SourceType.WEB,
            source_url="https://example.com/b",
            source_name="Test",
        )
        assert sc1.chunk_id != sc2.chunk_id

    def test_from_transcript_chunk_stable_id(self) -> None:
        """from_transcript_chunk produces same UUID regardless of text."""
        tc_a = TranscriptChunk(
            text="First transcription",
            source_url="https://youtube.com/watch?v=abc",
            source_name="Vid",
            start_seconds=30,
            end_seconds=60,
        )
        tc_b = TranscriptChunk(
            text="Completely different transcription",
            source_url="https://youtube.com/watch?v=abc",
            source_name="Vid",
            start_seconds=30,
            end_seconds=60,
        )
        assert SupportChunk.from_transcript_chunk(tc_a).chunk_id == \
            SupportChunk.from_transcript_chunk(tc_b).chunk_id

    def test_from_transcript_chunk_differs_by_timestamp(self) -> None:
        tc1 = TranscriptChunk(
            text="Same text",
            source_url="https://youtube.com/watch?v=abc",
            source_name="Vid",
            start_seconds=0,
            end_seconds=30,
        )
        tc2 = TranscriptChunk(
            text="Same text",
            source_url="https://youtube.com/watch?v=abc",
            source_name="Vid",
            start_seconds=30,
            end_seconds=60,
        )
        assert SupportChunk.from_transcript_chunk(tc1).chunk_id != \
            SupportChunk.from_transcript_chunk(tc2).chunk_id

    def test_from_web_chunk_stable_id(self) -> None:
        """from_web_chunk produces same UUID regardless of text when chunk_index set."""
        wc_a = WebChunk(
            text="First version of this chunk",
            source_url="https://docs.example.com/page",
            source_name="Docs",
            chunk_index=2,
        )
        wc_b = WebChunk(
            text="Completely rewritten chunk text",
            source_url="https://docs.example.com/page",
            source_name="Docs",
            chunk_index=2,
        )
        assert SupportChunk.from_web_chunk(wc_a).chunk_id == \
            SupportChunk.from_web_chunk(wc_b).chunk_id

    def test_from_web_chunk_differs_by_index(self) -> None:
        wc1 = WebChunk(
            text="Same text",
            source_url="https://docs.example.com/page",
            source_name="Docs",
            chunk_index=0,
        )
        wc2 = WebChunk(
            text="Same text",
            source_url="https://docs.example.com/page",
            source_name="Docs",
            chunk_index=1,
        )
        assert SupportChunk.from_web_chunk(wc1).chunk_id != \
            SupportChunk.from_web_chunk(wc2).chunk_id

    def test_from_web_chunk_without_index_falls_back_to_text(self) -> None:
        """WebChunks without chunk_index still get a UUID via model_post_init."""
        wc = WebChunk(
            text="Some text",
            source_url="https://docs.example.com/page",
            source_name="Docs",
        )
        sc = SupportChunk.from_web_chunk(wc)
        assert sc.chunk_id is not None

    def test_from_frame_chunk_stable_id(self) -> None:
        """from_frame_chunk produces same UUID regardless of LLM description."""
        tc_a = TranscriptChunk(
            text="Description A",
            source_url="https://youtube.com/watch?v=abc",
            source_name="Vid",
            start_seconds=60,
            end_seconds=90,
        )
        tc_b = TranscriptChunk(
            text="Description B — completely different",
            source_url="https://youtube.com/watch?v=abc",
            source_name="Vid",
            start_seconds=60,
            end_seconds=90,
        )
        id_a = SupportChunk.from_frame_chunk(tc_a).chunk_id
        id_b = SupportChunk.from_frame_chunk(tc_b).chunk_id
        assert id_a == id_b

    def test_from_frame_chunk_differs_by_timestamp(self) -> None:
        tc1 = TranscriptChunk(
            text="Same text",
            source_url="https://youtube.com/watch?v=abc",
            source_name="Vid",
            start_seconds=0,
            end_seconds=30,
        )
        tc2 = TranscriptChunk(
            text="Same text",
            source_url="https://youtube.com/watch?v=abc",
            source_name="Vid",
            start_seconds=30,
            end_seconds=60,
        )
        id1 = SupportChunk.from_frame_chunk(tc1).chunk_id
        id2 = SupportChunk.from_frame_chunk(tc2).chunk_id
        assert id1 != id2

    def test_from_screenshot_chunk_stable_id(self) -> None:
        """from_screenshot_chunk produces same UUID regardless of LLM description."""
        wc_a = WebChunk(
            text="Description A",
            source_url="https://docs.example.com/page",
            source_name="Docs",
            image_url="https://cdn.example.com/screen.png",
        )
        wc_b = WebChunk(
            text="Description B — completely different",
            source_url="https://docs.example.com/page",
            source_name="Docs",
            image_url="https://cdn.example.com/screen.png",
        )
        id_a = SupportChunk.from_screenshot_chunk(wc_a).chunk_id
        id_b = SupportChunk.from_screenshot_chunk(wc_b).chunk_id
        assert id_a == id_b

    def test_from_screenshot_chunk_differs_by_image_url(self) -> None:
        wc1 = WebChunk(
            text="Same text",
            source_url="https://docs.example.com/page",
            source_name="Docs",
            image_url="https://cdn.example.com/img_a.png",
        )
        wc2 = WebChunk(
            text="Same text",
            source_url="https://docs.example.com/page",
            source_name="Docs",
            image_url="https://cdn.example.com/img_b.png",
        )
        id1 = SupportChunk.from_screenshot_chunk(wc1).chunk_id
        id2 = SupportChunk.from_screenshot_chunk(wc2).chunk_id
        assert id1 != id2

    def test_from_screenshot_chunk_raises_without_image_url(self) -> None:
        import pytest
        wc = WebChunk(
            text="Some description",
            source_url="https://docs.example.com/page",
            source_name="Docs",
        )
        with pytest.raises(ValueError, match="image_url"):
            SupportChunk.from_screenshot_chunk(wc)


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
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        settings = AppSettings(
            _env_file=None,
            openrouter_api_key="test",
            firecrawl_api_key="test",
        )
        assert settings.llm_model == "google/gemini-3-flash-preview"
        assert settings.embedding_model == "nomic-embed-text-16k"
        assert settings.weaviate_url == "http://localhost:8080"
        assert settings.chunk_size == 400
        assert settings.top_k == 5
