"""Tests for the ingestion CLI orchestrator."""

from unittest.mock import MagicMock, patch

from multimodal_rag.ingest.__main__ import load_sources, run
from multimodal_rag.models.chunks import TranscriptChunk


class TestLoadSources:
    def test_loads_sources_yaml(self, tmp_path: object) -> None:
        sources = load_sources()
        assert len(sources.youtube) >= 1
        assert sources.youtube[0].name is not None


def _make_settings() -> MagicMock:
    settings = MagicMock()
    settings.log_level = "WARNING"
    settings.chunk_size = 400
    settings.firecrawl_api_key = "fake"
    settings.weaviate_url = "http://localhost:8080"
    settings.mistral_api_key = ""
    return settings


def _make_store() -> MagicMock:
    mock_store = MagicMock()
    mock_store.add_chunks.return_value = 1
    mock_store.count.return_value = 1
    return mock_store


def _setup_store_cls(mock_store_cls: MagicMock, mock_store: MagicMock) -> None:
    mock_store_cls.return_value.__enter__ = MagicMock(return_value=mock_store)
    mock_store_cls.return_value.__exit__ = MagicMock(return_value=False)


class TestRun:
    @patch("multimodal_rag.ingest.__main__.WeaviateStore")
    @patch("multimodal_rag.ingest.__main__.create_embeddings")
    @patch("multimodal_rag.ingest.__main__.fetch_transcript_chunks")
    @patch("multimodal_rag.ingest.__main__.load_sources")
    @patch("multimodal_rag.ingest.__main__.AppSettings")
    def test_ingests_youtube_per_video(
        self,
        mock_settings_cls: MagicMock,
        mock_load: MagicMock,
        mock_yt: MagicMock,
        mock_create_emb: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        mock_settings_cls.return_value = _make_settings()
        mock_create_emb.return_value = MagicMock()

        from multimodal_rag.models.sources import SourceConfig, YouTubeSource

        mock_load.return_value = SourceConfig(
            youtube=[
                YouTubeSource(
                    url="https://www.youtube.com/watch?v=abc",
                    name="Test",
                )
            ]
        )

        mock_yt.return_value = [
            TranscriptChunk(
                text="Hello",
                source_url="https://yt.com/watch?v=abc",
                source_name="Test",
                start_seconds=0,
                end_seconds=10,
            )
        ]

        mock_store = _make_store()
        _setup_store_cls(mock_store_cls, mock_store)

        run()

        mock_create_emb.assert_called_once()
        mock_yt.assert_called_once_with(
            video_url="https://www.youtube.com/watch?v=abc",
            source_name="Test",
            target_tokens=400,
            mistral_api_key="",
        )
        mock_store.ensure_collection.assert_called_once()
        mock_store.add_chunks.assert_called_once()

    @patch("multimodal_rag.ingest.__main__.WeaviateStore")
    @patch("multimodal_rag.ingest.__main__.create_embeddings")
    @patch("multimodal_rag.ingest.__main__.crawl_knowledge_base")
    @patch("multimodal_rag.ingest.__main__.fetch_transcript_chunks")
    @patch("multimodal_rag.ingest.__main__.load_sources")
    @patch("multimodal_rag.ingest.__main__.AppSettings")
    def test_ingests_kb_per_page(
        self,
        mock_settings_cls: MagicMock,
        mock_load: MagicMock,
        mock_yt: MagicMock,
        mock_crawl: MagicMock,
        mock_create_emb: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        mock_settings_cls.return_value = _make_settings()
        mock_create_emb.return_value = MagicMock()

        from multimodal_rag.models.sources import (
            KnowledgeBaseSource,
            SourceConfig,
        )

        mock_load.return_value = SourceConfig(
            youtube=[],
            knowledge_bases=[
                KnowledgeBaseSource(
                    url="https://docs.example.com",
                    name="Docs",
                )
            ],
        )
        mock_yt.return_value = []
        mock_crawl.return_value = [
            {
                "url": "https://docs.example.com/p1",
                "title": "P1",
                "content": "Page one content",
            },
            {
                "url": "https://docs.example.com/p2",
                "title": "P2",
                "content": "Page two content",
            },
        ]

        mock_store = _make_store()
        _setup_store_cls(mock_store_cls, mock_store)

        run()

        # add_chunks called once per page
        assert mock_store.add_chunks.call_count == 2

    @patch("multimodal_rag.ingest.__main__.WeaviateStore")
    @patch("multimodal_rag.ingest.__main__.create_embeddings")
    @patch("multimodal_rag.ingest.__main__.fetch_transcript_chunks")
    @patch("multimodal_rag.ingest.__main__.load_sources")
    @patch("multimodal_rag.ingest.__main__.AppSettings")
    def test_youtube_failure_continues(
        self,
        mock_settings_cls: MagicMock,
        mock_load: MagicMock,
        mock_yt: MagicMock,
        mock_create_emb: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        mock_settings_cls.return_value = _make_settings()
        mock_create_emb.return_value = MagicMock()

        from multimodal_rag.models.sources import SourceConfig, YouTubeSource

        mock_load.return_value = SourceConfig(
            youtube=[
                YouTubeSource(url="https://www.youtube.com/watch?v=fail", name="Bad"),
                YouTubeSource(url="https://www.youtube.com/watch?v=ok", name="Good"),
            ]
        )

        mock_yt.side_effect = [
            RuntimeError("transcript unavailable"),
            [
                TranscriptChunk(
                    text="Hello",
                    source_url="https://yt.com/watch?v=ok",
                    source_name="Good",
                    start_seconds=0,
                    end_seconds=10,
                )
            ],
        ]

        mock_store = _make_store()
        _setup_store_cls(mock_store_cls, mock_store)

        run()

        # First video fails, second succeeds
        assert mock_yt.call_count == 2
        mock_store.add_chunks.assert_called_once()

    @patch("multimodal_rag.ingest.__main__.WeaviateStore")
    @patch("multimodal_rag.ingest.__main__.create_embeddings")
    @patch("multimodal_rag.ingest.__main__.crawl_knowledge_base")
    @patch("multimodal_rag.ingest.__main__.fetch_transcript_chunks")
    @patch("multimodal_rag.ingest.__main__.load_sources")
    @patch("multimodal_rag.ingest.__main__.AppSettings")
    def test_kb_page_failure_continues(
        self,
        mock_settings_cls: MagicMock,
        mock_load: MagicMock,
        mock_yt: MagicMock,
        mock_crawl: MagicMock,
        mock_create_emb: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        mock_settings_cls.return_value = _make_settings()
        mock_create_emb.return_value = MagicMock()

        from multimodal_rag.models.sources import (
            KnowledgeBaseSource,
            SourceConfig,
        )

        mock_load.return_value = SourceConfig(
            youtube=[],
            knowledge_bases=[
                KnowledgeBaseSource(
                    url="https://docs.example.com",
                    name="Docs",
                )
            ],
        )
        mock_yt.return_value = []
        mock_crawl.return_value = [
            {
                "url": "https://docs.example.com/p1",
                "title": "P1",
                "content": "Page one",
            },
            {
                "url": "https://docs.example.com/p2",
                "title": "P2",
                "content": "Page two",
            },
        ]

        mock_store = _make_store()
        _setup_store_cls(mock_store_cls, mock_store)
        # First page store fails, second succeeds
        mock_store.add_chunks.side_effect = [RuntimeError("embed fail"), 1]

        run()

        assert mock_store.add_chunks.call_count == 2

    @patch("multimodal_rag.ingest.__main__.WeaviateStore")
    @patch("multimodal_rag.ingest.__main__.create_embeddings")
    @patch("multimodal_rag.ingest.__main__.fetch_transcript_chunks")
    @patch("multimodal_rag.ingest.__main__.load_sources")
    @patch("multimodal_rag.ingest.__main__.AppSettings")
    def test_no_chunks_opens_store_but_adds_nothing(
        self,
        mock_settings_cls: MagicMock,
        mock_load: MagicMock,
        mock_yt: MagicMock,
        mock_create_emb: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        mock_settings_cls.return_value = _make_settings()
        mock_create_emb.return_value = MagicMock()

        from multimodal_rag.models.sources import SourceConfig

        mock_load.return_value = SourceConfig(youtube=[])
        mock_yt.return_value = []

        mock_store = _make_store()
        _setup_store_cls(mock_store_cls, mock_store)

        run()

        mock_store.ensure_collection.assert_called_once()
        mock_store.add_chunks.assert_not_called()
