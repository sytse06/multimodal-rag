"""Tests for the ingestion CLI orchestrator."""

from unittest.mock import MagicMock, patch

from multimodal_rag.ingest.__main__ import load_sources, run
from multimodal_rag.models.chunks import TranscriptChunk


class TestLoadSources:
    def test_loads_sources_yaml(self, tmp_path: object) -> None:
        sources = load_sources()
        assert len(sources.youtube) >= 1
        assert sources.youtube[0].name is not None


class TestRun:
    @patch("multimodal_rag.ingest.__main__.WeaviateStore")
    @patch("multimodal_rag.ingest.__main__.create_embeddings")
    @patch("multimodal_rag.ingest.__main__.fetch_web_chunks")
    @patch("multimodal_rag.ingest.__main__.fetch_transcript_chunks")
    @patch("multimodal_rag.ingest.__main__.load_sources")
    @patch("multimodal_rag.ingest.__main__.AppSettings")
    def test_ingests_youtube_sources(
        self,
        mock_settings_cls: MagicMock,
        mock_load: MagicMock,
        mock_yt: MagicMock,
        mock_web: MagicMock,
        mock_create_emb: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        settings = MagicMock()
        settings.log_level = "WARNING"
        settings.chunk_size = 400
        settings.firecrawl_api_key = "fake"
        settings.weaviate_url = "http://localhost:8080"
        mock_settings_cls.return_value = settings
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
        mock_web.return_value = []

        mock_store = MagicMock()
        mock_store_cls.return_value.__enter__ = MagicMock(
            return_value=mock_store
        )
        mock_store_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_store.add_chunks.return_value = 1
        mock_store.count.return_value = 1

        run()

        mock_create_emb.assert_called_once_with(settings)
        mock_yt.assert_called_once()
        mock_store.ensure_collection.assert_called_once()
        mock_store.add_chunks.assert_called_once()

    @patch("multimodal_rag.ingest.__main__.WeaviateStore")
    @patch("multimodal_rag.ingest.__main__.create_embeddings")
    @patch("multimodal_rag.ingest.__main__.fetch_web_chunks")
    @patch("multimodal_rag.ingest.__main__.fetch_transcript_chunks")
    @patch("multimodal_rag.ingest.__main__.load_sources")
    @patch("multimodal_rag.ingest.__main__.AppSettings")
    def test_no_chunks_skips_store(
        self,
        mock_settings_cls: MagicMock,
        mock_load: MagicMock,
        mock_yt: MagicMock,
        mock_web: MagicMock,
        mock_create_emb: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        settings = MagicMock()
        settings.log_level = "WARNING"
        settings.chunk_size = 400
        settings.firecrawl_api_key = "fake"
        mock_settings_cls.return_value = settings

        from multimodal_rag.models.sources import SourceConfig

        mock_load.return_value = SourceConfig(youtube=[])
        mock_yt.return_value = []
        mock_web.return_value = []

        run()

        mock_store_cls.assert_not_called()
