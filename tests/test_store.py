"""Tests for store module (embeddings + Weaviate store)."""

from unittest.mock import MagicMock, patch

from multimodal_rag.models.chunks import (
    SourceType,
    SupportChunk,
    TranscriptChunk,
    WebChunk,
)
from multimodal_rag.store.embeddings import embed_texts


class TestEmbedTexts:
    @patch("multimodal_rag.store.embeddings.OpenAI")
    def test_returns_embeddings(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_item = MagicMock()
        mock_item.embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create.return_value = MagicMock(data=[mock_item])

        result = embed_texts(["hello"], api_key="fake", base_url="http://test")
        assert result == [[0.1, 0.2, 0.3]]
        mock_openai_cls.assert_called_once_with(api_key="fake", base_url="http://test")

    @patch("multimodal_rag.store.embeddings.OpenAI")
    def test_empty_list(self, mock_openai_cls: MagicMock) -> None:
        result = embed_texts([], api_key="fake")
        assert result == []
        mock_openai_cls.assert_not_called()

    @patch("multimodal_rag.store.embeddings.OpenAI")
    def test_batching(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_item = MagicMock()
        mock_item.embedding = [0.1]
        mock_client.embeddings.create.return_value = MagicMock(data=[mock_item])

        texts = [f"text_{i}" for i in range(150)]
        result = embed_texts(texts, api_key="fake")
        # 150 texts / 100 batch size = 2 calls
        assert mock_client.embeddings.create.call_count == 2
        assert len(result) == 2  # 1 per call (mock returns 1 per batch)


class TestSupportChunkConversion:
    def test_from_transcript_chunk(self) -> None:
        tc = TranscriptChunk(
            text="Hello world",
            source_url="https://yt.com/watch?v=abc",
            source_name="Test Video",
            start_seconds=90,
            end_seconds=120,
        )
        sc = SupportChunk.from_transcript_chunk(tc)
        assert sc.source_type == SourceType.VIDEO
        assert sc.timestamp_seconds == 90
        assert sc.text == "Hello world"
        assert sc.url_hash != ""

    def test_from_web_chunk(self) -> None:
        wc = WebChunk(
            text="Some content",
            source_url="https://example.com/page",
            source_name="Docs",
            section_heading="Getting Started",
        )
        sc = SupportChunk.from_web_chunk(wc)
        assert sc.source_type == SourceType.WEB
        assert sc.section_heading == "Getting Started"
        assert sc.timestamp_seconds is None
