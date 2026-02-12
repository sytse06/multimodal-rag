"""Tests for store module (embeddings + Weaviate store)."""

from unittest.mock import MagicMock

from multimodal_rag.models.chunks import (
    SourceType,
    SupportChunk,
    TranscriptChunk,
    WebChunk,
)
from multimodal_rag.store.embeddings import embed_texts


class TestEmbedTexts:
    def test_returns_embeddings(self) -> None:
        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        result = embed_texts(["hello"], embeddings=mock_emb)
        assert result == [[0.1, 0.2, 0.3]]
        mock_emb.embed_documents.assert_called_once_with(["hello"])

    def test_empty_list(self) -> None:
        mock_emb = MagicMock()
        result = embed_texts([], embeddings=mock_emb)
        assert result == []
        mock_emb.embed_documents.assert_not_called()

    def test_batching(self) -> None:
        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = [[0.1]]
        texts = [f"text_{i}" for i in range(150)]
        result = embed_texts(texts, embeddings=mock_emb)
        # 150 texts / 100 batch size = 2 calls
        assert mock_emb.embed_documents.call_count == 2
        assert len(result) == 2


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
