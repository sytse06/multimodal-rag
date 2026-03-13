"""Tests for store module (embeddings + Weaviate store)."""

from unittest.mock import MagicMock, patch

from multimodal_rag.models.chunks import (
    SourceType,
    SupportChunk,
    TranscriptChunk,
    WebChunk,
)
from multimodal_rag.store.embeddings import _MAX_WORDS, _RETRY_MAX_WORDS, embed_texts
from multimodal_rag.store.weaviate import WeaviateStore


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
        texts = [f"text_{i}" for i in range(12)]
        result = embed_texts(texts, embeddings=mock_emb)
        # 12 texts / 5 batch size = 3 calls
        assert mock_emb.embed_documents.call_count == 3
        assert len(result) == 3

    def test_truncates_long_texts(self) -> None:
        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = [[0.1]]
        long_text = "word " * 2000
        embed_texts([long_text], embeddings=mock_emb)
        called_texts = mock_emb.embed_documents.call_args[0][0]
        assert len(called_texts[0].split()) == _MAX_WORDS

    def test_context_length_retry(self) -> None:
        mock_emb = MagicMock()
        # First call raises context length error, retry succeeds
        mock_emb.embed_documents.side_effect = [
            Exception("the input length exceeds the context length"),
            [[0.1, 0.2]],
        ]
        long_text = "word " * 500
        result = embed_texts([long_text], embeddings=mock_emb)
        assert result == [[0.1, 0.2]]
        assert mock_emb.embed_documents.call_count == 2
        # Retry batch should be truncated to _RETRY_MAX_WORDS
        retry_texts = mock_emb.embed_documents.call_args_list[1][0][0]
        assert len(retry_texts[0].split()) == _RETRY_MAX_WORDS

    def test_non_context_error_propagates(self) -> None:
        mock_emb = MagicMock()
        mock_emb.embed_documents.side_effect = ConnectionError("network down")
        try:
            embed_texts(["hello"], embeddings=mock_emb)
            assert False, "Should have raised"
        except ConnectionError:
            pass


class TestWeaviateStoreDeleteBySourceType:
    def _make_store(self) -> tuple[WeaviateStore, MagicMock]:
        mock_client = MagicMock()
        mock_embeddings = MagicMock()
        patch_target = "multimodal_rag.store.weaviate.weaviate.connect_to_local"
        with patch(patch_target, return_value=mock_client):
            store = WeaviateStore.__new__(WeaviateStore)
            store._client = mock_client
            store._embeddings = mock_embeddings
        return store, mock_client

    def test_delete_video_chunks(self) -> None:
        store, mock_client = self._make_store()
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_result = MagicMock()
        mock_result.successful = 7
        mock_collection.data.delete_many.return_value = mock_result

        n = store.delete_by_source_type("video")

        assert n == 7
        mock_collection.data.delete_many.assert_called_once()
        call_kwargs = mock_collection.data.delete_many.call_args
        # verify the filter targets source_type property
        assert call_kwargs is not None

    def test_delete_web_chunks(self) -> None:
        store, mock_client = self._make_store()
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_result = MagicMock()
        mock_result.successful = 3
        mock_collection.data.delete_many.return_value = mock_result

        n = store.delete_by_source_type("web")

        assert n == 3

    def test_returns_zero_when_result_is_none(self) -> None:
        store, mock_client = self._make_store()
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_collection.data.delete_many.return_value = None

        n = store.delete_by_source_type("video")

        assert n == 0


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
