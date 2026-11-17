"""Tests for store module (embeddings + Weaviate store)."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from multimodal_rag.models.chunks import (
    SourceType,
    SupportChunk,
    TranscriptChunk,
    WebChunk,
)
from multimodal_rag.store.embeddings import _MAX_WORDS, _RETRY_MAX_WORDS, embed_texts
from multimodal_rag.store.weaviate import (
    EXPECTED_PROPERTIES,
    OPTIONAL_PROPERTIES,
    WeaviateStore,
)


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
            store._read_only = False
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


class TestWeaviateStoreConnection:
    def test_connects_to_local_url(self) -> None:
        client = MagicMock()
        with patch(
            "multimodal_rag.store.weaviate.weaviate.connect_to_local",
            return_value=client,
        ) as connect:
            store = WeaviateStore("http://localhost:9090", MagicMock())

        connect.assert_called_once_with(host="localhost", port=9090, grpc_port=50051)
        store.close()
        client.close.assert_called_once()

    def test_connects_to_cloud_with_api_key(self) -> None:
        client = MagicMock()
        with patch(
            "multimodal_rag.store.weaviate.weaviate.connect_to_weaviate_cloud",
            return_value=client,
        ) as connect:
            store = WeaviateStore(
                "https://example.weaviate.network",
                MagicMock(),
                weaviate_mode="cloud",
                weaviate_api_key="secret",
            )

        assert connect.call_args.kwargs["cluster_url"] == "example.weaviate.network"
        assert connect.call_args.kwargs["auth_credentials"].api_key == "secret"
        store.close()

    def test_cloud_requires_api_key(self) -> None:
        with patch(
            "multimodal_rag.store.weaviate.weaviate.connect_to_weaviate_cloud"
        ) as connect:
            try:
                WeaviateStore(
                    "https://example.weaviate.network",
                    MagicMock(),
                    weaviate_mode="cloud",
                )
            except ValueError as exc:
                assert "API key" in str(exc)
            else:
                raise AssertionError("Expected a missing API key error")
        connect.assert_not_called()

    def test_local_rejects_hosted_endpoint(self) -> None:
        with patch(
            "multimodal_rag.store.weaviate.weaviate.connect_to_local"
        ) as connect:
            with pytest.raises(ValueError, match="points to a hosted endpoint"):
                WeaviateStore(
                    "http://example.weaviate.cloud",
                    MagicMock(),
                    weaviate_mode="local",
                )
        connect.assert_not_called()


class TestWeaviateStoreCompatibility:
    def _make_store(
        self, vector: list[float], properties: dict[str, str] | None = None
    ) -> WeaviateStore:
        store = WeaviateStore.__new__(WeaviateStore)
        store._read_only = False
        collection = MagicMock()
        collection.config.get.return_value = SimpleNamespace(
            properties=[
                SimpleNamespace(name=name, data_type=SimpleNamespace(value=value))
                for name, value in (properties or EXPECTED_PROPERTIES).items()
            ]
        )
        collection.query.fetch_objects.return_value = SimpleNamespace(
            objects=[SimpleNamespace(vector=vector)]
        )
        client = MagicMock()
        client.collections.exists.return_value = True
        client.collections.get.return_value = collection
        store._client = client
        store._embeddings = MagicMock()
        store._tenant = None
        store._embed = MagicMock(return_value=[vector])
        return store

    def test_validates_schema_and_vector_dimension(self) -> None:
        store = self._make_store([0.1, 0.2])

        store.validate_compatibility(expected_vector_dimension=2)

        store._client.collections.get.return_value.query.fetch_objects.assert_called_once_with(
            limit=1, include_vector=True
        )
        store._embed.assert_called_once_with(["compatibility check"])

    def test_reports_missing_collection(self) -> None:
        store = self._make_store([0.1, 0.2])
        store._client.collections.exists.return_value = False

        with pytest.raises(ValueError, match="does not exist"):
            store.validate_compatibility(expected_vector_dimension=2)

    def test_reports_vector_dimension_mismatch(self) -> None:
        store = self._make_store([0.1, 0.2])

        with pytest.raises(ValueError, match="vector dimension"):
            store.validate_compatibility(expected_vector_dimension=3)

    def test_accepts_the_optional_sync_properties(self) -> None:
        store = self._make_store(
            [0.1, 0.2], {**EXPECTED_PROPERTIES, **OPTIONAL_PROPERTIES}
        )

        store.validate_compatibility(expected_vector_dimension=2)

    def test_accepts_unknown_extra_properties(self) -> None:
        store = self._make_store(
            [0.1, 0.2], {**EXPECTED_PROPERTIES, "future_field": "text"}
        )

        store.validate_compatibility(expected_vector_dimension=2)

    def test_rejects_a_missing_required_property(self) -> None:
        properties = {k: v for k, v in EXPECTED_PROPERTIES.items() if k != "text"}
        store = self._make_store([0.1, 0.2], properties)

        with pytest.raises(ValueError, match="schema is incompatible.*'text'"):
            store.validate_compatibility(expected_vector_dimension=2)

    def test_rejects_a_wrong_type_on_a_required_property(self) -> None:
        store = self._make_store(
            [0.1, 0.2], {**EXPECTED_PROPERTIES, "timestamp_seconds": "text"}
        )

        with pytest.raises(ValueError, match="timestamp_seconds"):
            store.validate_compatibility(expected_vector_dimension=2)

    def test_rejects_a_wrong_type_on_a_known_optional_property(self) -> None:
        store = self._make_store(
            [0.1, 0.2], {**EXPECTED_PROPERTIES, "image_url": "int"}
        )

        with pytest.raises(ValueError, match="image_url"):
            store.validate_compatibility(expected_vector_dimension=2)


class TestEnsureCollection:
    def test_creates_required_and_optional_properties(self) -> None:
        store = WeaviateStore.__new__(WeaviateStore)
        store._read_only = False
        client = MagicMock()
        client.collections.exists.return_value = False
        store._client = client

        store.ensure_collection()

        created = client.collections.create.call_args.kwargs["properties"]
        assert {p.name: p.dataType.value for p in created} == {
            **EXPECTED_PROPERTIES,
            **OPTIONAL_PROPERTIES,
        }

    def test_skips_an_existing_collection(self) -> None:
        store = WeaviateStore.__new__(WeaviateStore)
        store._read_only = False
        client = MagicMock()
        client.collections.exists.return_value = True
        store._client = client

        store.ensure_collection()

        client.collections.create.assert_not_called()


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


class TestIdentityFields:
    def _store(self, existing: list[str]) -> tuple[WeaviateStore, MagicMock]:
        store = WeaviateStore.__new__(WeaviateStore)
        store._read_only = False
        store._tenant = None
        collection = MagicMock()
        collection.config.get.return_value = SimpleNamespace(
            properties=[SimpleNamespace(name=n) for n in existing]
        )
        collection.batch.failed_objects = []
        client = MagicMock()
        client.collections.get.return_value = collection
        store._client = client
        store._embeddings = MagicMock()
        store._embed = MagicMock(return_value=[[0.1, 0.2]])
        return store, collection

    def test_add_chunks_writes_every_identity_field(self) -> None:
        store, collection = self._store([])
        lastmod = datetime(2026, 3, 1, tzinfo=timezone.utc)
        chunk = SupportChunk(
            text="t",
            source_type=SourceType.WEB,
            source_url="https://x.example/a",
            source_name="KB",
            source_lastmod=lastmod,
            embedded_video_ids=["v"],
        )

        store.add_chunks([chunk])

        batch = collection.batch.dynamic.return_value.__enter__.return_value
        props = batch.add_object.call_args.kwargs["properties"]
        assert props["document_id"] == chunk.document_id
        assert props["content_hash"] == chunk.content_hash
        assert props["source_lastmod"] == lastmod.isoformat()
        assert props["embedded_video_ids"] == ["v"]

    def test_add_chunks_writes_an_empty_list_not_a_missing_value(self) -> None:
        store, collection = self._store([])
        chunk = SupportChunk(
            text="t",
            source_type=SourceType.VIDEO,
            source_url="https://youtu.be/viJ-mmIjkhk",
            source_name="V",
        )

        store.add_chunks([chunk])

        batch = collection.batch.dynamic.return_value.__enter__.return_value
        props = batch.add_object.call_args.kwargs["properties"]
        assert props["embedded_video_ids"] == [] and props["source_lastmod"] is None

    def test_extend_schema_adds_only_the_missing_properties(self) -> None:
        have = [*EXPECTED_PROPERTIES, "document_id", "content_hash", "image_url"]
        store, collection = self._store(have)

        added = store.extend_schema()

        assert added == ["source_lastmod", "embedded_video_ids"]
        types = {
            c.args[0].name: c.args[0].dataType.value
            for c in collection.config.add_property.call_args_list
        }
        assert types == {"source_lastmod": "date", "embedded_video_ids": "text[]"}

    def test_extend_schema_is_refused_on_a_read_only_store(self) -> None:
        store, _ = self._store([])
        store._read_only = True

        with pytest.raises(PermissionError):
            store.extend_schema()

    def test_update_metadata_sends_only_the_named_properties(self) -> None:
        store, collection = self._store([])
        lastmod = datetime(2026, 3, 1, tzinfo=timezone.utc)

        count = store.update_metadata({"u1": {"source_lastmod": lastmod}})

        assert count == 1
        collection.data.update.assert_called_once_with(
            uuid="u1", properties={"source_lastmod": lastmod.isoformat()}
        )
        collection.data.replace.assert_not_called()
