"""Weaviate vector store for SupportChunk objects."""

import logging

import weaviate
import weaviate.classes.config as wvc
from weaviate.classes.query import MetadataQuery

from multimodal_rag.models.chunks import SourceType, SupportChunk
from multimodal_rag.store.embeddings import embed_texts

logger = logging.getLogger(__name__)

COLLECTION_NAME = "SupportChunk"


class WeaviateStore:
    """Manages SupportChunk storage and retrieval in Weaviate."""

    def __init__(
        self,
        weaviate_url: str,
        openrouter_api_key: str,
        openrouter_base_url: str = "https://openrouter.ai/api/v1",
        embedding_model: str = "openai/text-embedding-3-small",
    ) -> None:
        self._weaviate_url = weaviate_url
        self._api_key = openrouter_api_key
        self._base_url = openrouter_base_url
        self._embedding_model = embedding_model
        host = weaviate_url.replace("http://", "").split(":")[0]
        tail = weaviate_url.rsplit("/", 1)[-1]
        port = int(weaviate_url.split(":")[-1]) if ":" in tail else 8080
        self._client = weaviate.connect_to_local(host=host, port=port)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "WeaviateStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def ensure_collection(self) -> None:
        """Create the SupportChunk collection if it doesn't exist."""
        if self._client.collections.exists(COLLECTION_NAME):
            logger.info("Collection %s already exists", COLLECTION_NAME)
            return

        self._client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=wvc.Configure.Vectorizer.none(),
            properties=[
                wvc.Property(name="text", data_type=wvc.DataType.TEXT),
                wvc.Property(name="source_type", data_type=wvc.DataType.TEXT),
                wvc.Property(name="source_url", data_type=wvc.DataType.TEXT),
                wvc.Property(name="source_name", data_type=wvc.DataType.TEXT),
                wvc.Property(name="timestamp_seconds", data_type=wvc.DataType.INT),
                wvc.Property(name="section_heading", data_type=wvc.DataType.TEXT),
                wvc.Property(name="url_hash", data_type=wvc.DataType.TEXT),
                wvc.Property(name="ingested_at", data_type=wvc.DataType.DATE),
            ],
        )
        logger.info("Created collection %s", COLLECTION_NAME)

    def delete_collection(self) -> None:
        """Delete the SupportChunk collection."""
        if self._client.collections.exists(COLLECTION_NAME):
            self._client.collections.delete(COLLECTION_NAME)
            logger.info("Deleted collection %s", COLLECTION_NAME)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(
            texts,
            api_key=self._api_key,
            base_url=self._base_url,
            model=self._embedding_model,
        )

    def add_chunks(self, chunks: list[SupportChunk]) -> int:
        """Add SupportChunks to Weaviate with embeddings. Returns count added."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        vectors = self._embed(texts)

        collection = self._client.collections.get(COLLECTION_NAME)
        added = 0

        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(chunks, vectors):
                props = {
                    "text": chunk.text,
                    "source_type": chunk.source_type.value,
                    "source_url": chunk.source_url,
                    "source_name": chunk.source_name,
                    "timestamp_seconds": chunk.timestamp_seconds,
                    "section_heading": chunk.section_heading or "",
                    "url_hash": chunk.url_hash,
                    "ingested_at": chunk.ingested_at.isoformat(),
                }
                batch.add_object(
                    properties=props,
                    vector=vector,
                    uuid=chunk.chunk_id,
                )
                added += 1

        logger.info("Added %d chunks to %s", added, COLLECTION_NAME)
        return added

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Search for similar chunks, returning properties + distance."""
        vectors = self._embed([query])
        if not vectors:
            return []

        collection = self._client.collections.get(COLLECTION_NAME)
        response = collection.query.near_vector(
            near_vector=vectors[0],
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )

        results = []
        for obj in response.objects:
            props = dict(obj.properties)
            props["_distance"] = obj.metadata.distance if obj.metadata else None
            props["_uuid"] = str(obj.uuid)
            # Restore source_type as enum
            if "source_type" in props:
                props["source_type"] = SourceType(str(props["source_type"]))
            results.append(props)

        return results

    def count(self) -> int:
        """Return total number of objects in the collection."""
        collection = self._client.collections.get(COLLECTION_NAME)
        result = collection.aggregate.over_all(total_count=True)
        return result.total_count or 0
