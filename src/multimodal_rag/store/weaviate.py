"""Weaviate vector store for SupportChunk objects."""

import logging
from typing import Any
from urllib.parse import urlparse

import weaviate
import weaviate.classes.config as wvc
from langchain_core.embeddings import Embeddings
from weaviate.auth import AuthApiKey
from weaviate.classes.query import Filter, MetadataQuery

from multimodal_rag.models.chunks import SourceType, SupportChunk
from multimodal_rag.store.embeddings import embed_texts

logger = logging.getLogger(__name__)

COLLECTION_NAME = "SupportChunk"
EXPECTED_PROPERTIES = {
    "text": "text",
    "source_type": "text",
    "source_url": "text",
    "source_name": "text",
    "timestamp_seconds": "int",
    "section_heading": "text",
    "url_hash": "text",
    "ingested_at": "date",
}


class WeaviateStore:
    """Manages SupportChunk storage and retrieval in Weaviate."""

    def __init__(
        self,
        weaviate_url: str,
        embeddings: Embeddings,
        weaviate_mode: str = "local",
        weaviate_api_key: str = "",
        weaviate_tenant: str = "",
    ) -> None:
        self._embeddings = embeddings
        self._tenant = weaviate_tenant.strip() or None
        try:
            parsed = urlparse(weaviate_url)
            if weaviate_mode == "cloud":
                if parsed.scheme != "https" or not parsed.hostname:
                    raise ValueError("cloud Weaviate URL must be an https URL")
                if not weaviate_api_key:
                    raise ValueError("Weaviate API key is required in cloud mode")
                self._client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=parsed.hostname,
                    auth_credentials=AuthApiKey(weaviate_api_key),
                )
            elif weaviate_mode == "local":
                if parsed.scheme != "http" or not parsed.hostname:
                    raise ValueError("local Weaviate URL must be an http URL")
                if parsed.hostname.endswith((".weaviate.cloud", ".weaviate.network")):
                    raise ValueError(
                        "local Weaviate points to a hosted endpoint; set "
                        "weaviate_mode='cloud' for a hosted URL"
                    )
                self._client = weaviate.connect_to_local(
                    host=parsed.hostname,
                    port=parsed.port or 8080,
                )
            else:
                raise ValueError(f"Unsupported Weaviate mode: {weaviate_mode}")
        except ValueError:
            raise
        except Exception as exc:
            raise ConnectionError(
                f"Unable to connect to Weaviate ({weaviate_mode}) at "
                f"{weaviate_url}: {exc}"
            ) from exc

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
            vector_config=wvc.Configure.Vectors.self_provided(),
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

    def _collection(self) -> Any:
        collection = self._client.collections.get(COLLECTION_NAME)
        tenant = getattr(self, "_tenant", None)
        return collection.with_tenant(tenant) if tenant else collection

    def validate_compatibility(self, expected_vector_dimension: int = 768) -> None:
        """Validate collection, schema, vectors, and read access before inference."""
        try:
            if not self._client.collections.exists(COLLECTION_NAME):
                raise ValueError(
                    f"Weaviate collection {COLLECTION_NAME} does not exist"
                )
            collection = self._collection()
            config = collection.config.get()
            actual = {
                str(prop.name): str(prop.data_type.value)
                for prop in config.properties
            }
            if actual != EXPECTED_PROPERTIES:
                raise ValueError(
                    "Weaviate collection schema is incompatible: "
                    f"expected {EXPECTED_PROPERTIES}, found {actual}"
                )
            objects = collection.query.fetch_objects(limit=1, include_vector=True)
            if not objects.objects:
                raise ValueError(f"Weaviate collection {COLLECTION_NAME} is empty")
            vector = objects.objects[0].vector
            if isinstance(vector, dict):
                vector = vector.get("default")
            stored_dimension = len(vector) if vector is not None else 0
            if stored_dimension != expected_vector_dimension:
                raise ValueError(
                    "Weaviate vector dimension is incompatible: "
                    f"expected {expected_vector_dimension}, found {stored_dimension}"
                )
            query_vector = self._embed(["compatibility check"])[0]
            if len(query_vector) != stored_dimension:
                raise ValueError(
                    "Query embedding dimension is incompatible: "
                    f"expected {stored_dimension}, found {len(query_vector)}"
                )
            collection.aggregate.over_all(total_count=True)
        except ValueError:
            raise
        except Exception as exc:
            tenant = f" for tenant {self._tenant!r}" if self._tenant else ""
            raise ConnectionError(
                "Unable to validate Weaviate collection "
                f"{COLLECTION_NAME}{tenant}: {exc}"
            ) from exc

    def delete_by_source(self, source_url: str) -> int:
        """Delete all chunks matching source_url. Returns count deleted."""
        collection = self._collection()
        result = collection.data.delete_many(
            where=Filter.by_property("source_url").equal(source_url)
        )
        deleted = result.successful if result else 0
        logger.info("Deleted %d chunks for source: %s", deleted, source_url)
        return deleted

    def delete_by_source_type(self, source_type: str) -> int:
        """Delete all chunks matching source_type ('video' or 'web'). Returns count."""
        collection = self._collection()
        result = collection.data.delete_many(
            where=Filter.by_property("source_type").equal(source_type)
        )
        deleted = result.successful if result else 0
        logger.info("Deleted %d chunks of source_type '%s'", deleted, source_type)
        return deleted

    def delete_collection(self) -> None:
        """Delete the SupportChunk collection."""
        if self._client.collections.exists(COLLECTION_NAME):
            self._client.collections.delete(COLLECTION_NAME)
            logger.info("Deleted collection %s", COLLECTION_NAME)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts, embeddings=self._embeddings)

    def add_chunks(self, chunks: list[SupportChunk]) -> int:
        """Add SupportChunks to Weaviate with embeddings. Returns count added."""
        if not chunks:
            return 0

        texts = [f"{c.source_name}: {c.text}" for c in chunks]
        vectors = self._embed(texts)

        collection = self._collection()
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

        collection = self._collection()
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
        collection = self._collection()
        result = collection.aggregate.over_all(total_count=True)
        return result.total_count or 0
