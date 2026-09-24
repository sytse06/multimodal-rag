"""Export a fixed Weaviate collection for offline restore and sharing."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import weaviate
from pydantic import BaseModel, Field

from multimodal_rag.models.config import AppSettings
from multimodal_rag.store.weaviate import COLLECTION_NAME


class SnapshotManifest(BaseModel):
    """Compatibility metadata for a collection snapshot."""

    format_version: str = "1"
    collection_name: str
    object_count: int
    vector_dimension: int
    embedding_provider: str
    embedding_model: str
    weaviate_version: str
    created_at: datetime
    source_commit: str
    snapshot_file: str
    snapshot_sha256: str = Field(min_length=64, max_length=64)
    collection_schema: dict[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (datetime, UUID)):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            str(key): _jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def _schema(collection: Any) -> dict[str, Any]:
    config = collection.config.get()
    result: dict[str, Any] = {}
    for name in ("name", "description", "properties", "vectorizer_config"):
        if hasattr(config, name):
            result[name] = _jsonable(getattr(config, name))
    return result


def _source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def export_snapshot(
    client: Any,
    output_dir: Path,
    *,
    embedding_provider: str,
    embedding_model: str,
    collection_name: str = COLLECTION_NAME,
) -> SnapshotManifest:
    """Write collection objects as JSONL and return its compatibility manifest."""
    collection = client.collections.get(collection_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / f"{collection_name}.jsonl"
    object_count = 0
    vector_dimension = 0

    with snapshot_path.open("w", encoding="utf-8") as handle:
        for obj in collection.iterator(include_vector=True):
            vector = _jsonable(obj.vector)
            if isinstance(vector, list) and vector_dimension == 0:
                vector_dimension = len(vector)
            elif isinstance(vector, dict) and vector_dimension == 0:
                dimensions = [
                    len(value) for value in vector.values() if isinstance(value, list)
                ]
                if dimensions:
                    vector_dimension = dimensions[0]
            record = {
                "uuid": str(obj.uuid),
                "properties": _jsonable(obj.properties),
                "vector": vector,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            object_count += 1

    digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    meta = client.get_meta()
    manifest = SnapshotManifest(
        collection_name=collection_name,
        object_count=object_count,
        vector_dimension=vector_dimension,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        weaviate_version=str(meta.get("version", "unknown")),
        created_at=datetime.now(timezone.utc),
        source_commit=_source_commit(),
        snapshot_file=snapshot_path.name,
        snapshot_sha256=digest,
        collection_schema=_schema(collection),
    )
    (output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory outside Git in which to write the snapshot",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = AppSettings()
    client = weaviate.connect_to_local()
    try:
        manifest = export_snapshot(
            client,
            args.output,
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
        )
    finally:
        client.close()
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
