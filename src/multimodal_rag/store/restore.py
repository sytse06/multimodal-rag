"""Restore a fixed Weaviate collection snapshot into local Docker Weaviate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import weaviate
import weaviate.classes.config as wvc

from multimodal_rag.models.config import AppSettings
from multimodal_rag.store.snapshot import SnapshotManifest
from multimodal_rag.store.weaviate import COLLECTION_NAME


def _read_snapshot(snapshot_dir: Path) -> tuple[SnapshotManifest, list[dict[str, Any]]]:
    manifest = SnapshotManifest.model_validate_json(
        (snapshot_dir / "manifest.json").read_text(encoding="utf-8")
    )
    snapshot_path = snapshot_dir / manifest.snapshot_file
    digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    if digest != manifest.snapshot_sha256:
        raise ValueError("Snapshot checksum does not match manifest")
    records = [
        json.loads(line)
        for line in snapshot_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != manifest.object_count:
        raise ValueError(
            f"Snapshot object count mismatch: expected {manifest.object_count}, "
            f"found {len(records)}"
        )
    return manifest, records


def _validate_schema(collection: Any, manifest: SnapshotManifest) -> None:
    config = collection.config.get()
    actual = {
        str(prop.name): str(prop.data_type.value)
        for prop in config.properties
    }
    expected = {
        str(prop["name"]): str(prop["data_type"])
        for prop in manifest.collection_schema.get("properties", [])
    }
    if actual != expected:
        raise ValueError(
            f"Collection schema mismatch: expected {expected}, found {actual}"
        )


def _restore_vector(vector: Any) -> Any:
    """Convert a single exported named vector to Weaviate's list form."""
    if isinstance(vector, dict) and set(vector) == {"default"}:
        return vector["default"]
    return vector


def restore_snapshot(
    client: Any,
    snapshot_dir: Path,
    *,
    replace: bool = False,
    collection_name: str = COLLECTION_NAME,
) -> SnapshotManifest:
    """Restore and validate a snapshot into a local Weaviate client."""
    manifest, records = _read_snapshot(snapshot_dir)
    if manifest.collection_name != collection_name:
        raise ValueError(
            f"Snapshot collection is {manifest.collection_name}, "
            f"expected {collection_name}"
        )
    if client.collections.exists(collection_name):
        if not replace:
            raise ValueError(
                f"Collection {collection_name} already exists; use --replace explicitly"
            )
        client.collections.delete(collection_name)

    collection = client.collections.create(
        name=collection_name,
        vector_config=wvc.Configure.Vectors.self_provided(),
        properties=[
            wvc.Property(
                name=prop["name"], data_type=wvc.DataType(prop["data_type"])
            )
            for prop in manifest.collection_schema.get("properties", [])
        ],
    )
    _validate_schema(collection, manifest)
    for index, record in enumerate(records, start=1):
        try:
            collection.data.insert(
                properties=record["properties"],
                vector=_restore_vector(record["vector"]),
                uuid=UUID(record["uuid"]),
            )
        except Exception as exc:
            raise ValueError(f"Restore failed for object {index}: {exc}") from exc
    restored_count = collection.aggregate.over_all(total_count=True).total_count
    if restored_count != manifest.object_count:
        raise ValueError("Restored object count does not match manifest")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = AppSettings()
    parsed = urlparse(settings.weaviate_url)
    if settings.weaviate_mode != "local" or parsed.scheme != "http":
        raise ValueError("DATA-003 restore requires WEAVIATE_MODE=local")
    client = weaviate.connect_to_local(
        host=parsed.hostname or "localhost", port=parsed.port or 8080
    )
    try:
        manifest = restore_snapshot(client, args.snapshot_dir, replace=args.replace)
    finally:
        client.close()
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
