"""Tests for the fixed Weaviate collection snapshot exporter."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from multimodal_rag.store.snapshot import export_snapshot


def test_export_snapshot_writes_vectors_and_manifest(tmp_path: Path) -> None:
    objects = [
        SimpleNamespace(
            uuid=uuid4(),
            properties={"text": "first", "source_type": "web"},
            vector={"default": [0.1, 0.2, 0.3]},
        ),
        SimpleNamespace(
            uuid=uuid4(),
            properties={"text": "second", "source_type": "video"},
            vector={"default": [0.4, 0.5, 0.6]},
        ),
    ]
    collection = SimpleNamespace(
        config=SimpleNamespace(
            get=lambda: SimpleNamespace(
                name="SupportChunk",
                properties=[{"name": "text", "data_type": ["text"]}],
                vectorizer_config={"type": "none"},
            )
        ),
        iterator=lambda include_vector: iter(objects),
    )
    client = SimpleNamespace(
        collections=SimpleNamespace(get=lambda name: collection),
        get_meta=lambda: {"version": "1.28.4"},
    )

    manifest = export_snapshot(
        client,
        tmp_path,
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
    )

    snapshot = tmp_path / "SupportChunk.jsonl"
    manifest_path = tmp_path / "manifest.json"
    assert manifest.object_count == 2
    assert manifest.vector_dimension == 3
    assert manifest.embedding_model == "nomic-embed-text"
    assert manifest.snapshot_sha256 == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    first_record = json.loads(snapshot.read_text(encoding="utf-8").splitlines()[0])
    assert first_record["vector"] == {"default": [0.1, 0.2, 0.3]}
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["object_count"] == 2
