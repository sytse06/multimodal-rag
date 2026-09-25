"""Tests for guarded local Weaviate snapshot restore."""

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from multimodal_rag.models.config import AppSettings
from multimodal_rag.store.restore import _connect_target, restore_snapshot
from multimodal_rag.store.snapshot import export_snapshot


def _snapshot(tmp_path: Path) -> Path:
    source = SimpleNamespace(
        config=SimpleNamespace(
            get=lambda: SimpleNamespace(
                name="SupportChunk",
                properties=[SimpleNamespace(name="text", data_type="text")],
                vectorizer_config={"type": "none"},
            )
        ),
        iterator=lambda include_vector: iter(
            [
                SimpleNamespace(
                    uuid=uuid4(),
                    properties={"text": "hello"},
                    vector={"default": [0.1, 0.2]},
                )
            ]
        ),
    )
    client = SimpleNamespace(
        collections=SimpleNamespace(get=lambda name: source),
        get_meta=lambda: {"version": "1.28.4"},
    )
    output = tmp_path / "snapshot"
    export_snapshot(
        client,
        output,
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
    )
    return output


def test_restore_refuses_existing_collection(tmp_path: Path) -> None:
    snapshot_dir = _snapshot(tmp_path)
    client = SimpleNamespace(
        collections=SimpleNamespace(exists=lambda name: True)
    )

    with pytest.raises(ValueError, match="already exists"):
        restore_snapshot(client, snapshot_dir)


def test_restore_inserts_snapshot_records(tmp_path: Path) -> None:
    snapshot_dir = _snapshot(tmp_path)
    inserted: list[dict[str, object]] = []
    collection = SimpleNamespace(
        config=SimpleNamespace(
            get=lambda: SimpleNamespace(
                properties=[
                    SimpleNamespace(
                        name="text", data_type=SimpleNamespace(value="text")
                    )
                ]
            )
        ),
        data=SimpleNamespace(insert=lambda **record: inserted.append(record)),
        aggregate=SimpleNamespace(
            over_all=lambda total_count: SimpleNamespace(total_count=1)
        ),
    )
    collections = SimpleNamespace(
        exists=lambda name: False,
        create=lambda **kwargs: collection,
    )
    client = SimpleNamespace(collections=collections)

    manifest = restore_snapshot(client, snapshot_dir)

    assert manifest.object_count == 1
    assert len(inserted) == 1
    assert inserted[0]["vector"] == [0.1, 0.2]


def test_cloud_target_uses_admin_key(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings(
        _env_file=None,
        openrouter_api_key="test",
        weaviate_mode="cloud",
        weaviate_url="https://example.weaviate.network",
        weaviate_admin_api_key="admin-secret",
    )
    client = object()
    monkeypatch.setattr(
        "multimodal_rag.store.restore.weaviate.connect_to_weaviate_cloud",
        lambda **kwargs: client,
    )

    assert _connect_target(settings, "cloud") is client


def test_cloud_target_requires_admin_key() -> None:
    settings = AppSettings(
        _env_file=None,
        openrouter_api_key="test",
        weaviate_mode="cloud",
        weaviate_url="https://example.weaviate.network",
        weaviate_viewer_api_key="viewer-secret",
    )

    with pytest.raises(ValueError, match="WEAVIATE_ADMIN_API_KEY"):
        _connect_target(settings, "cloud")
