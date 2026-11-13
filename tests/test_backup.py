"""Tests for the snapshot backup to the state bucket."""

import base64
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from multimodal_rag.store import backup
from multimodal_rag.store.backup import backup_snapshot, upload_snapshot
from multimodal_rag.store.snapshot import SnapshotManifest


def _manifest(snapshot_file: str = "SupportChunk.jsonl") -> SnapshotManifest:
    return SnapshotManifest(
        collection_name="SupportChunk",
        object_count=1,
        vector_dimension=3,
        embedding_provider="ollama",
        embedding_model="nomic-embed-text",
        weaviate_version="1.39.13",
        created_at=datetime(2026, 11, 13, tzinfo=timezone.utc),
        source_commit="abc",
        snapshot_file=snapshot_file,
        snapshot_sha256="0" * 64,
        collection_schema={},
    )


def _md5(path: Path) -> str:
    return base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode()


def _settings(bucket: str = "state-bucket") -> MagicMock:
    settings = MagicMock()
    settings.state_bucket = bucket
    settings.gcp_project = "proj"
    settings.embedding_provider = "ollama"
    settings.embedding_model = "nomic-embed-text"
    return settings


def _fake_gcloud(calls: list[list[str]], tmp: Path, *, corrupt: bool = False):
    def run(args: list[str], project: str) -> str:
        calls.append(args)
        if args[1] == "objects":
            name = args[3].rsplit("/", 1)[1]
            return "bad" if corrupt else _md5(tmp / name)
        return ""

    return run


def test_upload_sends_data_before_manifest_and_verifies_each(tmp_path: Path) -> None:
    (tmp_path / "SupportChunk.jsonl").write_text("data")
    (tmp_path / "manifest.json").write_text("{}")
    calls: list[list[str]] = []
    with patch.object(backup, "_gcloud", _fake_gcloud(calls, tmp_path)):
        upload_snapshot(tmp_path, "gs://b/snapshots/x", "proj", _manifest())

    copies = [c[3].rsplit("/", 1)[1] for c in calls if c[1] == "cp"]
    assert copies == ["SupportChunk.jsonl", "manifest.json"]
    assert [c[1] for c in calls] == ["cp", "objects", "cp", "objects"]


def test_checksum_mismatch_stops_before_manifest(tmp_path: Path) -> None:
    (tmp_path / "SupportChunk.jsonl").write_text("data")
    (tmp_path / "manifest.json").write_text("{}")
    calls: list[list[str]] = []
    with patch.object(backup, "_gcloud", _fake_gcloud(calls, tmp_path, corrupt=True)):
        with pytest.raises(RuntimeError, match="Checksum mismatch"):
            upload_snapshot(tmp_path, "gs://b/snapshots/x", "proj", _manifest())

    assert not any("manifest.json" in " ".join(c) for c in calls)


def test_backup_requires_state_bucket() -> None:
    with pytest.raises(ValueError, match="STATE_BUCKET"):
        backup_snapshot(MagicMock(), _settings(""), target="cloud")


def test_backup_exports_then_uploads_to_timestamped_prefix() -> None:
    manifest = _manifest()
    seen: dict[str, object] = {}

    def fake_upload(dir_: Path, dest: str, project: str, m: SnapshotManifest) -> None:
        seen["dest"] = dest
        seen["dir_existed"] = dir_.exists()
        seen["dir"] = dir_

    with (
        patch.object(backup, "export_snapshot", return_value=manifest),
        patch.object(backup, "upload_snapshot", fake_upload),
    ):
        dest, result = backup_snapshot(
            MagicMock(),
            _settings(),
            target="cloud",
            now=datetime(2026, 11, 13, 14, 5, 9, tzinfo=timezone.utc),
        )

    assert dest == "gs://state-bucket/snapshots/20261113T140509Z-cloud"
    assert seen["dest"] == dest and seen["dir_existed"] is True
    assert result is manifest
    assert not Path(str(seen["dir"])).exists()


def test_gcloud_missing_and_failure_are_reported() -> None:
    with patch.object(backup.subprocess, "run", side_effect=FileNotFoundError):
        with pytest.raises(RuntimeError, match="gcloud CLI not found"):
            backup._gcloud(["storage", "cp"], "proj")
    err = subprocess.CalledProcessError(1, "gcloud", stderr="denied")
    with patch.object(backup.subprocess, "run", side_effect=err):
        with pytest.raises(RuntimeError, match="denied"):
            backup._gcloud(["storage", "cp", "a", "b"], "proj")

