"""Back up a Weaviate collection snapshot to the private state bucket."""

from __future__ import annotations

import argparse
import base64
import hashlib
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multimodal_rag.models.config import AppSettings
from multimodal_rag.store.restore import _connect_target
from multimodal_rag.store.snapshot import SnapshotManifest, export_snapshot

SNAPSHOT_PREFIX = "snapshots"


def _gcloud(args: list[str], project: str) -> str:
    command = ["gcloud", *args]
    if project:
        command += ["--project", project]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=600
        )
    except FileNotFoundError as exc:
        raise RuntimeError("gcloud CLI not found; install Google Cloud SDK") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"gcloud {' '.join(args[:3])} failed: {exc.stderr.strip()}"
        ) from exc
    return result.stdout.strip()


def _md5_base64(path: Path) -> str:
    return base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode()  # noqa: S324


def upload_snapshot(
    snapshot_dir: Path, destination: str, project: str, manifest: SnapshotManifest
) -> None:
    """Upload data first and the manifest last, then verify each object's MD5.

    The manifest marks a complete backup: a prefix without one is an aborted
    upload and must not be restored.
    """
    for name in (manifest.snapshot_file, "manifest.json"):
        local = snapshot_dir / name
        remote = f"{destination}/{name}"
        _gcloud(["storage", "cp", str(local), remote], project)
        stored = _gcloud(
            ["storage", "objects", "describe", remote, "--format=value(md5_hash)"],
            project,
        )
        if stored != _md5_base64(local):
            raise RuntimeError(f"Checksum mismatch after uploading {remote}")


def backup_snapshot(
    client: Any,
    settings: AppSettings,
    *,
    target: str,
    now: datetime | None = None,
) -> tuple[str, SnapshotManifest]:
    """Export the collection to a temporary directory and upload it."""
    bucket = settings.state_bucket
    if not bucket:
        raise ValueError("STATE_BUCKET is not set")
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    destination = f"gs://{bucket}/{SNAPSHOT_PREFIX}/{stamp}-{target}"
    with tempfile.TemporaryDirectory(prefix="snapshot-") as tmp:
        manifest = export_snapshot(
            client,
            Path(tmp),
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
        )
        upload_snapshot(Path(tmp), destination, settings.gcp_project, manifest)
    return destination, manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("local", "cloud"), default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = AppSettings()
    target = args.target or settings.weaviate_mode
    client = _connect_target(settings, target)
    try:
        destination, manifest = backup_snapshot(client, settings, target=target)
    finally:
        client.close()
    print(
        f"Backed up {manifest.object_count} objects "
        f"(sha256 {manifest.snapshot_sha256[:12]}) to {destination}"
    )


if __name__ == "__main__":
    main()
