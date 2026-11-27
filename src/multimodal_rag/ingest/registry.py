"""Pull and push the source registry between the state bucket and a local cache.

The registry (config/sources.yaml) lists the knowledge bases and videos to ingest. It
is private and lives in the state bucket; the local file is a cache. A push is guarded
by the object generation recorded at the last pull, so two machines cannot silently
overwrite each other.
"""

from __future__ import annotations

import argparse
import base64
import difflib
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

from multimodal_rag.gcloud import run_gcloud
from multimodal_rag.models.config import AppSettings
from multimodal_rag.models.sources import SourceConfig

REGISTRY_OBJECT = "registry/sources.yaml"
SOURCES_PATH = Path("config/sources.yaml")
GENERATION_PATH = Path("config/.sources.generation")
DESCRIBE_FORMAT = "--format=json(generation,md5_hash)"


def _url(settings: AppSettings) -> str:
    if not settings.state_bucket:
        raise ValueError("STATE_BUCKET is not set")
    return f"gs://{settings.state_bucket}/{REGISTRY_OBJECT}"


def _md5_hex(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324


def validate(path: Path) -> SourceConfig:
    """Parse the registry and reject files that would ingest nothing or twice."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = SourceConfig.model_validate(raw)
    if not config.youtube and not config.knowledge_bases:
        raise ValueError("Registry lists no sources")
    for label, urls in (
        ("youtube", [str(s.url) for s in config.youtube]),
        ("knowledge_bases", [str(s.url) for s in config.knowledge_bases]),
    ):
        duplicates = sorted({u for u in urls if urls.count(u) > 1})
        if duplicates:
            raise ValueError(f"Duplicate {label} URL: {duplicates[0]}")
    return config


def _remote(url: str, project: str) -> tuple[str, str] | None:
    """Return (generation, md5 hex) of the registry object, or None if absent."""
    try:
        out = run_gcloud(
            ["storage", "objects", "describe", url, DESCRIBE_FORMAT],
            project,
        )
    except RuntimeError as exc:
        if "404" in str(exc) or "No such object" in str(exc):
            return None
        raise
    data = json.loads(out)
    return str(data["generation"]), base64.b64decode(data["md5_hash"]).hex()


def _recorded_generation(path: Path) -> str | None:
    return path.read_text().strip() if path.exists() else None


NO_LOCAL_COPY = "no-local-copy"
NOT_IN_BUCKET = "not-in-bucket"
MATCHES = "matches"
UNPUSHED_EDITS = "unpushed-edits"
BUCKET_CHANGED = "bucket-changed"


def compare(
    settings: AppSettings,
    path: Path = SOURCES_PATH,
    generation_path: Path = GENERATION_PATH,
) -> str:
    """How the local registry file stands against the bucket's.

    Content decides ``matches``. When they differ, the recorded generation says who
    moved: unchanged generation means the local file was edited, a newer generation
    means the bucket changed (and the local file may have been edited too).
    """
    if not path.exists():
        return NO_LOCAL_COPY
    remote = _remote(_url(settings), settings.gcp_project)
    if remote is None:
        return NOT_IN_BUCKET
    generation, remote_md5 = remote
    if _md5_hex(path) == remote_md5:
        return MATCHES
    if _recorded_generation(generation_path) == generation:
        return UNPUSHED_EDITS
    return BUCKET_CHANGED


def diff(
    settings: AppSettings,
    path: Path = SOURCES_PATH,
    generation_path: Path = GENERATION_PATH,
) -> str:
    """A unified diff of the bucket's registry against the local file."""
    state = compare(settings, path, generation_path)
    if state == NO_LOCAL_COPY:
        raise RuntimeError(f"No local copy at {path}; run make sources-pull")
    if state == NOT_IN_BUCKET:
        raise RuntimeError("No registry in the bucket; create it with a push (INIT=1)")
    generation, _ = _remote(_url(settings), settings.gcp_project) or ("?", "")
    if state == MATCHES:
        return f"Identical to the bucket's registry (generation {generation})."
    with tempfile.TemporaryDirectory(prefix="registry-") as tmp:
        downloaded = Path(tmp) / "sources.yaml"
        run_gcloud(
            ["storage", "cp", _url(settings), str(downloaded)], settings.gcp_project
        )
        theirs = downloaded.read_text(encoding="utf-8").splitlines(keepends=True)
    ours = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines = list(
        difflib.unified_diff(
            theirs,
            ours,
            fromfile=f"bucket (generation {generation})",
            tofile=str(path),
        )
    )
    headline = {
        UNPUSHED_EDITS: "Your local file has edits that are not pushed"
        " (make sources-push publishes them):",
        BUCKET_CHANGED: "The bucket changed since your last pull; your file may also"
        " hold edits. Merge by hand, then push; a pull would refuse to overwrite them:",
    }[state]
    body = "".join(lines) or "(the files differ only in trailing whitespace)\n"
    return f"{headline}\n{body}".rstrip("\n")


def pull(
    settings: AppSettings,
    path: Path = SOURCES_PATH,
    generation_path: Path = GENERATION_PATH,
    *,
    force: bool = False,
) -> str:
    url = _url(settings)
    remote = _remote(url, settings.gcp_project)
    if remote is None:
        raise RuntimeError(f"No registry at {url}; create it with a push (INIT=1)")
    generation, remote_md5 = remote
    if path.exists() and _md5_hex(path) != remote_md5:
        if not force:
            raise RuntimeError(
                f"{path} has edits that are not in the registry. See them with"
                " make sources-diff and publish them with make sources-push."
            )
        shutil.copy2(path, path.with_name(path.name + ".prev"))
    with tempfile.TemporaryDirectory(prefix="registry-") as tmp:
        downloaded = Path(tmp) / "sources.yaml"
        run_gcloud(["storage", "cp", url, str(downloaded)], settings.gcp_project)
        validate(downloaded)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, path)
    generation_path.write_text(generation + "\n")
    return f"Pulled registry generation {generation} to {path}"


def push(
    settings: AppSettings,
    path: Path = SOURCES_PATH,
    generation_path: Path = GENERATION_PATH,
    *,
    init: bool = False,
) -> str:
    url = _url(settings)
    config = validate(path)
    remote = _remote(url, settings.gcp_project)
    recorded = _recorded_generation(generation_path)
    if remote is None:
        if not init:
            raise RuntimeError(f"No registry at {url}; run the first push with INIT=1")
        precondition = "0"
    else:
        generation, remote_md5 = remote
        if recorded is None:
            raise RuntimeError("No generation recorded; pull the registry first")
        if recorded != generation:
            raise RuntimeError(
                f"The registry changed since your last pull (yours {recorded},"
                f" remote {generation}); pull it and merge your edits"
            )
        if _md5_hex(path) == remote_md5:
            return f"Nothing to push: registry generation {generation} is identical"
        precondition = generation
    run_gcloud(
        ["storage", "cp", str(path), url, f"--if-generation-match={precondition}"],
        settings.gcp_project,
    )
    pushed = _remote(url, settings.gcp_project)
    if pushed is None:
        raise RuntimeError("Registry missing after upload")
    generation_path.write_text(pushed[0] + "\n")
    return (
        f"Pushed {len(config.youtube)} videos and {len(config.knowledge_bases)}"
        f" knowledge bases as generation {pushed[0]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pull", "push", "diff"))
    parser.add_argument("--init", action="store_true", help="first push only")
    parser.add_argument("--force", action="store_true", help="pull over local edits")
    args = parser.parse_args()
    settings = AppSettings()
    try:
        if args.action == "pull":
            message = pull(settings, force=args.force)
        elif args.action == "diff":
            message = diff(settings)
        else:
            message = push(settings, init=args.init)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(message)


if __name__ == "__main__":
    main()
