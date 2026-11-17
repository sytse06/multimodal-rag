"""Give every stored chunk its identity fields (SYNC-001). Local only, metadata only.

``DRY=1`` shows what would change and writes nothing. A real run takes a snapshot of
local, adds the missing properties with their declared types, updates the fields, and
then proves the object count and every text and vector are unchanged.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from multimodal_rag.models.config import AppSettings
from multimodal_rag.models.llm import create_embeddings
from multimodal_rag.store.snapshot import export_snapshot
from multimodal_rag.store.weaviate import (
    COLLECTION_NAME,
    OPTIONAL_PROPERTIES,
    WeaviateStore,
)
from multimodal_rag.sync.classify import Kind
from multimodal_rag.sync.convert import Conversion, convert, problems
from multimodal_rag.sync.lock import LockHeld, RunLock
from multimodal_rag.sync.report import build_inventory

EXIT_NEEDS_REVIEW = 10


def summary(conversion: Conversion, missing: list[str]) -> list[str]:
    kinds = ", ".join(f"{k} {n}" for k, n in sorted(conversion.by_type.items()))
    fields: dict[str, int] = {}
    for changed in conversion.updates.values():
        for name in changed:
            fields[name] = fields.get(name, 0) + 1
    lines = [
        f"{conversion.total} chunks ({kinds}) on {conversion.pages} pages",
        f"{len(conversion.updates)} to update, {conversion.already_current} current",
        f"pages with a sitemap lastmod: {conversion.pages_with_lastmod}"
        f" of {conversion.pages}",
        f"pages that link a video: {conversion.pages_with_videos}",
    ]
    lines += [f"  {name}: {n} chunks" for name, n in sorted(fields.items())]
    if missing:
        lines.append(f"properties to add to the schema: {', '.join(missing)}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true", help="show the changes only")
    parser.add_argument("--yes", action="store_true", help="do not ask first")
    parser.add_argument("--break-lock", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    settings = AppSettings()
    if settings.weaviate_mode != "local":
        print("Error: the conversion runs on the local collection only.")
        sys.exit(1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir: Path = args.runs_dir / f"convert-{stamp}"

    inventory = build_inventory(settings)
    lastmods = {i.url: i.lastmod for i in inventory.items if i.kind is Kind.UNCHANGED}
    with WeaviateStore(
        weaviate_url=str(settings.weaviate_url),
        embeddings=create_embeddings(settings),
    ) as store:
        config = store.client.collections.get(COLLECTION_NAME).config.get()
        have = {str(p.name) for p in config.properties}
        missing = [n for n in OPTIONAL_PROPERTIES if n not in have]
        before = store.read_objects()
        conversion = convert(before, lastmods)
        print("\n".join(summary(conversion, missing)))
        if args.dry:
            print("\nDry run: nothing was written.")
            return
        if not conversion.updates and not missing:
            print("\nNothing to convert.")
            return
        if not args.yes and input("\nConvert the LOCAL collection? [y/N] ") != "y":
            print("Nothing converted.")
            return

        lock = RunLock(settings.state_bucket, settings.gcp_project, f"convert-{stamp}")
        try:
            lock.acquire(break_lock=args.break_lock)
        except LockHeld as exc:
            print(f"Failed: {exc}")
            sys.exit(1)
        try:
            snapshot = run_dir / "snapshot-local"
            export_snapshot(
                store.client,
                snapshot,
                embedding_provider=settings.embedding_provider,
                embedding_model=settings.embedding_model,
            )
            store.extend_schema()
            store.update_metadata(conversion.updates)
            after = store.read_objects()
        finally:
            lock.release()

    found = problems(before, after, conversion)
    if found:
        print("\nFailed verification:")
        print("\n".join(f"  - {p}" for p in found))
        print(f"Roll back: make snapshot-restore SNAPSHOT_DIR={snapshot} REPLACE=1")
        sys.exit(1)
    print(f"\nConverted {len(conversion.updates)} chunks. Count, texts and vectors")
    print("are unchanged. Roll back with:")
    print(f"  make snapshot-restore SNAPSHOT_DIR={snapshot} REPLACE=1")


if __name__ == "__main__":
    main()
