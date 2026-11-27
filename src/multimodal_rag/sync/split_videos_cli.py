"""Cut the old hybrid video chunks down to their narration (SYNC-013). Local only.

``DRY=1`` shows what would change and writes nothing. A real run refuses unless the
embedding runtime is the pinned one, takes a lock and a snapshot of local, rewrites each
hybrid chunk's text to its narration under the same id with a fresh vector, and then
proves every other object is untouched and every new vector reproduces from its text.
The old descriptions are not kept as chunks; they stay in the snapshot.
"""

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from multimodal_rag.models.config import AppSettings
from multimodal_rag.models.llm import create_embeddings
from multimodal_rag.store.embeddings import embed_texts
from multimodal_rag.store.snapshot import export_snapshot
from multimodal_rag.store.weaviate import COLLECTION_NAME, WeaviateStore
from multimodal_rag.sync.embedding_check import apply_problems, inspect_embeddings
from multimodal_rag.sync.lock import LockHeld, RunLock
from multimodal_rag.sync.split_videos import (
    Row,
    Split,
    embedding_input,
    new_properties,
    plan_split,
    problems,
)


def summary(rows: list[Row], split: Split) -> list[str]:
    by_id = {u: p for u, p, _ in rows}
    videos = len(split.texts) + len(split.no_narration) + split.already_split
    lines = [
        f"{len(rows)} objects: {videos} video chunks and {split.other} others",
        f"{len(split.texts)} hybrid chunks to cut down to their narration",
        f"{split.already_split} video chunks already split (left as they are)",
        f"{len(split.no_narration)} with no narration (left as they are)",
    ]
    for uuid in split.no_narration:
        props = by_id[uuid]
        lines.append(
            f"    {props['source_name']} at {props['timestamp_seconds']} s"
        )
    if split.dropped_words:
        words = sorted(split.dropped_words.values())
        lines.append(
            f"descriptions left out: {sum(words)} words, median"
            f" {int(statistics.median(words))}, largest {words[-1]}"
        )
    lines.append(f"objects after: {len(rows)} (the count does not change)")
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
        print("Error: the split runs on the local collection only.")
        sys.exit(1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir: Path = args.runs_dir / f"split-{stamp}"

    embeddings = create_embeddings(settings)
    with WeaviateStore(
        weaviate_url=str(settings.weaviate_url), embeddings=embeddings
    ) as store:
        before = store.read_objects()
        split = plan_split(before)
        print("\n".join(summary(before, split)))
        if args.dry:
            print("\nDry run: nothing was written.")
            return
        if not split.texts:
            print("\nNothing to split.")
            return
        if not args.yes and input("\nSplit the LOCAL collection? [y/N] ") != "y":
            print("Nothing split.")
            return

        collection = store.client.collections.get(COLLECTION_NAME)
        refusals = apply_problems(inspect_embeddings(settings, collection, embeddings))
        if refusals:
            print("Refused:\n" + "\n".join(f"  - {r}" for r in refusals))
            sys.exit(1)

        lock = RunLock(settings.state_bucket, settings.gcp_project, f"split-{stamp}")
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
            props_by_id = {u: p for u, p, _ in before}
            uuids = sorted(split.texts)
            inputs = [embedding_input(props_by_id[u], split.texts[u]) for u in uuids]
            vectors = embed_texts(inputs, embeddings=embeddings)
            store.put_objects(
                [
                    (u, new_properties(props_by_id[u], split.texts[u]), v)
                    for u, v in zip(uuids, vectors)
                ]
            )
            after = store.read_objects()
            again = embed_texts(inputs, embeddings=embeddings)
        finally:
            lock.release()

    record = {
        "split": uuids,
        "no_narration": split.no_narration,
        "dropped_words": split.dropped_words,
        "snapshot": str(snapshot),
    }
    (run_dir / "split.json").write_text(json.dumps(record, indent=2))
    found = problems(before, after, split, dict(zip(uuids, again)))
    if found:
        print("\nFailed verification:")
        print("\n".join(f"  - {p}" for p in found[:20]))
        print(f"Roll back: make snapshot-restore SNAPSHOT_DIR={snapshot} REPLACE=1")
        sys.exit(1)
    print(f"\nSplit {len(uuids)} chunks. The count, every other object and every new")
    print(f"vector were verified. Record: {run_dir / 'split.json'}. Roll back with:")
    print(f"  make snapshot-restore SNAPSHOT_DIR={snapshot} REPLACE=1")


if __name__ == "__main__":
    main()
