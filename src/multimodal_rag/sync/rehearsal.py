"""Seed the rehearsal Weaviate from a hosted backup.

The rehearsal copy lets a promotion and its rollback run end to end on data that is
byte for byte the hosted state, without touching production.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import weaviate

from multimodal_rag.store.restore import restore_snapshot

REHEARSAL_HTTP = 8081
REHEARSAL_GRPC = 50052
REHEARSAL_URL = f"http://localhost:{REHEARSAL_HTTP}"


def seed(snapshot_dir: Path) -> str:
    """Restore a snapshot over the rehearsal collection; return the index it used.

    Hosted runs the hfresh index. The open-source server may refuse it, in which case
    hnsw is used and the index behaviour is covered by the canary write on hosted.
    """
    client = weaviate.connect_to_local(
        port=REHEARSAL_HTTP, grpc_port=REHEARSAL_GRPC
    )
    try:
        for index in ("hfresh", "hnsw"):
            try:
                restore_snapshot(
                    client, snapshot_dir, replace=True, vector_index=index
                )
                return index
            except Exception as exc:  # the server may not offer this index
                last = f"{index}: {type(exc).__name__}: {exc}"
        raise RuntimeError(f"could not restore into the rehearsal Weaviate ({last})")
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(f"seeded with the {seed(args.snapshot)} index")
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
