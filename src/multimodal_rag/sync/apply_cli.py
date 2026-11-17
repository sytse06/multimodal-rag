"""``make sync-apply``: apply a reviewed plan to the LOCAL collection.

Writes to local Weaviate only (the store refuses a hosted target). The plan must be
the one a person reviewed: its hash, the registry and the ledger must all still match.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import weaviate
from weaviate.classes.query import Filter

from multimodal_rag.checks import Status
from multimodal_rag.ingest import registry
from multimodal_rag.models.config import AppSettings
from multimodal_rag.models.llm import create_embeddings
from multimodal_rag.preflight import check_ollama
from multimodal_rag.retrieval_gate import gate_stores, load_canaries, run_gate
from multimodal_rag.store.snapshot import export_snapshot
from multimodal_rag.store.weaviate import COLLECTION_NAME, WeaviateStore
from multimodal_rag.sync.apply import (
    ApplyDeps,
    ApplyFailed,
    ApplyOptions,
    ApplyRefused,
    Counts,
    PageState,
    apply_plan,
)
from multimodal_rag.sync.diff import stored_from_objects
from multimodal_rag.sync.embedding_check import apply_problems, inspect_embeddings
from multimodal_rag.sync.ledger import Ledger, read_ledger, write_ledger
from multimodal_rag.sync.lock import LockHeld, RunLock
from multimodal_rag.sync.manifest import upload_run
from multimodal_rag.sync.plan import Plan
from multimodal_rag.sync.plan_cli import summary
from multimodal_rag.sync.policy import load_policy

EXIT_NEEDS_REVIEW = 10


def restore_command(snapshot: str) -> str:
    return f"make snapshot-restore SNAPSHOT_DIR={snapshot} REPLACE=1"


def _counts(counts: Counts | None) -> dict[str, int] | None:
    return {"total": counts.total, "video": counts.video} if counts else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run dir")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation")
    parser.add_argument("--accept-review", action="store_true")
    parser.add_argument("--break-lock", action="store_true")
    parser.add_argument("--no-snapshot", action="store_true")
    args = parser.parse_args()
    settings = AppSettings()
    run_dir: Path = args.run
    plan = Plan.model_validate_json((run_dir / "plan.json").read_text())
    bucket, project = settings.state_bucket, settings.gcp_project

    if settings.weaviate_mode != "local":
        print("Error: apply writes to local Docker only (WEAVIATE_MODE must be local)")
        sys.exit(1)
    ollama = check_ollama(settings)
    if ollama.status is Status.FAIL:
        print(f"Error: embeddings are unavailable: {ollama.detail}")
        sys.exit(1)

    print("\n".join(summary(plan)))
    question = "\nApply this plan to the LOCAL collection? [y/N] "
    if not args.yes and input(question) != "y":
        print("Nothing applied.")
        sys.exit(0)

    url = urlparse(str(settings.weaviate_url))
    client = weaviate.connect_to_local(
        host=url.hostname or "localhost", port=url.port or 8080
    )
    try:
        collection = client.collections.get(COLLECTION_NAME)

        def page_state(page_url: str) -> PageState:
            objs = collection.query.fetch_objects(
                filters=Filter.by_property("source_url").equal(page_url), limit=1000
            ).objects
            text = stored_from_objects(page_url, objs)
            texts = {i: c.text for i, c in text.items()}
            return PageState(texts, len(objs) - len(text))

        def counts() -> Counts:
            total = collection.aggregate.over_all(total_count=True).total_count or 0
            video = (
                collection.aggregate.over_all(
                    filters=Filter.by_property("source_type").equal("video"),
                    total_count=True,
                ).total_count
                or 0
            )
            return Counts(total, video)

        def registry_generation() -> str | None:
            remote = registry._remote(registry._url(settings), project)
            return remote[0] if remote else None

        def read() -> tuple[Ledger, str]:
            found = read_ledger(bucket, project)
            if found is None:
                raise ApplyRefused(["no ledger in the bucket"])
            return found

        def write(ledger: Ledger, if_generation: str) -> str:
            write_ledger(ledger, bucket, project, if_generation=if_generation)
            return read()[1]

        def snapshot() -> str:
            target = run_dir / "snapshot-local"
            export_snapshot(
                client,
                target,
                embedding_provider=settings.embedding_provider,
                embedding_model=settings.embedding_model,
            )
            return str(target)

        def gate(canaries):  # type: ignore[no-untyped-def]
            with gate_stores(settings) as (hosted, local):
                return run_gate(canaries, hosted, local, settings.top_k)

        embeddings = create_embeddings(settings)

        def embedding_check() -> tuple[dict[str, str], list[str]]:
            status = inspect_embeddings(settings, collection, embeddings)
            return status.fingerprint.as_dict(), apply_problems(status)

        with WeaviateStore(
            weaviate_url=str(settings.weaviate_url),
            embeddings=embeddings,
        ) as store:
            deps = ApplyDeps(
                store=store,
                page_state=page_state,
                counts=counts,
                registry_generation=registry_generation,
                read_ledger=read,
                write_ledger=write,
                lock=RunLock(bucket, project, plan.run_id),
                snapshot=snapshot,
                gate=gate,
                policy=load_policy(),
                chunk_size=settings.chunk_size,
                fixed_canaries=load_canaries(),
                embedding_check=embedding_check,
            )
            options = ApplyOptions(
                accept_review=args.accept_review,
                break_lock=args.break_lock,
                take_snapshot=not args.no_snapshot,
            )
            try:
                result = apply_plan(plan, run_dir, deps, options)
            except ApplyRefused as exc:
                print("Refused, nothing was written:")
                print("\n".join(f"  - {r}" for r in exc.reasons))
                sys.exit(EXIT_NEEDS_REVIEW if exc.needs_review else 1)
            except (ApplyFailed, LockHeld, RuntimeError) as exc:
                print(f"Failed: {exc}")
                snapshot_dir = run_dir / "snapshot-local"
                if snapshot_dir.exists():
                    print("The ledger was not updated. To roll back local:")
                    print(f"  {restore_command(str(snapshot_dir))}")
                sys.exit(1)
    finally:
        client.close()

    (run_dir / "apply.json").write_text(
        json.dumps(
            {
                "status": result.status,
                "inserted": result.inserted,
                "replaced": result.replaced,
                "deleted": result.deleted,
                "pages_already_applied": result.pages_already_applied,
                "counts_before": _counts(result.counts_before),
                "counts_after": _counts(result.counts_after),
                "ledger_generation": result.ledger_generation,
                "snapshot": result.snapshot,
                "gate": [
                    {"name": c.name, "status": c.status.value, "detail": c.detail}
                    for c in result.gate
                ],
            },
            indent=2,
        )
    )
    upload_run(run_dir, bucket, project)
    print(f"\nstatus: {result.status}")
    print(
        f"  inserted {result.inserted}, replaced {result.replaced},"
        f" deleted {result.deleted}"
    )
    if result.counts_before and result.counts_after:
        print(
            f"  collection {result.counts_before.total} ->"
            f" {result.counts_after.total} chunks"
        )
    for check in result.gate:
        print(f"  gate [{check.status.value}] {check.name}: {check.detail}")
    if result.status == "gate-failed":
        print("The ledger was not updated.")
        if result.snapshot:
            print(f"To roll back local: {restore_command(result.snapshot)}")
        sys.exit(1)
    print(f"  ledger generation {result.ledger_generation}")


if __name__ == "__main__":
    main()
