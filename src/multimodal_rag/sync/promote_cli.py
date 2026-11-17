"""``make promote-plan``, ``make promote`` and ``make promote-rollback``.

The target is ``hosted`` (the Weaviate Cloud cluster, written with the admin key) or
``rehearsal`` (a local copy of hosted to practise on). The plan only reads; the viewer
key is enough for it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from multimodal_rag.checks import Check
from multimodal_rag.models.config import AppSettings
from multimodal_rag.models.llm import create_embeddings
from multimodal_rag.retrieval_gate import Canary, load_canaries, run_gate
from multimodal_rag.store.backup import upload_snapshot
from multimodal_rag.store.fingerprint import embedding_fingerprint
from multimodal_rag.store.snapshot import export_snapshot
from multimodal_rag.store.weaviate import COLLECTION_NAME, WeaviateStore
from multimodal_rag.sync.apply import ApplyRefused
from multimodal_rag.sync.embedding_check import apply_problems, inspect_embeddings
from multimodal_rag.sync.lock import LockHeld, RunLock
from multimodal_rag.sync.policy import load_policy
from multimodal_rag.sync.promote import (
    PromoteDeps,
    PromoteFailed,
    PromoteOptions,
    PromotePlan,
    build_plan,
    load_snapshot,
    promote,
    rollback,
    to_objects,
)
from multimodal_rag.sync.rehearsal import REHEARSAL_GRPC, REHEARSAL_URL

EXIT_NEEDS_REVIEW = 10
RUNS = Path("runs")


@contextmanager
def open_target(
    settings: AppSettings, name: str, *, write: bool
) -> Iterator[WeaviateStore]:
    embeddings = create_embeddings(settings)
    if name == "rehearsal":
        store = WeaviateStore(
            weaviate_url=REHEARSAL_URL, embeddings=embeddings, grpc_port=REHEARSAL_GRPC
        )
    else:
        secret = settings.weaviate_viewer_api_key
        if write:
            secret = settings.weaviate_admin_api_key
        key = secret
        store = WeaviateStore(
            weaviate_url=settings.weaviate_hosted_url,
            embeddings=embeddings,
            weaviate_mode="cloud",
            weaviate_api_key=key.get_secret_value(),
            allow_hosted_write=write,
        )
    try:
        yield store
    finally:
        store.close()


@contextmanager
def open_local(settings: AppSettings) -> Iterator[WeaviateStore]:
    store = WeaviateStore(
        weaviate_url=str(settings.weaviate_url), embeddings=create_embeddings(settings)
    )
    try:
        yield store
    finally:
        store.close()


def summary(plan: PromotePlan) -> list[str]:
    lines = [
        f"promotion plan {plan.plan_hash[:16]} to {plan.target}",
        f"  local {plan.local_count} objects, {plan.target} {plan.target_count}",
        f"  add {len(plan.add)}, replace {len(plan.replace)} (text or properties),"
        f" refresh the vector of {len(plan.vector)}, delete {len(plan.delete)},"
        f" unchanged {plan.unchanged}",
        f"policy: {plan.policy.status}",
    ]
    lines += [f"  - {reason}" for reason in plan.policy.reasons]
    return lines


def _plan(settings: AppSettings, target_name: str) -> None:
    fingerprint = embedding_fingerprint(
        str(settings.embeddings.base_url), settings.embedding_model
    )
    with open_local(settings) as local, open_target(
        settings, target_name, write=False
    ) as target:
        plan = build_plan(
            to_objects(local.read_objects()),
            to_objects(target.read_objects()),
            target_name=target_name,
            policy=load_policy(),
            embedding=fingerprint.as_dict(),
            missing_properties=target.missing_properties(),
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS / f"promote-{target_name}-{stamp}"
    run_dir.mkdir(parents=True)
    (run_dir / "plan.json").write_text(plan.model_dump_json(indent=2))
    print("\n".join(summary(plan)))
    print(f"kept: {run_dir}")
    sys.exit(0 if plan.policy.status == "within" else EXIT_NEEDS_REVIEW)


def _run(settings: AppSettings, args: argparse.Namespace) -> None:
    run_dir: Path = args.run
    plan = PromotePlan.model_validate_json((run_dir / "plan.json").read_text())
    print("\n".join(summary(plan)))
    question = f"\nWrite these changes to the {plan.target.upper()} collection? [y/N] "
    if not args.yes and input(question) != "y":
        print("Nothing written.")
        sys.exit(0)
    bucket, project = settings.state_bucket, settings.gcp_project
    with open_local(settings) as local, open_target(
        settings, plan.target, write=True
    ) as target:

        def backup(label: str) -> str:
            directory = run_dir / f"snapshot-{plan.target}-{label}"
            manifest = export_snapshot(
                target.client,
                directory,
                embedding_provider=settings.embedding_provider,
                embedding_model=settings.embedding_model,
            )
            if plan.target != "hosted":
                return str(directory)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            destination = f"gs://{bucket}/snapshots/{stamp}-promote-{label}"
            upload_snapshot(directory, destination, project, manifest)
            return f"{directory} and {destination}"

        def gate(canaries: list[Canary]) -> list[Check]:
            return run_gate(canaries, local, target, settings.top_k)

        def embedding_check() -> tuple[dict[str, str], list[str]]:
            collection = local.client.collections.get(COLLECTION_NAME)
            embeddings = create_embeddings(settings)
            status = inspect_embeddings(settings, collection, embeddings)
            return status.fingerprint.as_dict(), apply_problems(status)

        deps = PromoteDeps(
            local=local,
            target=target,
            lock=RunLock(bucket, project, f"promote-{plan.target}"),
            backup=backup,
            gate=gate,
            embedding_check=embedding_check,
            canaries=load_canaries(),
            policy=load_policy(),
        )
        options = PromoteOptions(
            accept_review=args.accept_review,
            break_lock=args.break_lock,
            take_backup=not args.no_backup,
            canary_only=args.canary_only,
        )
        try:
            result = promote(plan, deps, options)
        except ApplyRefused as exc:
            print("Refused, nothing was written:")
            print("\n".join(f"  - {r}" for r in exc.reasons))
            sys.exit(EXIT_NEEDS_REVIEW if exc.needs_review else 1)
        except (PromoteFailed, LockHeld, RuntimeError) as exc:
            print(f"Failed: {exc}")
            print(f"The {plan.target} may be partly changed. Roll back with the")
            print("'before' snapshot (make promote-rollback), or build a new plan.")
            sys.exit(1)
    print(f"\nstatus: {result.status}")
    print(f"  written {result.written}, deleted {result.deleted}")
    if result.schema_added:
        print(f"  schema: added {', '.join(result.schema_added)} (cannot be removed)")
    backups = {"before": result.backup_before, "after": result.backup_after}
    for label, where in backups.items():
        if where:
            print(f"  backup {label}: {where}")
    for check in result.gate:
        print(f"  gate [{check.status.value}] {check.name}: {check.detail}")
    sys.exit(1 if result.status == "gate-failed" else 0)


def _rollback(settings: AppSettings, args: argparse.Namespace) -> None:
    snapshot = load_snapshot(args.snapshot)
    print(f"{len(snapshot)} objects in the snapshot; target {args.target}")
    question = "Put the target back exactly as the snapshot? [y/N] "
    if not args.yes and input(question) != "y":
        print("Nothing changed.")
        sys.exit(0)
    holder = f"rollback-{args.target}"
    lock = RunLock(settings.state_bucket, settings.gcp_project, holder)
    with open_target(settings, args.target, write=True) as target:
        left = rollback(snapshot, target, lock)
    if left:
        print(f"still different: {left[:5]}")
        sys.exit(1)
    print("rolled back: the target equals the snapshot")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--target", choices=("hosted", "rehearsal"), default="hosted")
    run = sub.add_parser("run")
    run.add_argument("--run", type=Path, required=True)
    for flag in ("yes", "accept-review", "break-lock", "no-backup", "canary-only"):
        run.add_argument(f"--{flag}", action="store_true")
    back = sub.add_parser("rollback")
    back.add_argument("--snapshot", type=Path, required=True)
    back.add_argument("--target", choices=("hosted", "rehearsal"), required=True)
    back.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    settings = AppSettings()
    if args.command == "plan":
        _plan(settings, args.target)
    elif args.command == "run":
        _run(settings, args)
    else:
        _rollback(settings, args)


if __name__ == "__main__":
    main()
