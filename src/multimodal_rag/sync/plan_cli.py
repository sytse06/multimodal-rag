"""``make sync-plan``: build, hash and keep a sync plan. Writes nothing to Weaviate."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import weaviate

from multimodal_rag.ingest import registry
from multimodal_rag.ingest.template import check_template
from multimodal_rag.models.config import AppSettings
from multimodal_rag.store.weaviate import COLLECTION_NAME
from multimodal_rag.sync.classify import Kind
from multimodal_rag.sync.fetch import fetch_pages, firecrawl_scraper, load_run, save_run
from multimodal_rag.sync.manifest import build_manifest, upload_run, write_plan_files
from multimodal_rag.sync.plan import Plan, PlanInputs, build_plan
from multimodal_rag.sync.policy import load_policy
from multimodal_rag.sync.preview import RUNS_DIR, page_chunk_ids, stored_chunks
from multimodal_rag.sync.probe import probe_url
from multimodal_rag.sync.report import build_inventory

EXIT_NEEDS_REVIEW = 10


def registry_state(settings: AppSettings) -> tuple[str | None, bool]:
    """Return (recorded registry generation, whether it is stale against the bucket)."""
    recorded = (
        registry.GENERATION_PATH.read_text().strip()
        if registry.GENERATION_PATH.exists()
        else None
    )
    remote = registry._remote(registry._url(settings), settings.gcp_project)
    return recorded, recorded is None or remote is None or remote[0] != recorded


def summary(plan: Plan) -> list[str]:
    t = plan.totals
    lines = [
        f"plan {plan.plan_hash[:16]}  run {plan.run_id}",
        f"  pages {t['pages']} (new {t['new_pages']}, content changed"
        f" {t['content_changed_pages']}, template-only {t['template_only_pages']}),"
        f" failures {t['failures']}",
        f"  positions: write {t['positions_to_write']}, delete"
        f" {t['positions_to_delete']}, skip {t['positions_skipped']};"
        f" frozen chunks untouched {t['frozen_chunks_untouched']}",
        f"  canaries for changed pages: {len(plan.canaries)}",
    ]
    for removal in plan.removals:
        hint = f" (may be renamed to {removal.rename_of})" if removal.rename_of else ""
        status = f" {removal.http_status}" if removal.http_status else ""
        lines.append(
            f"  removal candidate [{removal.evidence}{status}], needs confirmation:"
            f" {removal.url}{hint}; {removal.text_chunks} text +"
            f" {removal.frozen_chunks} screenshot chunks"
        )
    lines += [f"  {line.strip()}" for line in plan.template["lines"][1:]]
    lines.append(f"policy: {plan.policy.status}")
    lines += [f"  - {reason}" for reason in plan.policy.reasons]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--from-run", type=Path)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    settings = AppSettings()
    try:
        inventory = build_inventory(settings, verify=args.verify)
        if inventory.ledger_generation is None:
            raise RuntimeError("no ledger yet; run make sync-inventory INIT=1")
        registry_generation, registry_stale = registry_state(settings)
        wanted = [i for i in inventory.items if i.kind in (Kind.MODIFIED, Kind.NEW)]
        if args.only:
            wanted = [i for i in wanted if i.url in args.only]
        if args.from_run:
            run_dir, fetched = args.from_run, load_run(args.from_run)
        else:
            key = settings.firecrawl_api_key.get_secret_value()
            fetched = fetch_pages(
                [i.url for i in wanted], firecrawl_scraper(key), pause=1.0
            )
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_dir = RUNS_DIR / stamp
            save_run(run_dir, fetched)
        url = urlparse(str(settings.weaviate_url))
        client = weaviate.connect_to_local(
            host=url.hostname or "localhost", port=url.port or 8080
        )
        try:
            collection = client.collections.get(COLLECTION_NAME)
            plan = build_plan(
                PlanInputs(
                    run_id=run_dir.name,
                    registry_generation=registry_generation,
                    registry_stale=registry_stale,
                    ledger=inventory.ledger,
                    ledger_generation=inventory.ledger_generation,
                    items=inventory.items,
                    fetched=fetched,
                    kb_names={h: i["name"] for h, i in inventory.hosts.items()},
                    chunk_size=settings.chunk_size,
                    stored_loader=lambda u: stored_chunks(collection, u),
                    probe=probe_url,
                    chunk_ids_loader=lambda u: page_chunk_ids(collection, u),
                    template=check_template(
                        {p.url: p.markdown for p in fetched if p.markdown}
                    ),
                    policy=load_policy(),
                )
            )
        finally:
            client.close()
        write_plan_files(run_dir, plan, build_manifest(plan))
        where = str(run_dir)
        if not args.no_upload:
            where = upload_run(run_dir, settings.state_bucket, settings.gcp_project)
    except Exception as exc:  # report, then fail with a distinct exit code
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
    print(plan.model_dump_json(indent=2) if args.json else "\n".join(summary(plan)))
    print(f"kept: {where}")
    sys.exit(0 if plan.policy.status == "within" else EXIT_NEEDS_REVIEW)


if __name__ == "__main__":
    main()
