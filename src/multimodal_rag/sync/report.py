"""Read-only inventory report: what changed upstream since the ledger was written.

``make sync-inventory`` fetches the sitemaps, reads (or builds) the ledger and prints
what is modified, new and removed. It fetches no pages and writes nothing to Weaviate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import weaviate

from multimodal_rag.ingest import registry
from multimodal_rag.models.config import AppSettings
from multimodal_rag.store.weaviate import COLLECTION_NAME
from multimodal_rag.sync.classify import Item, Kind, classify
from multimodal_rag.sync.inventory import SitemapPage, knowledge_base_pages
from multimodal_rag.sync.ledger import (
    Ledger,
    build_baseline,
    read_ledger,
    write_ledger,
)

PLAUSIBLE_SHARE = 0.5


def plausible(pages: list[SitemapPage], stored: int) -> bool:
    """A sitemap that lists far fewer pages than are stored is treated as a failure."""
    return bool(pages) and len(pages) >= stored * PLAUSIBLE_SHARE


def load_baseline(settings: AppSettings) -> Ledger:
    url = urlparse(str(settings.weaviate_url))
    client = weaviate.connect_to_local(
        host=url.hostname or "localhost", port=url.port or 8080
    )
    try:
        return build_baseline(client.collections.get(COLLECTION_NAME).iterator())
    finally:
        client.close()


@dataclass
class Inventory:
    ledger: Ledger
    source: str
    hosts: dict[str, dict[str, Any]]
    items: list[Item]


def build_inventory(
    settings: AppSettings, *, verify: bool = False, init_ledger: bool = False
) -> Inventory:
    stored = read_ledger(settings.state_bucket, settings.gcp_project)
    if stored is None:
        ledger, source = load_baseline(settings), "baseline built from local, not saved"
        if init_ledger:
            write_ledger(
                ledger,
                settings.state_bucket,
                settings.gcp_project,
                if_generation="0",
            )
            source = "baseline written to the bucket (created once)"
    else:
        ledger, generation = stored
        source = f"bucket generation {generation}"

    roots = registry.validate(registry.SOURCES_PATH).knowledge_bases
    inventory: dict[str, list[SitemapPage]] = {}
    hosts: dict[str, dict[str, Any]] = {}
    for kb in roots:
        host, pages, out_of_scope = knowledge_base_pages(str(kb.url))
        held = sum(1 for p in ledger.pages.values() if p.host == host)
        ok = plausible(pages, held)
        hosts[host] = {
            "name": kb.name,
            "in_scope": len(pages),
            "out_of_scope": out_of_scope,
            "stored": held,
            "sitemap_ok": ok,
        }
        if ok:
            inventory[host] = pages
    items = classify(inventory, ledger, verify=verify)
    for host, info in hosts.items():
        mine = [i for i in items if urlparse(i.url).netloc == host]
        for kind in Kind:
            info[kind.value] = sum(1 for i in mine if i.kind is kind)
    return Inventory(ledger, source, hosts, items)


def run_inventory(
    settings: AppSettings, *, verify: bool = False, init_ledger: bool = False
) -> dict[str, Any]:
    inventory = build_inventory(settings, verify=verify, init_ledger=init_ledger)
    items = inventory.items
    return {
        "ledger": inventory.source,
        "hosts": inventory.hosts,
        "items": [
            {
                "url": i.url,
                "kind": i.kind.value,
                "reason": i.reason,
                "rename_of": i.rename_of,
            }
            for i in items
            if i.kind is not Kind.UNCHANGED
        ],
    }


def format_report(result: dict[str, Any]) -> str:
    lines = [f"Ledger: {result['ledger']}", ""]
    for host, info in result["hosts"].items():
        if not info["sitemap_ok"]:
            found = info["in_scope"]
            lines.append(f"{host}: sitemap unusable ({found} pages), skipped")
            continue
        counts = ", ".join(f"{k.value} {info[k.value]}" for k in Kind)
        lines.append(
            f"{host}: {info['in_scope']} in scope (+{info['out_of_scope']} outside"
            f" /article/), {info['stored']} stored | {counts}"
        )
    for kind in ("modified", "new", "removed"):
        rows = [i for i in result["items"] if i["kind"] == kind]
        if rows:
            lines += ["", f"{kind} ({len(rows)}):"]
            for row in rows:
                hint = (
                    f"  <- may be renamed to {row['rename_of']}"
                    if row["rename_of"]
                    else ""
                )
                lines.append(f"  {row['url']}  [{row['reason']}]{hint}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true", help="treat every page as modified"
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--init-ledger", action="store_true", help="save the baseline once"
    )
    args = parser.parse_args()
    try:
        result = run_inventory(
            AppSettings(), verify=args.verify, init_ledger=args.init_ledger
        )
    except Exception as exc:  # report, then fail with a distinct exit code
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, indent=2) if args.json else format_report(result))


if __name__ == "__main__":
    main()
