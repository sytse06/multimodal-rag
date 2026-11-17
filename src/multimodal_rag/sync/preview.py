"""Read-only preview of the web lane: fetch what changed and show the chunk difference.

``make sync-preview`` takes the pages the inventory marks modified or new, fetches them
with Firecrawl, chunks them with the same normaliser and chunker as ``make ingest``,
and compares them with the stored chunks by position. It writes nothing to Weaviate or
the bucket; the fetched pages are kept under ``runs/`` for the plan and the apply.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import weaviate
from weaviate.classes.query import Filter

from multimodal_rag.ingest.template import check_template
from multimodal_rag.ingest.web import chunk_page, normalise_page, page_fingerprint
from multimodal_rag.models.config import AppSettings
from multimodal_rag.store.weaviate import COLLECTION_NAME
from multimodal_rag.sync.classify import Kind
from multimodal_rag.sync.diff import (
    ADDED,
    CHANGED,
    IDENTICAL,
    REMOVED,
    TEMPLATE_ONLY,
    PageDiff,
    StoredChunk,
    diff_page,
    stored_from_objects,
)
from multimodal_rag.sync.fetch import (
    fetch_pages,
    firecrawl_scraper,
    load_run,
    save_run,
)
from multimodal_rag.sync.report import build_inventory

RUNS_DIR = Path("runs")
MAX_PAGES = 30  # a guard until the policy file exists
EXIT_NEEDS_REVIEW = 10


def page_chunk_ids(collection: Any, url: str) -> list[str]:
    """Every chunk the collection holds for a page, screenshots included."""
    response = collection.query.fetch_objects(
        filters=Filter.by_property("source_url").equal(url), limit=1000
    )
    return [str(o.uuid) for o in response.objects]


def stored_chunks(collection: Any, url: str) -> dict[int, StoredChunk]:
    response = collection.query.fetch_objects(
        filters=Filter.by_property("source_url").equal(url), limit=1000
    )
    return stored_from_objects(url, response.objects)


def describe(diff: PageDiff, fingerprint: str) -> str:
    return (
        f"  {diff.url.rsplit('/', 1)[-1][:58]:58} stored {diff.stored:2} -> fresh"
        f" {diff.fresh:2} | same {diff.count(IDENTICAL)}, template-only"
        f" {diff.count(TEMPLATE_ONLY)}, changed {diff.count(CHANGED)}, added"
        f" {diff.count(ADDED)}, removed {diff.count(REMOVED)} | {fingerprint[:10]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--only", action="append", default=[], help="limit to a URL")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES)
    parser.add_argument(
        "--from-run", type=Path, help="reuse the pages of an earlier fetch"
    )
    args = parser.parse_args()
    settings = AppSettings()

    inventory = build_inventory(settings, verify=args.verify)
    wanted = [
        i for i in inventory.items if i.kind in (Kind.MODIFIED, Kind.NEW)
    ]
    if args.only:
        wanted = [i for i in wanted if i.url in args.only]
    for removed in (i for i in inventory.items if i.kind is Kind.REMOVED):
        hint = f" (may be renamed to {removed.rename_of})" if removed.rename_of else ""
        print(f"removal candidate, not fetched: {removed.url}{hint}")
    if len(wanted) > args.max_pages:
        print(f"{len(wanted)} pages to fetch exceeds --max-pages {args.max_pages}")
        sys.exit(EXIT_NEEDS_REVIEW)

    names = {host: info["name"] for host, info in inventory.hosts.items()}
    if args.from_run:
        run_dir = args.from_run
        fetched = load_run(run_dir)
    else:
        key = settings.firecrawl_api_key.get_secret_value()
        fetched = fetch_pages(
            [i.url for i in wanted], firecrawl_scraper(key), pause=1.0
        )
        run_dir = RUNS_DIR / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        save_run(run_dir, fetched)

    url = urlparse(str(settings.weaviate_url))
    client = weaviate.connect_to_local(
        host=url.hostname or "localhost", port=url.port or 8080
    )
    try:
        collection = client.collections.get(COLLECTION_NAME)
        print(f"\nfetched {sum(p.markdown is not None for p in fetched)} of"
              f" {len(fetched)} pages into {run_dir}\n")
        for page in fetched:
            if page.markdown is None:
                print(f"  FAILED {page.url}: {page.error}")
                continue
            host = urlparse(page.url).netloc
            fresh = chunk_page(
                page.markdown, page.url, names[host], settings.chunk_size
            )
            title = normalise_page(page.markdown).title
            diff = diff_page(
                page.url, stored_chunks(collection, page.url), fresh, title=title
            )
            print(describe(diff, page_fingerprint(page.markdown)))
    finally:
        client.close()

    report = check_template({p.url: p.markdown for p in fetched if p.markdown})
    print()
    print("\n".join(report.lines()))
    failed = [p for p in fetched if p.markdown is None]
    if fetched and len(failed) == len(fetched):
        sys.exit(1)
    sys.exit(EXIT_NEEDS_REVIEW if report.needs_review or failed else 0)


if __name__ == "__main__":
    main()
