"""Fetch the pages a plan lists, one at a time, and keep them for a later apply."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from firecrawl import FirecrawlApp

Scrape = Callable[[str], str]


@dataclass(frozen=True)
class FetchedPage:
    url: str
    markdown: str | None
    error: str | None = None


def firecrawl_scraper(api_key: str) -> Scrape:
    """Scrape one page with the options the crawler uses (main content, markdown)."""
    app = FirecrawlApp(api_key=api_key)

    def scrape(url: str) -> str:
        document = app.scrape(url, formats=["markdown"], only_main_content=True)
        return document.markdown or ""

    return scrape


def fetch_pages(
    urls: list[str], scrape: Scrape, *, pause: float = 0.0
) -> list[FetchedPage]:
    """Fetch every URL. A failed or empty page is recorded and never aborts the rest."""
    results = []
    for index, url in enumerate(urls):
        if index and pause:
            time.sleep(pause)
        try:
            markdown = scrape(url)
        except Exception as exc:  # one bad page must not stop the run
            results.append(FetchedPage(url, None, f"{type(exc).__name__}: {exc}"))
            continue
        if not markdown.strip():
            results.append(FetchedPage(url, None, "empty page"))
        else:
            results.append(FetchedPage(url, markdown))
    return results


def load_run(run_dir: Path) -> list[FetchedPage]:
    """Reload the pages of an earlier fetch, so a later step reuses them unchanged."""
    summary = json.loads((run_dir / "fetch.json").read_text())
    return [
        FetchedPage(
            row["url"],
            (run_dir / "pages" / row["file"]).read_text(encoding="utf-8")
            if row["file"]
            else None,
            row["error"],
        )
        for row in summary
    ]


def file_name(url: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", url.split("://", 1)[-1]).strip("_") + ".md"


def save_run(run_dir: Path, pages: list[FetchedPage]) -> None:
    """Write each fetched page and a summary, so an apply can reuse them unchanged."""
    (run_dir / "pages").mkdir(parents=True, exist_ok=True)
    summary = []
    for page in pages:
        name = file_name(page.url)
        if page.markdown is not None:
            (run_dir / "pages" / name).write_text(page.markdown, encoding="utf-8")
        summary.append(
            {
                "url": page.url,
                "file": name if page.markdown is not None else None,
                "chars": len(page.markdown or ""),
                "error": page.error,
            }
        )
    (run_dir / "fetch.json").write_text(json.dumps(summary, indent=2))
