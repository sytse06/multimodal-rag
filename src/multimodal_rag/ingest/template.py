"""Detect a change of the site template across a set of fetched pages.

The normaliser removes the template's byline and footer by recognising their wording.
If the site changes that wording, the rules stop matching and the new boilerplate would
flow into every chunk. This report notices that from the pages themselves: how many
still match each rule, whether a cut looked wrong, and which paragraphs repeat across
many pages even after normalising (the signature of a new header, footer or banner).
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

from multimodal_rag.ingest.web import normalise_page

MIN_RULE_RATE = 0.9  # share of pages that should still match each rule
MIN_REPEAT_PAGES = 3  # a paragraph must repeat on at least this many pages
REPEAT_SHARE = 0.5  # ... and on at least this share of the fetched pages
MAX_BLOCK_CHARS = 300  # boilerplate is short; long repeats are real content
MIN_PAGE_CHARS = 200  # a normalised page shorter than this is suspect


@dataclass
class TemplateReport:
    pages: int
    byline_rate: float
    footer_rate: float
    suspicious: list[str] = field(default_factory=list)
    near_empty: list[str] = field(default_factory=list)
    repeated: list[tuple[str, int]] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return bool(
            self.pages
            and (
                self.byline_rate < MIN_RULE_RATE
                or self.footer_rate < MIN_RULE_RATE
                or self.suspicious
                or self.repeated
            )
        )

    def lines(self) -> list[str]:
        out = [
            f"template: {self.pages} pages, byline matched {self.byline_rate:.0%},"
            f" footer matched {self.footer_rate:.0%}"
        ]
        if self.byline_rate < MIN_RULE_RATE:
            out.append("  byline rule no longer matches: the template may have changed")
        if self.footer_rate < MIN_RULE_RATE:
            out.append("  footer rule no longer matches: the template may have changed")
        out += [f"  footer cut looked wrong, left whole: {u}" for u in self.suspicious]
        out += [
            f"  short page after normalising (content may be an embedded video): {u}"
            for u in self.near_empty
        ]
        out += [
            f"  repeated on {count} pages, not removed: {block[:90]!r}"
            for block, count in self.repeated
        ]
        return out


def _blocks(markdown: str) -> set[str]:
    found = set()
    for block in markdown.split("\n\n"):
        text = " ".join(block.split())
        if 20 <= len(text) <= MAX_BLOCK_CHARS and not text.startswith("#"):
            found.add(text)
    return found


def check_template(pages: dict[str, str]) -> TemplateReport:
    """Report on a set of raw scraped pages (url -> markdown)."""
    if not pages:
        return TemplateReport(0, 1.0, 1.0)
    normalised = {url: normalise_page(md) for url, md in pages.items()}
    count = len(pages)
    report = TemplateReport(
        pages=count,
        byline_rate=sum(n.byline_removed for n in normalised.values()) / count,
        footer_rate=sum(n.footer_removed for n in normalised.values()) / count,
        suspicious=[u for u, n in normalised.items() if n.footer_suspicious],
        near_empty=[
            u for u, n in normalised.items() if len(n.markdown) < MIN_PAGE_CHARS
        ],
    )
    seen: collections.Counter[str] = collections.Counter()
    for page in normalised.values():
        seen.update(_blocks(page.markdown))
    floor = max(MIN_REPEAT_PAGES, round(count * REPEAT_SHARE))
    report.repeated = sorted(
        ((block, n) for block, n in seen.items() if n >= floor),
        key=lambda item: -item[1],
    )
    return report
