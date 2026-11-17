"""Tests for the sitemap inventory, the page ledger and the classifier."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest

from multimodal_rag.sync import ledger as ledger_module
from multimodal_rag.sync.classify import Kind, classify, similarity
from multimodal_rag.sync.inventory import (
    SitemapPage,
    fetch_sitemap,
    in_scope,
    knowledge_base_pages,
    parse_sitemap,
)
from multimodal_rag.sync.ledger import (
    Ledger,
    PageEntry,
    build_baseline,
    read_ledger,
    split_layers,
    text_chunk_id,
    write_ledger,
)
from multimodal_rag.sync.report import plausible

NS = 'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'
MARCH = datetime(2026, 3, 13, tzinfo=timezone.utc)
HOST = "hydrosym.paro.nl"


def _urlset(*entries: tuple[str, str | None]) -> bytes:
    rows = "".join(
        f"<url><loc>{u}</loc>" + (f"<lastmod>{lm}</lastmod>" if lm else "") + "</url>"
        for u, lm in entries
    )
    return f"<urlset {NS}>{rows}</urlset>".encode()


def test_parse_sitemap_reads_lastmod_and_tolerates_a_missing_one() -> None:
    xml = _urlset(
        (f"https://{HOST}/article/1-a", "2026-03-17T13:06:22Z"),
        (f"https://{HOST}/article/2-b", None),
    )

    children, pages = parse_sitemap(xml)

    assert children == []
    assert pages[0].lastmod == datetime(2026, 3, 17, 13, 6, 22, tzinfo=timezone.utc)
    assert pages[1].lastmod is None


def test_a_sitemap_index_is_followed() -> None:
    index = (
        f"<sitemapindex {NS}><sitemap><loc>https://x/a.xml</loc></sitemap>"
        "</sitemapindex>"
    )
    files = {
        "https://x/index.xml": index.encode(),
        "https://x/a.xml": _urlset((f"https://{HOST}/article/1-a", None)),
    }

    pages = fetch_sitemap("https://x/index.xml", files.__getitem__)

    assert [p.url for p in pages] == [f"https://{HOST}/article/1-a"]


def test_scope_matches_the_exact_host_not_a_substring() -> None:
    hydroman_slug = "https://hydroman.paro.nl/article/9-hydrosym-to-hydroman"

    assert not in_scope(hydroman_slug, "hydrosym.paro.nl")
    assert in_scope(hydroman_slug, "hydroman.paro.nl")


def test_scope_is_the_crawlers_article_path() -> None:
    assert in_scope(f"https://{HOST}/article/1-a", HOST)
    assert not in_scope(f"https://{HOST}/category/1-a", HOST)
    assert not in_scope(f"https://{HOST}/", HOST)


def test_kb_pages_counts_what_it_leaves_out() -> None:
    xml = _urlset(
        (f"https://{HOST}/article/1-a", "2026-01-01T00:00:00Z"),
        (f"https://{HOST}/category/x", None),
        ("https://other.example/article/1", None),
    )

    host, pages, outside = knowledge_base_pages(
        f"https://{HOST}/article/", lambda _url: xml
    )

    assert (host, len(pages), outside) == (HOST, 1, 1)


def _obj(url: str, chunk_id: object, source_type: str = "web", day: int = 13) -> object:
    return SimpleNamespace(
        uuid=chunk_id,
        properties={
            "source_url": url,
            "source_type": source_type,
            "ingested_at": datetime(2026, 3, day, tzinfo=timezone.utc),
        },
    )


def test_screenshots_stored_as_web_are_told_apart_by_their_uuid() -> None:
    url = f"https://{HOST}/article/1-a"
    ids = [text_chunk_id(url, 0), text_chunk_id(url, 1), uuid4(), uuid4(), uuid4()]

    assert split_layers(url, ids) == (2, 3)


def test_baseline_ignores_videos_and_keeps_the_latest_ingest_date() -> None:
    url = f"https://{HOST}/article/1-a"
    objects = [
        _obj(url, text_chunk_id(url, 0), day=13),
        _obj(url, text_chunk_id(url, 1), day=15),
        _obj(url, uuid4()),
        _obj("https://youtu.be/abc", uuid4(), source_type="video"),
    ]

    ledger = build_baseline(objects)

    assert list(ledger.pages) == [url]
    entry = ledger.pages[url]
    assert (entry.text_chunks, entry.frozen_chunks) == (2, 1)
    assert entry.ingested_at.day == 15
    assert entry.fingerprint is None
    assert entry.host == HOST


def _entry(url: str, lastmod: datetime | None = None) -> PageEntry:
    return PageEntry(
        url=url,
        host=HOST,
        text_chunks=2,
        frozen_chunks=0,
        ingested_at=MARCH,
        lastmod=lastmod,
    )


def _ledger(*urls: str) -> Ledger:
    return Ledger(pages={u: _entry(u) for u in urls})


def _page(slug: str, lastmod: datetime | None) -> SitemapPage:
    return SitemapPage(f"https://{HOST}/article/{slug}", lastmod)


def _kinds(items: list) -> dict[str, Kind]:  # type: ignore[type-arg]
    return {i.url.rsplit("/", 1)[-1]: i.kind for i in items}


def test_classification_by_lastmod_against_the_ingest_date() -> None:
    ledger = _ledger(
        *(f"https://{HOST}/article/{s}" for s in ("old", "edited", "nolm", "gone"))
    )
    inventory = {
        HOST: [
            _page("old", datetime(2026, 1, 1, tzinfo=timezone.utc)),
            _page("edited", datetime(2026, 4, 1, tzinfo=timezone.utc)),
            _page("nolm", None),
            _page("fresh", datetime(2026, 5, 1, tzinfo=timezone.utc)),
        ]
    }

    kinds = _kinds(classify(inventory, ledger))

    assert kinds == {
        "old": Kind.UNCHANGED,
        "edited": Kind.MODIFIED,
        "nolm": Kind.MODIFIED,
        "fresh": Kind.NEW,
        "gone": Kind.REMOVED,
    }


def test_verify_marks_every_stored_page_modified() -> None:
    ledger = _ledger(f"https://{HOST}/article/old")
    inventory = {HOST: [_page("old", datetime(2026, 1, 1, tzinfo=timezone.utc))]}

    assert _kinds(classify(inventory, ledger, verify=True)) == {"old": Kind.MODIFIED}


def test_a_host_without_a_usable_sitemap_produces_no_removals() -> None:
    ledger = _ledger(f"https://{HOST}/article/old")

    assert classify({}, ledger) == []


def test_a_removed_page_is_matched_to_its_likely_rename() -> None:
    old = f"https://{HOST}/article/437-automatic-deployment-hydrosym-documentation"
    new = f"https://{HOST}/article/525-automated-deployment-update-guide"
    ledger = _ledger(old)
    inventory = {
        HOST: [
            _page("525-automated-deployment-update-guide", MARCH),
            _page("600-something-unrelated", MARCH),
        ]
    }

    removed = [i for i in classify(inventory, ledger) if i.kind is Kind.REMOVED]

    assert removed[0].rename_of == new


def test_unrelated_pages_get_no_rename_hint() -> None:
    a = f"https://{HOST}/article/1-print-bom"
    b = f"https://{HOST}/article/2-zoom-out"

    assert similarity(a, b) < 0.5


@pytest.mark.parametrize(
    ("listed", "stored", "ok"),
    [(0, 99, False), (40, 99, False), (60, 99, True), (109, 99, True), (5, 0, True)],
)
def test_an_implausibly_small_sitemap_is_rejected(
    listed: int, stored: int, ok: bool
) -> None:
    pages = [_page(str(i), None) for i in range(listed)]

    assert plausible(pages, stored) is ok


def test_ledger_round_trips_through_json() -> None:
    ledger = _ledger(f"https://{HOST}/article/1-a")

    assert Ledger.model_validate_json(ledger.model_dump_json()) == ledger


def test_the_baseline_is_created_once_with_a_zero_generation_precondition() -> None:
    calls: list[list[str]] = []
    def fake(args: list[str], project: str = "") -> str:
        calls.append(args)
        return ""

    with patch.object(ledger_module, "run_gcloud", fake):
        write_ledger(_ledger(), "bucket", "proj", if_generation="0")

    assert "--if-generation-match=0" in calls[0]
    assert calls[0][-2] == "gs://bucket/ledger/pages.json"


def test_a_missing_ledger_reads_as_none() -> None:
    def fake(args: list[str], project: str = "") -> str:
        raise RuntimeError("gcloud failed: 404 No such object")

    with patch.object(ledger_module, "run_gcloud", fake):
        assert read_ledger("bucket", "proj") is None
    assert ledger_module.LEDGER_OBJECT == "ledger/pages.json"


def test_a_removed_page_is_matched_to_a_replacement_that_is_already_stored() -> None:
    old = f"https://{HOST}/article/437-automatic-deployment-hydrosym-documentation"
    new = f"https://{HOST}/article/525-automated-deployment-update-guide"
    ledger = Ledger(pages={old: _entry(old), new: _entry(new)})
    inventory = {HOST: [_page("525-automated-deployment-update-guide", MARCH)]}

    removed = [i for i in classify(inventory, ledger) if i.kind is Kind.REMOVED]

    assert removed[0].rename_of == new
