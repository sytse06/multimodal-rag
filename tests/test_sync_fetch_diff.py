"""Tests for fetching pages and diffing them against stored chunks by position."""

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from multimodal_rag.models.chunks import WebChunk
from multimodal_rag.sync.diff import (
    ADDED,
    CHANGED,
    IDENTICAL,
    REMOVED,
    TEMPLATE_ONLY,
    StoredChunk,
    diff_page,
    stored_from_objects,
)
from multimodal_rag.sync.fetch import FetchedPage, fetch_pages, load_run, save_run
from multimodal_rag.sync.ledger import text_chunk_id

URL = "https://hydrosym.paro.nl/article/1-a"


def _fresh(*texts: str) -> list[WebChunk]:
    return [
        WebChunk(text=t, source_url=URL, source_name="KB", chunk_index=i)
        for i, t in enumerate(texts)
    ]


def _stored(*texts: str) -> dict[int, StoredChunk]:
    return {i: StoredChunk(text_chunk_id(URL, i), t) for i, t in enumerate(texts)}


def test_fetch_records_failures_and_empty_pages_and_keeps_going() -> None:
    def scrape(url: str) -> str:
        if url.endswith("boom"):
            raise RuntimeError("429 rate limited")
        return "" if url.endswith("empty") else "# Page\n\ncontent"

    pages = fetch_pages(["https://x/ok", "https://x/boom", "https://x/empty"], scrape)

    assert [p.markdown is not None for p in pages] == [True, False, False]
    assert "RuntimeError: 429" in (pages[1].error or "")
    assert pages[2].error == "empty page"


def test_save_run_keeps_each_page_and_a_summary(tmp_path: Path) -> None:
    pages = [
        FetchedPage("https://hydrosym.paro.nl/article/1-a", "# A\n\ntext"),
        FetchedPage("https://hydrosym.paro.nl/article/2-b", None, "empty page"),
    ]

    save_run(tmp_path / "run", pages)

    summary = json.loads((tmp_path / "run" / "fetch.json").read_text())
    assert summary[0]["chars"] == len("# A\n\ntext") and summary[0]["error"] is None
    assert summary[1]["file"] is None and summary[1]["error"] == "empty page"
    saved = tmp_path / "run" / "pages" / summary[0]["file"]
    assert saved.read_text() == "# A\n\ntext"


def test_identical_changed_added_and_removed_positions() -> None:
    diff = diff_page(URL, _stored("a", "b", "c", "d"), _fresh("a", "B", "c"))

    by_status = {c.index: c.status for c in diff.changes}
    assert by_status == {0: IDENTICAL, 1: CHANGED, 2: IDENTICAL, 3: REMOVED}
    assert (diff.stored, diff.fresh) == (4, 3)


def test_a_longer_page_adds_positions() -> None:
    diff = diff_page(URL, _stored("a"), _fresh("a", "b", "c"))

    assert [c.status for c in diff.changes] == [IDENTICAL, ADDED, ADDED]
    assert diff.count(ADDED) == 2


def test_a_page_with_no_stored_chunks_is_all_added() -> None:
    diff = diff_page(URL, {}, _fresh("a", "b"))

    assert diff.count(ADDED) == 2 and diff.stored == 0


def test_an_unchanged_page_is_reported_unchanged() -> None:
    assert diff_page(URL, _stored("a", "b"), _fresh("a", "b")).unchanged


def test_each_change_carries_the_uuid_of_its_position() -> None:
    diff = diff_page(URL, _stored("a"), _fresh("x"))

    assert diff.changes[0].uuid == text_chunk_id(URL, 0)


def test_screenshot_chunks_are_never_part_of_the_diff() -> None:
    objects = [
        SimpleNamespace(uuid=text_chunk_id(URL, 0), properties={"text": "page text"}),
        SimpleNamespace(uuid=uuid4(), properties={"text": "screenshot description"}),
        SimpleNamespace(uuid=uuid4(), properties={"text": "another screenshot"}),
    ]

    stored = stored_from_objects(URL, objects)

    assert list(stored) == [0]
    assert stored[0].text == "page text"
    diff = diff_page(URL, stored, _fresh("page text"))
    assert diff.stored == 1 and diff.unchanged


FOOTER = (
    " Did this answer your question?Thanks for the feedback YesNo"
    " Last updated on November 28, 2025 No results found"
)


def test_a_new_title_and_a_removed_footer_are_template_only() -> None:
    stored = _stored("first part", "last part" + FOOTER)
    fresh = _fresh("My Title\n\nfirst part", "last part")

    diff = diff_page(URL, stored, fresh, title="My Title")

    assert [c.status for c in diff.changes] == [TEMPLATE_ONLY, TEMPLATE_ONLY]
    assert not diff.content_changed


def test_a_real_edit_is_still_changed_next_to_template_differences() -> None:
    stored = _stored("first part", "last part" + FOOTER)
    fresh = _fresh("My Title\n\nfirst part, now edited", "last part")

    diff = diff_page(URL, stored, fresh, title="My Title")

    assert [c.status for c in diff.changes] == [CHANGED, TEMPLATE_ONLY]
    assert diff.content_changed


def test_added_and_removed_positions_count_as_content_changes() -> None:
    assert diff_page(URL, _stored("a"), _fresh("a", "b")).content_changed
    assert diff_page(URL, _stored("a", "b"), _fresh("a")).content_changed


def test_a_saved_run_loads_back_unchanged(tmp_path: Path) -> None:
    pages = [
        FetchedPage("https://hydrosym.paro.nl/article/1-a", "# A\n\ntext"),
        FetchedPage("https://hydrosym.paro.nl/article/2-b", None, "empty page"),
    ]
    save_run(tmp_path / "run", pages)

    assert load_run(tmp_path / "run") == pages
