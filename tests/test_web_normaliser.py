"""Tests for the page normaliser: byline and footer removal, title, fingerprint."""

import pytest

from multimodal_rag.ingest.web import (
    chunk_page,
    normalise_page,
    page_fingerprint,
)

FOOTER = (
    "Did this answer your question?Thanks for the feedbackThere was a problem "
    "submitting your feedback. Please try again later.\n\nYesNo\n\n"
    "Last updated on November 28, 2025\n\nNo results found\n\nStill stuck\n\n"
    "## Still have questions? We offer **unlimited support**.\n\n"
    "Contact us anytime at support@paro.nl.\n\n"
    "[Get in touch](mailto:support@paro.nl) "
    "[Watch tutorials now](https://paro.nl/tutorials)"
)


def _page(
    title: str = "Tutorial: How to create cut outs in Hydroman",
    byline: str = "HydroManUpdated 27 November 20252 min readPrint",
    body: str = "## Goal:\n\nCreating cut outs in your manifold.\n\n## Steps:\n\n"
    "Open the Modify menu and choose the cut outs tool.",
    footer: str = FOOTER,
) -> str:
    return f"# {title}\n\n{byline}\n\n{body}\n\n{footer}"


@pytest.mark.parametrize(
    "byline",
    [
        "HydroManUpdated 27 November 20252 min readPrint",
        "HydroSymUpdated 10 November 20254 min readPrint",
        "HydroCamUpdated 3 March 202610 min readPrint",
        "HydroSymUpdated 10 November 20254 min read Print",
    ],
)
def test_the_byline_is_removed_for_every_product(byline: str) -> None:
    page = normalise_page(_page(byline=byline))

    assert page.byline_removed
    assert "Updated" not in page.markdown
    assert "min read" not in page.markdown


def test_everything_from_the_feedback_widget_down_is_removed() -> None:
    page = normalise_page(_page())

    assert page.footer_removed
    for leaked in (
        "Did this answer",
        "Last updated on",
        "No results found",
        "Still stuck",
        "support@paro.nl",
        "unlimited support",
    ):
        assert leaked not in page.markdown
    assert page.markdown.endswith("choose the cut outs tool.")


def test_a_page_without_the_footer_marker_is_left_whole() -> None:
    page = normalise_page(_page(footer="Some closing remark."))

    assert not page.footer_removed
    assert page.markdown.endswith("Some closing remark.")


def test_the_title_is_returned_and_prefixes_only_the_first_chunk() -> None:
    chunks = chunk_page(_page(), "https://kb/a", "KB", target_tokens=10)

    assert len(chunks) > 1
    assert chunks[0].text.startswith("Tutorial: How to create cut outs in Hydroman\n\n")
    assert all("Tutorial: How to" not in c.text for c in chunks[1:])


def test_chunk_indexes_are_contiguous_from_zero() -> None:
    chunks = chunk_page(_page(), "https://kb/a", "KB", target_tokens=10)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_a_title_with_an_empty_byline_section_makes_no_empty_chunk() -> None:
    chunks = chunk_page(_page(), "https://kb/a", "KB")

    assert all(c.text.strip() for c in chunks)
    assert "Updated" not in " ".join(c.text for c in chunks)


def test_a_page_without_a_title_gets_no_prefix() -> None:
    markdown = "## Goal:\n\nCreating cut outs.\n\n" + FOOTER

    chunks = chunk_page(markdown, "https://kb/a", "KB")

    assert chunks[0].text == "Creating cut outs."


def test_images_stay_by_default_and_go_when_asked() -> None:
    body = "## Steps:\n\nOpen it.\n\n![Alternatieve tekst](https://cdn.example/a.png)\n\nDone."
    markdown = _page(body=body)

    assert "![Alternatieve tekst]" in normalise_page(markdown).markdown
    stripped = normalise_page(markdown, strip_images=True).markdown
    assert "cdn.example" not in stripped
    assert "Open it." in stripped and "Done." in stripped


def test_the_fingerprint_ignores_the_byline_and_footer() -> None:
    a = _page(byline="HydroManUpdated 27 November 20252 min readPrint")
    b = _page(
        byline="HydroManUpdated 3 January 20263 min readPrint",
        footer=FOOTER.replace("November 28, 2025", "January 3, 2026"),
    )

    assert page_fingerprint(a) == page_fingerprint(b)


def test_the_fingerprint_changes_when_the_content_changes() -> None:
    changed = _page(body="## Goal:\n\nCreating cut outs in your manifold, edited.")

    assert page_fingerprint(_page()) != page_fingerprint(changed)


def test_the_fingerprint_is_stable() -> None:
    assert page_fingerprint(_page()) == page_fingerprint(_page())
    assert len(page_fingerprint(_page())) == 64


def test_a_short_article_still_loses_its_much_longer_footer() -> None:
    page = normalise_page(_page(body="Short answer."))

    assert page.footer_removed and not page.footer_suspicious
    assert page.markdown.endswith("Short answer.")


def test_a_marker_far_from_the_end_is_not_trusted_and_nothing_is_cut() -> None:
    long_tail = "Real article text that follows the phrase. " * 60
    markdown = _page(footer=f"Did this answer your question?\n\n{long_tail}")

    page = normalise_page(markdown)

    assert page.footer_suspicious and not page.footer_removed
    assert "Real article text that follows" in page.markdown


def test_a_byline_lookalike_outside_the_page_head_is_kept() -> None:
    body = "x " * 400 + "\n\nHydroManUpdated 27 November 20252 min readPrint\n\nmore"

    page = normalise_page(_page(body=body))

    assert "min readPrint" in page.markdown


def test_ingest_and_sync_use_the_same_chunker() -> None:
    from multimodal_rag.ingest import __main__ as ingest_main
    from multimodal_rag.ingest import web

    assert ingest_main.chunk_page is web.chunk_page
