"""Tests for template-change detection across a set of fetched pages."""

from multimodal_rag.ingest.template import check_template

FOOTER = (
    "Did this answer your question?Thanks for the feedback\n\nYesNo\n\n"
    "Last updated on November 28, 2025\n\nNo results found"
)


def _pages(
    count: int = 8,
    byline: str = "HydroManUpdated 27 November 20252 min readPrint",
    footer: str = FOOTER,
    extra: str = "",
) -> dict[str, str]:
    def body(i: int) -> str:
        return f"Unique content about topic number {i}. " * 12

    return {
        f"https://kb/article/{i}": (
            f"# Page {i}\n\n{byline}\n\n## Goal:\n\n{body(i)}\n\n{extra}\n\n{footer}"
        )
        for i in range(count)
    }


def test_an_unchanged_template_needs_no_review() -> None:
    report = check_template(_pages())

    assert not report.needs_review
    assert report.byline_rate == 1.0 and report.footer_rate == 1.0
    assert report.repeated == []


def test_a_changed_byline_format_is_noticed() -> None:
    report = check_template(_pages(byline="By HydroMan · 27 Nov 2025 · 2 min read"))

    assert report.needs_review
    assert report.byline_rate == 0.0
    assert any("byline rule" in line for line in report.lines())


def test_a_changed_footer_phrase_is_noticed() -> None:
    report = check_template(_pages(footer="Was this article helpful?\n\nYes No"))

    assert report.needs_review
    assert report.footer_rate == 0.0
    assert any("footer rule" in line for line in report.lines())


def test_a_new_banner_on_most_pages_is_found_even_when_the_rules_match() -> None:
    banner = "New: try our AI assistant for instant answers!"
    report = check_template(_pages(extra=banner))

    assert report.needs_review
    assert report.repeated[0][0].startswith("New: try our AI assistant")
    assert report.repeated[0][1] == 8


def test_real_structural_headings_repeating_are_not_flagged() -> None:
    report = check_template(_pages())

    assert not any(block.startswith("#") for block, _ in report.repeated)


def test_a_marker_in_the_middle_of_an_article_is_flagged_and_left_alone() -> None:
    pages = _pages()
    pages["https://kb/article/0"] = (
        "# Page 0\n\nHydroManUpdated 27 November 20252 min readPrint\n\n"
        "Did this answer your question?\n\n" + "Article text continues here. " * 80
    )

    report = check_template(pages)

    assert report.suspicious == ["https://kb/article/0"]
    assert report.needs_review


def test_a_page_that_normalises_to_almost_nothing_is_flagged() -> None:
    pages = _pages()
    pages["https://kb/article/0"] = (
        "# Empty\n\nHydroManUpdated 1 May 20251 min readPrint\n\n" + FOOTER
    )

    assert check_template(pages).near_empty == ["https://kb/article/0"]


def test_no_pages_means_nothing_to_review() -> None:
    assert not check_template({}).needs_review


def test_a_short_page_alone_is_information_not_a_template_problem() -> None:
    pages = _pages()
    pages["https://kb/article/0"] = (
        "# Stub\n\nHydroManUpdated 1 May 20251 min readPrint\n\nOne sentence.\n\n"
        + FOOTER
    )

    report = check_template(pages)

    assert report.near_empty == ["https://kb/article/0"]
    assert not report.needs_review
    assert any("embedded video" in line for line in report.lines())
