"""Tests for the retrieval gate."""

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from multimodal_rag import retrieval_gate
from multimodal_rag.checks import Status
from multimodal_rag.models.chunks import SourceType
from multimodal_rag.models.query import SearchResult
from multimodal_rag.retrieval_gate import (
    Canary,
    best_rank,
    evaluate,
    from_changed_pages,
    load_canaries,
    run_gate,
)

GOOD = "https://kb.example.com/article/1-good"
OTHER = "https://kb.example.com/article/2-other"


def _results(*urls: str) -> list[SearchResult]:
    return [
        SearchResult(
            text="t",
            source_type=SourceType.WEB,
            source_url=url,
            source_name="kb",
            relevance_score=0.5,
        )
        for url in urls
    ]


def _canary() -> Canary:
    return Canary(question="How do I do the thing?", expect=[GOOD])


def test_best_rank_is_the_first_expected_position() -> None:
    assert best_rank(_results(OTHER, GOOD, OTHER), [GOOD]) == 2
    assert best_rank(_results(OTHER), [GOOD]) is None


def test_any_expected_source_counts() -> None:
    assert best_rank(_results(OTHER, GOOD), [OTHER, GOOD]) == 1


@pytest.mark.parametrize(
    ("before", "after", "status"),
    [
        (2, 2, Status.OK),
        (2, 4, Status.OK),
        (2, 5, Status.WARN),
        (2, None, Status.FAIL),
        (None, None, Status.FAIL),
        (None, 3, Status.OK),
        (5, 1, Status.OK),
    ],
)
def test_evaluate_status_by_rank_movement(
    before: int | None, after: int | None, status: Status
) -> None:
    assert evaluate(_canary(), before, after, top_k=10).status is status


def test_a_failure_says_where_the_source_used_to_rank() -> None:
    check = evaluate(_canary(), 2, None, top_k=10)

    assert "rank 2 before" in check.detail
    assert "top 10" in check.detail


def _gate(before: list[str], after: list[str]) -> Status:
    def fake_retrieve(question: str, store: str, top_k: int) -> list[SearchResult]:
        return _results(*(before if store == "before" else after))

    with patch.object(retrieval_gate, "retrieve", fake_retrieve):
        return run_gate([_canary()], "before", "after", top_k=10)[0].status


def test_the_pilot_crowd_out_failure_is_caught() -> None:
    crowd = [f"https://kb.example.com/screenshot/{i}" for i in range(10)]

    assert _gate([OTHER, GOOD], crowd) is Status.FAIL


def test_an_unchanged_ranking_passes() -> None:
    assert _gate([OTHER, GOOD], [OTHER, GOOD]) is Status.OK


def test_new_content_that_nudges_the_rank_does_not_fail() -> None:
    assert _gate([GOOD], ["https://kb.example.com/new", GOOD]) is Status.OK


def test_changed_pages_become_title_canaries() -> None:
    canaries = from_changed_pages([("Navigating in HydroSym", GOOD)])

    assert canaries == [Canary(question="Navigating in HydroSym", expect=[GOOD])]


def test_a_canary_needs_an_expected_source() -> None:
    with pytest.raises(ValidationError):
        Canary(question="q", expect=[])


def test_the_tracked_canary_file_loads() -> None:
    canaries = load_canaries(Path("config/canaries.yaml"))

    assert len(canaries) == 3
    assert all(c.expect for c in canaries)
