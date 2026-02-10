"""Tests for the retrieval module."""

from unittest.mock import MagicMock

from multimodal_rag.models.chunks import SourceType
from multimodal_rag.query.retriever import (
    _distance_to_score,
    format_context,
    retrieve,
)


class TestDistanceToScore:
    def test_zero_distance(self) -> None:
        assert _distance_to_score(0.0) == 1.0

    def test_max_distance(self) -> None:
        assert _distance_to_score(2.0) == 0.0

    def test_mid_distance(self) -> None:
        assert _distance_to_score(0.5) == 0.5

    def test_none_returns_zero(self) -> None:
        assert _distance_to_score(None) == 0.0

    def test_over_two_clamps_to_zero(self) -> None:
        assert _distance_to_score(2.5) == 0.0


class TestRetrieve:
    def _mock_store(self, hits: list[dict]) -> MagicMock:
        store = MagicMock()
        store.search.return_value = hits
        return store

    def test_returns_search_results(self) -> None:
        store = self._mock_store([
            {
                "text": "How to configure",
                "source_type": "video",
                "source_url": "https://yt.com/watch?v=abc",
                "source_name": "Tutorial",
                "timestamp_seconds": 42,
                "section_heading": "",
                "_distance": 0.3,
                "_uuid": "some-uuid",
            }
        ])
        results = retrieve("how to configure", store, top_k=5)
        assert len(results) == 1
        assert results[0].source_type == SourceType.VIDEO
        assert results[0].timestamp_seconds == 42
        assert results[0].relevance_score == 0.7
        store.search.assert_called_once_with("how to configure", top_k=5)

    def test_web_result(self) -> None:
        store = self._mock_store([
            {
                "text": "Installation guide",
                "source_type": "web",
                "source_url": "https://docs.example.com/install",
                "source_name": "Docs",
                "timestamp_seconds": None,
                "section_heading": "Setup",
                "_distance": 0.1,
                "_uuid": "uuid-2",
            }
        ])
        results = retrieve("install", store)
        assert results[0].source_type == SourceType.WEB
        assert results[0].section_heading == "Setup"
        assert results[0].timestamp_seconds is None

    def test_empty_results(self) -> None:
        store = self._mock_store([])
        results = retrieve("anything", store)
        assert results == []

    def test_respects_top_k(self) -> None:
        store = self._mock_store([])
        retrieve("q", store, top_k=3)
        store.search.assert_called_once_with("q", top_k=3)


class TestFormatContext:
    def test_formats_numbered_context(self) -> None:
        from multimodal_rag.models.query import SearchResult

        results = [
            SearchResult(
                text="First chunk",
                source_type=SourceType.VIDEO,
                source_url="https://yt.com/watch?v=abc",
                source_name="Video A",
                timestamp_seconds=30,
                relevance_score=0.9,
            ),
            SearchResult(
                text="Second chunk",
                source_type=SourceType.WEB,
                source_url="https://docs.example.com",
                source_name="Docs",
                section_heading="Intro",
                relevance_score=0.8,
            ),
        ]
        ctx = format_context(results)
        assert "[1] Video A @ 00:30" in ctx
        assert "First chunk" in ctx
        assert "[2] Docs" in ctx
        assert "Second chunk" in ctx

    def test_empty_results(self) -> None:
        assert format_context([]) == ""
