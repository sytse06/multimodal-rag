"""Tests for web knowledge base ingestion."""

from unittest.mock import MagicMock, patch

from multimodal_rag.ingest.web import (
    crawl_knowledge_base,
    fetch_web_chunks,
    split_by_sections,
)


class TestCrawlKnowledgeBase:
    def _mock_crawl_result(
        self, pages: list[dict[str, str]]
    ) -> dict[str, list[dict]]:
        return {
            "data": [
                {
                    "markdown": p["content"],
                    "metadata": {
                        "sourceURL": p["url"],
                        "title": p.get("title", ""),
                    },
                }
                for p in pages
            ]
        }

    @patch("multimodal_rag.ingest.web.FirecrawlApp")
    def test_returns_pages_with_content(self, mock_fc_cls: MagicMock) -> None:
        mock_app = MagicMock()
        mock_fc_cls.return_value = mock_app
        mock_app.crawl_url.return_value = self._mock_crawl_result(
            [
                {"url": "https://example.com/a", "title": "A", "content": "# Hello"},
                {"url": "https://example.com/b", "title": "B", "content": "World"},
            ]
        )
        pages = crawl_knowledge_base("https://example.com", "fake-key", limit=10)
        assert len(pages) == 2
        assert pages[0]["url"] == "https://example.com/a"
        assert pages[1]["content"] == "World"

    @patch("multimodal_rag.ingest.web.FirecrawlApp")
    def test_skips_empty_pages(self, mock_fc_cls: MagicMock) -> None:
        mock_app = MagicMock()
        mock_fc_cls.return_value = mock_app
        mock_app.crawl_url.return_value = self._mock_crawl_result(
            [
                {"url": "https://example.com/a", "content": "Real content"},
                {"url": "https://example.com/b", "content": "   "},
            ]
        )
        pages = crawl_knowledge_base("https://example.com", "fake-key")
        assert len(pages) == 1

    @patch("multimodal_rag.ingest.web.FirecrawlApp")
    def test_handles_empty_data(self, mock_fc_cls: MagicMock) -> None:
        mock_app = MagicMock()
        mock_fc_cls.return_value = mock_app
        mock_app.crawl_url.return_value = {"data": []}
        pages = crawl_knowledge_base("https://example.com", "fake-key")
        assert pages == []

    @patch("multimodal_rag.ingest.web.FirecrawlApp")
    def test_handles_non_dict_result(self, mock_fc_cls: MagicMock) -> None:
        mock_app = MagicMock()
        mock_fc_cls.return_value = mock_app
        mock_app.crawl_url.return_value = "unexpected"
        pages = crawl_knowledge_base("https://example.com", "fake-key")
        assert pages == []


class TestSplitBySections:
    def test_splits_on_headers(self) -> None:
        content = "## Intro\nSome intro text.\n## Details\nMore details here."
        chunks = split_by_sections(content, "https://ex.com", "Test")
        assert len(chunks) == 2
        assert chunks[0].section_heading == "Intro"
        assert chunks[1].section_heading == "Details"

    def test_no_headers_falls_back_to_tokens(self) -> None:
        content = "Just plain text without any markdown headers at all."
        chunks = split_by_sections(content, "https://ex.com", "Test")
        assert len(chunks) >= 1
        assert chunks[0].section_heading is None

    def test_preserves_pre_header_content(self) -> None:
        content = "Preamble text.\n\n## Section One\nSection content."
        chunks = split_by_sections(content, "https://ex.com", "Test")
        assert any("Preamble" in c.text for c in chunks)

    def test_respects_target_tokens(self) -> None:
        long_section = "word " * 500
        content = f"## Big Section\n{long_section}"
        chunks = split_by_sections(
            content, "https://ex.com", "Test", target_tokens=50
        )
        assert len(chunks) > 1

    def test_empty_section_skipped(self) -> None:
        content = "## Empty\n\n## Has Content\nSome text."
        chunks = split_by_sections(content, "https://ex.com", "Test")
        assert len(chunks) == 1
        assert chunks[0].section_heading == "Has Content"

    def test_chunk_metadata(self) -> None:
        content = "## Topic\nSome text about the topic."
        chunks = split_by_sections(content, "https://ex.com/page", "My KB")
        assert chunks[0].source_url == "https://ex.com/page"
        assert chunks[0].source_name == "My KB"

    def test_h1_h2_h3_all_split(self) -> None:
        content = "# H1\nText1.\n## H2\nText2.\n### H3\nText3."
        chunks = split_by_sections(content, "https://ex.com", "Test")
        assert len(chunks) == 3


class TestFetchWebChunks:
    @patch("multimodal_rag.ingest.web.crawl_knowledge_base")
    def test_returns_chunks(self, mock_crawl: MagicMock) -> None:
        mock_crawl.return_value = [
            {
                "url": "https://ex.com/page",
                "title": "Page",
                "content": "## Section\nSome content here.",
            }
        ]
        chunks = fetch_web_chunks(
            "https://ex.com", "Test KB", "fake-key", target_tokens=400
        )
        assert len(chunks) >= 1
        assert chunks[0].source_name == "Test KB"

    @patch("multimodal_rag.ingest.web.crawl_knowledge_base")
    def test_crawl_failure_returns_empty(self, mock_crawl: MagicMock) -> None:
        mock_crawl.side_effect = RuntimeError("Network error")
        chunks = fetch_web_chunks("https://ex.com", "Test KB", "fake-key")
        assert chunks == []

    @patch("multimodal_rag.ingest.web.crawl_knowledge_base")
    def test_no_pages_returns_empty(self, mock_crawl: MagicMock) -> None:
        mock_crawl.return_value = []
        chunks = fetch_web_chunks("https://ex.com", "Test KB", "fake-key")
        assert chunks == []

    @patch("multimodal_rag.ingest.web.crawl_knowledge_base")
    def test_multiple_pages_aggregated(self, mock_crawl: MagicMock) -> None:
        mock_crawl.return_value = [
            {"url": "https://ex.com/a", "title": "A", "content": "## S1\nText A."},
            {"url": "https://ex.com/b", "title": "B", "content": "## S2\nText B."},
        ]
        chunks = fetch_web_chunks("https://ex.com", "Test KB", "fake-key")
        urls = {c.source_url for c in chunks}
        assert "https://ex.com/a" in urls
        assert "https://ex.com/b" in urls
