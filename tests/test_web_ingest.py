"""Tests for web knowledge base ingestion."""

from unittest.mock import MagicMock, patch

from multimodal_rag.ingest.web import (
    crawl_knowledge_base,
    split_by_sections,
)


def _make_doc(url: str, title: str, markdown: str) -> MagicMock:
    """Create a mock Firecrawl Document."""
    doc = MagicMock()
    doc.markdown = markdown
    doc.metadata = MagicMock()
    doc.metadata.source_url = url
    doc.metadata.title = title
    return doc


def _make_crawl_result(pages: list[dict[str, str]]) -> MagicMock:
    """Create a mock CrawlJob result."""
    result = MagicMock()
    result.data = [
        _make_doc(p["url"], p.get("title", ""), p["content"]) for p in pages
    ]
    return result


class TestCrawlKnowledgeBase:
    @patch("multimodal_rag.ingest.web.ScrapeOptions")
    @patch("multimodal_rag.ingest.web.FirecrawlApp")
    def test_returns_pages_with_content(
        self, mock_fc_cls: MagicMock, _mock_opts: MagicMock
    ) -> None:
        mock_app = MagicMock()
        mock_fc_cls.return_value = mock_app
        mock_app.crawl.return_value = _make_crawl_result(
            [
                {"url": "https://example.com/a", "title": "A", "content": "# Hello"},
                {"url": "https://example.com/b", "title": "B", "content": "World"},
            ]
        )
        pages = crawl_knowledge_base("https://example.com", "fake-key", limit=10)
        assert len(pages) == 2
        assert pages[0]["url"] == "https://example.com/a"
        assert pages[1]["content"] == "World"

    @patch("multimodal_rag.ingest.web.ScrapeOptions")
    @patch("multimodal_rag.ingest.web.FirecrawlApp")
    def test_skips_empty_pages(
        self, mock_fc_cls: MagicMock, _mock_opts: MagicMock
    ) -> None:
        mock_app = MagicMock()
        mock_fc_cls.return_value = mock_app
        mock_app.crawl.return_value = _make_crawl_result(
            [
                {"url": "https://example.com/a", "content": "Real content"},
                {"url": "https://example.com/b", "content": "   "},
            ]
        )
        pages = crawl_knowledge_base("https://example.com", "fake-key")
        assert len(pages) == 1

    @patch("multimodal_rag.ingest.web.ScrapeOptions")
    @patch("multimodal_rag.ingest.web.FirecrawlApp")
    def test_handles_empty_data(
        self, mock_fc_cls: MagicMock, _mock_opts: MagicMock
    ) -> None:
        mock_app = MagicMock()
        mock_fc_cls.return_value = mock_app
        mock_app.crawl.return_value = _make_crawl_result([])
        pages = crawl_knowledge_base("https://example.com", "fake-key")
        assert pages == []

    @patch("multimodal_rag.ingest.web.ScrapeOptions")
    @patch("multimodal_rag.ingest.web.FirecrawlApp")
    def test_handles_no_metadata(
        self, mock_fc_cls: MagicMock, _mock_opts: MagicMock
    ) -> None:
        mock_app = MagicMock()
        mock_fc_cls.return_value = mock_app
        doc = MagicMock()
        doc.markdown = "Some content"
        doc.metadata = None
        result = MagicMock()
        result.data = [doc]
        mock_app.crawl.return_value = result
        pages = crawl_knowledge_base("https://example.com", "fake-key")
        assert len(pages) == 1
        assert pages[0]["url"] == "https://example.com"


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

    def test_chunk_index_is_sequential(self) -> None:
        content = "## A\nText A.\n## B\nText B.\n## C\nText C."
        chunks = split_by_sections(content, "https://ex.com", "Test")
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_chunk_index_sequential_across_sections(self) -> None:
        """Index must be global across sections, not reset per section."""
        long = "word " * 500
        content = f"## First\n{long}\n## Second\n{long}"
        chunks = split_by_sections(
            content, "https://ex.com", "Test", target_tokens=50
        )
        assert len(chunks) > 2
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_chunk_index_set_when_no_headers(self) -> None:
        content = "Just plain text without any headers."
        chunks = split_by_sections(content, "https://ex.com", "Test")
        assert all(c.chunk_index is not None for c in chunks)
        assert chunks[0].chunk_index == 0


