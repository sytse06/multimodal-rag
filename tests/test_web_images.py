"""Tests for web image extraction and description."""

from unittest.mock import MagicMock, patch

import pytest

from multimodal_rag.ingest.web_images import (
    describe_image,
    extract_image_urls,
    fetch_image_chunks,
)
from multimodal_rag.models.chunks import SupportChunk, WebChunk


class TestExtractImageUrls:
    def test_finds_markdown_images(self) -> None:
        markdown = "Some text\n![Alt text](https://example.com/img.png)\nMore text"
        result = extract_image_urls(markdown)
        assert result == ["https://example.com/img.png"]

    def test_finds_multiple_images(self) -> None:
        markdown = (
            "![A](https://example.com/a.jpg) "
            "![B](https://example.com/b.png)"
        )
        result = extract_image_urls(markdown)
        assert result == ["https://example.com/a.jpg", "https://example.com/b.png"]

    def test_ignores_relative_urls(self) -> None:
        markdown = "![Relative](/images/foo.png) ![Absolute](https://example.com/bar.png)"
        result = extract_image_urls(markdown)
        assert result == ["https://example.com/bar.png"]

    def test_deduplicates(self) -> None:
        url = "https://example.com/img.png"
        markdown = f"![A]({url}) ![B]({url})"
        result = extract_image_urls(markdown)
        assert result == [url]

    def test_empty_markdown_returns_empty(self) -> None:
        assert extract_image_urls("No images here.") == []

    def test_preserves_order(self) -> None:
        markdown = (
            "![C](https://c.com/img.png) "
            "![A](https://a.com/img.png) "
            "![B](https://b.com/img.png)"
        )
        result = extract_image_urls(markdown)
        assert result == [
            "https://c.com/img.png",
            "https://a.com/img.png",
            "https://b.com/img.png",
        ]


class TestDescribeImage:
    @patch("multimodal_rag.ingest.web_images.httpx.get")
    def test_downloads_and_calls_llm(self, mock_get: MagicMock) -> None:
        import base64

        image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        mock_response = MagicMock()
        mock_response.content = image_bytes
        mock_response.headers = {"content-type": "image/png"}
        mock_get.return_value = mock_response

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="A form with input fields")

        result = describe_image("https://example.com/screen.png", mock_llm)

        assert result == "A form with input fields"
        mock_get.assert_called_once_with(
            "https://example.com/screen.png", follow_redirects=True, timeout=30
        )

        message = mock_llm.invoke.call_args[0][0][0]
        image_block = next(
            b for b in message.content if b.get("type") == "image_url"
        )
        data_url = image_block["image_url"]["url"]
        assert data_url.startswith("data:image/png;base64,")
        b64_part = data_url.split(",", 1)[1]
        assert base64.b64decode(b64_part) == image_bytes

    @patch("multimodal_rag.ingest.web_images.httpx.get")
    def test_falls_back_to_jpeg_content_type(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.content = b"fakeimg"
        mock_response.headers = {}  # no content-type header
        mock_get.return_value = mock_response

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="A button")

        describe_image("https://example.com/img", mock_llm)

        message = mock_llm.invoke.call_args[0][0][0]
        image_block = next(
            b for b in message.content if b.get("type") == "image_url"
        )
        assert "image/jpeg" in image_block["image_url"]["url"]

    @patch("multimodal_rag.ingest.web_images.httpx.get")
    def test_raises_on_http_error(self, mock_get: MagicMock) -> None:
        import httpx

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock()
        )
        mock_get.return_value = mock_response

        mock_llm = MagicMock()
        with pytest.raises(httpx.HTTPStatusError):
            describe_image("https://example.com/missing.png", mock_llm)


class TestFetchImageChunks:
    @patch("multimodal_rag.ingest.web_images.describe_image")
    def test_returns_web_chunks_with_image_url(
        self, mock_describe: MagicMock
    ) -> None:
        markdown = (
            "![Screenshot](https://example.com/screen1.png)\n"
            "Some text\n"
            "![Another](https://example.com/screen2.png)"
        )
        mock_describe.side_effect = ["Screen one description", "Screen two description"]
        mock_llm = MagicMock()

        chunks = fetch_image_chunks(
            "https://docs.example.com/page",
            "Example Docs",
            markdown,
            mock_llm,
        )

        assert len(chunks) == 2
        assert all(isinstance(c, WebChunk) for c in chunks)
        assert chunks[0].image_url == "https://example.com/screen1.png"
        assert chunks[0].text == "Screen one description"
        assert chunks[0].source_url == "https://docs.example.com/page"
        assert chunks[0].source_name == "Example Docs"
        assert chunks[1].image_url == "https://example.com/screen2.png"

    @patch("multimodal_rag.ingest.web_images.describe_image")
    def test_skips_failed_images_and_continues(
        self, mock_describe: MagicMock
    ) -> None:
        markdown = (
            "![A](https://example.com/a.png) "
            "![B](https://example.com/b.png)"
        )
        mock_describe.side_effect = [
            RuntimeError("Download failed"),
            "Good description",
        ]
        mock_llm = MagicMock()

        chunks = fetch_image_chunks(
            "https://docs.example.com/page",
            "Docs",
            markdown,
            mock_llm,
        )

        assert len(chunks) == 1
        assert chunks[0].text == "Good description"
        assert chunks[0].image_url == "https://example.com/b.png"

    @patch("multimodal_rag.ingest.web_images.describe_image")
    def test_stable_chunk_id_for_same_image_url(
        self, mock_describe: MagicMock
    ) -> None:
        """chunk_id must be stable regardless of LLM description text."""
        markdown = "![A](https://example.com/screen.png)"
        mock_llm = MagicMock()

        mock_describe.return_value = "Description version A"
        chunks_a = fetch_image_chunks("https://p.com/page", "P", markdown, mock_llm)

        mock_describe.return_value = "Description version B"
        chunks_b = fetch_image_chunks("https://p.com/page", "P", markdown, mock_llm)

        id_a = SupportChunk.from_screenshot_chunk(chunks_a[0]).chunk_id
        id_b = SupportChunk.from_screenshot_chunk(chunks_b[0]).chunk_id

        assert id_a == id_b

    def test_empty_markdown_returns_empty(self) -> None:
        mock_llm = MagicMock()
        chunks = fetch_image_chunks(
            "https://p.com/page", "P", "No images here.", mock_llm
        )
        assert chunks == []
