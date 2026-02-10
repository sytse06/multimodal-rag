"""Tests for YouTube transcript ingestion."""

import pytest

from multimodal_rag.ingest.youtube import (
    chunk_segments,
    extract_video_id,
    fetch_transcript_chunks,
)


class TestExtractVideoId:
    def test_standard_url(self) -> None:
        url = "https://www.youtube.com/watch?v=KHo5xEaPyAI"
        assert extract_video_id(url) == "KHo5xEaPyAI"

    def test_url_with_timestamp(self) -> None:
        url = "https://www.youtube.com/watch?v=KHo5xEaPyAI&t=18s"
        assert extract_video_id(url) == "KHo5xEaPyAI"

    def test_short_url(self) -> None:
        url = "https://youtu.be/KHo5xEaPyAI"
        assert extract_video_id(url) == "KHo5xEaPyAI"

    def test_embed_url(self) -> None:
        url = "https://www.youtube.com/embed/KHo5xEaPyAI"
        assert extract_video_id(url) is not None

    def test_invalid_url(self) -> None:
        assert extract_video_id("https://example.com") is None

    def test_empty_string(self) -> None:
        assert extract_video_id("") is None


class TestChunkSegments:
    def _make_segments(
        self, texts: list[str], start: float = 0.0, gap: float = 3.0
    ) -> list[dict[str, float | str]]:
        segments = []
        t = start
        for text in texts:
            segments.append({"text": text, "start": t, "duration": gap})
            t += gap
        return segments

    def test_single_chunk_small_input(self) -> None:
        segments = self._make_segments(["Hello world", "This is a test"])
        chunks = chunk_segments(segments, "https://yt.com/watch?v=abc", "Test", 400)
        assert len(chunks) == 1
        assert "Hello world" in chunks[0].text
        assert "This is a test" in chunks[0].text

    def test_multiple_chunks(self) -> None:
        # ~50 words each, target 20 tokens → should split
        long_texts = ["word " * 20 for _ in range(10)]
        segments = self._make_segments(long_texts)
        chunks = chunk_segments(segments, "https://yt.com/watch?v=abc", "Test", 20)
        assert len(chunks) > 1

    def test_preserves_start_timestamp(self) -> None:
        segments = self._make_segments(["First segment", "Second segment"], start=90.0)
        chunks = chunk_segments(segments, "https://yt.com/watch?v=abc", "Test", 400)
        assert chunks[0].start_seconds == 90

    def test_empty_segments(self) -> None:
        chunks = chunk_segments([], "https://yt.com/watch?v=abc", "Test", 400)
        assert chunks == []

    def test_skips_empty_text(self) -> None:
        segments = [
            {"text": "", "start": 0.0, "duration": 3.0},
            {"text": "Actual content", "start": 3.0, "duration": 3.0},
        ]
        chunks = chunk_segments(segments, "https://yt.com/watch?v=abc", "Test", 400)
        assert len(chunks) == 1
        assert chunks[0].text == "Actual content"

    def test_chunk_has_correct_source_metadata(self) -> None:
        segments = self._make_segments(["Hello"])
        chunks = chunk_segments(
            segments, "https://yt.com/watch?v=abc", "My Video", 400
        )
        assert chunks[0].source_url == "https://yt.com/watch?v=abc"
        assert chunks[0].source_name == "My Video"


class TestFetchTranscriptChunksIntegration:
    """Integration tests against real YouTube API. Requires network."""

    @pytest.mark.integration
    def test_hydrosym_quickstart(self) -> None:
        chunks = fetch_transcript_chunks(
            video_url="https://www.youtube.com/watch?v=KHo5xEaPyAI",
            source_name="HydroSym Quickstart",
            target_tokens=400,
        )
        assert len(chunks) > 0
        assert all(c.source_name == "HydroSym Quickstart" for c in chunks)
        assert all(c.start_seconds >= 0 for c in chunks)
        assert "HydroSym" in chunks[0].text or "hydrosym" in chunks[0].text.lower()

    @pytest.mark.integration
    def test_invalid_video_returns_empty(self) -> None:
        chunks = fetch_transcript_chunks(
            video_url="https://www.youtube.com/watch?v=INVALIDVIDEO",
            source_name="Bad Video",
        )
        assert chunks == []

    def test_bad_url_returns_empty(self) -> None:
        chunks = fetch_transcript_chunks(
            video_url="not-a-url",
            source_name="Bad URL",
        )
        assert chunks == []
