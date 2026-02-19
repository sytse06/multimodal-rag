"""Tests for video keyframe extraction and description."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from multimodal_rag.ingest.video_frames import (
    describe_frame,
    download_video,
    extract_keyframes,
    fetch_frame_chunks,
)
from multimodal_rag.models.chunks import TranscriptChunk


class TestDownloadVideo:
    @patch("multimodal_rag.ingest.video_frames.yt_dlp.YoutubeDL")
    def test_calls_yt_dlp_with_correct_format(
        self, mock_ydl_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {"id": "abc123", "ext": "mp4"}
        video_file = tmp_path / "abc123.mp4"
        video_file.write_bytes(b"fake video")
        mock_ydl.prepare_filename.return_value = str(video_file)

        result = download_video("https://youtu.be/abc123", tmp_path)

        call_kwargs = mock_ydl_cls.call_args[0][0]
        assert call_kwargs["format"] == "bestvideo[ext=mp4]/bestvideo"
        assert result == video_file

    @patch("multimodal_rag.ingest.video_frames.yt_dlp.YoutubeDL")
    def test_falls_back_to_directory_scan_when_filename_wrong(
        self, mock_ydl_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {"id": "xyz", "ext": "webm"}
        mock_ydl.prepare_filename.return_value = str(tmp_path / "xyz.webm")

        actual_file = tmp_path / "xyz.mp4"
        actual_file.write_bytes(b"video data")

        result = download_video("https://youtu.be/xyz", tmp_path)

        assert result.exists()


class TestExtractKeyframes:
    @patch("multimodal_rag.ingest.video_frames.subprocess.run")
    def test_calls_ffmpeg_with_fps_filter(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake")
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()

        mock_run.return_value = MagicMock(returncode=0)

        extract_keyframes(video_path, frame_dir, interval_seconds=30)

        args = mock_run.call_args[0][0]
        assert "ffmpeg" in args
        assert "fps=1/30" in " ".join(args)

    @patch("multimodal_rag.ingest.video_frames.subprocess.run")
    def test_calls_ffmpeg_with_output_template(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake")
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()

        mock_run.return_value = MagicMock(returncode=0)

        extract_keyframes(video_path, frame_dir, interval_seconds=30)

        args = mock_run.call_args[0][0]
        joined = " ".join(args)
        assert "frame_" in joined
        assert ".jpg" in joined

    @patch("multimodal_rag.ingest.video_frames.subprocess.run")
    def test_returns_sorted_timestamps(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake")
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()

        mock_run.return_value = MagicMock(returncode=0)

        # Create fake frame files
        for i in range(1, 4):
            (frame_dir / f"frame_{i:04d}.jpg").write_bytes(b"jpeg")

        result = extract_keyframes(video_path, frame_dir, interval_seconds=30)

        assert len(result) == 3
        paths, timestamps = zip(*result)
        assert list(timestamps) == [0, 30, 60]
        assert all(p.suffix == ".jpg" for p in paths)

    @patch("multimodal_rag.ingest.video_frames.subprocess.run")
    def test_raises_on_ffmpeg_failure(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake")
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()

        with pytest.raises(subprocess.CalledProcessError):
            extract_keyframes(video_path, frame_dir)


class TestDescribeFrame:
    def test_sends_base64_image_to_llm(self, tmp_path: Path) -> None:
        import base64

        frame = tmp_path / "frame_0001.jpg"
        frame.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)  # minimal JPEG header

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="A screenshot showing a menu")

        result = describe_frame(frame, mock_llm)

        assert result == "A screenshot showing a menu"
        assert mock_llm.invoke.call_count == 1

        message = mock_llm.invoke.call_args[0][0][0]
        content = message.content
        # Find the image_url block
        image_block = next(b for b in content if b.get("type") == "image_url")
        data_url = image_block["image_url"]["url"]
        assert data_url.startswith("data:image/jpeg;base64,")

        # Verify the base64 is correct
        b64_part = data_url.split(",", 1)[1]
        assert base64.b64decode(b64_part) == frame.read_bytes()


class TestFetchFrameChunks:
    @patch("multimodal_rag.ingest.video_frames.describe_frame")
    @patch("multimodal_rag.ingest.video_frames.extract_keyframes")
    @patch("multimodal_rag.ingest.video_frames.download_video")
    def test_returns_transcript_chunks(
        self,
        mock_download: MagicMock,
        mock_extract: MagicMock,
        mock_describe: MagicMock,
        tmp_path: Path,
    ) -> None:
        video_file = tmp_path / "video.mp4"
        video_file.write_bytes(b"fake")
        mock_download.return_value = video_file

        frame1 = tmp_path / "frame_0001.jpg"
        frame2 = tmp_path / "frame_0002.jpg"
        frame1.write_bytes(b"jpeg1")
        frame2.write_bytes(b"jpeg2")
        mock_extract.return_value = [(frame1, 0), (frame2, 30)]
        mock_describe.side_effect = [
            "First frame description",
            "Second frame description",
        ]

        mock_llm = MagicMock()

        chunks = fetch_frame_chunks(
            "https://youtube.com/watch?v=abc",
            "Test Video",
            mock_llm,
            interval_seconds=30,
        )

        assert len(chunks) == 2
        assert all(isinstance(c, TranscriptChunk) for c in chunks)
        assert chunks[0].start_seconds == 0
        assert chunks[0].text == "First frame description"
        assert chunks[1].start_seconds == 30
        assert chunks[1].text == "Second frame description"
        assert chunks[0].source_url == "https://youtube.com/watch?v=abc"
        assert chunks[0].source_name == "Test Video"

    @patch("multimodal_rag.ingest.video_frames.describe_frame")
    @patch("multimodal_rag.ingest.video_frames.extract_keyframes")
    @patch("multimodal_rag.ingest.video_frames.download_video")
    def test_skips_failed_frames_and_continues(
        self,
        mock_download: MagicMock,
        mock_extract: MagicMock,
        mock_describe: MagicMock,
        tmp_path: Path,
    ) -> None:
        video_file = tmp_path / "video.mp4"
        video_file.write_bytes(b"fake")
        mock_download.return_value = video_file

        frame1 = tmp_path / "frame_0001.jpg"
        frame2 = tmp_path / "frame_0002.jpg"
        frame1.write_bytes(b"jpeg1")
        frame2.write_bytes(b"jpeg2")
        mock_extract.return_value = [(frame1, 0), (frame2, 30)]
        mock_describe.side_effect = [RuntimeError("LLM error"), "Good description"]

        mock_llm = MagicMock()

        chunks = fetch_frame_chunks(
            "https://youtube.com/watch?v=abc",
            "Test Video",
            mock_llm,
        )

        assert len(chunks) == 1
        assert chunks[0].text == "Good description"
        assert chunks[0].start_seconds == 30

    @patch("multimodal_rag.ingest.video_frames.describe_frame")
    @patch("multimodal_rag.ingest.video_frames.extract_keyframes")
    @patch("multimodal_rag.ingest.video_frames.download_video")
    def test_stable_chunk_id_for_same_url_and_timestamp(
        self,
        mock_download: MagicMock,
        mock_extract: MagicMock,
        mock_describe: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Chunk ID must be stable regardless of LLM description text."""
        from multimodal_rag.models.chunks import SupportChunk

        video_file = tmp_path / "video.mp4"
        video_file.write_bytes(b"fake")
        mock_download.return_value = video_file

        frame = tmp_path / "frame_0001.jpg"
        frame.write_bytes(b"jpeg")
        mock_extract.return_value = [(frame, 60)]
        mock_llm = MagicMock()

        url = "https://youtube.com/watch?v=abc"

        mock_describe.return_value = "Description version A"
        chunks_a = fetch_frame_chunks(url, "Vid", mock_llm)

        mock_describe.return_value = "Description version B"
        chunks_b = fetch_frame_chunks(url, "Vid", mock_llm)

        id_a = SupportChunk.from_frame_chunk(chunks_a[0]).chunk_id
        id_b = SupportChunk.from_frame_chunk(chunks_b[0]).chunk_id

        assert id_a == id_b
