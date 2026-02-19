"""Tests for Voxtral audio transcription fallback."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from youtube_transcript_api import (
    IpBlocked,
    NoTranscriptFound,
    TranscriptsDisabled,
)

from multimodal_rag.ingest.voxtral import (
    download_audio,
    fetch_voxtral_transcript,
    transcribe_with_voxtral,
)
from multimodal_rag.ingest.youtube import fetch_transcript_chunks

# ---------------------------------------------------------------------------
# voxtral.download_audio
# ---------------------------------------------------------------------------


class TestDownloadAudio:
    @patch("multimodal_rag.ingest.voxtral.yt_dlp.YoutubeDL")
    def test_calls_yt_dlp_with_correct_format(
        self, mock_ydl_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {"id": "abc123", "ext": "m4a"}
        mock_ydl.prepare_filename.return_value = str(tmp_path / "abc123.m4a")

        # Create a dummy audio file so the existence check passes
        (tmp_path / "abc123.m4a").write_bytes(b"fake audio")

        result = download_audio("https://youtu.be/abc123", tmp_path)

        # Verify YoutubeDL was constructed with the right format
        call_kwargs = mock_ydl_cls.call_args[0][0]
        assert call_kwargs["format"] == "bestaudio[ext=m4a]/bestaudio"
        assert result == tmp_path / "abc123.m4a"

    @patch("multimodal_rag.ingest.voxtral.yt_dlp.YoutubeDL")
    def test_falls_back_to_directory_scan_when_filename_wrong(
        self, mock_ydl_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {"id": "xyz", "ext": "webm"}
        # prepare_filename returns a path that doesn't exist
        mock_ydl.prepare_filename.return_value = str(tmp_path / "xyz.webm")

        # yt-dlp actually saved with a different name
        actual_file = tmp_path / "xyz.opus"
        actual_file.write_bytes(b"audio data")

        result = download_audio("https://youtu.be/xyz", tmp_path)

        assert result.exists()


# ---------------------------------------------------------------------------
# voxtral.transcribe_with_voxtral
# ---------------------------------------------------------------------------


class TestTranscribeWithVoxtral:
    def _make_segment(
        self, text: str, start: float, end: float
    ) -> MagicMock:
        seg = MagicMock()
        seg.text = text
        seg.start = start
        seg.end = end
        return seg

    @patch("multimodal_rag.ingest.voxtral.Mistral")
    def test_converts_segments_to_correct_format(
        self, mock_mistral_cls: MagicMock, tmp_path: Path
    ) -> None:
        audio_file = tmp_path / "audio.m4a"
        audio_file.write_bytes(b"fake")

        mock_client = MagicMock()
        mock_mistral_cls.return_value = mock_client

        response = MagicMock()
        response.segments = [
            self._make_segment("Hello world", 0.0, 2.5),
            self._make_segment("How are you", 2.5, 5.0),
        ]
        mock_client.audio.transcriptions.complete.return_value = response

        result = transcribe_with_voxtral(audio_file, "fake-key")

        assert len(result) == 2
        assert result[0] == {"text": "Hello world", "start": 0.0, "duration": 2.5}
        assert result[1] == {"text": "How are you", "start": 2.5, "duration": 2.5}

    @patch("multimodal_rag.ingest.voxtral.Mistral")
    def test_duration_equals_end_minus_start(
        self, mock_mistral_cls: MagicMock, tmp_path: Path
    ) -> None:
        audio_file = tmp_path / "audio.m4a"
        audio_file.write_bytes(b"fake")

        mock_client = MagicMock()
        mock_mistral_cls.return_value = mock_client

        response = MagicMock()
        response.segments = [self._make_segment("Test", 10.0, 13.7)]
        mock_client.audio.transcriptions.complete.return_value = response

        result = transcribe_with_voxtral(audio_file, "fake-key")

        assert pytest.approx(result[0]["duration"]) == 3.7

    @patch("multimodal_rag.ingest.voxtral.Mistral")
    def test_empty_segments_returns_empty_list(
        self, mock_mistral_cls: MagicMock, tmp_path: Path
    ) -> None:
        audio_file = tmp_path / "audio.m4a"
        audio_file.write_bytes(b"fake")

        mock_client = MagicMock()
        mock_mistral_cls.return_value = mock_client

        response = MagicMock()
        response.segments = []
        mock_client.audio.transcriptions.complete.return_value = response

        result = transcribe_with_voxtral(audio_file, "fake-key")

        assert result == []

    @patch("multimodal_rag.ingest.voxtral.Mistral")
    def test_passes_segment_timestamp_granularity(
        self, mock_mistral_cls: MagicMock, tmp_path: Path
    ) -> None:
        audio_file = tmp_path / "audio.m4a"
        audio_file.write_bytes(b"fake")

        mock_client = MagicMock()
        mock_mistral_cls.return_value = mock_client

        response = MagicMock()
        response.segments = [self._make_segment("Hi", 0.0, 1.0)]
        mock_client.audio.transcriptions.complete.return_value = response

        transcribe_with_voxtral(audio_file, "fake-key")

        call_kwargs = mock_client.audio.transcriptions.complete.call_args.kwargs
        assert call_kwargs["timestamp_granularities"] == ["segment"]
        assert call_kwargs["model"] == "voxtral-mini-latest"


# ---------------------------------------------------------------------------
# voxtral.fetch_voxtral_transcript
# ---------------------------------------------------------------------------


class TestFetchVoxtralTranscript:
    @patch("multimodal_rag.ingest.voxtral.transcribe_with_voxtral")
    @patch("multimodal_rag.ingest.voxtral.download_audio")
    def test_cleans_up_tempdir(
        self,
        mock_download: MagicMock,
        mock_transcribe: MagicMock,
    ) -> None:
        captured_tmpdir: list[Path] = []

        def fake_download(url: str, output_dir: Path) -> Path:
            captured_tmpdir.append(output_dir)
            f = output_dir / "audio.m4a"
            f.write_bytes(b"fake")
            return f

        mock_download.side_effect = fake_download
        mock_transcribe.return_value = [{"text": "Hi", "start": 0.0, "duration": 1.0}]

        fetch_voxtral_transcript("https://youtu.be/abc", "key")

        assert len(captured_tmpdir) == 1
        assert not captured_tmpdir[0].exists()

    @patch("multimodal_rag.ingest.voxtral.transcribe_with_voxtral")
    @patch("multimodal_rag.ingest.voxtral.download_audio")
    def test_returns_segments_from_transcribe(
        self,
        mock_download: MagicMock,
        mock_transcribe: MagicMock,
        tmp_path: Path,
    ) -> None:
        audio_file = tmp_path / "audio.m4a"
        audio_file.write_bytes(b"fake")
        mock_download.return_value = audio_file
        expected = [{"text": "Hello", "start": 0.0, "duration": 2.0}]
        mock_transcribe.return_value = expected

        result = fetch_voxtral_transcript("https://youtu.be/abc", "key")

        assert result == expected


# ---------------------------------------------------------------------------
# youtube.fetch_transcript_chunks — fallback behaviour
# ---------------------------------------------------------------------------


class TestFetchTranscriptChunksFallback:
    @patch("multimodal_rag.ingest.youtube.fetch_voxtral_transcript")
    @patch("multimodal_rag.ingest.youtube.fetch_transcript")
    def test_transcripts_disabled_falls_back_to_voxtral(
        self,
        mock_fetch: MagicMock,
        mock_voxtral: MagicMock,
    ) -> None:
        mock_fetch.side_effect = TranscriptsDisabled("vid")
        mock_voxtral.return_value = [
            {"text": "Hello from audio", "start": 0.0, "duration": 5.0}
        ]

        chunks = fetch_transcript_chunks(
            video_url="https://www.youtube.com/watch?v=abc12345678",
            source_name="Test",
            mistral_api_key="fake-key",
        )

        mock_voxtral.assert_called_once_with(
            "https://www.youtube.com/watch?v=abc12345678", "fake-key"
        )
        assert len(chunks) == 1
        assert "Hello from audio" in chunks[0].text

    @patch("multimodal_rag.ingest.youtube.fetch_voxtral_transcript")
    @patch("multimodal_rag.ingest.youtube.fetch_transcript")
    def test_no_transcript_found_falls_back_to_voxtral(
        self,
        mock_fetch: MagicMock,
        mock_voxtral: MagicMock,
    ) -> None:
        mock_fetch.side_effect = NoTranscriptFound("vid", [], {})
        mock_voxtral.return_value = [
            {"text": "Audio content", "start": 0.0, "duration": 3.0}
        ]

        chunks = fetch_transcript_chunks(
            video_url="https://www.youtube.com/watch?v=abc12345678",
            source_name="Test",
            mistral_api_key="fake-key",
        )

        mock_voxtral.assert_called_once()
        assert len(chunks) == 1

    @patch("multimodal_rag.ingest.youtube.fetch_voxtral_transcript")
    @patch("multimodal_rag.ingest.youtube.fetch_transcript")
    def test_ip_blocked_does_not_fall_back(
        self,
        mock_fetch: MagicMock,
        mock_voxtral: MagicMock,
    ) -> None:
        mock_fetch.side_effect = IpBlocked("vid")

        chunks = fetch_transcript_chunks(
            video_url="https://www.youtube.com/watch?v=abc12345678",
            source_name="Test",
            mistral_api_key="fake-key",
        )

        mock_voxtral.assert_not_called()
        assert chunks == []

    @patch("multimodal_rag.ingest.youtube.fetch_voxtral_transcript")
    @patch("multimodal_rag.ingest.youtube.fetch_transcript")
    def test_no_mistral_key_returns_empty_on_disabled(
        self,
        mock_fetch: MagicMock,
        mock_voxtral: MagicMock,
    ) -> None:
        mock_fetch.side_effect = TranscriptsDisabled("vid")

        chunks = fetch_transcript_chunks(
            video_url="https://www.youtube.com/watch?v=abc12345678",
            source_name="Test",
            mistral_api_key="",
        )

        mock_voxtral.assert_not_called()
        assert chunks == []

    @patch("multimodal_rag.ingest.youtube.fetch_voxtral_transcript")
    @patch("multimodal_rag.ingest.youtube.fetch_transcript")
    def test_voxtral_failure_returns_empty(
        self,
        mock_fetch: MagicMock,
        mock_voxtral: MagicMock,
    ) -> None:
        mock_fetch.side_effect = TranscriptsDisabled("vid")
        mock_voxtral.side_effect = RuntimeError("API error")

        chunks = fetch_transcript_chunks(
            video_url="https://www.youtube.com/watch?v=abc12345678",
            source_name="Test",
            mistral_api_key="fake-key",
        )

        assert chunks == []
