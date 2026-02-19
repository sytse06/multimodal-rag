"""Voxtral audio transcription fallback for videos without captions."""

import logging
import tempfile
from pathlib import Path

import yt_dlp
from mistralai import Mistral

logger = logging.getLogger(__name__)


def download_audio(video_url: str, output_dir: Path) -> Path:
    """Download best available audio from a YouTube URL to output_dir.

    Returns the path to the downloaded file.
    Raises yt_dlp.utils.DownloadError on failure.
    """
    ydl_opts: dict = {
        "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)

    filename = ydl.prepare_filename(info)
    audio_path = Path(filename)
    if not audio_path.exists():
        # yt-dlp may adjust extension after download; find the file
        candidates = list(output_dir.iterdir())
        if not candidates:
            raise FileNotFoundError(f"No audio file found in {output_dir}")
        audio_path = candidates[0]

    return audio_path


def transcribe_with_voxtral(
    audio_path: Path, api_key: str
) -> list[dict[str, float | str]]:
    """Transcribe an audio file using Mistral Voxtral Mini.

    Returns a list of segment dicts with keys: text, start, duration.
    """
    client = Mistral(api_key=api_key)
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.complete(
            model="voxtral-mini-latest",
            file={"file_name": audio_path.name, "content": f},
            timestamp_granularities=["segment"],
        )

    if not response.segments:
        text_preview = (response.text or "")[:120]
        logger.warning(
            "Voxtral returned no segments for %s (text: %r)",
            audio_path.name,
            text_preview,
        )
        return []

    return [
        {
            "text": seg.text,
            "start": float(seg.start),
            "duration": float(seg.end) - float(seg.start),
        }
        for seg in response.segments
    ]


def fetch_voxtral_transcript(
    video_url: str, api_key: str
) -> list[dict[str, float | str]]:
    """Download audio and transcribe via Voxtral. Cleans up temp files on exit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        logger.info("Downloading audio for Voxtral transcription: %s", video_url)
        audio_path = download_audio(video_url, tmp_path)
        logger.info("Transcribing %s with Voxtral", audio_path.name)
        return transcribe_with_voxtral(audio_path, api_key)
