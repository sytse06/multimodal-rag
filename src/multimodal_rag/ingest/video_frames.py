"""Video keyframe extraction and vision LLM description."""

import base64
import logging
import subprocess
import tempfile
from pathlib import Path

import yt_dlp
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from multimodal_rag.models.chunks import TranscriptChunk

logger = logging.getLogger(__name__)


def download_video(video_url: str, output_dir: Path) -> Path:
    """Download best available video from a YouTube URL to output_dir.

    Returns the path to the downloaded file.
    Raises yt_dlp.utils.DownloadError on failure.
    """
    ydl_opts: dict = {
        "format": "bestvideo[ext=mp4]/bestvideo",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)

    filename = ydl.prepare_filename(info)
    video_path = Path(filename)
    if not video_path.exists():
        candidates = list(output_dir.iterdir())
        if not candidates:
            raise FileNotFoundError(f"No video file found in {output_dir}")
        video_path = candidates[0]

    return video_path


def extract_keyframes(
    video_path: Path, output_dir: Path, interval_seconds: int = 30
) -> list[tuple[Path, int]]:
    """Extract one frame every interval_seconds from a video file.

    Returns a sorted list of (frame_path, timestamp_seconds) pairs.
    Raises subprocess.CalledProcessError on ffmpeg failure.
    """
    output_pattern = str(output_dir / "frame_%04d.jpg")
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vf",
            f"fps=1/{interval_seconds}",
            "-q:v",
            "2",
            output_pattern,
        ],
        check=True,
        capture_output=True,
    )

    frames = sorted(output_dir.glob("frame_*.jpg"))
    return [(frame, (i) * interval_seconds) for i, frame in enumerate(frames)]


def describe_frame(frame_path: Path, llm: BaseChatModel) -> str:
    """Describe a video frame using a vision LLM.

    Encodes the JPEG as base64 and sends it via an image_url content block.
    Returns the LLM's text description.
    """
    image_bytes = frame_path.read_bytes()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Describe what is shown in this video frame in detail. "
                    "Focus on any UI elements, text, menus, buttons, "
                    "or actions visible. "
                    "Be specific and concise."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            },
        ]
    )
    response = llm.invoke([message])
    return str(response.content)


def fetch_frame_chunks(
    video_url: str,
    video_title: str,
    llm: BaseChatModel,
    interval_seconds: int = 30,
) -> list[TranscriptChunk]:
    """Download a video, extract keyframes, and describe each with a vision LLM.

    Returns a list of TranscriptChunks where text is the frame description
    and start_seconds is the frame timestamp.
    """
    chunks: list[TranscriptChunk] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        video_dir = tmp_path / "video"
        frame_dir = tmp_path / "frames"
        video_dir.mkdir()
        frame_dir.mkdir()

        logger.info("Downloading video for frame extraction: %s", video_title)
        video_path = download_video(video_url, video_dir)

        frames = extract_keyframes(video_path, frame_dir, interval_seconds)
        logger.info("Extracted %d keyframes from %s", len(frames), video_title)

        for frame_path, timestamp in frames:
            try:
                description = describe_frame(frame_path, llm)
                chunks.append(
                    TranscriptChunk(
                        text=description,
                        source_url=video_url,
                        source_name=video_title,
                        start_seconds=timestamp,
                        end_seconds=timestamp + interval_seconds,
                    )
                )
            except Exception:
                logger.warning(
                    "Failed to describe frame at %ds for %s, skipping",
                    timestamp,
                    video_title,
                    exc_info=True,
                )

    return chunks
