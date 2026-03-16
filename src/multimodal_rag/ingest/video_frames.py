"""Video keyframe extraction and vision LLM description."""

import base64
import logging
import subprocess
import tempfile
from pathlib import Path

import yt_dlp
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from multimodal_rag.ingest.voxtral import transcribe_with_voxtral
from multimodal_rag.models.chunks import TranscriptChunk

logger = logging.getLogger(__name__)


def download_video(
    video_url: str, output_dir: Path, cookies_file: str = ""
) -> Path:
    """Download best available video from a YouTube URL to output_dir.

    Returns the path to the downloaded file.
    Raises yt_dlp.utils.DownloadError on failure.
    """
    ydl_opts: dict = {
        "format": "bestvideo[vcodec!=images]/best[vcodec!=images]/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "remote_components": "ejs:github",
    }
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file
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


_PROMPT_DESCRIBE = (
    "Describe what is shown in this video frame in detail. "
    "Focus on any UI elements, text, menus, buttons, or actions visible. "
    "Be specific and concise."
)

_PROMPT_TRANSCRIBE = (
    "Transcribe all text visible in this video frame exactly as written. "
    "Include any on-screen captions, labels, step descriptions, callouts, "
    "or text overlays. "
    "If no text is visible, respond with 'No text visible.'"
)


def describe_frame(
    frame_path: Path, llm: BaseChatModel, transcribe_mode: bool = False
) -> str:
    """Describe or transcribe a video frame using a vision LLM.

    In transcribe_mode, extracts on-screen text only (for silent screen recordings).
    Returns the LLM's text response.
    """
    image_bytes = frame_path.read_bytes()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = _PROMPT_TRANSCRIBE if transcribe_mode else _PROMPT_DESCRIBE
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            },
        ]
    )
    response = llm.invoke([message])
    return str(response.content)


def extract_audio(video_path: Path, output_dir: Path) -> Path:
    """Extract audio track from a video file to an m4a file via ffmpeg.

    Returns the path to the extracted audio file.
    Raises subprocess.CalledProcessError on ffmpeg failure.
    """
    audio_path = output_dir / (video_path.stem + ".m4a")
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "copy",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
    )
    return audio_path


def fetch_frame_chunks(
    video_url: str,
    video_title: str,
    llm: BaseChatModel,
    interval_seconds: int = 30,
    cookies_file: str = "",
    transcribe_mode: bool = False,
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
        video_path = download_video(video_url, video_dir, cookies_file=cookies_file)

        frames = extract_keyframes(video_path, frame_dir, interval_seconds)
        logger.info("Extracted %d keyframes from %s", len(frames), video_title)

        for frame_path, timestamp in frames:
            try:
                description = describe_frame(
                    frame_path, llm, transcribe_mode=transcribe_mode
                )
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


def fetch_fused_chunks(
    video_url: str,
    source_name: str,
    mistral_api_key: str,
    vision_llm: BaseChatModel | None,
    cookies_file: str = "",
    window_seconds: int = 30,
) -> list[TranscriptChunk]:
    """Download video once, transcribe with Voxtral, describe keyframes, merge per window."""  # noqa: E501
    chunks: list[TranscriptChunk] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        video_dir = tmp_path / "video"
        frame_dir = tmp_path / "frames"
        video_dir.mkdir()
        frame_dir.mkdir()

        logger.info("Downloading video for fusion: %s", source_name)
        video_path = download_video(video_url, video_dir, cookies_file=cookies_file)

        logger.info("Extracting audio for Voxtral: %s", source_name)
        audio_path = extract_audio(video_path, tmp_path)
        logger.info("Transcribing with Voxtral: %s", source_name)
        segments = transcribe_with_voxtral(audio_path, mistral_api_key)

        frames: list[tuple[Path, int]] = []
        if vision_llm is not None:
            frames = extract_keyframes(
                video_path, frame_dir, interval_seconds=window_seconds
            )
            logger.info("Extracted %d keyframes from %s", len(frames), source_name)

        if frames:
            window_starts = [ts for _, ts in frames]
        else:
            max_t = max(
                (float(s["start"]) + float(s["duration"]) for s in segments),
                default=0.0,
            )
            window_starts = list(
                range(0, int(max_t) + window_seconds, window_seconds)
            )

        for i, window_start in enumerate(window_starts):
            window_end = window_start + window_seconds

            window_segs = [
                s
                for s in segments
                if float(s["start"]) >= window_start
                and float(s["start"]) < window_end
            ]
            speech_text = " ".join(
                str(s["text"]).strip() for s in window_segs
            ).strip()

            frame_description: str | None = None
            if vision_llm is not None and i < len(frames):
                frame_path, _ = frames[i]
                try:
                    frame_description = describe_frame(frame_path, vision_llm)
                except Exception:
                    logger.warning(
                        "Failed to describe frame at %ds for %s, skipping visual",
                        window_start,
                        source_name,
                        exc_info=True,
                    )

            if not speech_text and frame_description is None:
                continue

            if speech_text and frame_description:
                text = f"[Transcript] {speech_text}\n[Visual] {frame_description}"
            elif speech_text:
                text = speech_text
            else:
                text = f"[Visual] {frame_description}"

            chunks.append(
                TranscriptChunk(
                    text=text,
                    source_url=video_url,
                    source_name=source_name,
                    start_seconds=window_start,
                    end_seconds=window_end,
                )
            )

    return chunks
