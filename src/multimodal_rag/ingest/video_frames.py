"""Video keyframe extraction and vision LLM description."""

import base64
import logging
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import yt_dlp
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from multimodal_rag.models.chunks import TranscriptChunk
from multimodal_rag.models.llm import message_text

logger = logging.getLogger(__name__)


def download_video(
    video_url: str,
    output_dir: Path,
    cookies_file: str = "",
    include_audio: bool = False,
) -> Path:
    """Download best available video from a YouTube URL to output_dir.

    Set include_audio=True to download a merged video+audio stream
    (for a file whose audio is needed).
    Returns the path to the downloaded file.
    Raises yt_dlp.utils.DownloadError on failure.
    """
    fmt = (
        "bestvideo+bestaudio/best"
        if include_audio
        else "bestvideo[vcodec!=images]/best[vcodec!=images]/best"
    )
    ydl_opts: dict = {
        "format": fmt,
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


# Bumped whenever the frame prompt changes; recorded in the run record, not stored.
FRAME_PROMPT_VERSION = "c1-2026-11"
# The cap the prompt asks for: 100, the value every measurement used. A real 60-word
# cap was tried on 12 frames and lost on-screen detail the questions ask for (SYNC-012),
# at a saving of about 70 output tokens per frame.
FRAME_DESCRIPTION_WORDS = 100

_PROMPT_DESCRIBE = (
    "This is a frame from a software tutorial video. "
    "Describe ONLY what is actively happening or being demonstrated: "
    "what action is being performed, what dialog or panel is open, "
    "what specific settings, values, or fields are visible in the active area, "
    "and what step in a workflow this appears to represent. "
    "Do NOT describe static UI chrome: title bar, menu bar, toolbar icons, "
    "sidebar tool palettes, or window controls — these are the same in every frame "
    "and add no value. "
    f"Write plain text in one paragraph of at most {FRAME_DESCRIPTION_WORDS} words. "
    "No markdown, no lists, no bold."
)

_PROMPT_TRANSCRIBE = (
    "Transcribe all text visible in this video frame exactly as written. "
    "Include any on-screen captions, labels, step descriptions, callouts, "
    "or text overlays. "
    "If no text is visible, respond with 'No text visible.'"
)


class DescriptionRejected(ValueError):
    """The vision model's description failed the guard twice."""


def description_problem(
    text: str, cap: int = FRAME_DESCRIPTION_WORDS
) -> str | None:
    """Why a frame description must be rejected, or None if it is acceptable.

    Rejects empty text, text far over the cap (the prompt asks for at most ``cap``
    words) and repetition loops: the stored descriptions included one of 42,843 words
    with a distinct-word ratio of 0.014, where healthy ones stay above 0.6.
    """
    words = text.split()
    if not words:
        return "empty"
    limit = cap * 3 // 2
    if len(words) > limit:
        return f"{len(words)} words, over {limit}"
    if len(words) >= 20 and len({w.lower() for w in words}) / len(words) < 0.3:
        return "a repetition loop"
    return None


def describe_frame(
    frame_path: Path, llm: BaseChatModel, transcribe_mode: bool = False
) -> str:
    """Describe or transcribe a video frame using a vision LLM.

    In transcribe_mode, extracts on-screen text only (for silent screen recordings).
    A description that fails ``description_problem`` is retried once, then raises
    ``DescriptionRejected``. Returns the LLM's text response.
    """
    image_bytes = frame_path.read_bytes()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = _PROMPT_TRANSCRIBE if transcribe_mode else _PROMPT_DESCRIBE
    # The image comes first: every measurement of the frame prompt sent it that way, and
    # text first gave noticeably different wording on the same frame.
    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            },
            {"type": "text", "text": prompt},
        ]
    )
    if transcribe_mode:
        return message_text(llm.invoke([message]))
    problem = None
    for _ in range(2):
        text = message_text(llm.invoke([message])).strip()
        problem = description_problem(text)
        if problem is None:
            return text
        logger.warning("Frame description rejected (%s)", problem)
    raise DescriptionRejected(f"the description was rejected twice ({problem})")



def fetch_frame_chunks(
    video_url: str,
    video_title: str,
    llm: BaseChatModel,
    interval_seconds: int = 30,
    cookies_file: str = "",
    transcribe_mode: bool = False,
    strict: bool = False,
) -> list[TranscriptChunk]:
    """Download a video, extract keyframes, and describe each with a vision LLM.

    Returns a list of TranscriptChunks where text is the frame description
    and start_seconds is the frame timestamp. With ``strict`` a frame that cannot be
    described fails the whole video, so nothing half-described is stored.
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
                if strict:
                    raise
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
    transcribe: Callable[[Path, Path], list[dict[str, float | str]]],
    vision_llm: BaseChatModel | None,
    cookies_file: str = "",
    window_seconds: int = 30,
    strict: bool = False,
) -> list[TranscriptChunk]:
    """Download video once, get its transcript, describe keyframes, window by window.

    Each window yields up to two chunks: its narration (``kind="transcript"``) and its
    on-screen description (``kind="visual"``).
    ``transcribe(video_path, work_dir)`` returns segments as ``{text, start, duration}``.

    With ``strict`` a frame that cannot be described fails the whole video instead of
    leaving that window transcript-only.
    """  # noqa: E501
    chunks: list[TranscriptChunk] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        video_dir = tmp_path / "video"
        frame_dir = tmp_path / "frames"
        video_dir.mkdir()
        frame_dir.mkdir()

        logger.info("Downloading video for fusion: %s", source_name)
        video_path = download_video(
            video_url, video_dir, cookies_file=cookies_file, include_audio=True
        )

        logger.info("Getting the transcript: %s", source_name)
        segments = transcribe(video_path, tmp_path)

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
                    if strict:
                        raise
                    logger.warning(
                        "Failed to describe frame at %ds for %s, skipping visual",
                        window_start,
                        source_name,
                        exc_info=True,
                    )

            # Narration and description are separate chunks of one window: a query
            # can match either without the other diluting it.
            narration = f"[Transcript] {speech_text}" if speech_text else ""
            visual = f"[Visual] {frame_description}" if frame_description else ""
            for kind, text in (("transcript", narration), ("visual", visual)):
                if text:
                    chunks.append(
                        TranscriptChunk(
                            text=text,
                            source_url=video_url,
                            source_name=source_name,
                            start_seconds=window_start,
                            end_seconds=window_end,
                            kind=kind,  # type: ignore[arg-type]
                        )
                    )

    return chunks
