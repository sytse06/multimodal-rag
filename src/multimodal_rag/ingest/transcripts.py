"""Where a video's transcript comes from: manual captions first, Gemini otherwise.

Both sources return segments as ``{text, start, duration}``, the shape the fused
30-second windows already consume, so a video looks the same whichever one supplied it.
Automatic YouTube captions are never used: they are weakest on product names.
"""

import logging
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from google import genai
from google.genai import types
from pydantic import BaseModel
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from multimodal_rag.ingest.youtube import _build_http_client, extract_video_id

logger = logging.getLogger(__name__)

TRANSCRIPTION_MODEL = "gemini-3.8-flash"
CAPTION_LANGUAGES = ["en", "en-GB", "en-US"]
PROMPT = (
    "Transcribe the speech in this audio verbatim, in English. Return a JSON array "
    "of segments, each with 'start' (MM:SS, when the segment starts) and 'text'. "
    "Make each segment one or two sentences. Do not summarise, translate or add "
    "anything."
)
_SOUND_TAG = re.compile(r"^\s*[\[(][^\])]*[\])]\s*$")  # "[Music]", "(applause)"

Segments = list[dict[str, float | str]]


@dataclass(frozen=True)
class Transcript:
    segments: Segments
    source: str  # "captions" or "gemini"
    model: str | None = None  # the model version Gemini reported


class _Segment(BaseModel):
    start: str
    text: str


def _captions(video_url: str, cookies_file: str) -> Any | None:
    """The video's manual English captions, or None when it has only automatic ones."""
    video_id = extract_video_id(video_url)
    if video_id is None:
        raise ValueError(f"not a YouTube video address: {video_url}")
    api = YouTubeTranscriptApi(http_client=_build_http_client(cookies_file))
    try:
        return api.list(video_id).find_manually_created_transcript(CAPTION_LANGUAGES)
    except (NoTranscriptFound, TranscriptsDisabled):
        return None


def has_manual_captions(video_url: str, cookies_file: str = "") -> bool:
    return _captions(video_url, cookies_file) is not None


def manual_captions(video_url: str, cookies_file: str = "") -> Segments | None:
    """Manual captions as segments, sound tags dropped; None when there are none."""
    transcript = _captions(video_url, cookies_file)
    if transcript is None:
        return None
    segments: Segments = []
    for snippet in transcript.fetch():
        text = " ".join(str(snippet.text).split())
        if text and not _SOUND_TAG.match(text):
            segments.append(
                {
                    "text": text,
                    "start": float(snippet.start),
                    "duration": float(snippet.duration),
                }
            )
    return segments or None


def to_seconds(stamp: str) -> float:
    """``MM:SS`` or ``HH:MM:SS`` as seconds."""
    parts = [float(p) for p in stamp.strip().split(":")]
    return float(sum(p * 60**i for i, p in enumerate(reversed(parts))))


def to_mono_wav(media_path: Path, output_dir: Path) -> Path:
    """Mono 16 kHz audio; a stereo track can make Gemini misjudge the length by ~10%."""
    wav = output_dir / (media_path.stem + ".mono.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(media_path), "-vn"]
        + ["-ac", "1", "-ar", "16000", str(wav)],
        check=True,
        capture_output=True,
    )
    return wav


def _audio_seconds(wav: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of"]
        + ["csv=p=0", str(wav)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return float(out)


def gemini_transcript(
    wav: Path, api_key: str, model: str = TRANSCRIPTION_MODEL
) -> Transcript:
    """Transcribe audio with Gemini at temperature 0. Raises if nothing comes back."""
    client = genai.Client(api_key=api_key)
    uploaded = client.files.upload(file=str(wav))
    contents: list[Any] = [uploaded, PROMPT]
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=list[_Segment],
        ),
    )
    parsed = [s for s in cast(list[_Segment], response.parsed or []) if s.text.strip()]
    if not parsed:
        raise RuntimeError(f"{model} returned no transcript for {wav.name}")
    starts = sorted((to_seconds(s.start), s.text.strip()) for s in parsed)
    length = _audio_seconds(wav)
    ends = [start for start, _ in starts[1:]] + [length]
    segments: Segments = [
        {"text": text, "start": start, "duration": max(end - start, 0.0)}
        for (start, text), end in zip(starts, ends)
    ]
    last = starts[-1][0]
    if last < length * 0.5:
        logger.warning(
            "%s: the transcript stops at %ds of %ds; check it",
            wav.name,
            int(last),
            int(length),
        )
    return Transcript(segments, "gemini", response.model_version or model)


def transcript_for(
    video_url: str, video_path: Path, work_dir: Path, gemini_api_key: str,
    cookies_file: str = "",
) -> Transcript:
    """Manual captions if the video has them, otherwise a Gemini transcription."""
    captions = manual_captions(video_url, cookies_file)
    if captions is not None:
        logger.info("Using manual captions for %s", video_url)
        return Transcript(captions, "captions")
    if not gemini_api_key:
        raise ValueError(
            f"{video_url} has no manual captions and GEMINI_API_KEY is not set"
        )
    logger.info("No manual captions for %s; transcribing with Gemini", video_url)
    started = time.time()
    transcript = gemini_transcript(to_mono_wav(video_path, work_dir), gemini_api_key)
    logger.info("Transcribed in %.0fs with %s", time.time() - started, transcript.model)
    return transcript
