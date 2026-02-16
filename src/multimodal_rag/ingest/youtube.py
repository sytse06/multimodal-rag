"""YouTube transcript fetching and chunking."""

import logging
import re

from youtube_transcript_api import YouTubeTranscriptApi

from multimodal_rag.models.chunks import TranscriptChunk

logger = logging.getLogger(__name__)


def extract_video_id(url: str) -> str | None:
    """Extract video ID from a YouTube URL."""
    patterns = [
        r"(?:v=|/v/|/embed/)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_transcript(video_id: str) -> list[dict[str, float | str]]:
    """Fetch timestamped transcript segments for a video."""
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id, languages=["en", "en-GB", "en-US"])
    return [
        {
            "text": str(snippet.text),
            "start": float(snippet.start),
            "duration": float(snippet.duration),
        }
        for snippet in transcript
    ]


def chunk_segments(
    segments: list[dict[str, float | str]],
    source_url: str,
    source_name: str,
    target_tokens: int = 400,
) -> list[TranscriptChunk]:
    """Group consecutive transcript segments into chunks of ~target_tokens.

    Estimates tokens as word_count * 1.3 (conservative for English text).
    """
    chunks: list[TranscriptChunk] = []
    current_texts: list[str] = []
    current_token_estimate = 0
    chunk_start: float = 0.0

    for i, segment in enumerate(segments):
        text = str(segment["text"]).strip()
        if not text:
            continue

        word_count = len(text.split())
        token_estimate = int(word_count * 1.3)

        if i == 0 or current_token_estimate == 0:
            chunk_start = float(segment["start"])

        if current_token_estimate + token_estimate > target_tokens and current_texts:
            last_seg_idx = max(0, i - 1)
            last_seg = segments[last_seg_idx]
            end = float(last_seg["start"]) + float(last_seg["duration"])
            chunks.append(
                TranscriptChunk(
                    text=" ".join(current_texts),
                    source_url=source_url,
                    source_name=source_name,
                    start_seconds=int(chunk_start),
                    end_seconds=int(end),
                )
            )
            current_texts = []
            current_token_estimate = 0
            chunk_start = float(segment["start"])

        current_texts.append(text)
        current_token_estimate += token_estimate

    if current_texts:
        last_seg = segments[-1]
        end = float(last_seg["start"]) + float(last_seg["duration"])
        chunks.append(
            TranscriptChunk(
                text=" ".join(current_texts),
                source_url=source_url,
                source_name=source_name,
                start_seconds=int(chunk_start),
                end_seconds=int(end),
            )
        )

    return chunks


def fetch_transcript_chunks(
    video_url: str,
    source_name: str,
    target_tokens: int = 400,
) -> list[TranscriptChunk]:
    """Fetch and chunk a YouTube video's transcript.

    Returns empty list if transcript is unavailable.
    """
    video_id = extract_video_id(video_url)
    if not video_id:
        logger.error("Could not extract video ID from URL: %s", video_url)
        return []

    try:
        segments = fetch_transcript(video_id)
    except Exception:
        logger.exception("Failed to fetch transcript for %s", video_url)
        return []

    if not segments:
        logger.warning("No transcript segments found for %s", video_url)
        return []

    chunks = chunk_segments(
        segments,
        source_url=video_url,
        source_name=source_name,
        target_tokens=target_tokens,
    )
    logger.info(
        "Processed %s: %d segments → %d chunks",
        source_name,
        len(segments),
        len(chunks),
    )
    return chunks
