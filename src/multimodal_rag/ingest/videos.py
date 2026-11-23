"""Ingest one video; shared by ``make ingest`` and the new-video lane."""

import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from multimodal_rag.ingest.transcripts import transcript_for
from multimodal_rag.ingest.video_frames import fetch_frame_chunks, fetch_fused_chunks
from multimodal_rag.models.chunks import SupportChunk
from multimodal_rag.models.config import AppSettings
from multimodal_rag.models.sources import YouTubeSource
from multimodal_rag.store.weaviate import WeaviateStore

logger = logging.getLogger(__name__)

FRAME_SECONDS = 5  # vision-only recordings are sampled densely
WINDOW_SECONDS = 30  # spoken videos: one frame and one transcript window per 30 s


@dataclass(frozen=True)
class Ingested:
    chunks: int
    transcript_source: str | None = None  # "captions" or "gemini"
    transcript_model: str | None = None


def _store(store: WeaviateStore, chunks: list[SupportChunk], label: str) -> int:
    if not chunks:
        logger.warning("[%s] No chunks produced, skipping", label)
        return 0
    added = store.add_chunks(chunks)
    logger.info("[%s] Stored %d chunks", label, added)
    return added


def ingest_video(
    store: WeaviateStore,
    video: YouTubeSource,
    settings: AppSettings,
    vision_llm: BaseChatModel | None,
    cookies_file: str,
    *,
    strict: bool = False,
) -> Ingested:
    """Fetch, chunk and store one video.

    The transcript is the video's manual captions when it has them and a Gemini
    transcription otherwise. ``strict`` makes a frame that cannot be described fail the
    video before anything is stored, because a video that already has chunks is never
    looked at again.
    """
    label = f"youtube:{video.name}"
    if video.skip_voxtral:
        # Silent screen recording: vision-only path, dense frames, no transcript
        logger.info("[%s] skip_voxtral=true — using vision-only path", label)
        if vision_llm is None:
            logger.warning(
                "[%s] skip_voxtral=true but VISION_MODEL not set, skipping", label
            )
            return Ingested(0)
        frames = fetch_frame_chunks(
            str(video.url),
            video.name,
            vision_llm,
            interval_seconds=FRAME_SECONDS,
            cookies_file=cookies_file,
            transcribe_mode=True,
            strict=strict,
        )
        chunks = [SupportChunk.from_frame_chunk(c) for c in frames]
        return Ingested(_store(store, chunks, f"{label} [frames]"))

    used: list[tuple[str, str | None]] = []

    def transcribe(video_path: Path, work_dir: Path) -> list[dict[str, float | str]]:
        transcript = transcript_for(
            str(video.url),
            video_path,
            work_dir,
            settings.credentials.gemini_api_key.get_secret_value(),
            cookies_file,
        )
        used.append((transcript.source, transcript.model))
        return transcript.segments

    fused = fetch_fused_chunks(
        video_url=str(video.url),
        source_name=video.name,
        transcribe=transcribe,
        vision_llm=vision_llm,
        cookies_file=cookies_file,
        window_seconds=WINDOW_SECONDS,
        strict=strict,
    )
    added = _store(store, [SupportChunk.from_fused_chunk(c) for c in fused], label)
    source, model = used[0] if used else (None, None)
    return Ingested(added, source, model)
