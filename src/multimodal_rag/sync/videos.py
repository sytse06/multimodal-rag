"""Which registry videos are new, what ingesting them costs, and proof it only added.

Pure logic. A video is new when no stored chunk carries its ``document_id``; the
registry URL shape does not matter, so ``youtu.be/<id>`` and ``watch?v=<id>`` are one.
"""

from dataclasses import dataclass, field
from typing import Any

from multimodal_rag.ingest.videos import FRAME_SECONDS, WINDOW_SECONDS
from multimodal_rag.models import identity
from multimodal_rag.models.sources import YouTubeSource

Row = tuple[str, dict[str, Any], list[float]]


@dataclass
class Selection:
    new: list[YouTubeSource] = field(default_factory=list)
    stored: list[YouTubeSource] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Estimate:
    name: str
    path: str  # fused (spoken) or frames (silent recording)
    transcript: str  # captions, gemini or none
    seconds: int | None
    vision_calls: int
    gemini_minutes: float  # audio sent to Gemini for transcription


def stored_video_documents(rows: list[Row]) -> tuple[set[str], list[str]]:
    """``document_id`` of every stored video, and the reasons the answer is unsafe."""
    held, problems = set(), []
    for uuid, props, _ in rows:
        if props["source_type"] != "video":
            continue
        if not props.get("document_id"):
            problems.append(f"{uuid} has no document_id; run make sync-convert first")
            break
        held.add(props["document_id"])
    return held, problems


def select_new(videos: list[YouTubeSource], stored: set[str]) -> Selection:
    """Split the registry's videos into new and stored, refusing what cannot be told."""
    out = Selection()
    seen: dict[str, str] = {}
    for video in videos:
        url = str(video.url)
        if not identity.is_video_url(url):
            out.refused.append(
                f"{url}: not a watch, youtu.be or embed link, so its identity would be"
                " a guess; register it as https://www.youtube.com/watch?v=<id>"
            )
            continue
        doc = identity.document_id(url)
        if doc in seen:
            out.refused.append(f"{url} and {seen[doc]} are the same video")
            continue
        seen[doc] = url
        (out.stored if doc in stored else out.new).append(video)
    return out


def estimate(
    video: YouTubeSource,
    seconds: int | None,
    *,
    manual_captions: bool,
    has_vision: bool,
) -> Estimate:
    """What one video costs on the path ingest will really take for it."""
    sampled = seconds is not None and has_vision

    def frames(interval: int) -> int:
        return seconds // interval + 1 if seconds is not None and sampled else 0

    if video.silent:
        calls = frames(FRAME_SECONDS)
        return Estimate(video.name, "frames", "none", seconds, calls, 0.0)
    transcript = "captions" if manual_captions else "gemini"
    minutes = 0.0 if manual_captions or seconds is None else round(seconds / 60, 2)
    return Estimate(
        video.name, "fused", transcript, seconds, frames(WINDOW_SECONDS), minutes
    )


def rollback_uuids(
    rows: list[Row], documents: set[str], recorded: list[str]
) -> list[str]:
    """Everything a run added: each object of its videos, plus any recorded UUID.

    The videos were new when the run began, so every object carrying their document id
    came from the run, even if it stopped before it could list what it wrote.
    """
    found = {u for u, p, _ in rows if p.get("document_id") in documents}
    return sorted(found | ({u for u, _, _ in rows} & set(recorded)))


def verify_added(
    before: list[Row], after: list[Row], documents: set[str]
) -> tuple[list[str], list[str]]:
    """Proof a run only added: (problems, UUIDs of the new objects).

    Every earlier object must be unchanged (properties and vector) and every new one
    must be a video chunk of one of ``documents``.
    """
    held = {u: (p, v) for u, p, v in after}
    problems = []
    for uuid, props, vector in before:
        if uuid not in held:
            problems.append(f"{uuid}: an existing object is gone")
        elif held[uuid][0] != props or held[uuid][1] != vector:
            problems.append(f"{uuid}: an existing object changed")
    known = {u for u, _, _ in before}
    added = sorted(u for u in held if u not in known)
    for uuid in added:
        props = held[uuid][0]
        if props["source_type"] != "video" or props.get("document_id") not in documents:
            problems.append(f"{uuid}: a new object that is not one of the new videos")
    return problems[:10], added
