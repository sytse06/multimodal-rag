"""Split the stored hybrid video chunks into narration only (SYNC-013).

Before SYNC-012 a video window was one chunk: ``[Transcript] <narration>`` followed by
``\\n[Visual] <description>``. New videos store the two as separate chunks; this module
converts the old ones by keeping each chunk's narration under its existing id and
leaving the old description out (the maintainer's decision: those descriptions are
mostly menu and toolbar inventory, and SYNC-012 part 3 will replace them). The old text
stays only in the snapshot taken first.

Pure logic: it reads objects as ``(uuid, properties, vector)`` and returns the new text
of every chunk to change. Embedding and writing happen in the command.
"""

from dataclasses import dataclass, field
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from multimodal_rag.models import identity

Row = tuple[str, dict[str, Any], list[float]]

NARRATION = "[Transcript] "
VISUAL = "[Visual] "
VISUAL_MARK = "\n" + VISUAL
COSINE_SAME = 0.9999


@dataclass
class Split:
    """What the split will do: new texts by uuid, and everything it leaves alone."""

    texts: dict[str, str] = field(default_factory=dict)
    no_narration: list[str] = field(default_factory=list)
    already_split: int = 0
    other: int = 0
    dropped_words: dict[str, int] = field(default_factory=dict)


def narration_of(text: str) -> str | None:
    """The narration of a hybrid chunk, or None if the chunk is not one to split.

    A hybrid chunk starts with ``[Transcript] `` and holds a ``[Visual] `` part. A chunk
    whose narration is empty is not split: it has nothing to keep.
    """
    if not text.startswith(NARRATION) or VISUAL_MARK not in text:
        return None
    narration = text.split(VISUAL_MARK, 1)[0]
    if not narration[len(NARRATION) :].strip():
        return None
    return narration


def fused_id(props: dict[str, Any]) -> str:
    """The id a window's narration chunk has: the same one the old hybrid chunk kept."""
    return str(
        uuid5(
            NAMESPACE_URL,
            f"{props['source_url']}|fused|{props['timestamp_seconds']}",
        )
    )


def plan_split(rows: list[Row]) -> Split:
    """Which video chunks to cut down to their narration, and which stay as they are.

    A chunk with only a description under the window's narration id is a window that had
    no narration: it is listed and left as it is. A description chunk under its own
    (``|visual|``) id belongs to a video that is already split.
    """
    result = Split()
    for uuid, props, _ in rows:
        if props.get("source_type") != "video":
            result.other += 1
            continue
        text = str(props.get("text", ""))
        narration = narration_of(text)
        if narration is not None:
            result.texts[uuid] = narration
            description = text[len(narration) + len(VISUAL_MARK) :]
            result.dropped_words[uuid] = len(description.split())
        elif VISUAL_MARK in text or (
            text.startswith(VISUAL) and uuid == fused_id(props)
        ):
            result.no_narration.append(uuid)
        else:
            result.already_split += 1
    return result


def new_properties(props: dict[str, Any], narration: str) -> dict[str, Any]:
    """The chunk's properties with its new text and the hash that describes it."""
    changed = dict(props)
    changed["text"] = narration
    changed["content_hash"] = identity.content_hash(narration)
    return changed


def embedding_input(props: dict[str, Any], narration: str) -> str:
    """What is embedded for a chunk: the video's name and its text, as at ingest."""
    return f"{props['source_name']}: {narration}"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = (sum(x * x for x in a) * sum(y * y for y in b)) ** 0.5
    return dot / norm if norm else 0.0


def problems(
    before: list[Row],
    after: list[Row],
    split: Split,
    fresh: dict[str, list[float]],
) -> list[str]:
    """Why the result is not what the split promised; empty when it is.

    ``fresh`` is a new embedding of every changed chunk's new text, made after the
    write, to prove the stored vectors can be reproduced from the stored text.
    """
    found: list[str] = []
    old = {u: (p, v) for u, p, v in before}
    new = {u: (p, v) for u, p, v in after}
    if len(before) != len(after):
        found.append(f"object count changed: {len(before)} before, {len(after)} after")
    for uuid in sorted(old.keys() - new.keys()):
        found.append(f"{uuid}: missing after the split")
    for uuid in sorted(new.keys() - old.keys()):
        found.append(f"{uuid}: should not exist")
    for uuid in sorted(old.keys() & new.keys()):
        before_props, before_vector = old[uuid]
        after_props, after_vector = new[uuid]
        if uuid not in split.texts:
            if before_props != after_props:
                found.append(f"{uuid}: properties changed but it was not to be split")
            elif _cosine(before_vector, after_vector) < COSINE_SAME:
                found.append(f"{uuid}: vector changed but it was not to be split")
            continue
        narration = split.texts[uuid]
        old_text = str(before_props["text"])
        if after_props.get("text") != narration:
            found.append(f"{uuid}: text is not the narration")
        if not (
            old_text.startswith(narration)
            and old_text[len(narration) :].startswith(VISUAL_MARK)
        ):
            found.append(f"{uuid}: the narration is not the start of the old text")
        if after_props.get("content_hash") != identity.content_hash(narration):
            found.append(f"{uuid}: content_hash does not match the new text")
        others = {k for k in before_props.keys() | after_props.keys()} - {
            "text",
            "content_hash",
        }
        if any(before_props.get(k) != after_props.get(k) for k in others):
            found.append(f"{uuid}: a property other than text changed")
        if _cosine(after_vector, fresh[uuid]) < COSINE_SAME:
            found.append(f"{uuid}: stored vector does not match its text")
    return found
