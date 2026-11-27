"""Tests for cutting the old hybrid video chunks down to their narration (SYNC-013)."""

from typing import Any
from uuid import NAMESPACE_URL, uuid5

from multimodal_rag.models import identity
from multimodal_rag.sync.split_videos import (
    Row,
    embedding_input,
    fused_id,
    narration_of,
    new_properties,
    plan_split,
    problems,
)

URL = "https://www.youtube.com/watch?v=abc"
HYBRID = "[Transcript] Press B to connect.\n[Visual] A menu bar and toolbar. " * 1


def _props(text: str, ts: int = 0, source_type: str = "video") -> dict[str, Any]:
    return {
        "text": text,
        "source_type": source_type,
        "source_url": URL,
        "source_name": "Tutorial",
        "timestamp_seconds": ts,
        "content_hash": identity.content_hash(text),
        "section_heading": "",
    }


def _row(text: str, ts: int = 0, vector: list[float] | None = None) -> Row:
    props = _props(text, ts)
    return (fused_id(props), props, vector or [1.0, 0.0])


def test_the_narration_is_everything_before_the_visual_part() -> None:
    assert narration_of(HYBRID) == "[Transcript] Press B to connect."


def test_only_a_transcript_then_visual_chunk_is_split() -> None:
    assert narration_of("[Transcript] Press B.") is None  # already narration only
    assert narration_of("[Visual] A menu.") is None  # a description alone
    assert narration_of("[Transcript] \n[Visual] A menu.") is None  # no narration
    assert narration_of("plain web text") is None


def test_the_plan_cuts_hybrids_and_leaves_everything_else() -> None:
    hybrid = _row(HYBRID, 0)
    narration_only = _row("[Transcript] Just words.", 30)
    visual_only_new = (
        str(uuid5(NAMESPACE_URL, f"{URL}|visual|30")),
        _props("[Visual] A description.", 30),
        [1.0, 0.0],
    )
    no_narration = _row("[Visual] A description under the narration id.", 60)
    web = ("w", _props("a page", source_type="web"), [1.0, 0.0])

    split = plan_split([hybrid, narration_only, visual_only_new, no_narration, web])

    assert split.texts == {hybrid[0]: "[Transcript] Press B to connect."}
    assert split.no_narration == [no_narration[0]]
    assert split.already_split == 2
    assert split.other == 1
    assert split.dropped_words[hybrid[0]] == len("A menu bar and toolbar.".split())


def test_a_new_description_chunk_is_not_a_window_without_narration() -> None:
    new_visual = (
        str(uuid5(NAMESPACE_URL, f"{URL}|visual|0")),
        _props("[Visual] A description."),
        [1.0, 0.0],
    )

    split = plan_split([new_visual])

    assert split.no_narration == []
    assert split.already_split == 1


def test_new_properties_change_only_the_text_and_its_hash() -> None:
    props = _props(HYBRID)
    changed = new_properties(props, "[Transcript] Press B to connect.")

    assert changed["text"] == "[Transcript] Press B to connect."
    assert changed["content_hash"] == identity.content_hash(changed["text"])
    assert {k: v for k, v in changed.items() if k not in ("text", "content_hash")} == {
        k: v for k, v in props.items() if k not in ("text", "content_hash")
    }
    assert props["text"] == HYBRID  # the input is not mutated


def test_the_embedding_input_names_the_video_like_ingest_does() -> None:
    props = _props(HYBRID)

    assert embedding_input(props, "[Transcript] Hi.") == "Tutorial: [Transcript] Hi."


def _applied(rows: list[Row]) -> tuple[list[Row], dict[str, list[float]], Any]:
    split = plan_split(rows)
    after: list[Row] = []
    fresh: dict[str, list[float]] = {}
    for uuid, props, vector in rows:
        if uuid in split.texts:
            vector = [0.0, 1.0]
            after.append((uuid, new_properties(props, split.texts[uuid]), vector))
            fresh[uuid] = vector
        else:
            after.append((uuid, dict(props), list(vector)))
    return after, fresh, split


def test_a_correct_split_has_no_problems() -> None:
    rows = [_row(HYBRID, 0), _row("[Transcript] Just words.", 30)]
    after, fresh, split = _applied(rows)

    assert problems(rows, after, split, fresh) == []


def test_verification_catches_a_lost_object_and_a_changed_bystander() -> None:
    rows = [_row(HYBRID, 0), _row("[Transcript] Just words.", 30)]
    after, fresh, split = _applied(rows)
    bystander = after[1]
    after[1] = (bystander[0], {**bystander[1], "text": "changed"}, bystander[2])
    lost = after[:1] + after[1:]
    del lost[0]

    found = problems(rows, after, split, fresh)
    assert any("not to be split" in p for p in found)
    found = problems(rows, lost, split, fresh)
    assert any("missing" in p for p in found)
    assert any("object count changed" in p for p in found)


def test_verification_catches_wrong_text_hash_and_vector() -> None:
    rows = [_row(HYBRID, 0)]
    after, fresh, split = _applied(rows)
    uuid, props, vector = after[0]
    wrong_text = (uuid, {**props, "text": "[Transcript] Something else."}, vector)
    wrong_hash = (uuid, {**props, "content_hash": "0" * 64}, vector)

    assert any(
        "not the narration" in p for p in problems(rows, [wrong_text], split, fresh)
    )
    assert any(
        "content_hash" in p for p in problems(rows, [wrong_hash], split, fresh)
    )
    drifted = {uuid: [1.0, 0.0]}  # a fresh embedding that disagrees with the stored one
    assert any(
        "vector does not match" in p for p in problems(rows, after, split, drifted)
    )


def test_verification_catches_a_changed_property_besides_the_text() -> None:
    rows = [_row(HYBRID, 0)]
    after, fresh, split = _applied(rows)
    uuid, props, vector = after[0]
    moved = (uuid, {**props, "timestamp_seconds": 999}, vector)

    assert any("other than text" in p for p in problems(rows, [moved], split, fresh))


def test_a_second_run_finds_nothing_left_to_split() -> None:
    rows = [_row(HYBRID, 0), _row("[Transcript] Just words.", 30)]
    after, _, _ = _applied(rows)

    again = plan_split(after)

    assert again.texts == {}
    assert again.already_split == 2
