from typing import Any

from pydantic import HttpUrl

from multimodal_rag.models import identity
from multimodal_rag.models.sources import YouTubeSource
from multimodal_rag.sync.videos import (
    Row,
    estimate,
    rollback_uuids,
    select_new,
    stored_video_documents,
    verify_added,
)
from multimodal_rag.sync.videos_cli import summary

ID = "yWuBQik8KPs"
WATCH = f"https://www.youtube.com/watch?v={ID}"
SHORT = f"https://youtu.be/{ID}"


def video(url: str, name: str = "V", **extra: Any) -> YouTubeSource:
    return YouTubeSource(url=HttpUrl(url), name=name, **extra)


def row(uuid: str, doc: str | None, kind: str = "video", vec: float = 0.1) -> Row:
    props: dict[str, Any] = {"source_type": kind, "text": uuid}
    if doc:
        props["document_id"] = doc
    return uuid, props, [vec, 0.2]


def test_a_video_with_stored_chunks_is_not_new() -> None:
    stored = {identity.document_id(WATCH)}

    chosen = select_new([video(WATCH), video("https://youtu.be/abcdefghijk")], stored)

    assert [str(v.url) for v in chosen.stored] == [WATCH]
    assert [str(v.url) for v in chosen.new] == ["https://youtu.be/abcdefghijk"]


def test_the_registry_url_shape_does_not_matter() -> None:
    stored = {identity.document_id(SHORT)}

    chosen = select_new([video(WATCH)], stored)

    assert chosen.new == [] and len(chosen.stored) == 1


def test_a_url_shape_it_cannot_identify_is_refused_not_ingested() -> None:
    chosen = select_new([video(f"https://www.youtube.com/shorts/{ID}")], set())

    assert chosen.new == []
    assert "identity would be a guess" in chosen.refused[0]


def test_two_registry_entries_for_one_video_are_refused() -> None:
    chosen = select_new([video(WATCH), video(SHORT, "again")], set())

    assert len(chosen.new) == 1
    assert "same video" in chosen.refused[0]


def test_stored_documents_come_from_video_chunks_only() -> None:
    rows = [row("a", "d1"), row("b", "d1"), row("c", "d2"), row("w", "dw", "web")]

    held, problems = stored_video_documents(rows)

    assert held == {"d1", "d2"} and problems == []


def test_a_stored_video_without_a_document_id_makes_the_answer_unsafe() -> None:
    held, problems = stored_video_documents([row("a", None)])

    assert problems and "make sync-convert" in problems[0]


def test_the_estimate_follows_the_path_and_transcript_source_ingest_will_take() -> None:
    captioned = estimate(video(WATCH), 124, manual_captions=True, has_vision=True)
    spoken = estimate(video(WATCH), 124, manual_captions=False, has_vision=True)
    silent = estimate(
        video(WATCH, silent=True), 124, manual_captions=False, has_vision=True
    )
    no_vision = estimate(video(WATCH), 124, manual_captions=True, has_vision=False)

    assert (captioned.transcript, captioned.gemini_minutes) == ("captions", 0.0)
    assert (spoken.path, spoken.transcript) == ("fused", "gemini")
    assert (spoken.vision_calls, spoken.gemini_minutes) == (5, 2.07)
    assert (silent.path, silent.transcript) == ("frames", "none")
    assert (silent.vision_calls, silent.gemini_minutes) == (25, 0.0)
    assert no_vision.vision_calls == 0


def test_a_run_that_only_added_the_new_video_verifies_clean() -> None:
    before = [row("a", "d1"), row("w", "dw", "web")]
    after = [*before, row("n1", "new"), row("n2", "new")]

    problems, added = verify_added(before, after, {"new"})

    assert problems == [] and added == ["n1", "n2"]


def test_verification_catches_a_change_a_loss_and_a_stray_object() -> None:
    before = [row("a", "d1"), row("b", "d1")]
    after = [row("a", "d1", vec=0.9), row("n1", "other"), row("n2", "new", "web")]

    problems, _ = verify_added(before, after, {"new"})

    text = " ".join(problems)
    assert "a: an existing object changed" in text
    assert "b: an existing object is gone" in text
    assert "n1: a new object that is not one of the new videos" in text
    assert "n2: a new object that is not one of the new videos" in text


def test_rollback_takes_all_of_the_planned_videos_without_a_record() -> None:
    rows = [
        row("a", "old"),
        row("n1", "new"),
        row("n2", "new"),
        row("w", "page", "web"),
    ]

    assert rollback_uuids(rows, {"new"}, []) == ["n1", "n2"]


def test_rollback_also_takes_recorded_uuids_that_still_exist() -> None:
    rows = [row("a", "old"), row("n1", "new")]

    assert rollback_uuids(rows, {"new"}, ["n1", "gone"]) == ["n1"]


def test_rollback_never_touches_other_videos() -> None:
    rows = [row("a", "old"), row("b", "old")]

    assert rollback_uuids(rows, {"new"}, []) == []


def test_an_unknown_length_is_reported_as_unknown_not_as_free() -> None:
    known = estimate(video(WATCH), 124, manual_captions=False, has_vision=True)
    unknown = estimate(
        video(SHORT, "second"), None, manual_captions=False, has_vision=True
    )

    lines = summary([known, unknown])

    assert any("second: length unknown" in line for line in lines)
    assert "incomplete: 1 with unknown length" in lines[-1]


def test_a_window_stored_as_a_narration_and_a_visual_chunk_verifies_clean() -> None:
    before = [row("a", "d1")]
    after = [*before, row("narration", "new"), row("visual", "new")]

    problems, added = verify_added(before, after, {"new"})

    assert problems == [] and added == ["narration", "visual"]
