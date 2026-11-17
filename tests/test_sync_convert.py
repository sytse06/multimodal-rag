from datetime import datetime, timezone
from typing import Any

from multimodal_rag.models import identity
from multimodal_rag.sync.convert import Row, checksum, convert, problems

PAGE = "https://hydrosym.paro.nl/article/1-intro"
VIDEO = "https://youtu.be/viJ-mmIjkhk"
VIDEO_DOC = identity.document_id("https://www.youtube.com/watch?v=viJ-mmIjkhk")
MARCH = datetime(2026, 3, 1, tzinfo=timezone.utc)


def row(uuid: str, url: str, kind: str, text: str, **extra: Any) -> Row:
    props = {"source_url": url, "source_type": kind, "text": text, **extra}
    return uuid, props, [0.1, 0.2]


def rows() -> list[Row]:
    return [
        row("a", PAGE, "web", f"intro, see {VIDEO}"),
        row("b", PAGE, "web", "second chunk"),
        row("c", PAGE, "web", "screenshot description"),
        row("v", VIDEO, "video", "transcript"),
    ]


def test_every_chunk_gets_a_document_id_and_a_content_hash() -> None:
    result = convert(rows(), {})

    assert set(result.updates) == {"a", "b", "c", "v"}
    assert result.updates["b"]["document_id"] == identity.document_id(PAGE)
    assert result.updates["b"]["content_hash"] == identity.content_hash("second chunk")


def test_a_pages_videos_go_on_every_chunk_of_the_page() -> None:
    result = convert(rows(), {})

    for uuid in ("a", "b", "c"):
        assert result.updates[uuid]["embedded_video_ids"] == [VIDEO_DOC]
    assert result.updates["v"]["embedded_video_ids"] == []
    assert result.pages_with_videos == 1


def test_the_video_and_its_short_link_share_one_document_id() -> None:
    result = convert(rows(), {})

    assert result.updates["v"]["document_id"] == VIDEO_DOC


def test_lastmod_is_set_only_for_pages_given_one() -> None:
    result = convert(rows(), {PAGE: MARCH})

    assert result.updates["b"]["source_lastmod"] == MARCH
    assert "source_lastmod" not in result.updates["v"]  # videos have none; null is null
    assert result.pages_with_lastmod == 1


def test_a_stored_lastmod_is_never_overwritten_with_nothing() -> None:
    held = [row("b", PAGE, "web", "second chunk", source_lastmod=MARCH)]

    result = convert(held, {})

    assert "source_lastmod" not in result.updates["b"]


def test_a_rerun_changes_nothing_and_never_drops_a_video() -> None:
    first = convert(rows(), {PAGE: MARCH})
    done = [
        (u, {**p, **first.updates.get(u, {})}, v) for u, p, v in rows()
    ]

    again = convert(done, {PAGE: MARCH})

    assert again.updates == {}
    assert again.already_current == 4


def test_the_checksum_ignores_metadata_but_sees_text_and_vectors() -> None:
    base = rows()
    metadata_only = [(u, {**p, "document_id": "x"}, v) for u, p, v in base]
    new_text = [(u, {**p, "text": p["text"] + "!"}, v) for u, p, v in base]
    new_vector = [(u, p, [0.9, 0.2]) for u, p, v in base]

    assert checksum(metadata_only) == checksum(base)
    assert checksum(new_text) != checksum(base)
    assert checksum(new_vector) != checksum(base)


def test_verification_reports_a_changed_vector_or_a_missing_field() -> None:
    before = rows()
    expected = convert(before, {})
    good = [(u, {**p, **expected.updates.get(u, {})}, v) for u, p, v in before]
    moved = [(u, p, [0.9, 0.9]) for u, p, v in good]

    assert problems(before, good, expected) == []
    assert "a text or a vector changed" in problems(before, moved, expected)
    assert any("not as planned" in p for p in problems(before, before, expected))
    assert any("count changed" in p for p in problems(before, good[:-1], expected))
