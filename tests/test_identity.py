from datetime import datetime, timezone

from multimodal_rag.ingest.web import chunk_page
from multimodal_rag.models import identity
from multimodal_rag.models.chunks import SourceType, SupportChunk, WebChunk

ID = "viJ-mmIjkhk"


def test_both_youtube_url_shapes_are_one_document() -> None:
    watch = f"https://www.youtube.com/watch?v={ID}"
    short = f"https://youtu.be/{ID}"
    embed = f"https://www.youtube.com/embed/{ID}"

    assert identity.document_id(watch) == identity.document_id(short)
    assert identity.document_id(watch) == identity.document_id(embed)


def test_a_timestamped_link_is_the_same_document() -> None:
    plain = f"https://www.youtube.com/watch?v={ID}"

    assert identity.document_id(plain + "&t=90s") == identity.document_id(plain)


def test_a_web_page_keeps_its_own_url_as_identity() -> None:
    url = "https://hydrosym.paro.nl/article/1-intro"

    assert identity.canonical_url(url) == url
    assert identity.document_id(url) != identity.document_id(url + "2")


def test_embedded_video_ids_are_the_videos_document_ids() -> None:
    text = f"see https://youtu.be/{ID} and https://www.youtube.com/embed/{ID}"

    assert identity.embedded_video_ids(text) == [
        identity.document_id(f"https://www.youtube.com/watch?v={ID}")
    ]
    assert identity.embedded_video_ids("no video here") == []


def test_content_hash_is_the_full_sha256_of_the_text() -> None:
    assert len(identity.content_hash("abc")) == 64
    assert identity.content_hash("abc") != identity.content_hash("abd")


def test_a_support_chunk_fills_its_identity_fields() -> None:
    chunk = SupportChunk(
        text="hello",
        source_type=SourceType.WEB,
        source_url="https://x.example/a",
        source_name="KB",
    )

    assert chunk.document_id == identity.document_id("https://x.example/a")
    assert chunk.content_hash == identity.content_hash("hello")
    assert chunk.embedded_video_ids == []
    assert chunk.source_lastmod is None


def test_page_values_travel_from_the_web_chunk_to_the_stored_chunk() -> None:
    lastmod = datetime(2026, 3, 1, tzinfo=timezone.utc)
    web = WebChunk(
        text="t",
        source_url="https://x.example/a",
        source_name="KB",
        chunk_index=0,
        source_lastmod=lastmod,
        embedded_video_ids=["v"],
    )

    stored = SupportChunk.from_web_chunk(web)

    assert stored.source_lastmod == lastmod
    assert stored.embedded_video_ids == ["v"]


def test_chunk_page_gives_every_chunk_the_pages_videos_and_lastmod() -> None:
    lastmod = datetime(2026, 3, 1, tzinfo=timezone.utc)
    markdown = (
        "# One\n\nfirst words here\n\n"
        f"# Two\n\nwatch https://youtu.be/{ID} now\n\n# Three\n\nlast"
    )

    chunks = chunk_page(markdown, "https://x.example/a", "KB", source_lastmod=lastmod)

    assert len(chunks) == 3
    wanted = [identity.document_id(f"https://www.youtube.com/watch?v={ID}")]
    assert all(c.embedded_video_ids == wanted for c in chunks)
    assert all(c.source_lastmod == lastmod for c in chunks)
