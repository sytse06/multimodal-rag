import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import HttpUrl

from multimodal_rag.ingest import videos
from multimodal_rag.ingest.video_frames import (
    FRAME_DESCRIPTION_WORDS,
    FRAME_PROMPT_VERSION,
    DescriptionRejected,
    describe_frame,
    description_problem,
)
from multimodal_rag.models.chunks import SupportChunk, TranscriptChunk
from multimodal_rag.models.sources import YouTubeSource

URL = "https://www.youtube.com/watch?v=yWuBQik8KPs"
GOOD = "A dialog titled Import is open and the first option is selected."


def test_a_normal_description_is_accepted() -> None:
    assert description_problem(GOOD) is None


def test_empty_text_is_rejected() -> None:
    assert description_problem("   ") == "empty"


def test_a_description_far_over_the_cap_is_rejected() -> None:
    limit = FRAME_DESCRIPTION_WORDS * 3 // 2

    at_limit = " ".join(f"w{i}" for i in range(limit))
    over = " ".join(f"w{i}" for i in range(limit + 1))

    assert description_problem(at_limit) is None
    assert "over" in str(description_problem(over))


def test_a_repetition_loop_is_rejected_even_when_short() -> None:
    loop = "a square with a camera icon. " * 5  # 30 words, 5 distinct, under the limit

    assert description_problem(loop) == "a repetition loop"


def test_a_short_text_with_repeats_is_not_called_a_loop() -> None:
    assert description_problem("Edit, then edit again.") is None


def llm_that_says(*texts: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.side_effect = [SimpleNamespace(content=t) for t in texts]
    return llm


def frame(tmp_path: Path) -> Path:
    path = tmp_path / "f.jpg"
    path.write_bytes(b"jpg")
    return path


def test_a_rejected_description_is_retried_once_and_the_second_is_used(
    tmp_path: Path,
) -> None:
    llm = llm_that_says("", GOOD)

    assert describe_frame(frame(tmp_path), llm) == GOOD
    assert llm.invoke.call_count == 2


def test_two_rejected_descriptions_raise(tmp_path: Path) -> None:
    llm = llm_that_says("", "x " * 400)

    with pytest.raises(DescriptionRejected, match="rejected twice"):
        describe_frame(frame(tmp_path), llm)
    assert llm.invoke.call_count == 2


def test_an_accepted_description_costs_one_call(tmp_path: Path) -> None:
    llm = llm_that_says(GOOD)

    describe_frame(frame(tmp_path), llm)

    assert llm.invoke.call_count == 1


def test_the_prompt_asks_for_plain_text_and_the_cap(tmp_path: Path) -> None:
    llm = llm_that_says(GOOD)

    describe_frame(frame(tmp_path), llm)

    blocks = llm.invoke.call_args.args[0][0].content
    prompt = next(b["text"] for b in blocks if b["type"] == "text")
    assert f"at most {FRAME_DESCRIPTION_WORDS} words" in prompt
    assert "No markdown" in prompt and "title bar" in prompt


def test_transcribing_a_silent_recording_is_not_judged_by_the_description_cap(
    tmp_path: Path,
) -> None:
    long_text = " ".join(f"label{i}" for i in range(500))
    llm = llm_that_says(long_text)

    assert describe_frame(frame(tmp_path), llm, transcribe_mode=True) == long_text


def chunk(text: str, kind: str = "transcript", start: int = 30) -> TranscriptChunk:
    return TranscriptChunk(
        text=text,
        source_url=URL,
        source_name="V",
        start_seconds=start,
        end_seconds=start + 30,
        kind=kind,  # type: ignore[arg-type]
    )


def test_the_visual_chunk_has_its_own_stable_id_at_the_same_moment() -> None:
    narration = SupportChunk.from_fused_chunk(chunk("[Transcript] hi"))
    visual = SupportChunk.from_visual_chunk(chunk("[Visual] a dialog", "visual"))
    again = SupportChunk.from_visual_chunk(chunk("[Visual] other words", "visual"))

    assert visual.chunk_id != narration.chunk_id
    assert visual.chunk_id == again.chunk_id
    assert (visual.source_url, visual.timestamp_seconds) == (
        narration.source_url,
        narration.timestamp_seconds,
    )
    assert visual.document_id == narration.document_id


def test_the_visual_id_differs_from_every_other_id_scheme() -> None:
    c = chunk("x")
    ids = {
        SupportChunk.from_visual_chunk(c).chunk_id,
        SupportChunk.from_fused_chunk(c).chunk_id,
        SupportChunk.from_frame_chunk(c).chunk_id,
        SupportChunk.from_transcript_chunk(c).chunk_id,
    }

    assert len(ids) == 4


def source() -> YouTubeSource:
    return YouTubeSource(url=HttpUrl(URL), name="V")


class Store:
    def __init__(self) -> None:
        self.added: list[SupportChunk] = []

    def add_chunks(self, chunks: list[SupportChunk]) -> int:
        self.added.extend(chunks)
        return len(chunks)


def settings() -> Any:
    s = MagicMock()
    s.credentials.gemini_api_key.get_secret_value.return_value = "k"
    return s


def ingest_with(
    monkeypatch: pytest.MonkeyPatch, chunks: list[TranscriptChunk], **kw: bool
) -> tuple[Store, Any]:
    monkeypatch.setattr(videos, "fetch_fused_chunks", lambda **k: chunks)
    store = Store()
    result = videos.ingest_video(store, source(), settings(), None, "", **kw)  # type: ignore[arg-type]
    return store, result


def test_ingest_builds_a_narration_chunk_and_a_visual_chunk_per_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, result = ingest_with(
        monkeypatch,
        [chunk("[Transcript] hi", "transcript"), chunk("[Visual] a dialog", "visual")],
    )

    assert result.chunks == 2
    assert [c.text for c in store.added] == ["[Transcript] hi", "[Visual] a dialog"]
    narration_id = SupportChunk.from_fused_chunk(chunk("x")).chunk_id
    visual_id = SupportChunk.from_visual_chunk(chunk("x")).chunk_id
    assert [c.chunk_id for c in store.added] == [narration_id, visual_id]


def test_the_run_record_carries_the_frame_prompt_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, result = ingest_with(monkeypatch, [chunk("[Transcript] hi")])

    assert result.frame_prompt_version == FRAME_PROMPT_VERSION


def test_a_chunk_the_embedder_would_cut_fails_a_strict_run_and_stores_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    too_long = chunk("[Transcript] " + " ".join(["word"] * 450))
    monkeypatch.setattr(videos, "fetch_fused_chunks", lambda **k: [too_long])
    store = Store()

    with pytest.raises(ValueError, match="would be cut at 400"):
        videos.ingest_video(store, source(), settings(), None, "", strict=True)  # type: ignore[arg-type]

    assert store.added == []


def test_a_chunk_the_embedder_would_cut_only_warns_when_not_strict(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    too_long = chunk("[Transcript] " + " ".join(["word"] * 450))
    caplog.set_level(logging.WARNING)

    store, result = ingest_with(monkeypatch, [too_long])

    assert result.chunks == 1 and "would be cut at 400" in caplog.text


def test_a_gemini_reply_in_content_blocks_becomes_plain_text(tmp_path: Path) -> None:
    blocks = [
        {"type": "thinking", "thinking": "private reasoning"},
        {"type": "text", "text": "A dialog "},
        {"type": "text", "text": "is open."},
    ]
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content=blocks)

    text = describe_frame(frame(tmp_path), llm)

    assert text == "A dialog is open."
    assert "type" not in text and "thinking" not in text


def test_a_plain_string_reply_is_unchanged(tmp_path: Path) -> None:
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content=GOOD)

    assert describe_frame(frame(tmp_path), llm) == GOOD


def test_a_silent_recordings_transcription_also_reads_blocks(tmp_path: Path) -> None:
    llm = MagicMock()
    blocks = [{"type": "text", "text": "Step 3"}]
    llm.invoke.return_value = SimpleNamespace(content=blocks)

    assert describe_frame(frame(tmp_path), llm, transcribe_mode=True) == "Step 3"


def test_the_image_is_sent_before_the_prompt(tmp_path: Path) -> None:
    llm = llm_that_says(GOOD)

    describe_frame(frame(tmp_path), llm)

    blocks = llm.invoke.call_args.args[0][0].content
    assert [b["type"] for b in blocks] == ["image_url", "text"]
