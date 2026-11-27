from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import HttpUrl

from multimodal_rag.ingest import video_frames, videos
from multimodal_rag.ingest.transcripts import Transcript
from multimodal_rag.models.chunks import SupportChunk, TranscriptChunk
from multimodal_rag.models.sources import YouTubeSource

URL = "https://www.youtube.com/watch?v=yWuBQik8KPs"


def source(**extra: Any) -> YouTubeSource:
    return YouTubeSource(url=HttpUrl(URL), name="V", **extra)


def settings(gemini_key: str = "g-key") -> Any:
    s = MagicMock()
    s.credentials.gemini_api_key.get_secret_value.return_value = gemini_key
    return s


def piece(start: int = 0) -> TranscriptChunk:
    return TranscriptChunk(
        text="t",
        source_url=URL,
        source_name="V",
        start_seconds=start,
        end_seconds=start + 30,
    )


class Store:
    def __init__(self) -> None:
        self.added: list[SupportChunk] = []

    def add_chunks(self, chunks: list[SupportChunk]) -> int:
        self.added.extend(chunks)
        return len(chunks)


def ingest(store: Store, video: YouTubeSource, cfg: Any, llm: Any, **kw: bool) -> Any:
    return videos.ingest_video(store, video, cfg, llm, "", **kw)  # type: ignore[arg-type]


def fused_that_asks_for_the_transcript(
    monkeypatch: pytest.MonkeyPatch, transcript: Transcript
) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fused(**kwargs: Any) -> list[TranscriptChunk]:
        seen.update(kwargs)
        kwargs["transcribe"](Path("v.mp4"), Path("."))
        return [piece(0), piece(30)]

    monkeypatch.setattr(videos, "fetch_fused_chunks", fused)
    monkeypatch.setattr(videos, "transcript_for", lambda *a, **k: transcript)
    return seen


def test_a_spoken_video_takes_the_fused_path_and_passes_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segments: list[dict[str, float | str]] = [
        {"text": "x", "start": 0.0, "duration": 5.0}
    ]
    seen = fused_that_asks_for_the_transcript(
        monkeypatch, Transcript(segments, "captions")
    )
    store = Store()

    result = ingest(store, source(), settings(), MagicMock(), strict=True)

    assert result.chunks == 2
    assert seen["strict"] is True and seen["window_seconds"] == 30
    assert {c.document_id for c in store.added} == {store.added[0].document_id}


def test_the_run_records_where_the_transcript_came_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fused_that_asks_for_the_transcript(
        monkeypatch, Transcript([], "gemini", "gemini-3.8-flash")
    )

    result = ingest(Store(), source(), settings(), MagicMock())

    assert (result.transcript_source, result.transcript_model) == (
        "gemini",
        "gemini-3.8-flash",
    )


def test_manual_captions_are_recorded_without_a_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fused_that_asks_for_the_transcript(monkeypatch, Transcript([], "captions"))

    result = ingest(Store(), source(), settings(), None)

    assert (result.transcript_source, result.transcript_model) == ("captions", None)


def test_the_gemini_key_is_passed_to_the_transcript_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys: list[str] = []

    def fused(**kwargs: Any) -> list[TranscriptChunk]:
        kwargs["transcribe"](Path("v.mp4"), Path("."))
        return [piece()]

    def transcript_for(
        url: str, path: Path, work: Path, key: str, cookies: str
    ) -> Any:
        keys.append(key)
        return Transcript([], "captions")

    monkeypatch.setattr(videos, "fetch_fused_chunks", fused)
    monkeypatch.setattr(videos, "transcript_for", transcript_for)

    ingest(Store(), source(), settings("secret-g"), None)

    assert keys == ["secret-g"]


def test_a_silent_recording_without_vision_is_skipped_not_half_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def never(*args: Any, **kwargs: Any) -> list[TranscriptChunk]:
        pytest.fail("must not fetch frames without a vision model")

    monkeypatch.setattr(videos, "fetch_frame_chunks", never)
    store = Store()

    result = ingest(store, source(silent=True), settings(), None)

    assert result.chunks == 0 and result.transcript_source is None
    assert store.added == []


def test_a_failure_in_strict_mode_stores_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(**kwargs: Any) -> list[TranscriptChunk]:
        raise RuntimeError("frame 60s could not be described")

    monkeypatch.setattr(videos, "fetch_fused_chunks", boom)
    store = Store()

    with pytest.raises(RuntimeError, match="frame 60s"):
        ingest(store, source(), settings(), MagicMock(), strict=True)

    assert store.added == []


def fused_with_one_bad_frame(
    monkeypatch: pytest.MonkeyPatch, *, strict: bool
) -> list[TranscriptChunk]:
    monkeypatch.setattr(video_frames, "download_video", lambda *a, **k: Path("v.mp4"))
    segments = [{"start": 0, "duration": 5, "text": "hello"}]
    frames = [(Path("f0.jpg"), 0), (Path("f1.jpg"), 30)]
    monkeypatch.setattr(video_frames, "extract_keyframes", lambda *a, **k: frames)

    def describe(path: Path, llm: Any, **kwargs: Any) -> str:
        if path.name == "f1.jpg":
            raise RuntimeError("vision failed")
        return "a screen"

    monkeypatch.setattr(video_frames, "describe_frame", describe)
    return video_frames.fetch_fused_chunks(
        URL, "V", lambda video, work: segments, MagicMock(), strict=strict
    )


def test_a_failed_frame_drops_only_its_own_visual_chunk_unless_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = fused_with_one_bad_frame(monkeypatch, strict=False)

    # window 0 keeps its narration and description; window 1 had no speech and its
    # frame failed, so it yields nothing
    assert [(c.kind, c.start_seconds) for c in chunks] == [
        ("transcript", 0),
        ("visual", 0),
    ]


def test_a_failed_frame_fails_the_whole_video_when_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="vision failed"):
        fused_with_one_bad_frame(monkeypatch, strict=True)
