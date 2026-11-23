from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled

from multimodal_rag.ingest import transcripts

URL = "https://www.youtube.com/watch?v=yWuBQik8KPs"


def snippet(text: str, start: float, duration: float = 3.0) -> Any:
    return SimpleNamespace(text=text, start=start, duration=duration)


def fake_api(monkeypatch: pytest.MonkeyPatch, manual: Any) -> None:
    listing = MagicMock()
    if isinstance(manual, Exception):
        listing.find_manually_created_transcript.side_effect = manual
    else:
        listing.find_manually_created_transcript.return_value = manual
    api = MagicMock()
    api.list.return_value = listing
    monkeypatch.setattr(transcripts, "YouTubeTranscriptApi", lambda **k: api)
    monkeypatch.setattr(transcripts, "_build_http_client", lambda cookies: None)


def captions(*snippets: Any) -> Any:
    return SimpleNamespace(fetch=lambda: list(snippets))


def test_manual_captions_become_segments_and_sound_tags_are_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_api(
        monkeypatch,
        captions(
            snippet("[Music]", 0),
            snippet("Welcome to\nHydroMan", 3),
            snippet("(applause)", 6),
        ),
    )

    segments = transcripts.manual_captions(URL)

    assert segments == [{"text": "Welcome to HydroMan", "start": 3.0, "duration": 3.0}]


@pytest.mark.parametrize(
    "error", [TranscriptsDisabled("x"), NoTranscriptFound("x", [], None)]
)
def test_a_video_without_manual_captions_gives_none(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    fake_api(monkeypatch, error)

    assert transcripts.manual_captions(URL) is None
    assert transcripts.has_manual_captions(URL) is False


def test_a_network_failure_is_not_mistaken_for_no_captions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_api(monkeypatch, ConnectionError("blocked"))

    with pytest.raises(ConnectionError):
        transcripts.manual_captions(URL)


def test_an_address_that_is_not_a_video_is_refused() -> None:
    with pytest.raises(ValueError, match="not a YouTube video"):
        transcripts.manual_captions("https://example.com/page")


def test_timestamps_parse_in_both_formats() -> None:
    assert transcripts.to_seconds("01:15") == 75.0
    assert transcripts.to_seconds("1:02:03") == 3723.0


def test_manual_captions_win_and_gemini_is_not_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_api(monkeypatch, captions(snippet("Hello", 0)))
    monkeypatch.setattr(
        transcripts, "gemini_transcript", lambda *a, **k: pytest.fail("called")
    )

    result = transcripts.transcript_for(URL, Path("v.mp4"), Path("."), "key")

    assert (result.source, result.model) == ("captions", None)


def test_gemini_is_the_fallback_and_its_model_is_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_api(monkeypatch, NoTranscriptFound("x", [], None))
    seen: dict[str, Any] = {}
    monkeypatch.setattr(transcripts, "to_mono_wav", lambda media, out: out / "a.wav")

    def gemini(wav: Path, key: str) -> transcripts.Transcript:
        seen.update(wav=wav, key=key)
        return transcripts.Transcript([], "gemini", "gemini-3.8-flash")

    monkeypatch.setattr(transcripts, "gemini_transcript", gemini)

    result = transcripts.transcript_for(URL, Path("v.mp4"), tmp_path, "secret")

    assert (result.source, result.model) == ("gemini", "gemini-3.8-flash")
    assert seen == {"wav": tmp_path / "a.wav", "key": "secret"}


def test_no_captions_and_no_gemini_key_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_api(monkeypatch, NoTranscriptFound("x", [], None))

    with pytest.raises(ValueError, match="GEMINI_API_KEY is not set"):
        transcripts.transcript_for(URL, Path("v.mp4"), Path("."), "")


def gemini_response(segments: list[tuple[str, str]], version: str = "m-1") -> Any:
    parsed = [transcripts._Segment(start=s, text=t) for s, t in segments]
    return SimpleNamespace(parsed=parsed, model_version=version)


def fake_gemini(
    monkeypatch: pytest.MonkeyPatch, response: Any, length: float
) -> MagicMock:
    client = MagicMock()
    client.models.generate_content.return_value = response
    monkeypatch.setattr(transcripts.genai, "Client", lambda api_key: client)
    monkeypatch.setattr(transcripts, "_audio_seconds", lambda wav: length)
    return client


def test_gemini_segments_get_durations_from_the_next_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = fake_gemini(
        monkeypatch, gemini_response([("00:10", "B"), ("00:00", "A")]), length=25.0
    )

    result = transcripts.gemini_transcript(Path("a.wav"), "key")

    assert result.segments == [
        {"text": "A", "start": 0.0, "duration": 10.0},
        {"text": "B", "start": 10.0, "duration": 15.0},
    ]
    assert result.model == "m-1"
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.temperature == 0


def test_an_empty_gemini_answer_fails_the_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_gemini(monkeypatch, gemini_response([("00:00", "  ")]), length=10.0)

    with pytest.raises(RuntimeError, match="returned no transcript"):
        transcripts.gemini_transcript(Path("a.wav"), "key")
