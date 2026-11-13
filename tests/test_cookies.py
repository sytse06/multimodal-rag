"""Tests for resolving the YouTube cookies file from a local path or Secret Manager."""

import stat
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from multimodal_rag.ingest.cookies import youtube_cookies
from multimodal_rag.models import AppSettings


def _settings(**values: str) -> AppSettings:
    return AppSettings(
        _env_file=None, llm_provider="ollama", embedding_provider="ollama", **values
    )


def _completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


def test_without_a_secret_the_configured_file_is_used_without_gcloud() -> None:
    settings = _settings(youtube_cookies_file="config/cookies.txt")

    with patch("multimodal_rag.ingest.cookies.subprocess.run") as run:
        with youtube_cookies(settings) as path:
            assert path == "config/cookies.txt"

    run.assert_not_called()


def test_without_any_setting_the_path_is_empty() -> None:
    with youtube_cookies(_settings()) as path:
        assert path == ""


def test_secret_is_read_with_the_right_command() -> None:
    settings = _settings(youtube_cookies_secret="YT_COOKIES", gcp_project="my-project")

    with patch(
        "multimodal_rag.ingest.cookies.subprocess.run", return_value=_completed("data")
    ) as run:
        with youtube_cookies(settings):
            pass

    command = run.call_args.args[0]
    assert command[:5] == ["gcloud", "secrets", "versions", "access", "latest"]
    assert command[command.index("--secret") + 1] == "YT_COOKIES"
    assert command[command.index("--project") + 1] == "my-project"


def test_secret_becomes_a_private_temp_file_that_is_removed() -> None:
    settings = _settings(youtube_cookies_secret="YT_COOKIES")

    with patch(
        "multimodal_rag.ingest.cookies.subprocess.run",
        return_value=_completed("# Netscape cookie data"),
    ):
        with youtube_cookies(settings) as path:
            cookie_file = Path(path)
            assert cookie_file.read_text() == "# Netscape cookie data"
            assert stat.S_IMODE(cookie_file.stat().st_mode) == 0o600

    assert not cookie_file.exists()


def test_temp_file_is_removed_when_the_block_raises() -> None:
    settings = _settings(youtube_cookies_secret="YT_COOKIES")
    seen: list[Path] = []

    with patch(
        "multimodal_rag.ingest.cookies.subprocess.run", return_value=_completed("x")
    ):
        with pytest.raises(ValueError, match="boom"):
            with youtube_cookies(settings) as path:
                seen.append(Path(path))
                raise ValueError("boom")

    assert not seen[0].exists()


def test_secret_wins_over_a_configured_file() -> None:
    settings = _settings(
        youtube_cookies_file="config/old.txt", youtube_cookies_secret="YT_COOKIES"
    )

    with patch(
        "multimodal_rag.ingest.cookies.subprocess.run", return_value=_completed("x")
    ):
        with youtube_cookies(settings) as path:
            assert path != "config/old.txt"


def test_missing_gcloud_gives_an_actionable_error() -> None:
    settings = _settings(youtube_cookies_secret="YT_COOKIES")

    with patch(
        "multimodal_rag.ingest.cookies.subprocess.run", side_effect=FileNotFoundError
    ):
        with pytest.raises(RuntimeError, match="gcloud CLI not found"):
            with youtube_cookies(settings):
                pass


def test_failed_read_names_the_secret_and_project() -> None:
    settings = _settings(youtube_cookies_secret="YT_COOKIES", gcp_project="my-project")
    error = subprocess.CalledProcessError(1, "gcloud", stderr="PERMISSION_DENIED")

    with patch("multimodal_rag.ingest.cookies.subprocess.run", side_effect=error):
        with pytest.raises(RuntimeError, match="YT_COOKIES.*my-project.*PERMISSION"):
            with youtube_cookies(settings):
                pass


def test_ingest_run_passes_the_resolved_cookies_to_the_pipeline() -> None:
    from multimodal_rag.ingest import __main__ as ingest_main

    settings = _settings(youtube_cookies_secret="YT_COOKIES")

    with (
        patch.object(ingest_main, "AppSettings", return_value=settings),
        patch.object(ingest_main, "configure_logging"),
        patch.object(ingest_main, "_run") as run_pipeline,
        patch(
            "multimodal_rag.ingest.cookies.subprocess.run",
            return_value=_completed("x"),
        ),
    ):
        ingest_main.run()

    passed_settings, cookies_file = run_pipeline.call_args.args
    assert passed_settings is settings
    assert cookies_file.endswith(".txt") and "yt-cookies-" in cookies_file
    assert not Path(cookies_file).exists()
