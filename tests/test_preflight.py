"""Tests for the read-only sync preflight."""

import time
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from multimodal_rag import preflight
from multimodal_rag.models.config import AppSettings
from multimodal_rag.preflight import (
    Check,
    Status,
    check_backup,
    check_config,
    check_cookies,
    check_keys,
    check_ollama,
    compare_collections,
    format_report,
    run_preflight,
)


def _settings(**overrides: object) -> AppSettings:
    values: dict[str, object] = {
        "_env_file": None,
        "openrouter_api_key": "or-secret-value",
        "gemini_api_key": "gem-secret-value",
        "weaviate_admin_api_key": "admin-secret-value",
    }
    values.update(overrides)
    return AppSettings(**values)  # type: ignore[arg-type]


def test_report_lists_each_check_and_counts() -> None:
    report = format_report(
        [
            Check("alpha", Status.OK, "fine"),
            Check("beta", Status.FAIL, "broken"),
            Check("gamma", Status.MANUAL, "look"),
        ]
    )

    assert "[  OK  ] alpha" in report
    assert "[ FAIL ] beta" in report
    assert "1 ok, 0 warnings, 1 failed, 1 manual" in report


def test_identical_collections_are_ok() -> None:
    assert compare_collections({"a": "1"}, {"a": "1"}).status is Status.OK


def test_differences_are_counted_by_kind() -> None:
    check = compare_collections(
        {"a": "1", "b": "2", "c": "3"}, {"a": "1", "b": "X", "d": "4"}
    )

    assert check.status is Status.WARN
    assert "1 only local, 1 only hosted, 1 changed" in check.detail


def test_keys_report_presence_and_never_the_value() -> None:
    checks = check_keys(_settings())
    report = format_report(checks)

    by_name = {c.name: c for c in checks}
    assert by_name["OPENROUTER_API_KEY"].status is Status.OK
    assert by_name["FIRECRAWL_API_KEY"].status is Status.FAIL
    assert by_name["MISTRAL_API_KEY"].status is Status.WARN
    assert "or-secret-value" not in report
    assert "admin-secret-value" not in report


def test_ollama_model_must_be_pulled() -> None:
    tags = {"models": [{"name": "nomic-embed-text:latest"}]}
    with patch.object(preflight, "_get_json", return_value=tags):
        assert check_ollama(_settings(embedding_provider="ollama")).status is Status.OK
    with patch.object(preflight, "_get_json", return_value={"models": []}):
        result = check_ollama(_settings(embedding_provider="ollama"))
        assert result.status is Status.FAIL


def test_ollama_unreachable_fails() -> None:
    with patch.object(
        preflight, "_get_json", side_effect=urllib.error.URLError("refused")
    ):
        result = check_ollama(_settings(embedding_provider="ollama"))
        assert result.status is Status.FAIL


def test_config_flags_a_vision_model_that_is_not_an_openrouter_image_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with patch.object(preflight, "_openrouter_vision_ids", return_value={"good/model"}):
        bad = {c.name: c for c in check_config(_settings(vision_model="made/up"))}
        good = {c.name: c for c in check_config(_settings(vision_model="good/model"))}

    assert bad["vision model"].status is Status.FAIL
    assert good["vision model"].status is Status.OK


def test_config_reports_duplicate_env_keys_by_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("LLM_MODEL=a\nLLM_MODEL=b\n")

    checks = {c.name: c for c in check_config(_settings())}

    assert checks[".env duplicates"].status is Status.WARN
    assert "LLM_MODEL" in checks[".env duplicates"].detail
    assert "a" not in checks[".env duplicates"].detail.replace("LLM_MODEL", "")


def _cookie_file(tmp_path: Path, expires: int) -> Path:
    path = tmp_path / "cookies.txt"
    path.write_text(f".youtube.com\tTRUE\t/\tTRUE\t{expires}\tLOGIN_INFO\tvalue\n")
    return path


def _with_cookies(path: Path):  # type: ignore[no-untyped-def]
    from contextlib import contextmanager

    @contextmanager
    def fake(settings: AppSettings):  # type: ignore[no-untyped-def]
        yield str(path)

    return patch.object(preflight, "youtube_cookies", fake)


@pytest.mark.parametrize(
    ("days", "status"),
    [(200, Status.OK), (5, Status.WARN), (-3, Status.FAIL)],
)
def test_cookie_expiry_decides_the_status(
    tmp_path: Path, days: int, status: Status
) -> None:
    path = _cookie_file(tmp_path, int(time.time()) + days * 86400 + 60)
    settings = _settings(youtube_cookies_secret="YOUTUBE_COOKIES")
    with _with_cookies(path):
        assert check_cookies(settings).status is status


def test_backup_age_is_reported() -> None:
    fresh = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y%m%dT%H%M%SZ")
    stale = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y%m%dT%H%M%SZ")
    settings = _settings(state_bucket="b", gcp_project="p")

    with patch.object(preflight, "run_gcloud", return_value=f"gs://b/snapshots/{fresh}-cloud/"):
        assert check_backup(settings).status is Status.OK
    with patch.object(preflight, "run_gcloud", return_value=f"gs://b/snapshots/{stale}-cloud/"):
        assert check_backup(settings).status is Status.WARN
    with patch.object(preflight, "run_gcloud", return_value=""):
        assert check_backup(settings).status is Status.FAIL


def test_fingerprints_change_when_text_changes() -> None:
    def collection(text: str) -> MagicMock:
        obj = SimpleNamespace(uuid="u1", properties={"text": text, "source_url": "x"})
        return MagicMock(iterator=lambda: iter([obj]))

    assert preflight.fingerprints(collection("a")) != preflight.fingerprints(
        collection("b")
    )
    assert preflight.fingerprints(collection("a")) == preflight.fingerprints(
        collection("a")
    )


def test_a_crashing_probe_becomes_a_failed_check_not_an_exception() -> None:
    def boom() -> Check:
        raise RuntimeError("gcloud exploded")

    result = preflight._safe("backup", boom)

    assert result[0].status is Status.FAIL
    assert "gcloud exploded" in result[0].detail


def test_local_only_skips_cloud_checks_and_keeps_the_manual_item() -> None:
    cloud = MagicMock(side_effect=AssertionError("cloud check must not run"))
    with (
        patch.object(preflight, "check_cookies", cloud),
        patch.object(preflight, "check_registry", cloud),
        patch.object(preflight, "check_backup", cloud),
        patch.object(preflight, "check_weaviate", cloud),
        patch.object(preflight, "check_tools", return_value=[]),
        patch.object(preflight, "check_ollama", return_value=Check("o", Status.OK, "")),
        patch.object(preflight, "check_config", return_value=[]),
    ):
        checks = run_preflight(_settings(), local_only=True)

    assert any(c.status is Status.MANUAL for c in checks)
    cloud.assert_not_called()


def test_exit_code_is_nonzero_only_when_a_check_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["preflight", "--local-only"])
    cases = (((Status.OK, Status.WARN), 0), ((Status.OK, Status.FAIL), 1))
    for statuses, code in cases:
        checks = [Check(f"c{i}", s, "") for i, s in enumerate(statuses)]
        with (
            patch.object(preflight, "AppSettings"),
            patch.object(preflight, "run_preflight", return_value=checks),
            pytest.raises(SystemExit) as exit_info,
        ):
            preflight.main()
        assert exit_info.value.code == code
    assert "ok" in capsys.readouterr().out
