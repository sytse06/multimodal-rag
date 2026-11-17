"""Precedence of explicit arguments, real environment variables and .env."""

from pathlib import Path

import pytest

from multimodal_rag.models.config import AppSettings

HOSTED = "https://cluster.weaviate.cloud"


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("WEAVIATE_URL", "WEAVIATE_MODE", "WEAVIATE_ADMIN_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"
    path.write_text(
        "OPENROUTER_API_KEY=test-key\n"
        "WEAVIATE_URL=http://localhost:8080\n"
        "WEAVIATE_MODE=local\n"
    )
    monkeypatch.chdir(tmp_path)
    return path


def test_dotenv_is_used_when_no_environment_variable(env_file: Path) -> None:
    settings = AppSettings()

    assert str(settings.weaviate_url).startswith("http://localhost:8080")
    assert settings.weaviate_mode == "local"


def test_environment_variable_overrides_dotenv(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WEAVIATE_URL", HOSTED)
    monkeypatch.setenv("WEAVIATE_MODE", "cloud")
    monkeypatch.setenv("WEAVIATE_ADMIN_API_KEY", "admin-key")

    settings = AppSettings()

    assert str(settings.weaviate_url).startswith(HOSTED)
    assert settings.weaviate_mode == "cloud"


def test_explicit_argument_overrides_environment_and_dotenv(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WEAVIATE_URL", HOSTED)

    settings = AppSettings(weaviate_url="http://explicit:9000")

    assert str(settings.weaviate_url).startswith("http://explicit:9000")


def test_environment_variable_not_in_dotenv_still_applies(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STATE_BUCKET", "my-bucket")

    assert AppSettings().state_bucket == "my-bucket"


def test_vision_model_in_dotenv_is_read(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VISION_MODEL", raising=False)
    env_file.write_text(env_file.read_text() + "VISION_MODEL=google/gemini-2.5-flash\n")

    assert AppSettings().vision_model == "google/gemini-2.5-flash"


def test_vision_model_environment_variable_overrides_dotenv(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file.write_text(env_file.read_text() + "VISION_MODEL=from-dotenv\n")
    monkeypatch.setenv("VISION_MODEL", "from-environment")

    assert AppSettings().vision_model == "from-environment"


def test_embedding_base_url_moves_embeddings_but_not_the_chat_endpoint(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file.write_text(
        env_file.read_text()
        + "EMBEDDING_PROVIDER=ollama\nOLLAMA_BASE_URL=http://localhost:11434\n"
        + "EMBEDDING_BASE_URL=http://127.0.0.1:11435\n"
    )
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)

    settings = AppSettings()

    assert str(settings.embeddings.base_url).startswith("http://127.0.0.1:11435")
    assert settings.ollama_base_url == "http://localhost:11434"


def test_embeddings_follow_the_ollama_url_when_no_override_is_set(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file.write_text(
        env_file.read_text()
        + "EMBEDDING_PROVIDER=ollama\nOLLAMA_BASE_URL=http://localhost:11434\n"
    )
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)

    assert str(AppSettings().embeddings.base_url).startswith("http://localhost:11434")
