"""Tests for the typed provider/model registry."""

from multimodal_rag.models.config import AppSettings
from multimodal_rag.models.providers import (
    available_models,
    available_providers,
    model_choices,
    provider_available,
    provider_choices,
    selection_from_key,
)


def test_available_models_only_include_configured_providers() -> None:
    settings = AppSettings(
        _env_file=None,
        openrouter_api_key="router-key",
        llm_provider="openrouter",
    )

    providers = {model.provider for model in available_models(settings)}

    assert providers == {"openrouter", "ollama"}


def test_active_model_is_added_to_registry() -> None:
    settings = AppSettings(
        _env_file=None,
        openrouter_api_key="router-key",
        llm_provider="openrouter",
        llm_model="custom/model",
    )

    models = {(model.provider, model.model) for model in available_models(settings)}

    assert ("openrouter", "custom/model") in models


def test_model_choices_keep_provider_explicit() -> None:
    settings = AppSettings(
        _env_file=None,
        openrouter_api_key="router-key",
        llm_provider="openrouter",
    )

    choices = model_choices(settings)

    assert all(":" in value for _, value in choices)
    assert selection_from_key(choices[0][1], settings).provider == "openrouter"


def test_provider_choices_only_include_configured_providers() -> None:
    settings = AppSettings(
        _env_file=None,
        openrouter_api_key="router-key",
        llm_provider="openrouter",
    )

    assert available_providers(settings) == ("openrouter", "ollama")
    assert provider_choices(settings) == (
        ("OpenRouter", "openrouter"),
        ("Ollama", "ollama"),
    )


def test_model_choices_can_be_filtered_by_provider() -> None:
    settings = AppSettings(
        _env_file=None,
        openrouter_api_key="router-key",
        openai_api_key="openai-key",
        llm_provider="openrouter",
    )

    choices = model_choices(settings, provider="openai")

    assert choices == (("OpenAI — GPT-4o Mini", "openai:gpt-4o-mini"),)


def test_provider_availability_never_exposes_credentials() -> None:
    settings = AppSettings(
        _env_file=None,
        openrouter_api_key="super-secret",
        llm_provider="openrouter",
    )

    assert provider_available(settings, "openrouter")
    assert all(
        "super-secret" not in repr(model) for model in available_models(settings)
    )
