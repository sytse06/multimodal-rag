"""Tests for the typed provider/model registry."""

from multimodal_rag.models.config import AppSettings
from multimodal_rag.models.providers import (
    available_models,
    model_choices,
    provider_available,
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
