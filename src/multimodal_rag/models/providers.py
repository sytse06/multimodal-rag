"""Typed chat-provider registry used by the factory and application UI."""

from collections.abc import Iterable

from multimodal_rag.models.config import (
    AppSettings,
    ChatProvider,
    ModelCapabilities,
    ModelSelection,
    ModelSpec,
)

CHAT_MODEL_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec(
        provider="openrouter",
        model="openai/gpt-6-luna",
        display_name="OpenRouter — GPT-6 Luna",
        capabilities=ModelCapabilities(reasoning=True),
    ),
    ModelSpec(
        provider="openrouter",
        model="qwen/qwen3.5-35b-a3b",
        display_name="OpenRouter — Qwen 3.5 35B",
    ),
    ModelSpec(
        provider="openrouter",
        model="deepseek/deepseek-v4.1-flash",
        display_name="OpenRouter — DeepSeek 4.1-flash",
        capabilities=ModelCapabilities(reasoning=True),
    ),
    ModelSpec(
        provider="openrouter",
        model="mistralai/mistral-small-2603",
        display_name="OpenRouter — Mistral Small 2603",
        capabilities=ModelCapabilities(vision=True),
    ),
    ModelSpec(
        provider="openai",
        model="gpt-6-luna",
        display_name="OpenAI — GPT-6 Luna",
    ),
    ModelSpec(
        provider="gemini",
        model="gemini-3.6-flash",
        display_name="Gemini — 3.6 Flash",
        capabilities=ModelCapabilities(vision=True),
    ),
    ModelSpec(
        provider="ollama",
        model="qwen3.5:9b",
        display_name="Ollama — Qwen 3.5 9B",
    ),
    ModelSpec(
        provider="ollama",
        model="nemotron-3.5-lightning:30b-mlx",
        display_name="Ollama — Nemotron 3.5 Lightning 30B MLX",
    ),
    ModelSpec(
        provider="nvidia",
        model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        display_name="NVIDIA — Nemotron 3 Nano omni 30b a3b reasoning",
        capabilities=ModelCapabilities(vision=True),
    ),
    ModelSpec(
        provider="nvidia",
        model="nvidia/nemotron-3-ultra-550b-a55b",
        display_name="NVIDIA — Nemotron 3 Ultra 550B A55B",
        capabilities=ModelCapabilities(reasoning=True),
    ),
)


def provider_available(settings: AppSettings, provider: str) -> bool:
    """Return whether a provider has enough validated configuration to use."""
    if provider == "ollama":
        return True
    if provider == settings.chat.provider:
        return bool(settings.chat.api_key.get_secret_value())
    credentials = getattr(settings.credentials, f"{provider}_api_key", None)
    return bool(credentials and credentials.get_secret_value())


def _with_active_model(
    settings: AppSettings, models: Iterable[ModelSpec]
) -> tuple[ModelSpec, ...]:
    entries = list(models)
    active = ModelSelection(provider=settings.chat.provider, model=settings.chat.model)
    if not any(
        entry.provider == active.provider and entry.model == active.model
        for entry in entries
    ):
        entries.append(
            ModelSpec(
                provider=active.provider,
                model=active.model,
                display_name=f"{active.provider.title()} — {active.model}",
            )
        )
    return tuple(entries)


def available_models(settings: AppSettings) -> tuple[ModelSpec, ...]:
    """Return registry entries for providers usable with the current settings."""
    models = (
        model
        for model in CHAT_MODEL_REGISTRY
        if provider_available(settings, model.provider)
    )
    return _with_active_model(settings, models)


def available_providers(settings: AppSettings) -> tuple[ChatProvider, ...]:
    """Return providers with enough configuration for a chat request."""
    providers: tuple[ChatProvider, ...] = (
        "openrouter",
        "openai",
        "gemini",
        "ollama",
        "nvidia",
    )
    return tuple(
        provider
        for provider in providers
        if provider_available(settings, provider)
    )


def provider_choices(settings: AppSettings) -> tuple[tuple[str, str], ...]:
    """Return Gradio-safe display/value pairs for available providers."""
    labels = {
        "openrouter": "OpenRouter",
        "openai": "OpenAI",
        "gemini": "Gemini",
        "ollama": "Ollama",
        "nvidia": "NVIDIA",
    }
    return tuple(
        (labels[provider], provider) for provider in available_providers(settings)
    )


def model_choices(
    settings: AppSettings, provider: ChatProvider | None = None
) -> tuple[tuple[str, str], ...]:
    """Return Gradio-safe display/value pairs for available chat models."""
    return tuple(
        (model.display_name, model.selection_key)
        for model in available_models(settings)
        if provider is None or model.provider == provider
    )


def selection_from_key(key: str, settings: AppSettings) -> ModelSelection:
    """Resolve a UI selection key against the validated available registry."""
    for model in available_models(settings):
        if model.selection_key == key:
            return ModelSelection(provider=model.provider, model=model.model)
    raise ValueError(f"Unknown or unavailable model selection: {key}")
