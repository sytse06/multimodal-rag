"""LangChain model factories for provider-agnostic LLM and embedding access."""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from multimodal_rag.models.config import AppSettings, ModelSelection


def _provider_credentials(settings: AppSettings, provider: str) -> SecretStr | None:
    if provider == "ollama":
        return None
    credentials = (
        settings.chat.api_key
        if provider == settings.chat.provider
        else getattr(settings.credentials, f"{provider}_api_key")
    )
    if not isinstance(credentials, SecretStr):
        raise ValueError(f"Unsupported chat provider: {provider}")
    if not credentials.get_secret_value():
        raise ValueError(f"chat.{provider}.api_key is required")
    return credentials


def _provider_endpoint(settings: AppSettings, provider: str) -> str:
    if provider == settings.chat.provider:
        return str(settings.chat.base_url)
    return str(getattr(settings.endpoints, f"{provider}_base_url"))


def create_chat_model(
    settings: AppSettings, selection: ModelSelection | None = None
) -> BaseChatModel:
    """Create a LangChain chat model from an explicit provider/model selection."""
    selection = selection or ModelSelection(
        provider=settings.chat.provider,
        model=settings.chat.model,
    )
    provider = selection.provider
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=selection.model,
            base_url=_provider_endpoint(settings, provider),
            temperature=settings.chat.temperature,
        )
    api_key = _provider_credentials(settings, provider)
    endpoint = _provider_endpoint(settings, provider)
    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(
            model=selection.model,
            api_key=api_key,
            base_url=endpoint,
            temperature=settings.chat.temperature,
            timeout=int(settings.chat.timeout * 1000),
            max_retries=settings.chat.max_retries,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=selection.model,
            api_key=api_key,
            base_url=endpoint,
            temperature=settings.chat.temperature,
            timeout=settings.chat.timeout,
            max_retries=settings.chat.max_retries,
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=selection.model,
            api_key=api_key,
            client_options={"api_endpoint": endpoint},
            temperature=settings.chat.temperature,
            request_timeout=settings.chat.timeout,
            retries=settings.chat.max_retries,
        )
    raise ValueError(f"Unsupported chat provider: {provider}")


def create_vision_llm(settings: AppSettings) -> BaseChatModel:
    """Create a vision-capable LangChain chat model via OpenRouter.

    Always uses OpenRouter (no Ollama vision path).
    Raises ValueError if vision_model is not configured.
    """
    if not settings.vision_model:
        raise ValueError("vision_model is not configured in settings")
    from langchain_openrouter import ChatOpenRouter

    return ChatOpenRouter(
        model=settings.vision_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=0.3,
    )


def create_embeddings(settings: AppSettings) -> Embeddings:
    """Create a LangChain embeddings instance based on the configured provider."""
    if settings.embeddings.provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.embeddings.model,
            base_url=str(settings.embeddings.base_url),
        )
    if settings.embeddings.provider != "openrouter":
        raise ValueError(
            f"Embedding provider '{settings.embeddings.provider}' is configured but"
            " not yet supported by the current model factory"
        )
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embeddings.model,
        api_key=settings.embeddings.api_key,
        base_url=str(settings.embeddings.base_url),
    )
