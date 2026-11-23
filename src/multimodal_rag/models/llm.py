"""LangChain model factories for provider-agnostic LLM and embedding access."""

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from multimodal_rag.models.config import (
    AppSettings,
    ModelSelection,
    NvidiaProviderConfig,
)


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

        # gpt-6 models reject any non-default temperature with a 400.
        sampling: dict[str, Any] = (
            {}
            if selection.model.startswith("gpt-6")
            else {"temperature": settings.chat.temperature}
        )
        # langchain-openai only requests streamed usage by default when no base_url is
        # passed; we always pass one, so ask for it explicitly.
        return ChatOpenAI(
            model=selection.model,
            api_key=api_key,
            base_url=endpoint,
            timeout=settings.chat.timeout,
            max_retries=settings.chat.max_retries,
            stream_usage=True,
            **sampling,
        )
    if provider == "nvidia":
        from langchain_openai import ChatOpenAI

        nvidia_config = (
            settings.chat.config
            if isinstance(settings.chat.config, NvidiaProviderConfig)
            else NvidiaProviderConfig()
        )
        return ChatOpenAI(
            model=selection.model,
            api_key=api_key,
            base_url=endpoint,
            temperature=settings.chat.temperature,
            timeout=settings.chat.timeout,
            max_retries=settings.chat.max_retries,
            extra_body={
                **nvidia_config.extra_body,
                "max_tokens": nvidia_config.max_tokens,
            },
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=selection.model,
            api_key=api_key,
            base_url=endpoint,
            temperature=settings.chat.temperature,
            request_timeout=settings.chat.timeout,
            retries=settings.chat.max_retries,
        )
    raise ValueError(f"Unsupported chat provider: {provider}")


def message_text(message: Any) -> str:
    """The text of a chat reply.

    OpenRouter returns a string; Gemini 3 returns a list of content blocks, and
    ``str()`` of that list would put ``[{'type': 'text', ...}]`` into a stored
    description. Only text blocks count: thinking blocks are left out.
    """
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block
        if isinstance(block, str)
        else str(block.get("text", ""))
        for block in content
        if isinstance(block, str)
        or (isinstance(block, dict) and block.get("type") == "text")
    )


GEMINI_VISION_PREFIX = "gemini:"
# Room for the answer plus the model's own reasoning: Gemini 3 counts thinking tokens
# against the output limit, and the measured descriptions needed this much headroom.
GEMINI_VISION_MAX_OUTPUT_TOKENS = 600


def create_vision_llm(settings: AppSettings) -> BaseChatModel:
    """Create a vision-capable LangChain chat model.

    ``VISION_MODEL=gemini:<model>`` calls Gemini directly with the settings the frame
    prompt was measured with (temperature 0, low thinking, a capped output); any other
    value is an OpenRouter model id and goes through OpenRouter as before.
    Raises ValueError if vision_model is not configured.
    """
    if not settings.vision_model:
        raise ValueError("vision_model is not configured in settings")
    if settings.vision_model.startswith(GEMINI_VISION_PREFIX):
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = settings.credentials.gemini_api_key
        if not key.get_secret_value():
            raise ValueError("GEMINI_API_KEY is required for a gemini: vision model")
        return ChatGoogleGenerativeAI(
            model=settings.vision_model.removeprefix(GEMINI_VISION_PREFIX),
            api_key=key,
            temperature=0,
            thinking_level="low",
            max_output_tokens=GEMINI_VISION_MAX_OUTPUT_TOKENS,
        )
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
