"""LangChain model factories for provider-agnostic LLM and embedding access."""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from multimodal_rag.models.config import AppSettings


def create_chat_model(settings: AppSettings) -> BaseChatModel:
    """Create a LangChain chat model based on the configured provider."""
    if settings.chat.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.chat.model,
            base_url=str(settings.chat.base_url),
            temperature=settings.chat.temperature,
        )
    if settings.chat.provider != "openrouter":
        raise ValueError(
            f"Chat provider '{settings.chat.provider}' is configured but not yet"
            " supported by the current model factory"
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.chat.model,
        api_key=settings.chat.api_key,
        base_url=str(settings.chat.base_url),
        temperature=settings.chat.temperature,
    )


def create_vision_llm(settings: AppSettings) -> BaseChatModel:
    """Create a vision-capable LangChain chat model via OpenRouter.

    Always uses OpenRouter (no Ollama vision path).
    Raises ValueError if vision_model is not configured.
    """
    if not settings.vision_model:
        raise ValueError("vision_model is not configured in settings")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
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
