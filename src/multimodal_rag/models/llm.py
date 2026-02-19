"""LangChain model factories for provider-agnostic LLM and embedding access."""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from multimodal_rag.models.config import AppSettings


def create_chat_model(settings: AppSettings) -> BaseChatModel:
    """Create a LangChain chat model based on the configured provider."""
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=0.3,
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=SecretStr(settings.openrouter_api_key),
        base_url=settings.openrouter_base_url,
        temperature=0.3,
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
        api_key=SecretStr(settings.openrouter_api_key),
        base_url=settings.openrouter_base_url,
        temperature=0.3,
    )


def create_embeddings(settings: AppSettings) -> Embeddings:
    """Create a LangChain embeddings instance based on the configured provider."""
    if settings.embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=SecretStr(settings.openrouter_api_key),
        base_url=settings.openrouter_base_url,
    )
