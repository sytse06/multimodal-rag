"""Tests for LangChain model factory functions."""

from unittest.mock import patch

from multimodal_rag.models.config import AppSettings
from multimodal_rag.models.llm import create_chat_model, create_embeddings


class TestCreateChatModel:
    @patch.dict(
        "os.environ",
        {"OPENROUTER_API_KEY": "test-key", "LLM_PROVIDER": "openrouter"},
        clear=False,
    )
    def test_openrouter_returns_chatopenai(self) -> None:
        from langchain_openai import ChatOpenAI

        settings = AppSettings(openrouter_api_key="test-key", llm_provider="openrouter")
        llm = create_chat_model(settings)
        assert isinstance(llm, ChatOpenAI)

    @patch.dict(
        "os.environ",
        {"LLM_PROVIDER": "ollama"},
        clear=False,
    )
    def test_ollama_returns_chatollama(self) -> None:
        from langchain_ollama import ChatOllama

        settings = AppSettings(llm_provider="ollama")
        llm = create_chat_model(settings)
        assert isinstance(llm, ChatOllama)


class TestCreateEmbeddings:
    @patch.dict(
        "os.environ",
        {"OPENROUTER_API_KEY": "test-key", "EMBEDDING_PROVIDER": "openrouter"},
        clear=False,
    )
    def test_openrouter_returns_openaiembeddings(self) -> None:
        from langchain_openai import OpenAIEmbeddings

        settings = AppSettings(
            openrouter_api_key="test-key", embedding_provider="openrouter"
        )
        emb = create_embeddings(settings)
        assert isinstance(emb, OpenAIEmbeddings)

    @patch.dict(
        "os.environ",
        {"EMBEDDING_PROVIDER": "ollama"},
        clear=False,
    )
    def test_ollama_returns_ollamaembeddings(self) -> None:
        from langchain_ollama import OllamaEmbeddings

        settings = AppSettings(embedding_provider="ollama")
        emb = create_embeddings(settings)
        assert isinstance(emb, OllamaEmbeddings)
