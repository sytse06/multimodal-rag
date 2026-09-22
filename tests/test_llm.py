"""Tests for LangChain model factory functions."""

from unittest.mock import patch

from multimodal_rag.models.config import AppSettings, ModelSelection
from multimodal_rag.models.llm import create_chat_model, create_embeddings


class TestCreateChatModel:
    @patch.dict(
        "os.environ",
        {"OPENROUTER_API_KEY": "test-key", "LLM_PROVIDER": "openrouter"},
        clear=False,
    )
    def test_openrouter_returns_chatopenrouter(self) -> None:
        from langchain_openrouter import ChatOpenRouter

        settings = AppSettings(
            _env_file=None,
            openrouter_api_key="test-key",
            llm_provider="openrouter",
        )
        llm = create_chat_model(settings)
        assert isinstance(llm, ChatOpenRouter)
        assert llm.client.sdk_configuration.timeout_ms == 60_000

    @patch.dict(
        "os.environ",
        {"LLM_PROVIDER": "ollama"},
        clear=False,
    )
    def test_ollama_returns_chatollama(self) -> None:
        from langchain_ollama import ChatOllama

        settings = AppSettings(
            _env_file=None,
            llm_provider="ollama",
            embedding_provider="ollama",
        )
        llm = create_chat_model(settings)
        assert isinstance(llm, ChatOllama)
        assert llm.model == "nemotron-3.5-lightning:30b-mlx"

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "test-key", "LLM_PROVIDER": "openai"},
        clear=False,
    )
    def test_openai_returns_chatopenai(self) -> None:
        from langchain_openai import ChatOpenAI

        settings = AppSettings(
            _env_file=None,
            openai_api_key="test-key",
            llm_provider="openai",
            embedding_provider="ollama",
        )
        llm = create_chat_model(settings)
        assert isinstance(llm, ChatOpenAI)

    @patch.dict(
        "os.environ",
        {"GEMINI_API_KEY": "test-key", "LLM_PROVIDER": "gemini"},
        clear=False,
    )
    def test_gemini_returns_chat_google_generative_ai(self) -> None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        settings = AppSettings(
            _env_file=None,
            gemini_api_key="test-key",
            llm_provider="gemini",
            embedding_provider="ollama",
        )
        llm = create_chat_model(settings)
        assert isinstance(llm, ChatGoogleGenerativeAI)

    @patch.dict(
        "os.environ",
        {
            "OPENROUTER_API_KEY": "router-key",
            "GEMINI_API_KEY": "gemini-key",
            "LLM_PROVIDER": "openrouter",
        },
        clear=False,
    )
    def test_explicit_selection_does_not_infer_provider_from_model(self) -> None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        settings = AppSettings(
            _env_file=None,
            openrouter_api_key="router-key",
            llm_provider="openrouter",
            gemini_api_key="gemini-key",
        )
        llm = create_chat_model(
            settings,
            ModelSelection(provider="gemini", model="gemini-3.6-flash"),
        )
        assert isinstance(llm, ChatGoogleGenerativeAI)


class TestCreateEmbeddings:
    @patch.dict(
        "os.environ",
        {"OPENROUTER_API_KEY": "test-key", "EMBEDDING_PROVIDER": "openrouter"},
        clear=False,
    )
    def test_openrouter_returns_openaiembeddings(self) -> None:
        from langchain_openai import OpenAIEmbeddings

        settings = AppSettings(
            _env_file=None,
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

        settings = AppSettings(
            _env_file=None,
            llm_provider="ollama",
            embedding_provider="ollama",
        )
        emb = create_embeddings(settings)
        assert isinstance(emb, OllamaEmbeddings)
