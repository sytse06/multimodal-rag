"""Application settings via Pydantic BaseSettings."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Provider selection
    llm_provider: Literal["openrouter", "ollama"] = "openrouter"
    embedding_provider: Literal["openrouter", "ollama"] = "openrouter"

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # Models
    llm_model: str = "google/gemini-3-flash-preview"
    embedding_model: str = "nomic-embed-text-16k"

    # Weaviate
    weaviate_url: str = "http://localhost:8080"

    # Firecrawl
    firecrawl_api_key: str = ""

    # Mistral (Voxtral audio transcription fallback)
    mistral_api_key: str = ""

    # Ingestion
    chunk_size: int = 400
    chunk_overlap: int = 50
    top_k: int = 5

    # App
    app_env: str = "development"
    log_level: str = "DEBUG"
