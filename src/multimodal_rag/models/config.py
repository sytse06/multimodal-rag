"""Application settings via Pydantic BaseSettings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Models
    llm_model: str = "openai/gpt-4o-mini"
    embedding_model: str = "openai/text-embedding-3-small"

    # Weaviate
    weaviate_url: str = "http://localhost:8080"

    # Firecrawl
    firecrawl_api_key: str = ""

    # Ingestion
    chunk_size: int = 400
    chunk_overlap: int = 50
    top_k: int = 5

    # App
    app_env: str = "development"
    log_level: str = "DEBUG"
