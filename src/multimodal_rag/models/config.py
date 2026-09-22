"""Typed application settings loaded from environment variables."""

import os
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

ChatProvider: TypeAlias = Literal["openrouter", "openai", "gemini", "ollama"]
EmbeddingProvider: TypeAlias = Literal["openrouter", "ollama"]


class ModelCapabilities(BaseModel):
    """Capabilities advertised by a registered chat model."""

    streaming: bool = True
    vision: bool = False
    reasoning: bool = False


class ModelSpec(BaseModel):
    """Provider-qualified model metadata used by the factory and UI."""

    provider: ChatProvider
    model: str
    display_name: str
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)

    @field_validator("model", "display_name")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model metadata must not be empty")
        return value.strip()

    @property
    def selection_key(self) -> str:
        return f"{self.provider}:{self.model}"


class ModelSelection(BaseModel):
    """An explicit provider/model choice made by the application or UI."""

    provider: ChatProvider
    model: str

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("selected model must not be empty")
        return value.strip()


class _ProviderConfig(BaseModel):
    """Shared provider configuration and validation."""

    model: str
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    base_url: AnyHttpUrl
    timeout: float = 60.0
    max_retries: int = 2

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be empty")
        return value.strip()

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout must be greater than zero")
        return value

    @field_validator("max_retries")
    @classmethod
    def validate_max_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("max_retries must be zero or greater")
        return value


class OpenRouterProviderConfig(_ProviderConfig):
    provider: Literal["openrouter"] = "openrouter"
    model: str = "google/gemini-3-flash-preview"
    base_url: AnyHttpUrl = AnyHttpUrl("https://openrouter.ai/api/v1")


class OpenAIProviderConfig(_ProviderConfig):
    provider: Literal["openai"] = "openai"
    model: str = "gpt-4o-mini"
    base_url: AnyHttpUrl = AnyHttpUrl("https://api.openai.com/v1")


class GeminiProviderConfig(_ProviderConfig):
    provider: Literal["gemini"] = "gemini"
    model: str = "gemini-2.5-flash"
    base_url: AnyHttpUrl = AnyHttpUrl("https://generativelanguage.googleapis.com")


class OllamaProviderConfig(_ProviderConfig):
    provider: Literal["ollama"] = "ollama"
    model: str = "llama3.2"
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:11434")


ProviderConfig: TypeAlias = Annotated[
    OpenRouterProviderConfig
    | OpenAIProviderConfig
    | GeminiProviderConfig
    | OllamaProviderConfig,
    Field(discriminator="provider"),
]


class ChatSettings(BaseModel):
    """Active chat provider and generation controls."""

    config: ProviderConfig = Field(default_factory=OpenRouterProviderConfig)
    temperature: float = 0.3

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if not 0 <= value <= 2:
            raise ValueError("temperature must be between 0 and 2")
        return value

    @model_validator(mode="after")
    def validate_credentials(self) -> "ChatSettings":
        if self.provider != "ollama" and not self.api_key.get_secret_value():
            raise ValueError(f"chat.{self.provider}.api_key is required")
        return self

    @property
    def provider(self) -> ChatProvider:
        return self.config.provider

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def api_key(self) -> SecretStr:
        return self.config.api_key

    @property
    def base_url(self) -> AnyHttpUrl:
        return self.config.base_url

    @property
    def timeout(self) -> float:
        return self.config.timeout

    @property
    def max_retries(self) -> int:
        return self.config.max_retries


class EmbeddingSettings(BaseModel):
    """Independent embedding provider configuration."""

    provider: EmbeddingProvider = "openrouter"
    model: str = "nomic-embed-text"
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    base_url: AnyHttpUrl = AnyHttpUrl("https://openrouter.ai/api/v1")
    timeout: float = 60.0
    max_retries: int = 2

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("embedding model must not be empty")
        return value.strip()

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("embedding timeout must be greater than zero")
        return value

    @field_validator("max_retries")
    @classmethod
    def validate_max_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("embedding max_retries must be zero or greater")
        return value

    @model_validator(mode="after")
    def validate_credentials(self) -> "EmbeddingSettings":
        if self.provider == "openrouter" and not self.api_key.get_secret_value():
            raise ValueError("embeddings.openrouter_api_key is required")
        return self


class InfrastructureSettings(BaseModel):
    """External services and their credentials."""

    weaviate_url: AnyHttpUrl = AnyHttpUrl("http://localhost:8080")
    firecrawl_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    mistral_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    youtube_cookies_file: str = ""


class IngestionSettings(BaseModel):
    """Chunking controls for ingestion."""

    chunk_size: int = 400
    chunk_overlap: int = 50
    top_k: int = 10

    @field_validator("chunk_size", "top_k")
    @classmethod
    def validate_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("chunk_size and top_k must be greater than zero")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, value: int) -> int:
        if value < 0:
            raise ValueError("chunk_overlap must be zero or greater")
        return value


class VisionSettings(BaseModel):
    """Optional visual grounding configuration."""

    model: str = ""


class RuntimeSettings(BaseModel):
    """Application runtime and sharing controls."""

    app_env: str = "development"
    log_level: str = "INFO"
    gradio_share: bool = False


class CredentialSettings(BaseModel):
    """Credentials for configured providers, kept separate from active models."""

    openrouter_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    openai_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    gemini_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))


class ProviderEndpointSettings(BaseModel):
    """Endpoints for providers that support custom gateways or local servers."""

    openrouter_base_url: AnyHttpUrl = AnyHttpUrl("https://openrouter.ai/api/v1")
    openai_base_url: AnyHttpUrl = AnyHttpUrl("https://api.openai.com/v1")
    gemini_base_url: AnyHttpUrl = AnyHttpUrl(
        "https://generativelanguage.googleapis.com"
    )
    ollama_base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:11434")


class AppSettings(BaseSettings):
    """Complete application configuration with legacy env compatibility."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    chat: ChatSettings = Field(default_factory=ChatSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    infrastructure: InfrastructureSettings = Field(
        default_factory=InfrastructureSettings
    )
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    credentials: CredentialSettings = Field(default_factory=CredentialSettings)
    endpoints: ProviderEndpointSettings = Field(
        default_factory=ProviderEndpointSettings
    )

    @classmethod
    def _value(cls, values: dict[str, object], name: str, default: object) -> object:
        if name in values:
            return values[name]
        if name.upper() in values:
            return values[name.upper()]
        return os.environ.get(name.upper(), default)

    @classmethod
    def _provider_payload(
        cls,
        values: dict[str, object],
        provider: object,
        model: object,
    ) -> dict[str, object]:
        provider_name = str(provider)
        key_name = f"{provider_name}_api_key"
        url_name = f"{provider_name}_base_url"
        defaults: dict[str, object] = {
            "openrouter": "https://openrouter.ai/api/v1",
            "openai": "https://api.openai.com/v1",
            "gemini": "https://generativelanguage.googleapis.com",
            "ollama": "http://localhost:11434",
        }
        if provider_name not in defaults:
            raise ValueError(f"Unsupported chat provider: {provider_name}")
        return {
            "provider": provider_name,
            "model": model,
            "api_key": cls._value(values, key_name, ""),
            "base_url": cls._value(values, url_name, defaults[provider_name]),
            "timeout": cls._value(values, "llm_timeout", 60.0),
            "max_retries": cls._value(values, "llm_max_retries", 2),
        }

    @classmethod
    def _normalise_legacy_values(cls, values: object) -> dict[str, object]:
        data = dict(values) if isinstance(values, dict) else {}

        if "chat" not in data:
            provider = cls._value(data, "llm_provider", "openrouter")
            model = cls._value(data, "llm_model", "google/gemini-3-flash-preview")
            data["chat"] = {
                "config": cls._provider_payload(data, provider, model),
                "temperature": cls._value(
                    data, "llm_temperature", cls._value(data, "temperature", 0.3)
                ),
            }
        if "embeddings" not in data:
            provider = cls._value(data, "embedding_provider", "openrouter")
            data["embeddings"] = {
                "provider": provider,
                "model": cls._value(data, "embedding_model", "nomic-embed-text"),
                "api_key": cls._value(
                    data,
                    f"{provider}_api_key",
                    cls._value(data, "openrouter_api_key", ""),
                ),
                "base_url": cls._value(
                    data,
                    f"{provider}_base_url",
                    (
                        "http://localhost:11434"
                        if provider == "ollama"
                        else cls._value(
                            data,
                            "openrouter_base_url",
                            "https://openrouter.ai/api/v1",
                        )
                    ),
                ),
                "timeout": cls._value(data, "embedding_timeout", 60.0),
                "max_retries": cls._value(data, "embedding_max_retries", 2),
            }
        if "infrastructure" not in data:
            data["infrastructure"] = {
                "weaviate_url": cls._value(
                    data, "weaviate_url", "http://localhost:8080"
                ),
                "firecrawl_api_key": cls._value(data, "firecrawl_api_key", ""),
                "mistral_api_key": cls._value(data, "mistral_api_key", ""),
                "youtube_cookies_file": cls._value(data, "youtube_cookies_file", ""),
            }
        if "ingestion" not in data:
            data["ingestion"] = {
                "chunk_size": cls._value(data, "chunk_size", 400),
                "chunk_overlap": cls._value(data, "chunk_overlap", 50),
                "top_k": cls._value(data, "top_k", 10),
            }
        if "vision" not in data:
            data["vision"] = {"model": cls._value(data, "vision_model", "")}
        if "runtime" not in data:
            data["runtime"] = {
                "app_env": cls._value(data, "app_env", "development"),
                "log_level": cls._value(data, "log_level", "INFO"),
                "gradio_share": cls._value(data, "gradio_share", False),
            }
        if "credentials" not in data:
            data["credentials"] = {
                "openrouter_api_key": cls._value(data, "openrouter_api_key", ""),
                "openai_api_key": cls._value(data, "openai_api_key", ""),
                "gemini_api_key": cls._value(data, "gemini_api_key", ""),
            }
        if "endpoints" not in data:
            data["endpoints"] = {
                "openrouter_base_url": cls._value(
                    data, "openrouter_base_url", "https://openrouter.ai/api/v1"
                ),
                "openai_base_url": cls._value(
                    data, "openai_base_url", "https://api.openai.com/v1"
                ),
                "gemini_base_url": cls._value(
                    data,
                    "gemini_base_url",
                    "https://generativelanguage.googleapis.com",
                ),
                "ollama_base_url": cls._value(
                    data, "ollama_base_url", "http://localhost:11434"
                ),
            }
        return data

    @model_validator(mode="before")
    @classmethod
    def normalise_legacy_values(cls, values: object) -> dict[str, object]:
        return cls._normalise_legacy_values(values)

    # Temporary read-only accessors keep the ingestion and store migration small.
    @property
    def llm_provider(self) -> ChatProvider:
        return self.chat.provider

    @property
    def llm_model(self) -> str:
        return self.chat.model

    @property
    def openrouter_api_key(self) -> SecretStr:
        return self.credentials.openrouter_api_key

    @property
    def openrouter_base_url(self) -> str:
        return str(self.endpoints.openrouter_base_url).rstrip("/")

    @property
    def ollama_base_url(self) -> str:
        return str(self.endpoints.ollama_base_url).rstrip("/")

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        return self.embeddings.provider

    @property
    def embedding_model(self) -> str:
        return self.embeddings.model

    @property
    def weaviate_url(self) -> str:
        return str(self.infrastructure.weaviate_url).rstrip("/")

    @property
    def firecrawl_api_key(self) -> SecretStr:
        return self.infrastructure.firecrawl_api_key

    @property
    def mistral_api_key(self) -> SecretStr:
        return self.infrastructure.mistral_api_key

    @property
    def youtube_cookies_file(self) -> str:
        return self.infrastructure.youtube_cookies_file

    @property
    def chunk_size(self) -> int:
        return self.ingestion.chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self.ingestion.chunk_overlap

    @property
    def top_k(self) -> int:
        return self.ingestion.top_k

    @property
    def vision_model(self) -> str:
        return self.vision.model

    @property
    def app_env(self) -> str:
        return self.runtime.app_env

    @property
    def log_level(self) -> str:
        return self.runtime.log_level

    @property
    def gradio_share(self) -> bool:
        return self.runtime.gradio_share
