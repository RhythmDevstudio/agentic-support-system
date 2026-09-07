"""Application configuration.

All configuration is environment-driven. No credential, endpoint or model ID is
ever hard-coded in application logic.

The `resolve_*` helpers implement the "auto" provider-selection contract: with an
empty environment the system boots into fully offline deterministic mode instead
of crashing, which is what lets the entire test suite run without credentials.
"""

from __future__ import annotations

import functools
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = Path(__file__).resolve().parent / "policies"


class LLMProviderName(StrEnum):
    AUTO = "auto"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    DETERMINISTIC = "deterministic"


class EmbeddingProviderName(StrEnum):
    AUTO = "auto"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    HASHING = "hashing"


class VectorStoreName(StrEnum):
    AUTO = "auto"
    PGVECTOR = "pgvector"
    SQLITE = "sqlite"


class WebSearchProviderName(StrEnum):
    AUTO = "auto"
    TAVILY = "tavily"
    FIXTURE = "fixture"
    NONE = "none"


class DomainMode(StrEnum):
    """How hard the official-source policy is enforced for web research."""

    STRICT = "strict"  # hard allowlist - non-official results are discarded
    PREFER = "prefer"  # official domains boosted, others permitted but down-ranked


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- Application ---------------------------------------------------------
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["console", "json"] = "console"

    # -- LLM -----------------------------------------------------------------
    llm_provider: LLMProviderName = LLMProviderName.AUTO
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    llm_synthesis_model: str = "gpt-5.6-sol"
    llm_classification_model: str = "gpt-5.6-terra"
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 2000
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2

    azure_openai_api_key: SecretStr | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_synthesis_deployment: str | None = None
    azure_openai_classification_deployment: str | None = None
    azure_openai_embedding_deployment: str | None = None

    # -- Embeddings ----------------------------------------------------------
    embedding_provider: EmbeddingProviderName = EmbeddingProviderName.AUTO
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 64

    # -- Storage -------------------------------------------------------------
    vector_store: VectorStoreName = VectorStoreName.AUTO
    database_url: SecretStr | None = None
    pgvector_hnsw_m: int = 16
    pgvector_hnsw_ef_construction: int = 64
    pgvector_hnsw_ef_search: int = 100
    sqlite_path: Path = Path("./data/support_agent.db")

    # -- Web research --------------------------------------------------------
    web_search_provider: WebSearchProviderName = WebSearchProviderName.AUTO
    tavily_api_key: SecretStr | None = None
    web_search_max_results: int = 6
    web_search_depth: Literal["basic", "advanced"] = "advanced"
    web_search_timeout_seconds: float = 30.0
    web_search_domain_mode: DomainMode = DomainMode.STRICT

    # -- Retrieval -----------------------------------------------------------
    retrieval_top_k: int = 8
    retrieval_min_score: float = 0.15
    retrieval_hybrid_alpha: float = 0.65
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 120

    # -- Agent budgets -------------------------------------------------------
    agent_max_iterations: int = 4
    agent_max_tool_calls: int = 12
    agent_tool_timeout_seconds: float = 45.0
    agent_total_timeout_seconds: float = 180.0

    # -- Guardrails ----------------------------------------------------------
    classification_confidence_floor: float = 0.55
    enable_injection_scanning: bool = True
    enable_citation_validation: bool = True
    require_approval_for_high_risk: bool = True

    # -- Observability -------------------------------------------------------
    tracing_provider: Literal["none", "langsmith", "otel"] = "none"
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "agentic-support-system"
    otel_exporter_otlp_endpoint: str | None = None
    redact_pii_in_logs: bool = True

    # -- API -----------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: str = "*"

    # -- Paths ---------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    policy_dir: Path = POLICY_DIR

    @field_validator(
        "openai_api_key",
        "azure_openai_api_key",
        "tavily_api_key",
        "langsmith_api_key",
        "database_url",
        mode="before",
    )
    @classmethod
    def _empty_string_is_none(cls, value: object) -> object:
        """Treat `KEY=` in a .env file as unset rather than as an empty secret."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("retrieval_hybrid_alpha")
    @classmethod
    def _validate_alpha(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("retrieval_hybrid_alpha must be between 0.0 and 1.0")
        return value

    @field_validator("classification_confidence_floor")
    @classmethod
    def _validate_floor(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("classification_confidence_floor must be between 0.0 and 1.0")
        return value

    # -- Resolution of "auto" providers --------------------------------------

    def resolve_llm_provider(self) -> LLMProviderName:
        """Pick the concrete LLM provider, degrading to offline rather than failing."""
        if self.llm_provider is not LLMProviderName.AUTO:
            return self.llm_provider
        if self.openai_api_key is not None:
            return LLMProviderName.OPENAI
        if self.azure_openai_api_key is not None and self.azure_openai_endpoint:
            return LLMProviderName.AZURE_OPENAI
        return LLMProviderName.DETERMINISTIC

    def resolve_embedding_provider(self) -> EmbeddingProviderName:
        if self.embedding_provider is not EmbeddingProviderName.AUTO:
            return self.embedding_provider
        if self.openai_api_key is not None:
            return EmbeddingProviderName.OPENAI
        if self.azure_openai_api_key is not None and self.azure_openai_embedding_deployment:
            return EmbeddingProviderName.AZURE_OPENAI
        return EmbeddingProviderName.HASHING

    def resolve_vector_store(self) -> VectorStoreName:
        """pgvector only when a DSN is configured *and* the driver is importable."""
        if self.vector_store is not VectorStoreName.AUTO:
            return self.vector_store
        if self.database_url is not None and _postgres_extras_available():
            return VectorStoreName.PGVECTOR
        return VectorStoreName.SQLITE

    def resolve_web_search_provider(self) -> WebSearchProviderName:
        if self.web_search_provider is not WebSearchProviderName.AUTO:
            return self.web_search_provider
        if self.tavily_api_key is not None and _tavily_available():
            return WebSearchProviderName.TAVILY
        return WebSearchProviderName.FIXTURE

    def is_offline_mode(self) -> bool:
        """True when no live external service is in play - useful for eval labelling."""
        return (
            self.resolve_llm_provider() is LLMProviderName.DETERMINISTIC
            and self.resolve_web_search_provider() is not WebSearchProviderName.TAVILY
        )

    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.api_cors_origins.split(",") if origin.strip()]

    def sqlite_absolute_path(self) -> Path:
        path = self.sqlite_path
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _postgres_extras_available() -> bool:
    try:
        import pgvector  # noqa: F401
        import psycopg  # noqa: F401
    except ImportError:
        return False
    return True


def _tavily_available() -> bool:
    try:
        import tavily  # noqa: F401
    except ImportError:
        return False
    return True


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Cache is cleared in tests via `reset_settings`."""
    return Settings()


def reset_settings() -> None:
    """Drop the cached settings so a test can rebuild them from a patched environment."""
    get_settings.cache_clear()
