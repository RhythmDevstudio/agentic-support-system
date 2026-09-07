"""Shared test fixtures.

The suite runs with no credentials and no Postgres. `_isolated_environment`
strips any real keys out of the environment so a developer with a populated
`.env` gets the same deterministic results as CI.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.config import policies as policies_module
from app.config.settings import Settings, reset_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Credentials that must not leak into a test run.
_CREDENTIAL_VARS = (
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "TAVILY_API_KEY",
    "LANGSMITH_API_KEY",
    "DATABASE_URL",
)


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for var in _CREDENTIAL_VARS:
        monkeypatch.delenv(var, raising=False)
    # Force provider selection down the offline path regardless of local .env.
    monkeypatch.setenv("LLM_PROVIDER", "deterministic")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing")
    monkeypatch.setenv("VECTOR_STORE", "sqlite")
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "fixture")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    reset_settings()
    policies_module.reset_policies()
    yield
    reset_settings()
    policies_module.reset_policies()


@pytest.fixture
def settings() -> Settings:
    """Settings built from the isolated test environment."""
    # _env_file=None so a developer's local .env cannot influence assertions.
    return Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def vendor_registry() -> policies_module.VendorRegistry:
    return policies_module.load_vendor_registry()


@pytest.fixture
def routing_policy() -> policies_module.RoutingPolicy:
    return policies_module.load_routing_policy()


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Drop the forced provider overrides so a test can exercise `auto` resolution."""
    for var in (
        "LLM_PROVIDER",
        "EMBEDDING_PROVIDER",
        "VECTOR_STORE",
        "WEB_SEARCH_PROVIDER",
        *_CREDENTIAL_VARS,
    ):
        monkeypatch.delenv(var, raising=False)
    os.environ.pop("LLM_PROVIDER", None)
    reset_settings()
    return monkeypatch
