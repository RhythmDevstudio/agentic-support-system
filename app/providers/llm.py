"""LLM providers behind one protocol.

Every call returns a validated Pydantic object. Structured outputs use the
current OpenAI API surface: `client.responses.parse(..., text_format=Model)` with
the result read from `response.output_parsed`, and safety refusals detected as a
`refusal` content item rather than being mistaken for a malformed answer.

Model IDs are never hard-coded here; they come from settings.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from app.config.settings import LLMProviderName, Settings, get_settings
from app.observability.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when a model call cannot produce a valid result."""


class LLMRefusalError(LLMError):
    """Raised when the model declines the request for safety reasons."""


@dataclass(frozen=True)
class LLMUsage:
    """Token accounting. Zeros when the provider does not report usage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: LLMUsage) -> LLMUsage:
        return LLMUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass(frozen=True)
class LLMResult[TValue: BaseModel]:
    """A parsed model response plus the observability metadata around it."""

    value: TValue
    model: str
    provider: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    latency_ms: float = 0.0
    call_name: str = ""


class ModelRole:
    """Which configured model a call should use."""

    SYNTHESIS = "synthesis"
    CLASSIFICATION = "classification"


class LLMProvider(ABC):
    """Produces validated structured output from a prompt."""

    name: str

    @abstractmethod
    async def parse(
        self,
        *,
        instructions: str,
        input_text: str,
        schema: type[T],
        role: str = ModelRole.CLASSIFICATION,
        temperature: float | None = None,
        call_name: str = "",
    ) -> LLMResult[T]:
        """Call the model and return an instance of `schema`."""

    def is_offline(self) -> bool:
        return False

    def model_for_role(self, role: str) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OpenAI / Azure OpenAI
# ---------------------------------------------------------------------------


class OpenAIProvider(LLMProvider):
    """Structured outputs via the OpenAI Responses API."""

    name = "openai"

    def __init__(self, settings: Settings, client: Any = None) -> None:
        self._settings = settings
        self._client = client if client is not None else self._build_client(settings)

    @staticmethod
    def _build_client(settings: Settings) -> Any:
        try:
            if settings.resolve_llm_provider() is LLMProviderName.AZURE_OPENAI:
                from openai import AsyncAzureOpenAI

                if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
                    raise LLMError(
                        "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are required "
                        "for the azure_openai provider"
                    )
                return AsyncAzureOpenAI(
                    api_key=settings.azure_openai_api_key.get_secret_value(),
                    azure_endpoint=settings.azure_openai_endpoint,
                    api_version=settings.azure_openai_api_version,
                    timeout=settings.llm_timeout_seconds,
                    max_retries=settings.llm_max_retries,
                )

            from openai import AsyncOpenAI

            if not settings.openai_api_key:
                raise LLMError("OPENAI_API_KEY is required for the openai provider")
            return AsyncOpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
                base_url=settings.openai_base_url or None,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
            )
        except ImportError as exc:  # pragma: no cover - openai is a hard dependency
            raise LLMError("The openai package is not installed") from exc

    def model_for_role(self, role: str) -> str:
        settings = self._settings
        if settings.resolve_llm_provider() is LLMProviderName.AZURE_OPENAI:
            # Azure addresses deployments, not model names.
            deployment = (
                settings.azure_openai_synthesis_deployment
                if role == ModelRole.SYNTHESIS
                else settings.azure_openai_classification_deployment
            )
            if not deployment:
                raise LLMError(f"No Azure OpenAI deployment configured for role '{role}'")
            return deployment
        return (
            settings.llm_synthesis_model
            if role == ModelRole.SYNTHESIS
            else settings.llm_classification_model
        )

    @staticmethod
    def _extract_refusal(response: Any) -> str | None:
        """Find a safety refusal, which arrives as a content item rather than an error."""
        for output in getattr(response, "output", None) or []:
            if getattr(output, "type", None) != "message":
                continue
            for item in getattr(output, "content", None) or []:
                if getattr(item, "type", None) == "refusal":
                    return str(getattr(item, "refusal", "")) or "Model refused the request"
        return None

    @staticmethod
    def _extract_usage(response: Any) -> LLMUsage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return LLMUsage()
        return LLMUsage(
            prompt_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        )

    async def parse(
        self,
        *,
        instructions: str,
        input_text: str,
        schema: type[T],
        role: str = ModelRole.CLASSIFICATION,
        temperature: float | None = None,
        call_name: str = "",
    ) -> LLMResult[T]:
        model = self.model_for_role(role)
        started = time.perf_counter()

        request: dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": input_text},
            ],
            "text_format": schema,
        }
        effective_temperature = (
            temperature if temperature is not None else self._settings.llm_temperature
        )
        if effective_temperature is not None:
            request["temperature"] = effective_temperature
        if self._settings.llm_max_output_tokens:
            request["max_output_tokens"] = self._settings.llm_max_output_tokens

        try:
            response = await self._client.responses.parse(**request)
        except Exception as exc:
            # Some models reject `temperature`; retry once without it rather than
            # failing the whole run over an unsupported sampling parameter.
            if "temperature" in str(exc).lower() and "temperature" in request:
                logger.debug("retrying_without_temperature", model=model, call=call_name)
                request.pop("temperature")
                try:
                    response = await self._client.responses.parse(**request)
                except Exception as retry_exc:
                    raise LLMError(f"Model call '{call_name}' failed: {retry_exc}") from retry_exc
            else:
                raise LLMError(f"Model call '{call_name}' failed: {exc}") from exc

        refusal = self._extract_refusal(response)
        if refusal:
            raise LLMRefusalError(refusal)

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise LLMError(
                f"Model call '{call_name}' returned no parsed output matching {schema.__name__}"
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        usage = self._extract_usage(response)
        logger.info(
            "llm_call_complete",
            call=call_name,
            model=model,
            latency_ms=round(elapsed_ms, 1),
            total_tokens=usage.total_tokens,
        )
        return LLMResult(
            value=parsed,
            model=model,
            provider=self.name,
            usage=usage,
            latency_ms=elapsed_ms,
            call_name=call_name,
        )


class AzureOpenAIProvider(OpenAIProvider):
    """Azure OpenAI. Same API surface, deployment-addressed models."""

    name = "azure_openai"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Construct the configured LLM provider, degrading to offline if unavailable."""
    settings = settings or get_settings()
    resolved = settings.resolve_llm_provider()

    if resolved is LLMProviderName.DETERMINISTIC:
        from app.providers.deterministic_llm import DeterministicLLMProvider

        return DeterministicLLMProvider(settings)

    try:
        if resolved is LLMProviderName.AZURE_OPENAI:
            return AzureOpenAIProvider(settings)
        return OpenAIProvider(settings)
    except LLMError as exc:
        from app.providers.deterministic_llm import DeterministicLLMProvider

        logger.warning(
            "llm_provider_fallback",
            requested=str(resolved),
            reason=str(exc),
            using="deterministic",
        )
        return DeterministicLLMProvider(settings)
