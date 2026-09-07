"""Embedding providers behind one protocol.

`HashingEmbeddingProvider` is not a stub that returns noise. It is a feature-hashing
lexical embedder: deterministic, dependency-free, and it produces genuine cosine
similarity for texts that share vocabulary. That makes offline retrieval tests
meaningful rather than vacuous - a query about refunds really does rank the refund
policy chunk first. It is lexical, not semantic, so it will not match paraphrases
the way a real embedding model does; that limitation is stated in the eval output
rather than hidden.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from app.config.settings import EmbeddingProviderName, Settings, get_settings
from app.observability.logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced."""


class EmbeddingProvider(ABC):
    """Turns text into vectors. Implementations must be batch-safe and deterministic
    for identical input."""

    name: str
    dimensions: int

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query. Kept separate because some models use a
        different prefix or instruction for queries."""

    def is_offline(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Offline deterministic provider
# ---------------------------------------------------------------------------


class HashingEmbeddingProvider(EmbeddingProvider):
    """Feature-hashing embedder over word unigrams, bigrams and character n-grams.

    Character n-grams give partial robustness to morphology and to code-mixed
    Hindi/English text, which matters for the ticket corpus.
    """

    name = "hashing"

    def __init__(self, dimensions: int = 1536, char_ngram: int = 4) -> None:
        self.dimensions = dimensions
        self._char_ngram = char_ngram

    def is_offline(self) -> bool:
        return True

    @staticmethod
    def _bucket(feature: str, dimensions: int) -> tuple[int, float]:
        """Map a feature to a bucket and a stable +/-1 sign.

        The sign reduces collision bias: colliding features cancel rather than
        always reinforcing.
        """
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % dimensions, 1.0 if (value >> 63) & 1 else -1.0

    def _features(self, text: str) -> list[str]:
        tokens = _TOKEN_PATTERN.findall(text.lower())
        features: list[str] = list(tokens)
        features.extend(f"{a}_{b}" for a, b in itertools.pairwise(tokens))
        compact = "".join(tokens)
        n = self._char_ngram
        if len(compact) >= n:
            features.extend(
                f"#{compact[i : i + n]}" for i in range(0, len(compact) - n + 1, 2)
            )
        return features

    def _vector(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        features = self._features(text)
        if not features:
            return vector.tolist()

        counts: dict[str, int] = {}
        for feature in features:
            counts[feature] = counts.get(feature, 0) + 1

        for feature, count in counts.items():
            index, sign = self._bucket(feature, self.dimensions)
            # Sublinear term frequency: damps the effect of repeated boilerplate.
            vector[index] += sign * (1.0 + math.log(count))

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


# ---------------------------------------------------------------------------
# OpenAI / Azure OpenAI
# ---------------------------------------------------------------------------


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embeddings via the OpenAI embeddings API.

    `dimensions` is passed through so text-embedding-3-* can be shortened, which
    keeps the pgvector index small without changing model.
    """

    name = "openai"

    def __init__(self, settings: Settings, client: object | None = None) -> None:
        self._settings = settings
        self.dimensions = settings.embedding_dimensions
        self._model = settings.embedding_model
        self._batch_size = settings.embedding_batch_size
        self._client = client or self._build_client(settings)

    @staticmethod
    def _build_client(settings: Settings) -> object:
        provider = settings.resolve_embedding_provider()
        try:
            if provider is EmbeddingProviderName.AZURE_OPENAI:
                from openai import AsyncAzureOpenAI

                if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
                    raise EmbeddingError("Azure OpenAI endpoint and API key are required")
                return AsyncAzureOpenAI(
                    api_key=settings.azure_openai_api_key.get_secret_value(),
                    azure_endpoint=settings.azure_openai_endpoint,
                    api_version=settings.azure_openai_api_version,
                    timeout=settings.llm_timeout_seconds,
                    max_retries=settings.llm_max_retries,
                )

            from openai import AsyncOpenAI

            if not settings.openai_api_key:
                raise EmbeddingError("OPENAI_API_KEY is required for the openai embedder")
            return AsyncOpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
                base_url=settings.openai_base_url or None,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
            )
        except ImportError as exc:  # pragma: no cover - openai is a hard dependency
            raise EmbeddingError("The openai package is not installed") from exc

    def _model_name(self) -> str:
        if self._settings.resolve_embedding_provider() is EmbeddingProviderName.AZURE_OPENAI:
            return self._settings.azure_openai_embedding_deployment or self._model
        return self._model

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            # The API rejects empty strings; substitute a single space and let
            # the caller's min-score threshold discard the result.
            cleaned = [text if text.strip() else " " for text in batch]
            try:
                response = await self._client.embeddings.create(  # type: ignore[attr-defined]
                    model=self._model_name(),
                    input=cleaned,
                    dimensions=self.dimensions,
                )
            except Exception as exc:
                raise EmbeddingError(f"Embedding request failed: {exc}") from exc
            vectors.extend(item.embedding for item in response.data)
        return vectors

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts)

    async def embed_query(self, text: str) -> list[float]:
        result = await self._embed([text])
        if not result:
            raise EmbeddingError("Embedding provider returned no vector for the query")
        return result[0]


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------


def cosine_similarity(a: list[float] | NDArray[np.float32], b: list[float]) -> float:
    """Cosine similarity between two vectors, safe for zero vectors."""
    left = np.asarray(a, dtype=np.float32)
    right = np.asarray(b, dtype=np.float32)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def cosine_similarity_matrix(
    matrix: NDArray[np.float32], query: list[float]
) -> NDArray[np.float32]:
    """Cosine similarity of a query against every row of a matrix at once."""
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    query_vector = np.asarray(query, dtype=np.float32)
    query_norm = float(np.linalg.norm(query_vector))
    if query_norm == 0.0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    row_norms = np.linalg.norm(matrix, axis=1)
    row_norms[row_norms == 0.0] = 1.0
    return (matrix @ query_vector) / (row_norms * query_norm)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Construct the configured embedding provider.

    Falls back to the offline embedder rather than raising, so a missing key
    degrades the system instead of breaking it.
    """
    settings = settings or get_settings()
    resolved = settings.resolve_embedding_provider()

    if resolved is EmbeddingProviderName.HASHING:
        return HashingEmbeddingProvider(dimensions=settings.embedding_dimensions)

    try:
        return OpenAIEmbeddingProvider(settings)
    except EmbeddingError as exc:
        logger.warning(
            "embedding_provider_fallback",
            requested=str(resolved),
            reason=str(exc),
            using="hashing",
        )
        return HashingEmbeddingProvider(dimensions=settings.embedding_dimensions)
