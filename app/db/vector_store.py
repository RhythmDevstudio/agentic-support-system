"""Vector store protocol.

Two implementations satisfy this interface: `PgVectorStore` (the production
target - PostgreSQL with pgvector and HNSW) and `SqliteVectorStore` (development
and CI, where no database server is available). Both are exercised by the same
conformance test suite, so switching between them is a configuration change
rather than a code change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.documents import Chunk, Document, SourceType


class MatchType(StrEnum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class SearchResult(BaseModel):
    """A retrieved chunk with its score and how it was matched."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float
    match_type: MatchType = MatchType.VECTOR
    vector_score: float | None = None
    keyword_score: float | None = None
    rank: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchFilters(BaseModel):
    """Optional metadata filters applied inside the store, not after retrieval."""

    model_config = ConfigDict(frozen=True)

    document_ids: tuple[str, ...] = ()
    source_types: tuple[SourceType, ...] = ()
    version: str | None = None

    def is_empty(self) -> bool:
        return not self.document_ids and not self.source_types and self.version is None


class VectorStoreError(RuntimeError):
    """Raised when a store operation fails."""


class VectorStore(ABC):
    """Persistence and retrieval for documents, chunks and their embeddings."""

    name: str

    @abstractmethod
    async def initialize(self) -> None:
        """Create schema if absent. Must be idempotent."""

    @abstractmethod
    async def upsert_document(self, document: Document) -> None:
        """Insert or replace a document record."""

    @abstractmethod
    async def upsert_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> int:
        """Insert or replace chunks with their embeddings. Returns rows written."""

    @abstractmethod
    async def vector_search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 8,
        filters: SearchFilters | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Nearest-neighbour search by cosine similarity."""

    @abstractmethod
    async def keyword_search(
        self,
        query_text: str,
        *,
        top_k: int = 8,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        """Lexical search. Complements vector search for exact terms, IDs and codes."""

    @abstractmethod
    async def get_document(self, document_id: str) -> Document | None: ...

    @abstractmethod
    async def list_documents(self) -> list[Document]: ...

    @abstractmethod
    async def get_chunk(self, chunk_id: str) -> Chunk | None: ...

    @abstractmethod
    async def delete_document(self, document_id: str) -> int:
        """Delete a document and its chunks. Returns chunks removed."""

    @abstractmethod
    async def count_chunks(self) -> int: ...

    @abstractmethod
    async def close(self) -> None: ...

    async def health(self) -> dict[str, Any]:
        """Lightweight status for the health endpoint."""
        try:
            chunks = await self.count_chunks()
        except Exception as exc:
            return {"store": self.name, "healthy": False, "error": str(exc)}
        return {"store": self.name, "healthy": True, "chunks": chunks}
