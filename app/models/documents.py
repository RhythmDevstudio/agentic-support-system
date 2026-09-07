"""Document and chunk domain models.

Metadata is preserved end to end, from the parsed source file through to the
citation shown to a user. A chunk that cannot say which document, page and
section it came from cannot be cited, so these fields are carried everywhere
rather than reconstructed later.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SourceType(StrEnum):
    """Where a piece of knowledge came from. Drives citation rendering and trust."""

    INTERNAL_DOCUMENT = "internal_document"
    OFFICIAL_DOCUMENTATION = "official_documentation"
    OFFICIAL_API_REFERENCE = "official_api_reference"
    OFFICIAL_REPOSITORY = "official_repository"
    AUTHORITATIVE_SECONDARY = "authoritative_secondary"
    COMMUNITY = "community"

    @property
    def is_internal(self) -> bool:
        return self is SourceType.INTERNAL_DOCUMENT

    @property
    def is_official(self) -> bool:
        return self in {
            SourceType.OFFICIAL_DOCUMENTATION,
            SourceType.OFFICIAL_API_REFERENCE,
            SourceType.OFFICIAL_REPOSITORY,
        }


class DocumentFormat(StrEnum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def content_hash(text: str) -> str:
    """Stable content fingerprint, used for dedupe and for deterministic IDs."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Document(BaseModel):
    """A source document before chunking."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    title: str
    source_type: SourceType = SourceType.INTERNAL_DOCUMENT
    document_format: DocumentFormat = DocumentFormat.TEXT
    source_path: str | None = None
    source_url: str | None = None
    publisher: str | None = None
    version: str = "1"
    language: str = "en"
    checksum: str = ""
    page_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    ingested_at: datetime = Field(default_factory=_utcnow)

    @classmethod
    def make_id(cls, source: str, version: str = "1") -> str:
        """Deterministic document ID so re-ingesting the same file replaces it."""
        return hashlib.sha256(f"{source}::{version}".encode()).hexdigest()[:16]


class Chunk(BaseModel):
    """A retrievable unit of a document, carrying full citation provenance."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    content: str
    ordinal: int

    # Provenance - all optional because not every format supplies every field,
    # but whatever the parser knows is preserved.
    document_title: str = ""
    source_type: SourceType = SourceType.INTERNAL_DOCUMENT
    source_path: str | None = None
    source_url: str | None = None
    publisher: str | None = None
    page: int | None = None
    section: str | None = None
    heading_path: tuple[str, ...] = ()
    version: str = "1"

    token_estimate: int = 0
    checksum: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def make_id(cls, document_id: str, ordinal: int, text: str) -> str:
        """Chunk IDs are content-addressed: identical re-ingest yields identical IDs."""
        digest = hashlib.sha256(f"{document_id}:{ordinal}:{text}".encode()).hexdigest()
        return f"{document_id}-{ordinal:04d}-{digest[:8]}"

    def citation_label(self) -> str:
        """Human-readable location within the document, e.g. 'p. 4, Refund Windows'."""
        parts: list[str] = []
        if self.page is not None:
            parts.append(f"p. {self.page}")
        if self.section:
            parts.append(self.section)
        elif self.heading_path:
            parts.append(" > ".join(self.heading_path))
        return ", ".join(parts)


class ParsedPage(BaseModel):
    """One page or logical block emitted by a parser, before chunking."""

    model_config = ConfigDict(frozen=True)

    text: str
    page: int | None = None
    heading_path: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Parser output: normalised text blocks plus whatever metadata was recoverable."""

    model_config = ConfigDict(frozen=True)

    title: str
    document_format: DocumentFormat
    pages: tuple[ParsedPage, ...]
    source_path: str | None = None
    source_url: str | None = None
    publisher: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)


class IngestionResult(BaseModel):
    """Outcome of ingesting one document - reported back through the API."""

    document_id: str
    title: str
    chunks_created: int
    chunks_embedded: int
    source_path: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
