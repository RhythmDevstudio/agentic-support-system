"""Ingestion pipeline: parse -> clean -> chunk -> embed -> index.

Ingestion is deliberately resilient: one malformed file in a directory must not
abort the batch. Failures are collected and reported per document so an operator
can see exactly what did and did not land in the index.
"""

from __future__ import annotations

from pathlib import Path

from app.config.settings import Settings, get_settings
from app.db.vector_store import VectorStore
from app.models.documents import (
    Document,
    DocumentFormat,
    IngestionResult,
    ParsedDocument,
    SourceType,
    content_hash,
)
from app.observability.logging import get_logger
from app.providers.embeddings import EmbeddingProvider
from app.rag.ingestion.chunking import chunk_document
from app.rag.ingestion.parsers import (
    SUPPORTED_EXTENSIONS,
    ParserError,
    parse_content,
    parse_file,
)

logger = get_logger(__name__)


class IngestionPipeline:
    """Turns source documents into indexed, retrievable chunks."""

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingProvider,
        settings: Settings | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._settings = settings or get_settings()

    async def _index(
        self,
        parsed: ParsedDocument,
        *,
        document_id: str,
        source_type: SourceType,
        publisher: str | None,
        version: str,
        language: str,
        source_path: str | None,
        source_url: str | None,
    ) -> IngestionResult:
        warnings: list[str] = []

        if not parsed.pages:
            return IngestionResult(
                document_id=document_id,
                title=parsed.title,
                chunks_created=0,
                chunks_embedded=0,
                source_path=source_path,
                skipped=True,
                skip_reason="No extractable text content",
            )

        document = Document(
            document_id=document_id,
            title=parsed.title,
            source_type=source_type,
            document_format=parsed.document_format,
            source_path=source_path,
            source_url=source_url,
            publisher=publisher,
            version=version,
            language=language,
            checksum=content_hash(parsed.full_text),
            page_count=parsed.metadata.get("page_count"),
            metadata=dict(parsed.metadata),
        )

        chunks = chunk_document(
            parsed,
            document,
            max_tokens=self._settings.chunk_size_tokens,
            overlap_tokens=self._settings.chunk_overlap_tokens,
        )
        if not chunks:
            return IngestionResult(
                document_id=document_id,
                title=parsed.title,
                chunks_created=0,
                chunks_embedded=0,
                source_path=source_path,
                skipped=True,
                skip_reason="Document produced no chunks",
            )

        # Re-ingesting replaces rather than duplicates: chunk IDs are content
        # addressed, so an edited document would otherwise leave orphans behind.
        existing = await self._store.get_document(document_id)
        if existing is not None:
            removed = await self._store.delete_document(document_id)
            if removed:
                warnings.append(f"Replaced {removed} previously indexed chunks")

        await self._store.upsert_document(document)

        embeddings = await self._embedder.embed_documents([chunk.content for chunk in chunks])
        written = await self._store.upsert_chunks(chunks, embeddings)

        logger.info(
            "document_ingested",
            document_id=document_id,
            title=parsed.title,
            chunks=written,
            source_type=str(source_type),
        )

        return IngestionResult(
            document_id=document_id,
            title=parsed.title,
            chunks_created=len(chunks),
            chunks_embedded=written,
            source_path=source_path,
            warnings=warnings,
        )

    async def ingest_file(
        self,
        path: Path,
        *,
        source_type: SourceType = SourceType.INTERNAL_DOCUMENT,
        publisher: str | None = None,
        version: str = "1",
        language: str = "en",
        source_url: str | None = None,
    ) -> IngestionResult:
        """Parse and index a single file."""
        parsed = parse_file(path)
        return await self._index(
            parsed,
            document_id=Document.make_id(str(path), version),
            source_type=source_type,
            publisher=publisher,
            version=version,
            language=language,
            source_path=str(path),
            source_url=source_url,
        )

    async def ingest_text(
        self,
        content: str,
        *,
        title: str,
        document_format: DocumentFormat = DocumentFormat.TEXT,
        source_type: SourceType = SourceType.INTERNAL_DOCUMENT,
        publisher: str | None = None,
        version: str = "1",
        language: str = "en",
        source_url: str | None = None,
        document_id: str | None = None,
    ) -> IngestionResult:
        """Parse and index in-memory content (API uploads, fetched pages)."""
        parsed = parse_content(
            content, document_format=document_format, title=title, source_url=source_url
        )
        return await self._index(
            parsed,
            document_id=document_id or Document.make_id(source_url or title, version),
            source_type=source_type,
            publisher=publisher,
            version=version,
            language=language,
            source_path=None,
            source_url=source_url,
        )

    async def ingest_directory(
        self,
        directory: Path,
        *,
        source_type: SourceType = SourceType.INTERNAL_DOCUMENT,
        publisher: str | None = None,
        recursive: bool = True,
    ) -> list[IngestionResult]:
        """Ingest every supported file in a directory.

        One failing file is recorded as a skipped result; the batch continues.
        """
        if not directory.exists():
            raise ParserError(f"Directory not found: {directory}")

        pattern = "**/*" if recursive else "*"
        paths = sorted(
            path
            for path in directory.glob(pattern)
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and not path.name.startswith("._")  # macOS AppleDouble sidecars
        )

        results: list[IngestionResult] = []
        for path in paths:
            try:
                results.append(
                    await self.ingest_file(
                        path, source_type=source_type, publisher=publisher
                    )
                )
            except (ParserError, OSError) as exc:
                logger.warning("ingestion_failed", path=str(path), error=str(exc))
                results.append(
                    IngestionResult(
                        document_id="",
                        title=path.name,
                        chunks_created=0,
                        chunks_embedded=0,
                        source_path=str(path),
                        skipped=True,
                        skip_reason=str(exc),
                    )
                )
        return results
