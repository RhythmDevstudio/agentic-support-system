"""PostgreSQL + pgvector store - the production target.

Vector search uses an HNSW index over cosine distance; keyword search uses
PostgreSQL full-text search with `ts_rank_cd`. Both are combined by the same
hybrid fusion code that serves the SQLite store, so retrieval behaviour is
consistent across backends.

`psycopg` and `pgvector` are optional dependencies (`uv sync --extra postgres`),
so every import is deferred and a missing driver produces a clear error rather
than an ImportError at module load.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from app.db.vector_store import (
    MatchType,
    SearchFilters,
    SearchResult,
    VectorStore,
    VectorStoreError,
)
from app.models.documents import Chunk, Document, DocumentFormat, SourceType
from app.observability.logging import get_logger

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

logger = get_logger(__name__)


def _require_drivers() -> None:
    try:
        import pgvector  # noqa: F401
        import psycopg  # noqa: F401
        import psycopg_pool  # noqa: F401
    except ImportError as exc:
        raise VectorStoreError(
            "PostgreSQL support requires the 'postgres' extra. "
            "Install it with: uv pip install -e '.[postgres]'"
        ) from exc


class PgVectorStore(VectorStore):
    name = "pgvector"

    def __init__(
        self,
        dsn: str,
        *,
        dimensions: int = 1536,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
        hnsw_ef_search: int = 100,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ) -> None:
        _require_drivers()
        self._dsn = dsn
        self._dimensions = dimensions
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._hnsw_ef_search = hnsw_ef_search
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pool: AsyncConnectionPool | None = None

    # -- lifecycle -----------------------------------------------------------

    async def _get_pool(self) -> AsyncConnectionPool:
        if self._pool is not None:
            return self._pool

        from pgvector.psycopg import register_vector_async
        from psycopg_pool import AsyncConnectionPool

        async def configure(connection: Any) -> None:
            # Vector adaptation must be registered per connection.
            await register_vector_async(connection)

        pool = AsyncConnectionPool(
            self._dsn,
            min_size=self._min_pool_size,
            max_size=self._max_pool_size,
            configure=configure,
            open=False,
        )
        await pool.open(wait=True)
        self._pool = pool
        return pool

    async def initialize(self) -> None:
        """Create the extension, tables and indexes. Idempotent."""
        pool = await self._get_pool()
        async with pool.connection() as connection:
            await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            for statement in self._schema_statements():
                await connection.execute(statement)
        logger.info("pgvector_initialized", dimensions=self._dimensions)

    def _schema_statements(self) -> list[str]:
        return [
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id     TEXT PRIMARY KEY,
                title           TEXT NOT NULL,
                source_type     TEXT NOT NULL,
                document_format TEXT NOT NULL,
                source_path     TEXT,
                source_url      TEXT,
                publisher       TEXT,
                version         TEXT NOT NULL DEFAULT '1',
                language        TEXT NOT NULL DEFAULT 'en',
                checksum        TEXT NOT NULL DEFAULT '',
                page_count      INTEGER,
                metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
                ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id        TEXT PRIMARY KEY,
                document_id     TEXT NOT NULL
                                REFERENCES documents(document_id) ON DELETE CASCADE,
                content         TEXT NOT NULL,
                ordinal         INTEGER NOT NULL,
                document_title  TEXT NOT NULL DEFAULT '',
                source_type     TEXT NOT NULL,
                source_path     TEXT,
                source_url      TEXT,
                publisher       TEXT,
                page            INTEGER,
                section         TEXT,
                heading_path    JSONB NOT NULL DEFAULT '[]'::jsonb,
                version         TEXT NOT NULL DEFAULT '1',
                token_estimate  INTEGER NOT NULL DEFAULT 0,
                checksum        TEXT NOT NULL DEFAULT '',
                metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
                content_tsv     TSVECTOR GENERATED ALWAYS AS
                                (to_tsvector('english', content)) STORED
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                chunk_id   TEXT PRIMARY KEY
                           REFERENCES document_chunks(chunk_id) ON DELETE CASCADE,
                dimensions INTEGER NOT NULL,
                embedding  vector({self._dimensions}) NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_source_type ON document_chunks(source_type)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON document_chunks USING GIN(content_tsv)",
            f"""
            CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
            ON chunk_embeddings USING hnsw (embedding vector_cosine_ops)
            WITH (m = {self._hnsw_m}, ef_construction = {self._hnsw_ef_construction})
            """,
        ]

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # -- writes --------------------------------------------------------------

    async def upsert_document(self, document: Document) -> None:
        pool = await self._get_pool()
        async with pool.connection() as connection:
            await connection.execute(
                """
                INSERT INTO documents (
                    document_id, title, source_type, document_format, source_path,
                    source_url, publisher, version, language, checksum, page_count,
                    metadata, ingested_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (document_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    source_type = EXCLUDED.source_type,
                    document_format = EXCLUDED.document_format,
                    source_path = EXCLUDED.source_path,
                    source_url = EXCLUDED.source_url,
                    publisher = EXCLUDED.publisher,
                    version = EXCLUDED.version,
                    language = EXCLUDED.language,
                    checksum = EXCLUDED.checksum,
                    page_count = EXCLUDED.page_count,
                    metadata = EXCLUDED.metadata,
                    ingested_at = EXCLUDED.ingested_at
                """,
                (
                    document.document_id,
                    document.title,
                    str(document.source_type),
                    str(document.document_format),
                    document.source_path,
                    document.source_url,
                    document.publisher,
                    document.version,
                    document.language,
                    document.checksum,
                    document.page_count,
                    json.dumps(document.metadata),
                    document.ingested_at,
                ),
            )

    async def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Chunk/embedding count mismatch: {len(chunks)} chunks, "
                f"{len(embeddings)} embeddings"
            )
        if not chunks:
            return 0

        import numpy as np

        pool = await self._get_pool()
        async with pool.connection() as connection, connection.transaction():
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                if len(embedding) != self._dimensions:
                    raise VectorStoreError(
                        f"Embedding has {len(embedding)} dimensions but the index "
                        f"expects {self._dimensions}"
                    )
                await connection.execute(
                    """
                    INSERT INTO document_chunks (
                        chunk_id, document_id, content, ordinal, document_title,
                        source_type, source_path, source_url, publisher, page,
                        section, heading_path, version, token_estimate, checksum, metadata
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        ordinal = EXCLUDED.ordinal,
                        document_title = EXCLUDED.document_title,
                        section = EXCLUDED.section,
                        heading_path = EXCLUDED.heading_path,
                        token_estimate = EXCLUDED.token_estimate,
                        checksum = EXCLUDED.checksum,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.content,
                        chunk.ordinal,
                        chunk.document_title,
                        str(chunk.source_type),
                        chunk.source_path,
                        chunk.source_url,
                        chunk.publisher,
                        chunk.page,
                        chunk.section,
                        json.dumps(list(chunk.heading_path)),
                        chunk.version,
                        chunk.token_estimate,
                        chunk.checksum,
                        json.dumps(chunk.metadata),
                    ),
                )
                await connection.execute(
                    """
                    INSERT INTO chunk_embeddings (chunk_id, dimensions, embedding)
                    VALUES (%s,%s,%s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        dimensions = EXCLUDED.dimensions,
                        embedding = EXCLUDED.embedding
                    """,
                    (chunk.chunk_id, len(embedding), np.asarray(embedding, dtype=np.float32)),
                )
        return len(chunks)

    # -- reads ---------------------------------------------------------------

    @staticmethod
    def _filter_clause(filters: SearchFilters | None) -> tuple[str, list[Any]]:
        if filters is None or filters.is_empty():
            return "", []
        clauses: list[str] = []
        params: list[Any] = []
        if filters.document_ids:
            clauses.append("c.document_id = ANY(%s)")
            params.append(list(filters.document_ids))
        if filters.source_types:
            clauses.append("c.source_type = ANY(%s)")
            params.append([str(item) for item in filters.source_types])
        if filters.version is not None:
            clauses.append("c.version = %s")
            params.append(filters.version)
        return (" AND " + " AND ".join(clauses)) if clauses else "", params

    @staticmethod
    def _row_to_chunk(row: dict[str, Any]) -> Chunk:
        heading_path = row.get("heading_path") or []
        if isinstance(heading_path, str):
            heading_path = json.loads(heading_path)
        metadata = row.get("metadata") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return Chunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            content=row["content"],
            ordinal=row["ordinal"],
            document_title=row["document_title"],
            source_type=SourceType(row["source_type"]),
            source_path=row["source_path"],
            source_url=row["source_url"],
            publisher=row["publisher"],
            page=row["page"],
            section=row["section"],
            heading_path=tuple(heading_path),
            version=row["version"],
            token_estimate=row["token_estimate"],
            checksum=row["checksum"],
            metadata=metadata,
        )

    async def vector_search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 8,
        filters: SearchFilters | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        import numpy as np
        from psycopg.rows import dict_row

        if len(query_embedding) != self._dimensions:
            raise VectorStoreError(
                f"Query embedding has {len(query_embedding)} dimensions but the "
                f"index stores {self._dimensions}. Re-ingest after changing the "
                "embedding model."
            )

        clause, params = self._filter_clause(filters)
        pool = await self._get_pool()
        async with pool.connection() as connection:
            # Recall/latency trade-off, tuned per query rather than per index.
            await connection.execute(f"SET LOCAL hnsw.ef_search = {self._hnsw_ef_search}")
            async with connection.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(
                    f"""
                    SELECT c.*, 1 - (e.embedding <=> %s) AS score
                    FROM chunk_embeddings e
                    JOIN document_chunks c ON c.chunk_id = e.chunk_id
                    WHERE 1=1 {clause}
                    ORDER BY e.embedding <=> %s
                    LIMIT %s
                    """,
                    [
                        np.asarray(query_embedding, dtype=np.float32),
                        *params,
                        np.asarray(query_embedding, dtype=np.float32),
                        top_k,
                    ],
                )
                rows = await cursor.fetchall()

        results: list[SearchResult] = []
        for rank, row in enumerate(rows, start=1):
            score = float(row["score"])
            if score < min_score:
                continue
            results.append(
                SearchResult(
                    chunk=self._row_to_chunk(row),
                    score=score,
                    vector_score=score,
                    match_type=MatchType.VECTOR,
                    rank=rank,
                )
            )
        return results

    async def keyword_search(
        self,
        query_text: str,
        *,
        top_k: int = 8,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        from psycopg.rows import dict_row

        terms = [term for term in query_text.split() if len(term) > 1]
        if not terms:
            return []
        # websearch_to_tsquery tolerates arbitrary user text without raising.
        clause, params = self._filter_clause(filters)

        pool = await self._get_pool()
        async with (
            pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
                await cursor.execute(
                    f"""
                    SELECT c.*,
                           ts_rank_cd(c.content_tsv, websearch_to_tsquery('english', %s))
                               AS score
                    FROM document_chunks c
                    WHERE c.content_tsv @@ websearch_to_tsquery('english', %s) {clause}
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    [query_text, query_text, *params, top_k],
                )
                rows = await cursor.fetchall()

        results: list[SearchResult] = []
        for rank, row in enumerate(rows, start=1):
            raw = float(row["score"])
            # ts_rank_cd is unbounded; squash into 0..1 so it fuses with cosine.
            score = raw / (1.0 + raw)
            results.append(
                SearchResult(
                    chunk=self._row_to_chunk(row),
                    score=score,
                    keyword_score=score,
                    match_type=MatchType.KEYWORD,
                    rank=rank,
                )
            )
        return results

    async def get_document(self, document_id: str) -> Document | None:
        from psycopg.rows import dict_row

        pool = await self._get_pool()
        async with (
            pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
                await cursor.execute(
                    "SELECT * FROM documents WHERE document_id = %s", (document_id,)
                )
                row = await cursor.fetchone()
        return self._row_to_document(row) if row else None

    @staticmethod
    def _row_to_document(row: dict[str, Any]) -> Document:
        metadata = row.get("metadata") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        ingested = row["ingested_at"]
        return Document(
            document_id=row["document_id"],
            title=row["title"],
            source_type=SourceType(row["source_type"]),
            document_format=DocumentFormat(row["document_format"]),
            source_path=row["source_path"],
            source_url=row["source_url"],
            publisher=row["publisher"],
            version=row["version"],
            language=row["language"],
            checksum=row["checksum"],
            page_count=row["page_count"],
            metadata=metadata,
            ingested_at=ingested if isinstance(ingested, datetime) else datetime.now(),
        )

    async def list_documents(self) -> list[Document]:
        from psycopg.rows import dict_row

        pool = await self._get_pool()
        async with (
            pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
                await cursor.execute("SELECT * FROM documents ORDER BY title")
                rows = await cursor.fetchall()
        return [self._row_to_document(row) for row in rows]

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        from psycopg.rows import dict_row

        pool = await self._get_pool()
        async with (
            pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
                await cursor.execute(
                    "SELECT * FROM document_chunks WHERE chunk_id = %s", (chunk_id,)
                )
                row = await cursor.fetchone()
        return self._row_to_chunk(row) if row else None

    async def delete_document(self, document_id: str) -> int:
        pool = await self._get_pool()
        async with pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT COUNT(*) FROM document_chunks WHERE document_id = %s", (document_id,)
            )
            row = await cursor.fetchone()
            count = int(row[0]) if row else 0
            # ON DELETE CASCADE removes chunks and embeddings.
            await connection.execute(
                "DELETE FROM documents WHERE document_id = %s", (document_id,)
            )
        return count

    async def count_chunks(self) -> int:
        pool = await self._get_pool()
        async with pool.connection() as connection:
            cursor = await connection.execute("SELECT COUNT(*) FROM document_chunks")
            row = await cursor.fetchone()
        return int(row[0]) if row else 0
