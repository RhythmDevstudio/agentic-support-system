"""SQLite vector store for development, CI and no-Docker environments.

Vectors are stored as float32 blobs and scored with numpy. That is honest about
what it is: linear scan, correct but O(n) per query. For the MVP knowledge base
(hundreds to low thousands of chunks) it is comfortably fast, and it means the
whole system - agent loop, guardrails, citations, evaluation - can be exercised
with no database server present.

Keyword search uses FTS5 when the local SQLite build provides it, falling back to
a LIKE scan otherwise, so the hybrid retrieval path behaves the same either way.

`PgVectorStore` is the production implementation; both pass the same conformance
tests in `tests/integration/test_vector_store_conformance.py`.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.db.vector_store import (
    MatchType,
    SearchFilters,
    SearchResult,
    VectorStore,
    VectorStoreError,
)
from app.models.documents import Chunk, Document, DocumentFormat, SourceType
from app.observability.logging import get_logger
from app.providers.embeddings import cosine_similarity_matrix

logger = get_logger(__name__)

_SCHEMA = """
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
    metadata        TEXT NOT NULL DEFAULT '{}',
    ingested_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id        TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    content         TEXT NOT NULL,
    ordinal         INTEGER NOT NULL,
    document_title  TEXT NOT NULL DEFAULT '',
    source_type     TEXT NOT NULL,
    source_path     TEXT,
    source_url      TEXT,
    publisher       TEXT,
    page            INTEGER,
    section         TEXT,
    heading_path    TEXT NOT NULL DEFAULT '[]',
    version         TEXT NOT NULL DEFAULT '1',
    token_estimate  INTEGER NOT NULL DEFAULT 0,
    checksum        TEXT NOT NULL DEFAULT '',
    metadata        TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source_type ON document_chunks(source_type);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id    TEXT PRIMARY KEY REFERENCES document_chunks(chunk_id) ON DELETE CASCADE,
    dimensions  INTEGER NOT NULL,
    vector      BLOB NOT NULL
);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    chunk_id UNINDEXED,
    content,
    tokenize = 'unicode61'
);
"""


def _to_blob(vector: list[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class SqliteVectorStore(VectorStore):
    name = "sqlite"

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._connection: sqlite3.Connection | None = None
        self._fts_enabled = False
        self._lock = asyncio.Lock()
        # Cached embedding matrix; invalidated on write.
        self._matrix: np.ndarray | None = None
        self._matrix_ids: list[str] = []

    # -- lifecycle -----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection
        if str(self._path) != ":memory:":
            self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            str(self._path), check_same_thread=False, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        self._connection = connection
        return connection

    def _initialize_sync(self) -> None:
        connection = self._connect()
        connection.executescript(_SCHEMA)
        try:
            connection.executescript(_FTS_SCHEMA)
            self._fts_enabled = True
        except sqlite3.OperationalError as exc:
            # FTS5 is optional in a SQLite build; degrade to LIKE search.
            logger.warning("fts5_unavailable", error=str(exc), fallback="like_scan")
            self._fts_enabled = False

    async def initialize(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._initialize_sync)

    async def close(self) -> None:
        async with self._lock:
            if self._connection is not None:
                await asyncio.to_thread(self._connection.close)
                self._connection = None
            self._matrix = None
            self._matrix_ids = []

    # -- writes --------------------------------------------------------------

    def _upsert_document_sync(self, document: Document) -> None:
        self._connect().execute(
            """
            INSERT INTO documents (
                document_id, title, source_type, document_format, source_path,
                source_url, publisher, version, language, checksum, page_count,
                metadata, ingested_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(document_id) DO UPDATE SET
                title=excluded.title,
                source_type=excluded.source_type,
                document_format=excluded.document_format,
                source_path=excluded.source_path,
                source_url=excluded.source_url,
                publisher=excluded.publisher,
                version=excluded.version,
                language=excluded.language,
                checksum=excluded.checksum,
                page_count=excluded.page_count,
                metadata=excluded.metadata,
                ingested_at=excluded.ingested_at
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
                document.ingested_at.isoformat(),
            ),
        )

    async def upsert_document(self, document: Document) -> None:
        async with self._lock:
            await asyncio.to_thread(self._upsert_document_sync, document)

    def _upsert_chunks_sync(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        connection = self._connect()
        connection.execute("BEGIN")
        try:
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                connection.execute(
                    """
                    INSERT INTO document_chunks (
                        chunk_id, document_id, content, ordinal, document_title,
                        source_type, source_path, source_url, publisher, page,
                        section, heading_path, version, token_estimate, checksum, metadata
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        content=excluded.content,
                        ordinal=excluded.ordinal,
                        document_title=excluded.document_title,
                        section=excluded.section,
                        heading_path=excluded.heading_path,
                        token_estimate=excluded.token_estimate,
                        checksum=excluded.checksum,
                        metadata=excluded.metadata
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
                connection.execute(
                    """
                    INSERT INTO chunk_embeddings (chunk_id, dimensions, vector)
                    VALUES (?,?,?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        dimensions=excluded.dimensions, vector=excluded.vector
                    """,
                    (chunk.chunk_id, len(embedding), _to_blob(embedding)),
                )
                if self._fts_enabled:
                    connection.execute(
                        "DELETE FROM chunk_fts WHERE chunk_id = ?", (chunk.chunk_id,)
                    )
                    connection.execute(
                        "INSERT INTO chunk_fts (chunk_id, content) VALUES (?,?)",
                        (chunk.chunk_id, chunk.content),
                    )
            connection.execute("COMMIT")
        except Exception as exc:
            connection.execute("ROLLBACK")
            raise VectorStoreError(f"Failed to write chunks: {exc}") from exc
        return len(chunks)

    async def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Chunk/embedding count mismatch: {len(chunks)} chunks, "
                f"{len(embeddings)} embeddings"
            )
        if not chunks:
            return 0
        async with self._lock:
            written = await asyncio.to_thread(self._upsert_chunks_sync, chunks, embeddings)
            self._matrix = None  # invalidate cache
            self._matrix_ids = []
        return written

    # -- reads ---------------------------------------------------------------

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
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
            heading_path=tuple(json.loads(row["heading_path"] or "[]")),
            version=row["version"],
            token_estimate=row["token_estimate"],
            checksum=row["checksum"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    @staticmethod
    def _filter_clause(filters: SearchFilters | None) -> tuple[str, list[Any]]:
        if filters is None or filters.is_empty():
            return "", []
        clauses: list[str] = []
        params: list[Any] = []
        if filters.document_ids:
            placeholders = ",".join("?" * len(filters.document_ids))
            clauses.append(f"c.document_id IN ({placeholders})")
            params.extend(filters.document_ids)
        if filters.source_types:
            placeholders = ",".join("?" * len(filters.source_types))
            clauses.append(f"c.source_type IN ({placeholders})")
            params.extend(str(item) for item in filters.source_types)
        if filters.version is not None:
            clauses.append("c.version = ?")
            params.append(filters.version)
        return (" AND " + " AND ".join(clauses)) if clauses else "", params

    def _load_matrix_sync(self, filters: SearchFilters | None) -> tuple[np.ndarray, list[str]]:
        clause, params = self._filter_clause(filters)
        # The cache only applies to unfiltered queries, which is the common path.
        cacheable = not clause
        if cacheable and self._matrix is not None:
            return self._matrix, self._matrix_ids

        rows = (
            self._connect()
            .execute(
                f"""
                SELECT e.chunk_id AS chunk_id, e.vector AS vector
                FROM chunk_embeddings e
                JOIN document_chunks c ON c.chunk_id = e.chunk_id
                WHERE 1=1 {clause}
                """,
                params,
            )
            .fetchall()
        )
        if not rows:
            return np.zeros((0, 0), dtype=np.float32), []

        ids = [row["chunk_id"] for row in rows]
        vectors = [_from_blob(row["vector"]) for row in rows]
        width = max(vector.shape[0] for vector in vectors)
        matrix = np.zeros((len(vectors), width), dtype=np.float32)
        for index, vector in enumerate(vectors):
            matrix[index, : vector.shape[0]] = vector

        if cacheable:
            self._matrix, self._matrix_ids = matrix, ids
        return matrix, ids

    def _chunks_by_ids_sync(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = (
            self._connect()
            .execute(
                f"SELECT c.* FROM document_chunks c WHERE c.chunk_id IN ({placeholders})",
                chunk_ids,
            )
            .fetchall()
        )
        return {row["chunk_id"]: self._row_to_chunk(row) for row in rows}

    def _vector_search_sync(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: SearchFilters | None,
        min_score: float,
    ) -> list[SearchResult]:
        matrix, ids = self._load_matrix_sync(filters)
        if matrix.size == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        if query.shape[0] != matrix.shape[1]:
            # Dimension drift usually means the embedding model changed after
            # ingestion. Say so plainly rather than returning silent nonsense.
            raise VectorStoreError(
                f"Query embedding has {query.shape[0]} dimensions but the index "
                f"stores {matrix.shape[1]}. Re-ingest documents after changing "
                "the embedding model."
            )

        scores = cosine_similarity_matrix(matrix, query_embedding)
        top_indices = np.argsort(-scores)[: max(top_k, 0)]
        selected = [(ids[i], float(scores[i])) for i in top_indices if scores[i] >= min_score]

        chunks = self._chunks_by_ids_sync([chunk_id for chunk_id, _ in selected])
        results: list[SearchResult] = []
        for rank, (chunk_id, score) in enumerate(selected, start=1):
            chunk = chunks.get(chunk_id)
            if chunk is None:
                continue
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    vector_score=score,
                    match_type=MatchType.VECTOR,
                    rank=rank,
                )
            )
        return results

    async def vector_search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 8,
        filters: SearchFilters | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        async with self._lock:
            return await asyncio.to_thread(
                self._vector_search_sync, query_embedding, top_k, filters, min_score
            )

    def _keyword_search_sync(
        self, query_text: str, top_k: int, filters: SearchFilters | None
    ) -> list[SearchResult]:
        clause, filter_params = self._filter_clause(filters)
        connection = self._connect()
        terms = [term for term in query_text.lower().split() if len(term) > 1]
        if not terms:
            return []

        rows: list[sqlite3.Row] = []
        if self._fts_enabled:
            # OR the terms so partial matches still surface; bm25() is ascending
            # (lower is better), so negate it into a descending score.
            match_expression = " OR ".join(f'"{term}"' for term in terms)
            try:
                rows = connection.execute(
                    f"""
                    SELECT c.*, bm25(chunk_fts) AS bm25_score
                    FROM chunk_fts
                    JOIN document_chunks c ON c.chunk_id = chunk_fts.chunk_id
                    WHERE chunk_fts MATCH ? {clause}
                    ORDER BY bm25_score ASC
                    LIMIT ?
                    """,
                    [match_expression, *filter_params, top_k],
                ).fetchall()
            except sqlite3.OperationalError as exc:
                logger.warning("fts_query_failed", error=str(exc), fallback="like_scan")
                rows = []

        if not rows:
            like_clauses = " OR ".join("LOWER(c.content) LIKE ?" for _ in terms)
            rows = connection.execute(
                f"""
                SELECT c.*, 0.0 AS bm25_score
                FROM document_chunks c
                WHERE ({like_clauses}) {clause}
                LIMIT ?
                """,
                [*[f"%{term}%" for term in terms], *filter_params, top_k],
            ).fetchall()

        results: list[SearchResult] = []
        for rank, row in enumerate(rows, start=1):
            chunk = self._row_to_chunk(row)
            # Both query branches project a bm25_score column, so it is always present.
            # Normalise to 0..1 so it can be fused with cosine scores.
            raw = row["bm25_score"]
            if raw:
                score = 1.0 / (1.0 + abs(float(raw)))
            else:
                lowered = chunk.content.lower()
                score = sum(1 for term in terms if term in lowered) / len(terms)
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    keyword_score=score,
                    match_type=MatchType.KEYWORD,
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
        async with self._lock:
            return await asyncio.to_thread(
                self._keyword_search_sync, query_text, top_k, filters
            )

    def _get_document_sync(self, document_id: str) -> Document | None:
        row = (
            self._connect()
            .execute("SELECT * FROM documents WHERE document_id = ?", (document_id,))
            .fetchone()
        )
        return self._row_to_document(row) if row else None

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> Document:
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
            metadata=json.loads(row["metadata"] or "{}"),
            ingested_at=datetime.fromisoformat(row["ingested_at"]),
        )

    async def get_document(self, document_id: str) -> Document | None:
        async with self._lock:
            return await asyncio.to_thread(self._get_document_sync, document_id)

    async def list_documents(self) -> list[Document]:
        async with self._lock:
            rows = await asyncio.to_thread(
                lambda: self._connect()
                .execute("SELECT * FROM documents ORDER BY title")
                .fetchall()
            )
        return [self._row_to_document(row) for row in rows]

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        async with self._lock:
            found = await asyncio.to_thread(self._chunks_by_ids_sync, [chunk_id])
        return found.get(chunk_id)

    def _delete_document_sync(self, document_id: str) -> int:
        connection = self._connect()
        chunk_ids = [
            row["chunk_id"]
            for row in connection.execute(
                "SELECT chunk_id FROM document_chunks WHERE document_id = ?", (document_id,)
            ).fetchall()
        ]
        if self._fts_enabled and chunk_ids:
            placeholders = ",".join("?" * len(chunk_ids))
            connection.execute(
                f"DELETE FROM chunk_fts WHERE chunk_id IN ({placeholders})", chunk_ids
            )
        connection.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        return len(chunk_ids)

    async def delete_document(self, document_id: str) -> int:
        async with self._lock:
            removed = await asyncio.to_thread(self._delete_document_sync, document_id)
            self._matrix = None
            self._matrix_ids = []
        return removed

    async def count_chunks(self) -> int:
        async with self._lock:
            row = await asyncio.to_thread(
                lambda: self._connect()
                .execute("SELECT COUNT(*) AS n FROM document_chunks")
                .fetchone()
            )
        return int(row["n"]) if row else 0
