"""Vector store construction from configuration."""

from __future__ import annotations

from app.config.settings import Settings, VectorStoreName, get_settings
from app.db.sqlite_store import SqliteVectorStore
from app.db.vector_store import VectorStore, VectorStoreError
from app.observability.logging import get_logger

logger = get_logger(__name__)


def build_vector_store(settings: Settings | None = None) -> VectorStore:
    """Construct the configured vector store.

    When `VECTOR_STORE=pgvector` is requested explicitly but the driver or DSN is
    missing, this raises rather than silently falling back - an operator who asked
    for Postgres should not discover at query time that they got SQLite. The
    `auto` setting is where graceful degradation happens.
    """
    settings = settings or get_settings()
    resolved = settings.resolve_vector_store()

    if resolved is VectorStoreName.PGVECTOR:
        if settings.database_url is None:
            raise VectorStoreError("VECTOR_STORE=pgvector requires DATABASE_URL to be set")
        from app.db.pgvector_store import PgVectorStore

        return PgVectorStore(
            dsn=settings.database_url.get_secret_value(),
            dimensions=settings.embedding_dimensions,
            hnsw_m=settings.pgvector_hnsw_m,
            hnsw_ef_construction=settings.pgvector_hnsw_ef_construction,
            hnsw_ef_search=settings.pgvector_hnsw_ef_search,
        )

    path = settings.sqlite_absolute_path()
    logger.debug("vector_store_selected", store="sqlite", path=str(path))
    return SqliteVectorStore(path)
