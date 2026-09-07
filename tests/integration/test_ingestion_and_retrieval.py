"""Phase 2: end-to-end ingestion and hybrid retrieval.

Exercises the real pipeline - parse, chunk, embed, index, retrieve - against the
real seed documents, using the offline embedder and the SQLite store.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.db.sqlite_store import SqliteVectorStore
from app.db.vector_store import MatchType, SearchFilters, VectorStoreError
from app.models.documents import Document, DocumentFormat, SourceType
from app.providers.embeddings import HashingEmbeddingProvider
from app.rag.ingestion.pipeline import IngestionPipeline
from app.rag.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion

pytestmark = pytest.mark.integration


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteVectorStore]:
    instance = SqliteVectorStore(tmp_path / "kb.db")
    await instance.initialize()
    yield instance
    await instance.close()


@pytest.fixture
def embedder() -> HashingEmbeddingProvider:
    return HashingEmbeddingProvider(dimensions=1024)


@pytest.fixture
async def populated(
    store: SqliteVectorStore,
    embedder: HashingEmbeddingProvider,
    project_root: Path,
) -> SqliteVectorStore:
    pipeline = IngestionPipeline(store, embedder)
    results = await pipeline.ingest_directory(
        project_root / "data" / "documents", publisher="Internal Knowledge Base"
    )
    assert any(result.chunks_embedded > 0 for result in results), "seed corpus failed to ingest"
    return store


class TestIngestion:
    async def test_ingests_all_seed_documents(
        self, store: SqliteVectorStore, embedder: HashingEmbeddingProvider, project_root: Path
    ) -> None:
        pipeline = IngestionPipeline(store, embedder)
        results = await pipeline.ingest_directory(project_root / "data" / "documents")

        ingested = [result for result in results if not result.skipped]
        assert len(ingested) >= 4
        assert all(result.chunks_embedded == result.chunks_created for result in ingested)
        assert await store.count_chunks() > 0

    async def test_ingests_each_supported_format(
        self, store: SqliteVectorStore, embedder: HashingEmbeddingProvider, tmp_path: Path
    ) -> None:
        (tmp_path / "a.md").write_text("# Markdown Doc\n\nMarkdown body content.")
        (tmp_path / "b.txt").write_text("Plain text body content.")
        (tmp_path / "c.html").write_text("<html><body><h1>HTML Doc</h1><p>Html body.</p></body>")

        pipeline = IngestionPipeline(store, embedder)
        results = await pipeline.ingest_directory(tmp_path)

        assert len([r for r in results if not r.skipped]) == 3
        formats = {doc.document_format for doc in await store.list_documents()}
        assert formats == {DocumentFormat.MARKDOWN, DocumentFormat.TEXT, DocumentFormat.HTML}

    async def test_reingestion_replaces_rather_than_duplicates(
        self, store: SqliteVectorStore, embedder: HashingEmbeddingProvider, tmp_path: Path
    ) -> None:
        path = tmp_path / "policy.md"
        path.write_text("# Policy\n\nOriginal body text about refunds.")
        pipeline = IngestionPipeline(store, embedder)

        await pipeline.ingest_file(path)
        first_count = await store.count_chunks()

        path.write_text("# Policy\n\nRevised body text about refunds and cancellations.")
        result = await pipeline.ingest_file(path)

        assert await store.count_chunks() == first_count
        assert result.warnings  # reports that it replaced prior chunks
        assert len(await store.list_documents()) == 1

    async def test_one_bad_file_does_not_abort_the_batch(
        self, store: SqliteVectorStore, embedder: HashingEmbeddingProvider, tmp_path: Path
    ) -> None:
        (tmp_path / "good.md").write_text("# Good\n\nUsable content here.")
        (tmp_path / "empty.md").write_text("")

        results = await IngestionPipeline(store, embedder).ingest_directory(tmp_path)

        assert len(results) == 2
        assert any(not r.skipped for r in results)
        assert any(r.skipped for r in results)

    async def test_ingests_in_memory_text(
        self, store: SqliteVectorStore, embedder: HashingEmbeddingProvider
    ) -> None:
        result = await IngestionPipeline(store, embedder).ingest_text(
            "# Uploaded\n\nContent supplied through the API.",
            title="Uploaded Doc",
            document_format=DocumentFormat.MARKDOWN,
            source_url="https://example.com/uploaded",
        )
        assert result.chunks_embedded > 0
        document = await store.get_document(result.document_id)
        assert document is not None
        assert document.source_url == "https://example.com/uploaded"

    async def test_skips_applesingle_sidecar_files(
        self, store: SqliteVectorStore, embedder: HashingEmbeddingProvider, tmp_path: Path
    ) -> None:
        (tmp_path / "real.md").write_text("# Real\n\nContent.")
        (tmp_path / "._real.md").write_bytes(b"\x00\x05\x16\x07binary junk")

        results = await IngestionPipeline(store, embedder).ingest_directory(tmp_path)
        assert len(results) == 1


class TestVectorStore:
    async def test_metadata_survives_the_round_trip(self, populated: SqliteVectorStore) -> None:
        documents = await populated.list_documents()
        refund = next(d for d in documents if "Refund" in d.title)
        assert refund.source_type is SourceType.INTERNAL_DOCUMENT
        assert refund.source_path is not None
        assert refund.checksum

    async def test_rejects_mismatched_embedding_count(
        self, store: SqliteVectorStore
    ) -> None:
        document = Document(document_id="d1", title="D")
        await store.upsert_document(document)
        with pytest.raises(VectorStoreError, match="mismatch"):
            await store.upsert_chunks([], [[0.1, 0.2]])

    async def test_rejects_query_with_wrong_dimensions(
        self, populated: SqliteVectorStore
    ) -> None:
        """Dimension drift means the embedding model changed after ingestion."""
        with pytest.raises(VectorStoreError, match="dimensions"):
            await populated.vector_search([0.1, 0.2, 0.3], top_k=3)

    async def test_delete_removes_document_and_chunks(
        self, populated: SqliteVectorStore
    ) -> None:
        documents = await populated.list_documents()
        target = documents[0]
        before = await populated.count_chunks()

        removed = await populated.delete_document(target.document_id)

        assert removed > 0
        assert await populated.count_chunks() == before - removed
        assert await populated.get_document(target.document_id) is None

    async def test_filters_by_document_id(
        self, populated: SqliteVectorStore, embedder: HashingEmbeddingProvider
    ) -> None:
        documents = await populated.list_documents()
        target = next(d for d in documents if "Refund" in d.title)
        embedding = await embedder.embed_query("refund")

        results = await populated.vector_search(
            embedding, top_k=10, filters=SearchFilters(document_ids=(target.document_id,))
        )
        assert results
        assert all(r.chunk.document_id == target.document_id for r in results)

    async def test_keyword_search_finds_exact_terms(
        self, populated: SqliteVectorStore
    ) -> None:
        results = await populated.keyword_search("chargeback", top_k=5)
        assert results
        assert any("chargeback" in r.chunk.content.lower() for r in results)


class TestHybridRetrieval:
    async def test_finds_refund_policy_for_a_policy_question(
        self, populated: SqliteVectorStore, embedder: HashingEmbeddingProvider
    ) -> None:
        """TEST 3 precondition: the refund policy must be retrievable."""
        retriever = HybridRetriever(populated, embedder, top_k=5)
        results = await retriever.search("How do we process refunds according to company policy?")

        assert results
        titles = {result.chunk.document_title for result in results}
        assert any("Refund" in title for title in titles)

    async def test_finds_duplicate_charge_guidance(
        self, populated: SqliteVectorStore, embedder: HashingEmbeddingProvider
    ) -> None:
        retriever = HybridRetriever(populated, embedder, top_k=5)
        results = await retriever.search("customer was charged twice duplicate charge")

        assert results
        combined = " ".join(result.chunk.content.lower() for result in results)
        assert "duplicate" in combined

    async def test_finds_login_guidance_from_html_document(
        self, populated: SqliteVectorStore, embedder: HashingEmbeddingProvider
    ) -> None:
        retriever = HybridRetriever(populated, embedder, top_k=5)
        results = await retriever.search("customer cannot log in account lockout password reset")

        assert results
        combined = " ".join(result.chunk.content.lower() for result in results)
        assert "lockout" in combined or "password" in combined

    async def test_results_carry_citation_metadata(
        self, populated: SqliteVectorStore, embedder: HashingEmbeddingProvider
    ) -> None:
        """Retrieval must return enough provenance to build a citation."""
        retriever = HybridRetriever(populated, embedder, top_k=3)
        results = await retriever.search("refund approval authority")

        assert results
        for result in results:
            assert result.chunk.document_id
            assert result.chunk.document_title
            assert result.chunk.chunk_id
            assert result.chunk.source_type is SourceType.INTERNAL_DOCUMENT

    async def test_hybrid_results_are_marked_as_fused(
        self, populated: SqliteVectorStore, embedder: HashingEmbeddingProvider
    ) -> None:
        retriever = HybridRetriever(populated, embedder, top_k=5)
        results = await retriever.search("refund policy duplicate charge")
        assert any(result.match_type is MatchType.HYBRID for result in results)

    async def test_empty_query_returns_nothing(
        self, populated: SqliteVectorStore, embedder: HashingEmbeddingProvider
    ) -> None:
        retriever = HybridRetriever(populated, embedder)
        assert await retriever.search("   ") == []

    async def test_search_on_empty_index_returns_nothing(
        self, store: SqliteVectorStore, embedder: HashingEmbeddingProvider
    ) -> None:
        retriever = HybridRetriever(store, embedder)
        assert await retriever.search("anything at all") == []

    async def test_respects_top_k(
        self, populated: SqliteVectorStore, embedder: HashingEmbeddingProvider
    ) -> None:
        retriever = HybridRetriever(populated, embedder)
        assert len(await retriever.search("refund", top_k=2)) <= 2


class TestReciprocalRankFusion:
    def _result(self, chunk_id: str, score: float):  # type: ignore[no-untyped-def]
        from app.db.vector_store import SearchResult
        from app.models.documents import Chunk

        return SearchResult(
            chunk=Chunk(chunk_id=chunk_id, document_id="d", content="c", ordinal=0),
            score=score,
        )

    def test_item_ranked_by_both_channels_wins(self) -> None:
        vector = [self._result("a", 0.9), self._result("b", 0.8)]
        keyword = [self._result("b", 0.95), self._result("c", 0.7)]

        fused = reciprocal_rank_fusion(vector, keyword, alpha=0.5, top_k=3)

        assert fused[0].chunk.chunk_id == "b"
        assert fused[0].metadata["matched_vector"] is True
        assert fused[0].metadata["matched_keyword"] is True

    def test_alpha_one_ignores_keyword_ordering(self) -> None:
        vector = [self._result("a", 0.9)]
        keyword = [self._result("z", 0.99)]
        fused = reciprocal_rank_fusion(vector, keyword, alpha=1.0, top_k=2)
        assert fused[0].chunk.chunk_id == "a"

    def test_alpha_zero_ignores_vector_ordering(self) -> None:
        vector = [self._result("a", 0.99)]
        keyword = [self._result("z", 0.5)]
        fused = reciprocal_rank_fusion(vector, keyword, alpha=0.0, top_k=2)
        assert fused[0].chunk.chunk_id == "z"

    def test_deduplicates_across_channels(self) -> None:
        vector = [self._result("a", 0.9)]
        keyword = [self._result("a", 0.8)]
        fused = reciprocal_rank_fusion(vector, keyword, top_k=5)
        assert len(fused) == 1

    def test_empty_inputs_produce_empty_output(self) -> None:
        assert reciprocal_rank_fusion([], [], top_k=5) == []
