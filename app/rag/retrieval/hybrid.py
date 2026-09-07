"""Hybrid retrieval: dense vector search fused with lexical search.

Neither channel alone is sufficient for a support knowledge base. Vector search
misses exact identifiers (error codes, API names, SKU strings); keyword search
misses paraphrase. Fusing them recovers both.

Fusion uses **Reciprocal Rank Fusion**, which combines rankings rather than raw
scores. That matters here because cosine similarity and BM25/ts_rank live on
different, non-comparable scales, and a weighted sum of them is arbitrary. RRF
only needs the ordering to be meaningful within each channel.
"""

from __future__ import annotations

from app.db.vector_store import (
    MatchType,
    SearchFilters,
    SearchResult,
    VectorStore,
)
from app.observability.logging import get_logger
from app.providers.embeddings import EmbeddingProvider

logger = get_logger(__name__)

# Standard RRF damping constant. Larger values flatten the influence of rank.
RRF_K = 60


def reciprocal_rank_fusion(
    vector_results: list[SearchResult],
    keyword_results: list[SearchResult],
    *,
    alpha: float = 0.65,
    k: int = RRF_K,
    top_k: int = 8,
) -> list[SearchResult]:
    """Fuse two ranked lists.

    `alpha` weights the vector channel (1.0 = vector only, 0.0 = keyword only).
    """
    scores: dict[str, float] = {}
    best: dict[str, SearchResult] = {}
    vector_scores: dict[str, float] = {}
    keyword_scores: dict[str, float] = {}

    for rank, result in enumerate(vector_results, start=1):
        chunk_id = result.chunk.chunk_id
        scores[chunk_id] = scores.get(chunk_id, 0.0) + alpha / (k + rank)
        best[chunk_id] = result
        vector_scores[chunk_id] = result.score

    for rank, result in enumerate(keyword_results, start=1):
        chunk_id = result.chunk.chunk_id
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 - alpha) / (k + rank)
        best.setdefault(chunk_id, result)
        keyword_scores[chunk_id] = result.score

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    fused: list[SearchResult] = []
    for rank, (chunk_id, fused_score) in enumerate(ordered, start=1):
        source = best[chunk_id]
        fused.append(
            SearchResult(
                chunk=source.chunk,
                score=fused_score,
                vector_score=vector_scores.get(chunk_id),
                keyword_score=keyword_scores.get(chunk_id),
                match_type=MatchType.HYBRID,
                rank=rank,
                metadata={
                    "matched_vector": chunk_id in vector_scores,
                    "matched_keyword": chunk_id in keyword_scores,
                },
            )
        )
    return fused


class HybridRetriever:
    """Retrieval facade over a `VectorStore` and an `EmbeddingProvider`."""

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingProvider,
        *,
        top_k: int = 8,
        min_score: float = 0.15,
        alpha: float = 0.65,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._top_k = top_k
        self._min_score = min_score
        self._alpha = alpha

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: SearchFilters | None = None,
        use_hybrid: bool = True,
    ) -> list[SearchResult]:
        """Retrieve chunks for a query.

        A failure in one channel degrades to the other rather than failing the
        whole retrieval - a search that returns keyword-only results is far more
        useful to the agent than an exception.
        """
        limit = top_k or self._top_k
        if not query.strip():
            return []

        # Over-fetch per channel so fusion has material to work with.
        channel_limit = max(limit * 2, limit + 4)

        vector_results: list[SearchResult] = []
        keyword_results: list[SearchResult] = []

        try:
            embedding = await self._embedder.embed_query(query)
            vector_results = await self._store.vector_search(
                embedding,
                top_k=channel_limit,
                filters=filters,
                # Filtering happens after fusion so a strong keyword match is not
                # discarded for having a weak cosine score.
                min_score=0.0,
            )
        except Exception as exc:
            logger.warning("vector_search_failed", error=str(exc), query_length=len(query))

        if use_hybrid:
            try:
                keyword_results = await self._store.keyword_search(
                    query, top_k=channel_limit, filters=filters
                )
            except Exception as exc:
                logger.warning("keyword_search_failed", error=str(exc))

        if not vector_results and not keyword_results:
            return []

        if not use_hybrid or not keyword_results:
            results = [
                result for result in vector_results if result.score >= self._min_score
            ][:limit]
            return results
        if not vector_results:
            return keyword_results[:limit]

        return reciprocal_rank_fusion(
            vector_results,
            keyword_results,
            alpha=self._alpha,
            top_k=limit,
        )
