"""Phase 2: embedding providers.

The offline embedder must produce *real* similarity structure, otherwise every
retrieval test downstream is vacuous. These tests assert that property directly.
"""

from __future__ import annotations

import pytest

from app.providers.embeddings import (
    HashingEmbeddingProvider,
    cosine_similarity,
    cosine_similarity_matrix,
)


@pytest.fixture
def embedder() -> HashingEmbeddingProvider:
    return HashingEmbeddingProvider(dimensions=512)


class TestHashingEmbeddingProvider:
    async def test_produces_correct_dimensions(self, embedder: HashingEmbeddingProvider) -> None:
        vector = await embedder.embed_query("refund policy")
        assert len(vector) == 512

    async def test_is_deterministic(self, embedder: HashingEmbeddingProvider) -> None:
        first = await embedder.embed_query("duplicate charge refund")
        second = await embedder.embed_query("duplicate charge refund")
        assert first == second

    async def test_vectors_are_normalised(self, embedder: HashingEmbeddingProvider) -> None:
        vector = await embedder.embed_query("some reasonably long piece of text here")
        magnitude = sum(value * value for value in vector) ** 0.5
        assert magnitude == pytest.approx(1.0, abs=1e-5)

    async def test_related_text_scores_higher_than_unrelated(
        self, embedder: HashingEmbeddingProvider
    ) -> None:
        """The core property that makes offline retrieval tests meaningful."""
        query = await embedder.embed_query("how do we process refunds")
        related = await embedder.embed_query(
            "refunds are processed after approval by a billing manager"
        )
        unrelated = await embedder.embed_query(
            "kubernetes pod scheduling uses node affinity rules"
        )
        assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)

    async def test_identical_text_is_maximally_similar(
        self, embedder: HashingEmbeddingProvider
    ) -> None:
        text = "verified duplicate charges are refunded in full"
        vectors = await embedder.embed_documents([text, text])
        assert cosine_similarity(vectors[0], vectors[1]) == pytest.approx(1.0, abs=1e-5)

    async def test_batch_matches_single_embedding(
        self, embedder: HashingEmbeddingProvider
    ) -> None:
        single = await embedder.embed_query("login failure")
        batch = await embedder.embed_documents(["login failure"])
        assert batch[0] == single

    async def test_empty_text_produces_zero_vector(
        self, embedder: HashingEmbeddingProvider
    ) -> None:
        vector = await embedder.embed_query("")
        assert all(value == 0.0 for value in vector)

    async def test_empty_batch_returns_empty_list(
        self, embedder: HashingEmbeddingProvider
    ) -> None:
        assert await embedder.embed_documents([]) == []

    async def test_handles_devanagari_and_code_mixed_text(
        self, embedder: HashingEmbeddingProvider
    ) -> None:
        """Ticket text is frequently Hindi or romanized Hindi."""
        hindi = await embedder.embed_query("मेरा अकाउंट लॉगिन नहीं हो रहा")
        hinglish = await embedder.embed_query("mera account login nahi ho raha")
        assert len(hindi) == 512
        assert len(hinglish) == 512
        assert any(value != 0.0 for value in hinglish)

    async def test_reports_itself_as_offline(self, embedder: HashingEmbeddingProvider) -> None:
        assert embedder.is_offline() is True


class TestSimilarityHelpers:
    def test_orthogonal_vectors_score_zero(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_score_minus_one(self) -> None:
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_is_safe(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_matrix_similarity_matches_pairwise(self) -> None:
        import numpy as np

        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        query = [1.0, 0.0]
        scores = cosine_similarity_matrix(matrix, query)
        assert scores[0] == pytest.approx(1.0, abs=1e-6)
        assert scores[1] == pytest.approx(0.0, abs=1e-6)
        assert scores[2] == pytest.approx(0.7071, abs=1e-3)

    def test_empty_matrix_returns_empty_scores(self) -> None:
        import numpy as np

        assert cosine_similarity_matrix(np.zeros((0, 0), dtype=np.float32), [1.0]).size == 0
