"""The `search_internal_documentation` tool.

Wraps hybrid retrieval and converts results into `Evidence` carrying the full
provenance needed for a citation: document title, section, page, version and
chunk ID.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.db.vector_store import SearchFilters
from app.models.evidence import Evidence, RetrievalChannel
from app.observability.logging import get_logger
from app.rag.retrieval.hybrid import HybridRetriever

logger = get_logger(__name__)

MAX_CHUNK_CHARS = 4000


class InternalSearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    evidence: tuple[Evidence, ...] = ()
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence)

    def summary(self) -> str:
        if self.error:
            return f"search_internal_documentation failed: {self.error}"
        return f"{len(self.evidence)} internal passage(s) retrieved"


class InternalDocumentationTool:
    def __init__(self, retriever: HybridRetriever) -> None:
        self._retriever = retriever

    async def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        filters: SearchFilters | None = None,
    ) -> InternalSearchResult:
        """Search the internal knowledge base.

        Retrieval failure is returned rather than raised so the agent can fall
        back to official documentation and report the gap.
        """
        if not query.strip():
            return InternalSearchResult(query=query, error="Empty search query")

        retrieved_at = datetime.now(UTC)
        try:
            results = await self._retriever.search(query, top_k=top_k, filters=filters)
        except Exception as exc:
            logger.warning("internal_search_failed", error=str(exc))
            return InternalSearchResult(query=query, error=str(exc))

        evidence = tuple(
            Evidence(
                evidence_id="",  # assigned by the run's EvidenceRegistry
                content=result.chunk.content[:MAX_CHUNK_CHARS],
                channel=RetrievalChannel.INTERNAL_DOCS,
                source_type=result.chunk.source_type,
                title=result.chunk.document_title or "Internal document",
                url=result.chunk.source_url,
                publisher=result.chunk.publisher or "Internal Knowledge Base",
                document_id=result.chunk.document_id,
                chunk_id=result.chunk.chunk_id,
                page=result.chunk.page,
                section=result.chunk.section,
                version=result.chunk.version,
                score=result.score,
                # Internal policy is authoritative for company questions; tier 1
                # alongside first-party vendor docs, which cannot describe our rules.
                authority_tier=1,
                retrieved_at=retrieved_at,
                retrieval_query=query,
                metadata={
                    "match_type": str(result.match_type),
                    "heading_path": list(result.chunk.heading_path),
                    "source_path": result.chunk.source_path,
                },
            )
            for result in results
        )

        logger.info("internal_search_complete", results=len(evidence))
        return InternalSearchResult(query=query, evidence=evidence, retrieved_at=retrieved_at)
