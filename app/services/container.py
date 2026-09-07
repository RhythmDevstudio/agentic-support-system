"""Dependency wiring.

One place that builds the object graph from configuration, so the API, the demo
script and the tests all construct the system identically.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.agents.graph import SupportAgent
from app.agents.state import AgentState, initial_state
from app.config.policies import get_routing_policy
from app.config.settings import Settings, get_settings
from app.db.factory import build_vector_store
from app.db.vector_store import VectorStore
from app.models.documents import SourceType
from app.observability.logging import configure_logging, get_logger
from app.providers.embeddings import EmbeddingProvider, build_embedding_provider
from app.providers.llm import LLMProvider, build_llm_provider
from app.rag.ingestion.pipeline import IngestionPipeline
from app.rag.retrieval.hybrid import HybridRetriever
from app.routing.engine import RoutingEngine
from app.schemas.agent import validate_vocabulary_matches_policy
from app.tools.documentation.search import InternalDocumentationTool
from app.tools.tickets.adapters import MockTicketAdapter, TicketAdapter
from app.tools.tickets.tools import TicketTools
from app.tools.web_research.tool import WebResearchTool

logger = get_logger(__name__)


@dataclass
class Container:
    """Constructed application services."""

    settings: Settings
    llm: LLMProvider
    embedder: EmbeddingProvider
    store: VectorStore
    retriever: HybridRetriever
    internal_docs: InternalDocumentationTool
    web_research: WebResearchTool
    ticket_adapter: TicketAdapter
    ticket_tools: TicketTools
    routing: RoutingEngine
    agent: SupportAgent
    ingestion: IngestionPipeline

    async def initialize(self, *, ingest_seed_documents: bool = False) -> None:
        await self.store.initialize()
        if ingest_seed_documents:
            await self.ingest_seed_documents()

    async def ingest_seed_documents(self) -> int:
        """Ingest data/documents if the knowledge base is empty."""
        if await self.store.count_chunks() > 0:
            return 0
        documents_dir = Path(self.settings.data_dir) / "documents"
        if not documents_dir.exists():
            logger.warning("seed_documents_missing", path=str(documents_dir))
            return 0
        results = await self.ingestion.ingest_directory(
            documents_dir,
            source_type=SourceType.INTERNAL_DOCUMENT,
            publisher="Internal Knowledge Base",
        )
        total = sum(result.chunks_embedded for result in results)
        logger.info("seed_documents_ingested", documents=len(results), chunks=total)
        return total

    async def close(self) -> None:
        await self.store.close()

    def new_state(
        self,
        request: str,
        *,
        thread_id: str | None = None,
        ticket_id: str | None = None,
        customer_id: str | None = None,
    ) -> AgentState:
        return initial_state(
            request,
            thread_id=thread_id or str(uuid.uuid4()),
            ticket_id=ticket_id,
            customer_id=customer_id,
            max_iterations=self.settings.agent_max_iterations,
            max_tool_calls=self.settings.agent_max_tool_calls,
        )

    def health(self) -> dict[str, Any]:
        return {
            "llm_provider": self.llm.name,
            "llm_offline": self.llm.is_offline(),
            "embedding_provider": self.embedder.name,
            "vector_store": self.store.name,
            "web_search_provider": self.web_research.provider_name,
            "ticket_adapter": self.ticket_adapter.name,
            "offline_mode": self.settings.is_offline_mode(),
        }


def build_container(settings: Settings | None = None) -> Container:
    """Construct the application from configuration."""
    settings = settings or get_settings()
    configure_logging(settings)

    # Fail fast if the code vocabulary and the routing policy have drifted apart.
    validate_vocabulary_matches_policy()

    llm = build_llm_provider(settings)
    embedder = build_embedding_provider(settings)
    store = build_vector_store(settings)

    retriever = HybridRetriever(
        store,
        embedder,
        top_k=settings.retrieval_top_k,
        min_score=settings.retrieval_min_score,
        alpha=settings.retrieval_hybrid_alpha,
    )
    internal_docs = InternalDocumentationTool(retriever)
    web_research = WebResearchTool(settings=settings)

    ticket_adapter = MockTicketAdapter()
    routing_policy = get_routing_policy()
    ticket_tools = TicketTools(ticket_adapter, routing_policy)
    routing = RoutingEngine(
        routing_policy, confidence_floor=settings.classification_confidence_floor
    )

    agent = SupportAgent(
        llm=llm,
        internal_docs=internal_docs,
        web_research=web_research,
        ticket_tools=ticket_tools,
        routing_engine=routing,
        settings=settings,
    )

    return Container(
        settings=settings,
        llm=llm,
        embedder=embedder,
        store=store,
        retriever=retriever,
        internal_docs=internal_docs,
        web_research=web_research,
        ticket_adapter=ticket_adapter,
        ticket_tools=ticket_tools,
        routing=routing,
        agent=agent,
        ingestion=IngestionPipeline(store, embedder, settings),
    )
