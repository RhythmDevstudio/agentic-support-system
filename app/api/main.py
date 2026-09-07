"""FastAPI application."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app import __version__
from app.api.schemas import (
    ChatRequest,
    CitationOut,
    ClassificationOut,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    ResearchRequest,
    ResearchResponse,
    TriageRequest,
    TriageResponse,
)
from app.config.settings import get_settings
from app.models.documents import DocumentFormat
from app.observability.logging import get_logger
from app.services.container import Container, build_container

logger = get_logger(__name__)

_container: Container | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _container
    _container = build_container()
    await _container.initialize(ingest_seed_documents=True)
    logger.info("api_started", **_container.health())
    yield
    await _container.close()
    _container = None


def get_container() -> Container:
    if _container is None:  # pragma: no cover - only before startup
        raise HTTPException(status_code=503, detail="Application is still starting")
    return _container


app = FastAPI(
    title="Agentic AI Support & Knowledge Research System",
    description=(
        "Grounded documentation research with verified citations, and support "
        "ticket triage with deterministic routing."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_WEB_UI_PATH = Path(__file__).resolve().parents[1] / "web" / "index.html"


@app.get("/", include_in_schema=False)
async def web_ui() -> FileResponse:
    """The simple ticket/question web page - ask a question or submit a ticket
    without needing to use curl or the Swagger docs directly."""
    return FileResponse(_WEB_UI_PATH)


def _citations_out(state: dict[str, Any]) -> list[CitationOut]:
    return [
        CitationOut(
            marker=citation.marker,
            title=citation.title,
            url=citation.url,
            publisher=citation.publisher,
            document_id=citation.document_id,
            page=citation.page,
            section=citation.section,
            source_type=str(citation.source_type),
            authority_tier=citation.authority_tier,
            retrieved_at=citation.retrieved_at,
        )
        for citation in state.get("citations", [])
    ]


def _research_response(state: dict[str, Any]) -> ResearchResponse:
    return ResearchResponse(
        answer=state.get("answer", ""),
        citations=_citations_out(state),
        insufficient_evidence=state.get("insufficient_evidence", False),
        confidence=state.get("confidence", 0.0),
        caveats=state.get("caveats", []),
        conflicts=[conflict.model_dump() for conflict in state.get("conflicts", [])],
        vendor=state.get("vendor_key"),
        decision_log=state.get("decision_log", []),
        tools_used=[item.tool for item in state.get("tool_invocations", [])],
        evidence_count=len(state.get("evidence", [])),
        quarantined_count=len(state.get("quarantined_evidence", [])),
        errors=[error.message for error in state.get("errors", [])],
    )


def _triage_response(state: dict[str, Any]) -> TriageResponse:
    decision = state.get("routing_decision")
    if decision is None:
        raise HTTPException(status_code=500, detail="Routing produced no decision")
    validated = state.get("validated_classification")

    return TriageResponse(
        classification=ClassificationOut(
            intent=decision.intent,
            urgency=decision.urgency,
            language=decision.language,
            language_code=decision.language_code,
            queue=decision.queue,
            confidence=decision.confidence,
            secondary_intent=validated.secondary_intent if validated else None,
        ),
        queue_display_name=decision.queue_display_name,
        sla_hours=decision.sla_hours,
        requires_escalation=decision.requires_escalation,
        requires_human_approval=decision.requires_human_approval,
        approval_reasons=list(decision.approval_reasons),
        applied_rules=list(decision.applied_rules),
        language_supported=decision.language_supported,
        policy_adjustments=list(validated.adjustments) if validated else [],
        decision_log=state.get("decision_log", []),
        actions_taken=[
            item.summary for item in state.get("tool_invocations", []) if item.ok
        ],
    )


@app.get("/api/health", response_model=HealthResponse)
async def health(container: Container = Depends(get_container)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        providers=container.health(),
        knowledge_base=await container.store.health(),
    )


@app.post("/api/research", response_model=ResearchResponse)
async def research(
    request: ResearchRequest, container: Container = Depends(get_container)
) -> ResearchResponse:
    state = container.new_state(request.question, thread_id=request.thread_id)
    result = await container.agent.run(state)
    return _research_response(dict(result))


@app.post("/api/chat")
async def chat(
    request: ChatRequest, container: Container = Depends(get_container)
) -> dict[str, Any]:
    """Single entry point: the agent decides research vs triage."""
    state = container.new_state(
        request.message, thread_id=request.thread_id, ticket_id=request.ticket_id
    )
    result = dict(await container.agent.run(state))

    if result.get("routing_decision") is not None:
        return {"workflow": "triage", "result": _triage_response(result).model_dump()}
    return {"workflow": "research", "result": _research_response(result).model_dump()}


@app.post("/api/tickets/triage", response_model=TriageResponse)
async def triage_ticket(
    request: TriageRequest, container: Container = Depends(get_container)
) -> TriageResponse:
    text = f"{request.subject}\n{request.text}".strip()
    state = container.new_state(
        text, ticket_id=request.ticket_id, customer_id=request.customer_id
    )
    state["workflow"] = "triage"
    result = await container.agent.run(state)
    return _triage_response(dict(result))


@app.post("/api/tickets/{ticket_id}/process", response_model=TriageResponse)
async def process_ticket(
    ticket_id: str, container: Container = Depends(get_container)
) -> TriageResponse:
    """Fetch a ticket from the ticket system, triage it, and apply the routing."""
    fetched = await container.ticket_tools.get_ticket(ticket_id)
    if not fetched.ok:
        raise HTTPException(status_code=404, detail=fetched.error)

    ticket = fetched.data
    state = container.new_state(
        ticket.text(), ticket_id=ticket_id, customer_id=ticket.customer_id
    )
    state["workflow"] = "triage"
    result = await container.agent.run(state)
    return _triage_response(dict(result))


@app.post("/api/documents/ingest", response_model=IngestResponse)
async def ingest_documents(
    request: IngestRequest, container: Container = Depends(get_container)
) -> IngestResponse:
    if request.directory:
        directory = Path(request.directory)
        if not directory.is_absolute():
            directory = Path(container.settings.data_dir).parent / directory
        if not directory.exists():
            raise HTTPException(status_code=400, detail=f"Directory not found: {directory}")
        results = await container.ingestion.ingest_directory(directory)
    elif request.content:
        try:
            document_format = DocumentFormat(request.document_format)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"Unsupported format '{request.document_format}'"
            ) from exc
        results = [
            await container.ingestion.ingest_text(
                request.content,
                title=request.title,
                document_format=document_format,
                source_url=request.source_url,
            )
        ]
    else:
        raise HTTPException(status_code=400, detail="Provide either 'content' or 'directory'")

    return IngestResponse(
        documents=len([item for item in results if not item.skipped]),
        chunks_created=sum(item.chunks_created for item in results),
        chunks_embedded=sum(item.chunks_embedded for item in results),
        skipped=[
            {"title": item.title, "reason": item.skip_reason or ""}
            for item in results
            if item.skipped
        ],
    )
