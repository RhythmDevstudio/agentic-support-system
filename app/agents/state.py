"""Agent state.

The state carries everything the run accumulates: evidence, tool results,
citations, classification, routing, errors and approval status. Reducers are
attached where a node contributes to a list rather than replacing it.
"""

from __future__ import annotations

import operator
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

from app.models.evidence import Citation, Evidence
from app.routing.engine import RoutingDecision, ValidatedClassification
from app.schemas.agent import SourceConflict, TicketClassification


class ToolInvocation(BaseModel):
    """Audit record of one tool call."""

    model_config = ConfigDict(frozen=True)

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    summary: str = ""
    latency_ms: float = 0.0
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentError(BaseModel):
    """A recoverable failure. Collected rather than raised, so the agent can
    report degraded results instead of dying."""

    model_config = ConfigDict(frozen=True)

    stage: str
    message: str
    recoverable: bool = True
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ApprovalState(BaseModel):
    model_config = ConfigDict(frozen=True)

    required: bool = False
    granted: bool = False
    reasons: tuple[str, ...] = ()
    requested_action: str | None = None
    decided_by: str | None = None


class Budget(BaseModel):
    """Hard limits on a run. Checked by nodes before doing more work."""

    model_config = ConfigDict(frozen=True)

    max_iterations: int = 4
    max_tool_calls: int = 12
    iterations_used: int = 0
    tool_calls_used: int = 0

    @property
    def iterations_exhausted(self) -> bool:
        return self.iterations_used >= self.max_iterations

    @property
    def tool_calls_exhausted(self) -> bool:
        return self.tool_calls_used >= self.max_tool_calls

    def spend_iteration(self) -> Budget:
        return self.model_copy(update={"iterations_used": self.iterations_used + 1})

    def spend_tool_calls(self, count: int = 1) -> Budget:
        return self.model_copy(update={"tool_calls_used": self.tool_calls_used + count})


class AgentState(TypedDict, total=False):
    """The graph's state."""

    # Input
    messages: Annotated[list, add_messages]
    request: str
    ticket_id: str | None
    customer_id: str | None
    customer_tier: str | None
    thread_id: str

    # Understanding
    workflow: str
    vendor_key: str | None
    vendor_display_name: str | None
    needs_retrieval: bool
    search_internal: bool
    search_official: bool
    queries: Annotated[list[str], operator.add]
    decision_log: Annotated[list[str], operator.add]

    # Evidence
    evidence: Annotated[list[Evidence], operator.add]
    quarantined_evidence: Annotated[list[Evidence], operator.add]
    evidence_sufficient: bool
    missing_information: str
    # Queries the grading step proposed, consumed by the refine step.
    refined_queries: list[str]

    # Output
    answer: str
    citations: list[Citation]
    conflicts: list[SourceConflict]
    caveats: list[str]
    insufficient_evidence: bool
    confidence: float

    # Triage
    classification: TicketClassification | None
    validated_classification: ValidatedClassification | None
    routing_decision: RoutingDecision | None

    # Bookkeeping
    tool_invocations: Annotated[list[ToolInvocation], operator.add]
    errors: Annotated[list[AgentError], operator.add]
    approval: ApprovalState
    budget: Budget
    started_at: datetime


def initial_state(
    request: str,
    *,
    thread_id: str,
    ticket_id: str | None = None,
    customer_id: str | None = None,
    max_iterations: int = 4,
    max_tool_calls: int = 12,
) -> AgentState:
    return AgentState(
        messages=[],
        request=request,
        ticket_id=ticket_id,
        customer_id=customer_id,
        customer_tier=None,
        thread_id=thread_id,
        queries=[],
        decision_log=[],
        evidence=[],
        quarantined_evidence=[],
        evidence_sufficient=False,
        missing_information="",
        refined_queries=[],
        answer="",
        citations=[],
        conflicts=[],
        caveats=[],
        insufficient_evidence=False,
        confidence=0.0,
        classification=None,
        validated_classification=None,
        routing_decision=None,
        tool_invocations=[],
        errors=[],
        approval=ApprovalState(),
        budget=Budget(max_iterations=max_iterations, max_tool_calls=max_tool_calls),
        started_at=datetime.now(UTC),
    )
