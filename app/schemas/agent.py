"""Structured output schemas for every model call.

Every LLM interaction in this system returns a validated Pydantic object, never
free text that later has to be parsed with regexes. That is what makes the
guardrail layer possible: the model's output is checked against a schema before
any code acts on it.

The ticket vocabulary (`Intent`, `Urgency`) is duplicated here as enums so the
model is *structurally* unable to emit a value outside the closed set. Duplication
between code and `routing.yaml` risks drift, so `validate_vocabulary_matches_policy`
asserts the two agree and is called at application startup.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config.policies import RoutingPolicy, get_routing_policy


class WorkflowKind(StrEnum):
    RESEARCH = "research"
    TRIAGE = "triage"


class Intent(StrEnum):
    """Closed ticket intent vocabulary. Mirrors `intents` in routing.yaml."""

    PAYMENT_ISSUE = "payment_issue"
    REFUND_REQUEST = "refund_request"
    LOGIN_ISSUE = "login_issue"
    PASSWORD_RESET = "password_reset"
    TECHNICAL_ISSUE = "technical_issue"
    BUG_REPORT = "bug_report"
    ACCOUNT_ISSUE = "account_issue"
    CANCELLATION = "cancellation"
    FEATURE_REQUEST = "feature_request"
    SECURITY_ISSUE = "security_issue"
    GENERAL_QUERY = "general_query"


class Urgency(StrEnum):
    """Closed urgency vocabulary. Mirrors `urgency_levels` in routing.yaml."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class VocabularyDriftError(RuntimeError):
    """Raised when the code enums and the routing policy have diverged."""


def validate_vocabulary_matches_policy(policy: RoutingPolicy | None = None) -> None:
    """Fail at startup if `routing.yaml` and the enums above disagree.

    Without this check, adding an intent to the YAML would silently produce a
    routing rule the model can never trigger.
    """
    policy = policy or get_routing_policy()

    code_intents = {member.value for member in Intent}
    policy_intents = set(policy.intents)
    if code_intents != policy_intents:
        raise VocabularyDriftError(
            "Intent enum and routing.yaml disagree. "
            f"Only in code: {sorted(code_intents - policy_intents)}; "
            f"only in policy: {sorted(policy_intents - code_intents)}"
        )

    code_urgencies = {member.value for member in Urgency}
    policy_urgencies = set(policy.urgency_levels)
    if code_urgencies != policy_urgencies:
        raise VocabularyDriftError(
            "Urgency enum and routing.yaml disagree. "
            f"Only in code: {sorted(code_urgencies - policy_urgencies)}; "
            f"only in policy: {sorted(policy_urgencies - code_urgencies)}"
        )


# ---------------------------------------------------------------------------
# Research workflow
# ---------------------------------------------------------------------------


class QueryUnderstanding(BaseModel):
    """First model call: read the request and plan retrieval."""

    model_config = ConfigDict(frozen=True)

    workflow: WorkflowKind = Field(
        description="Whether this is a knowledge question or an incoming support ticket."
    )
    needs_retrieval: bool = Field(
        description="False only for greetings or meta questions that need no evidence."
    )
    search_internal_docs: bool = Field(
        description="True when company-specific policy, SOP or product knowledge is needed."
    )
    search_official_docs: bool = Field(
        description="True when authoritative external vendor documentation is needed."
    )
    vendor_hint: str | None = Field(
        default=None,
        description=(
            "Suggested vendor key, e.g. 'microsoft_azure'. A hint only: unknown "
            "values are discarded and domains are never taken from the model."
        ),
    )
    search_queries: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="Focused search queries. Not a restatement of the question.",
    )
    decision_summary: str = Field(
        default="",
        max_length=500,
        description="One or two sentences on the retrieval plan. Never chain-of-thought.",
    )

    @field_validator("search_queries")
    @classmethod
    def _drop_blank_queries(cls, value: list[str]) -> list[str]:
        return [query.strip() for query in value if query and query.strip()]


class EvidenceGrade(BaseModel):
    """Per-passage relevance judgement."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(description="The evidence handle being graded, e.g. 'E3'.")
    relevant: bool
    relevance_score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=300)


class EvidenceAssessment(BaseModel):
    """Whether the gathered evidence can support an answer, and what is missing."""

    model_config = ConfigDict(frozen=True)

    grades: list[EvidenceGrade] = Field(default_factory=list)
    sufficient: bool = Field(
        description="True only when the evidence can fully support a grounded answer."
    )
    missing_information: str = Field(
        default="",
        max_length=500,
        description="What is still needed. Required when sufficient is false.",
    )
    refined_queries: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Rewritten queries for another retrieval pass.",
    )

    def relevant_ids(self) -> set[str]:
        return {grade.evidence_id for grade in self.grades if grade.relevant}


class GroundedAnswer(BaseModel):
    """A synthesised answer with inline evidence handles."""

    model_config = ConfigDict(frozen=True)

    answer: str = Field(
        description=(
            "The answer. Every factual claim carries an inline handle such as [E2]. "
            "Never write a URL, page number or source title directly."
        )
    )
    cited_evidence_ids: list[str] = Field(
        default_factory=list,
        description="Handles used in the answer, e.g. ['E1', 'E3'].",
    )
    insufficient_evidence: bool = Field(
        default=False,
        description="True when the evidence cannot support a grounded answer.",
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    caveats: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Limitations, staleness or gaps the reader should know about.",
    )


class SourceConflict(BaseModel):
    """A detected disagreement between two pieces of evidence."""

    model_config = ConfigDict(frozen=True)

    topic: str = Field(max_length=200, description="What the sources disagree about.")
    internal_evidence_id: str | None = None
    external_evidence_id: str | None = None
    internal_claim: str = Field(default="", max_length=400)
    external_claim: str = Field(default="", max_length=400)
    preferred_evidence_id: str | None = Field(
        default=None, description="Which source should be preferred, and why, in `resolution`."
    )
    resolution: str = Field(default="", max_length=400)
    affects_business_decision: bool = Field(
        default=False,
        description="True when acting on the wrong version would have business impact.",
    )


class ConflictAssessment(BaseModel):
    """Conflict detection across the gathered evidence."""

    model_config = ConfigDict(frozen=True)

    conflicts: list[SourceConflict] = Field(default_factory=list, max_length=5)

    @property
    def has_conflicts(self) -> bool:
        return bool(self.conflicts)


# ---------------------------------------------------------------------------
# Triage workflow
# ---------------------------------------------------------------------------


class TicketClassification(BaseModel):
    """The model's read of a ticket.

    Deliberately contains **no queue field**. Queue selection is a deterministic
    policy decision made from this classification, not something the model emits.
    """

    model_config = ConfigDict(frozen=True)

    intent: Intent
    urgency: Urgency
    language: str = Field(
        description="Detected language name or code, e.g. 'English', 'Hindi', 'Hinglish'."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    secondary_intent: Intent | None = Field(
        default=None, description="Set when the ticket plausibly spans two intents."
    )
    rationale: str = Field(
        default="",
        max_length=400,
        description="Brief justification. A decision summary, not chain-of-thought.",
    )
    key_signals: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="Short phrases from the ticket that drove the classification.",
    )
    customer_sentiment: str = Field(default="neutral", max_length=32)
    suggested_actions: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Recommended actions. Execution remains subject to approval policy.",
    )
