"""API request and response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CitationOut(BaseModel):
    marker: str
    title: str
    url: str | None = None
    publisher: str | None = None
    document_id: str | None = None
    page: int | None = None
    section: str | None = None
    source_type: str
    authority_tier: int
    retrieved_at: datetime


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = None


class ResearchResponse(BaseModel):
    answer: str
    citations: list[CitationOut] = Field(default_factory=list)
    insufficient_evidence: bool = False
    confidence: float = 0.0
    caveats: list[str] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    vendor: str | None = None
    decision_log: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    quarantined_count: int = 0
    errors: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = None
    ticket_id: str | None = None


class TriageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000, description="Ticket body.")
    subject: str = ""
    ticket_id: str | None = None
    customer_id: str | None = None


class ClassificationOut(BaseModel):
    intent: str
    urgency: str
    language: str
    language_code: str
    queue: str
    confidence: float
    secondary_intent: str | None = None


class TriageResponse(BaseModel):
    classification: ClassificationOut
    queue_display_name: str
    sla_hours: int
    requires_escalation: bool = False
    requires_human_approval: bool = False
    approval_reasons: list[str] = Field(default_factory=list)
    applied_rules: list[str] = Field(default_factory=list)
    language_supported: bool = True
    policy_adjustments: list[str] = Field(default_factory=list)
    decision_log: list[str] = Field(default_factory=list)
    actions_taken: list[str] = Field(default_factory=list)


class IngestRequest(BaseModel):
    content: str | None = Field(default=None, max_length=2_000_000)
    title: str = "Uploaded document"
    document_format: str = "markdown"
    source_url: str | None = None
    directory: str | None = Field(
        default=None, description="Server-side directory to ingest instead of inline content."
    )


class IngestResponse(BaseModel):
    documents: int
    chunks_created: int
    chunks_embedded: int
    skipped: list[dict[str, str]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    version: str
    providers: dict[str, Any]
    knowledge_base: dict[str, Any]
