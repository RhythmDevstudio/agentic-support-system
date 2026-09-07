"""Evidence and citation models - the backbone of the anti-fabrication guarantee.

The contract this module encodes:

* `Evidence` is created **only** by a retrieval tool, and always records what was
  actually fetched: a real URL or document ID, and a real retrieval timestamp.
* The model is shown evidence under short handles (`E1`, `E2`, ...) and cites
  them by handle. It is never asked to produce a URL, title or page number.
* `Citation` is constructed from an `Evidence` record by deterministic code
  (`Citation.from_evidence`). There is no code path that builds a citation from
  model-generated text, so a source the agent never retrieved cannot be cited.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from app.models.documents import SourceType

# Handles the model uses to reference evidence, e.g. "[E3]".
EVIDENCE_HANDLE_PATTERN = re.compile(r"\[(E\d+)\]")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RetrievalChannel(StrEnum):
    """Which tool produced this evidence."""

    INTERNAL_DOCS = "internal_docs"
    WEB_RESEARCH = "web_research"


class QuarantineReason(StrEnum):
    PROMPT_INJECTION = "prompt_injection"
    BLOCKED_DOMAIN = "blocked_domain"
    OFF_ALLOWLIST_DOMAIN = "off_allowlist_domain"
    EMPTY_CONTENT = "empty_content"


class Evidence(BaseModel):
    """A single retrieved passage plus its verified provenance.

    Frozen: once retrieval records what it fetched, nothing downstream may
    rewrite the provenance.
    """

    model_config = ConfigDict(frozen=True)

    evidence_id: str  # short handle exposed to the model: "E1", "E2", ...
    content: str
    channel: RetrievalChannel
    source_type: SourceType

    # Provenance. Populated from the retrieval record, never from model output.
    title: str = ""
    url: str | None = None
    publisher: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    section: str | None = None
    version: str | None = None

    # Ranking signals.
    score: float = 0.0
    authority_tier: int = 5
    vendor_key: str | None = None

    # Audit.
    retrieved_at: datetime = Field(default_factory=_utcnow)
    retrieval_query: str = ""

    # Guardrail state.
    quarantined: bool = False
    quarantine_reason: QuarantineReason | None = None
    quarantine_detail: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def domain(self) -> str | None:
        if not self.url:
            return None
        netloc = urlparse(self.url).netloc.lower()
        return netloc.removeprefix("www.") or None

    @property
    def is_usable(self) -> bool:
        """Quarantined or empty evidence never reaches generation."""
        return not self.quarantined and bool(self.content.strip())

    def locator(self) -> str:
        """Human-readable position inside the source."""
        parts: list[str] = []
        if self.page is not None:
            parts.append(f"p. {self.page}")
        if self.section:
            parts.append(self.section)
        return ", ".join(parts)

    def quarantine(
        self, reason: QuarantineReason, detail: str | None = None
    ) -> Evidence:
        """Return a quarantined copy. Frozen model, so this replaces rather than mutates."""
        return self.model_copy(
            update={
                "quarantined": True,
                "quarantine_reason": reason,
                "quarantine_detail": detail,
            }
        )


class Citation(BaseModel):
    """A verified citation. Only ever built from an `Evidence` record."""

    model_config = ConfigDict(frozen=True)

    marker: str  # "[1]" as rendered to the user
    evidence_id: str  # the "E3" handle it resolved from
    title: str
    url: str | None = None
    publisher: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    section: str | None = None
    version: str | None = None
    source_type: SourceType
    authority_tier: int = 5
    retrieved_at: datetime

    @classmethod
    def from_evidence(cls, evidence: Evidence, index: int) -> Citation:
        """Build a citation from a retrieval record.

        This is the only constructor used in production code. Every field is
        copied from `evidence`; nothing is derived from model output.
        """
        return cls(
            marker=f"[{index}]",
            evidence_id=evidence.evidence_id,
            title=evidence.title or (evidence.domain or "Untitled source"),
            url=evidence.url,
            publisher=evidence.publisher,
            document_id=evidence.document_id,
            chunk_id=evidence.chunk_id,
            page=evidence.page,
            section=evidence.section,
            version=evidence.version,
            source_type=evidence.source_type,
            authority_tier=evidence.authority_tier,
            retrieved_at=evidence.retrieved_at,
        )

    def render(self) -> str:
        """Format for display, e.g. '[1] Microsoft Learn - Agentic Retrieval (p. 3)'."""
        head = f"{self.marker} "
        if self.publisher and self.publisher not in self.title:
            head += f"{self.publisher} - "
        head += self.title
        locator = ", ".join(
            part
            for part in (
                f"p. {self.page}" if self.page is not None else "",
                self.section or "",
            )
            if part
        )
        if locator:
            head += f" ({locator})"
        if self.url:
            head += f"\n    {self.url}"
        return head


class EvidenceRegistry:
    """Assigns and resolves evidence handles for a single agent run.

    Handles are allocated here and nowhere else, which is what makes
    "does `[E7]` correspond to something we actually retrieved?" answerable.
    """

    def __init__(self) -> None:
        self._by_handle: dict[str, Evidence] = {}
        self._by_fingerprint: dict[str, str] = {}
        self._counter = 0

    def __len__(self) -> int:
        return len(self._by_handle)

    def __contains__(self, handle: str) -> bool:
        return handle in self._by_handle

    @staticmethod
    def _fingerprint(
        content: str, url: str | None, chunk_id: str | None
    ) -> str:
        return f"{chunk_id or ''}|{url or ''}|{hash(content.strip())}"

    def next_handle(self) -> str:
        self._counter += 1
        return f"E{self._counter}"

    def register(self, evidence: Evidence) -> Evidence:
        """Add evidence, deduplicating identical passages onto one handle."""
        fingerprint = self._fingerprint(evidence.content, evidence.url, evidence.chunk_id)
        existing = self._by_fingerprint.get(fingerprint)
        if existing is not None:
            return self._by_handle[existing]

        handle = evidence.evidence_id or self.next_handle()
        if handle in self._by_handle:
            handle = self.next_handle()
        stored = evidence.model_copy(update={"evidence_id": handle})
        self._by_handle[handle] = stored
        self._by_fingerprint[fingerprint] = handle
        return stored

    def register_all(self, items: list[Evidence]) -> list[Evidence]:
        return [self.register(item) for item in items]

    def resolve(self, handle: str) -> Evidence | None:
        return self._by_handle.get(handle)

    def all(self) -> list[Evidence]:
        return list(self._by_handle.values())

    def usable(self) -> list[Evidence]:
        return [item for item in self._by_handle.values() if item.is_usable]

    @classmethod
    def from_evidence(cls, items: list[Evidence]) -> EvidenceRegistry:
        registry = cls()
        # Preserve handles that were already assigned upstream.
        for item in items:
            if item.evidence_id.startswith("E") and item.evidence_id[1:].isdigit():
                registry._counter = max(registry._counter, int(item.evidence_id[1:]))
        registry.register_all(items)
        return registry


def extract_handles(text: str) -> list[str]:
    """Pull evidence handles out of generated text, preserving first-seen order."""
    seen: dict[str, None] = {}
    for match in EVIDENCE_HANDLE_PATTERN.finditer(text):
        seen.setdefault(match.group(1), None)
    return list(seen)
