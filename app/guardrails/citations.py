"""Citation validation - the anti-fabrication guarantee.

Rules enforced here, deterministically:

1. Every `[E#]` handle in an answer must resolve to evidence that was actually
   retrieved during this run. Unresolvable handles are stripped.
2. Citation objects are built from the retrieval record via
   `Citation.from_evidence`. No field is ever taken from model output, so a URL
   the agent never fetched cannot be emitted.
3. Quarantined evidence cannot be cited.
4. An answer whose handles were all invalid is downgraded to an explicit
   "insufficient verified evidence" response rather than being shown as fact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.models.evidence import (
    EVIDENCE_HANDLE_PATTERN,
    Citation,
    Evidence,
    EvidenceRegistry,
    extract_handles,
)
from app.observability.logging import get_logger
from app.schemas.agent import GroundedAnswer

logger = get_logger(__name__)

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I could not find sufficient verified evidence to answer this question. "
    "Rather than guess, I am reporting the gap: no retrieved source supported "
    "the claims needed for an answer."
)

# A bare URL in model output is a fabrication risk - the model was told to cite
# handles, never to write URLs.
_URL_IN_ANSWER = re.compile(r"https?://\S+")


@dataclass
class CitationValidationResult:
    """Outcome of validating one answer."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    used_evidence: list[Evidence] = field(default_factory=list)
    invalid_handles: list[str] = field(default_factory=list)
    stripped_urls: list[str] = field(default_factory=list)
    quarantined_handles: list[str] = field(default_factory=list)
    insufficient_evidence: bool = False

    @property
    def is_valid(self) -> bool:
        return not self.invalid_handles and not self.stripped_urls

    @property
    def has_citations(self) -> bool:
        return bool(self.citations)

    def summary(self) -> str:
        parts = [f"{len(self.citations)} verified citation(s)"]
        if self.invalid_handles:
            parts.append(f"{len(self.invalid_handles)} fabricated handle(s) removed")
        if self.stripped_urls:
            parts.append(f"{len(self.stripped_urls)} model-written URL(s) removed")
        if self.quarantined_handles:
            parts.append(f"{len(self.quarantined_handles)} quarantined source(s) refused")
        return "; ".join(parts)


class CitationValidator:
    """Resolves and renumbers evidence handles into verified citations."""

    def __init__(self, registry: EvidenceRegistry) -> None:
        self._registry = registry

    def validate(self, answer: GroundedAnswer) -> CitationValidationResult:
        text = answer.answer

        if answer.insufficient_evidence:
            return CitationValidationResult(
                answer=text or INSUFFICIENT_EVIDENCE_MESSAGE,
                insufficient_evidence=True,
            )

        invalid: list[str] = []
        quarantined: list[str] = []
        resolved: dict[str, Evidence] = {}

        for handle in extract_handles(text):
            evidence = self._registry.resolve(handle)
            if evidence is None:
                invalid.append(handle)
            elif not evidence.is_usable:
                quarantined.append(handle)
            else:
                resolved[handle] = evidence

        if invalid:
            logger.warning(
                "fabricated_citation_handles_removed",
                handles=invalid,
                known=len(self._registry),
            )
        if quarantined:
            logger.warning("quarantined_evidence_citation_refused", handles=quarantined)

        # Renumber surviving handles to reader-facing [1], [2], ... in first-use order.
        numbering = {handle: index for index, handle in enumerate(resolved, start=1)}
        citations = [
            Citation.from_evidence(resolved[handle], numbering[handle]) for handle in numbering
        ]

        cleaned = self._rewrite_handles(text, numbering, drop=set(invalid) | set(quarantined))
        cleaned, stripped_urls = self._strip_urls(cleaned)
        cleaned = _tidy_whitespace(cleaned)

        insufficient = not citations
        if insufficient:
            logger.warning("answer_had_no_verifiable_citations")
            cleaned = INSUFFICIENT_EVIDENCE_MESSAGE

        return CitationValidationResult(
            answer=cleaned,
            citations=citations,
            used_evidence=[resolved[handle] for handle in numbering],
            invalid_handles=invalid,
            stripped_urls=stripped_urls,
            quarantined_handles=quarantined,
            insufficient_evidence=insufficient,
        )

    @staticmethod
    def _rewrite_handles(
        text: str, numbering: dict[str, int], drop: set[str]
    ) -> str:
        def replace(match: re.Match[str]) -> str:
            handle = match.group(1)
            if handle in drop:
                return ""
            index = numbering.get(handle)
            return f"[{index}]" if index is not None else ""

        return EVIDENCE_HANDLE_PATTERN.sub(replace, text)

    @staticmethod
    def _strip_urls(text: str) -> tuple[str, list[str]]:
        """Remove URLs the model wrote itself.

        Real URLs reach the user through the citation list, built from the
        retrieval record. A URL in the answer body was invented by the model.
        """
        found = _URL_IN_ANSWER.findall(text)
        if not found:
            return text, []
        return _URL_IN_ANSWER.sub("", text), found


def _tidy_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def render_citations(citations: list[Citation]) -> str:
    """Render the sources block shown to the user."""
    if not citations:
        return ""
    return "Sources:\n" + "\n".join(citation.render() for citation in citations)
