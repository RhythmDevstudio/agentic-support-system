"""Offline deterministic "LLM".

Rule-based stand-in that satisfies the same `LLMProvider` contract, so the whole
system runs and is testable with no API key. It reads the evidence out of the
`<untrusted_document>` fences in the prompt and produces schema-valid output.

What it is: real lexical relevance grading, genuinely extractive synthesis (every
sentence in an answer is copied from evidence and tagged with that evidence's
handle), and keyword-based ticket classification covering English, Hindi and
Hinglish.

What it is not: a language model. It cannot paraphrase, reason about novel
phrasing, or handle questions whose wording does not overlap the evidence. Its
purpose is to prove the wiring, guardrails and control flow are correct.
Accuracy numbers from this provider are not meaningful - set OPENAI_API_KEY for
that, and the same code paths run against a real model with no changes.
"""

from __future__ import annotations

import re
import time
from typing import Any, TypeVar

from pydantic import BaseModel

from app.agents.prompts.templates import parse_evidence_blocks
from app.config.settings import Settings, get_settings
from app.providers.llm import LLMError, LLMProvider, LLMResult, LLMUsage, ModelRole
from app.schemas.agent import (
    ConflictAssessment,
    EvidenceAssessment,
    EvidenceGrade,
    GroundedAnswer,
    Intent,
    QueryUnderstanding,
    SourceConflict,
    TicketClassification,
    Urgency,
    WorkflowKind,
)

T = TypeVar("T", bound=BaseModel)

_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_SECTION = re.compile(
    r"^([A-Z][A-Z _]+):\n(.*?)(?=\n[A-Z][A-Z _]+:\n|\Z)", re.DOTALL | re.MULTILINE
)

_STOPWORDS = frozenset(
    """a an the is are was were be been being do does did doing have has had how what
    when where which who whom why can could should would will shall may might must i we you
    he she it they them us our your their this that these those to of in on at by for with
    from as and or but if then than so not no yes my me about into over under between during
    before after above below up down out off again further once here there all any both each
    few more most other some such only own same too very just now get got""".split()
)

# Devanagari block.
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_ARABIC = re.compile(r"[؀-ۿ]")
_CJK = re.compile(r"[一-鿿぀-ヿ]")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")

# Romanized Hindi markers. Presence of several signals Hinglish.
_HINGLISH_MARKERS = frozenset(
    """nahi nahin hain raha rha rahi karo kiya gaya gayi mera meri mujhe muje kya kaise
    kyun chahiye hua hui bhi aur liye jaldi turant paisa paise kata batao karna krna kripya
    thik theek accha bilkul abhi wala wali hum aap tum nhi""".split()
)


def _tokens(text: str) -> list[str]:
    return [word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS]


def _content_terms(text: str) -> set[str]:
    return {word for word in _tokens(text) if len(word) > 2}


def _overlap_score(query: str, text: str) -> float:
    """Fraction of query content terms present in `text`."""
    query_terms = _content_terms(query)
    if not query_terms:
        return 0.0
    lowered = text.lower()
    return sum(1 for term in query_terms if term in lowered) / len(query_terms)


def _sections(prompt: str) -> dict[str, str]:
    """Split a built prompt back into its labelled sections."""
    return {name.strip(): body.strip() for name, body in _SECTION.findall(prompt)}


class DeterministicLLMProvider(LLMProvider):
    name = "deterministic"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def is_offline(self) -> bool:
        return True

    def model_for_role(self, role: str) -> str:
        return f"deterministic-{role}"

    async def parse(
        self,
        *,
        instructions: str,
        input_text: str,
        schema: type[T],
        role: str = ModelRole.CLASSIFICATION,
        temperature: float | None = None,
        call_name: str = "",
    ) -> LLMResult[T]:
        started = time.perf_counter()

        handlers: dict[type[BaseModel], Any] = {
            QueryUnderstanding: self._understand,
            EvidenceAssessment: self._grade,
            GroundedAnswer: self._synthesise,
            ConflictAssessment: self._detect_conflicts,
            TicketClassification: self._classify,
        }
        handler = handlers.get(schema)
        if handler is None:
            raise LLMError(
                f"The offline provider has no handler for {schema.__name__}. "
                "Set OPENAI_API_KEY to use a real model."
            )

        value = handler(input_text)
        return LLMResult(
            value=value,
            model=self.model_for_role(role),
            provider=self.name,
            usage=LLMUsage(),
            latency_ms=(time.perf_counter() - started) * 1000,
            call_name=call_name,
        )

    # -- understanding -------------------------------------------------------

    def _understand(self, prompt: str) -> QueryUnderstanding:
        from app.tools.web_research.domain_policy import VendorDetector

        request = _sections(prompt).get("REQUEST", prompt)
        lowered = request.lower()

        vendor_match = VendorDetector().detect(request)
        vendor_key = vendor_match.key if vendor_match else None

        # First person plural / possessive language signals a company-policy question.
        internal_markers = (
            "our ", "our company", "we ", "company policy", "internal", "sop",
            "our policy", "do we ", "does our", "my company",
        )
        wants_internal = any(marker in lowered for marker in internal_markers)

        is_ticket = self._looks_like_ticket(request)

        return QueryUnderstanding(
            workflow=WorkflowKind.TRIAGE if is_ticket else WorkflowKind.RESEARCH,
            needs_retrieval=not is_ticket,
            search_internal_docs=wants_internal or vendor_key is None,
            search_official_docs=vendor_key is not None,
            vendor_hint=vendor_key,
            search_queries=self._make_queries(request),
            decision_summary=(
                f"Detected {vendor_match.vendor.display_name} - searching official documentation."
                if vendor_match
                else "No specific vendor detected - searching internal documentation."
            ),
        )

    @staticmethod
    def _looks_like_ticket(text: str) -> bool:
        lowered = text.lower()
        first_person_problem = any(
            phrase in lowered
            for phrase in (
                "my account", "my payment", "my card", "i was charged", "i cannot",
                "i can't", "i am unable", "mera ", "mujhe ", "meri ", "help me",
                "please refund", "i need a refund", "not working for me",
            )
        )
        asks_how = lowered.strip().startswith(("what", "how", "why", "when", "where", "which"))
        return first_person_problem and not asks_how

    @staticmethod
    def _make_queries(request: str) -> list[str]:
        terms = [term for term in _tokens(request) if len(term) > 2]
        if not terms:
            return [request.strip()[:120]]
        queries = [" ".join(terms[:8])]
        if len(terms) > 4:
            queries.append(" ".join(terms[:4]))
        return queries[:2]

    # -- grading -------------------------------------------------------------

    def _grade(self, prompt: str) -> EvidenceAssessment:
        sections = _sections(prompt)
        question = sections.get("QUESTION", "")
        blocks = parse_evidence_blocks(prompt)

        grades: list[EvidenceGrade] = []
        for block in blocks:
            score = _overlap_score(question, f"{block.get('title', '')} {block['content']}")
            grades.append(
                EvidenceGrade(
                    evidence_id=block.get("id", ""),
                    relevant=score >= 0.3,
                    relevance_score=round(min(score, 1.0), 3),
                    reason=(
                        "Shares substantive terminology with the question."
                        if score >= 0.3
                        else "Little overlap with the question."
                    ),
                )
            )

        relevant = [grade for grade in grades if grade.relevant]
        best = max((grade.relevance_score for grade in grades), default=0.0)
        sufficient = bool(relevant) and best >= 0.45

        return EvidenceAssessment(
            grades=grades,
            sufficient=sufficient,
            missing_information=(
                ""
                if sufficient
                else "Retrieved passages do not directly address the question."
            ),
            refined_queries=[] if sufficient else self._refine(question),
        )

    @staticmethod
    def _refine(question: str) -> list[str]:
        terms = [term for term in _tokens(question) if len(term) > 3]
        if not terms:
            return []
        # A narrower query on the most specific-looking terms.
        return [" ".join(sorted(terms, key=len, reverse=True)[:5])]

    # -- synthesis -----------------------------------------------------------

    def _synthesise(self, prompt: str) -> GroundedAnswer:
        sections = _sections(prompt)
        question = sections.get("QUESTION", "")
        blocks = parse_evidence_blocks(prompt)

        if not blocks:
            return GroundedAnswer(
                answer=(
                    "I could not find sufficient evidence to answer this question. "
                    "No supporting documentation was retrieved."
                ),
                cited_evidence_ids=[],
                insufficient_evidence=True,
                confidence=0.0,
                caveats=["No evidence was available."],
            )

        # Rank blocks by authority first, then relevance - the same ordering rule
        # the real model is instructed to follow.
        ranked = sorted(
            blocks,
            key=lambda block: (
                int(block.get("authority_tier", 5)),
                -_overlap_score(question, block["content"]),
            ),
        )

        picked: list[tuple[str, str]] = []
        seen: set[str] = set()
        for block in ranked[:4]:
            handle = block.get("id", "")
            for sentence in self._best_sentences(question, block["content"], limit=2):
                key = sentence.lower()[:60]
                if key in seen:
                    continue
                seen.add(key)
                picked.append((sentence, handle))
            if len(picked) >= 5:
                break

        if not picked:
            return GroundedAnswer(
                answer=(
                    "The retrieved documentation does not directly answer this question. "
                    "I do not have sufficient evidence to give a grounded answer."
                ),
                cited_evidence_ids=[],
                insufficient_evidence=True,
                confidence=0.1,
                caveats=["Retrieved evidence did not address the question."],
            )

        answer = " ".join(f"{sentence} [{handle}]" for sentence, handle in picked)
        cited = list(dict.fromkeys(handle for _, handle in picked))

        return GroundedAnswer(
            answer=answer,
            cited_evidence_ids=cited,
            insufficient_evidence=False,
            confidence=round(min(0.4 + 0.1 * len(picked), 0.85), 2),
            caveats=[
                "Answer assembled by the offline extractive provider; "
                "set OPENAI_API_KEY for fluent synthesis."
            ],
        )

    @staticmethod
    def _best_sentences(question: str, content: str, *, limit: int = 2) -> list[str]:
        sentences = [
            sentence.strip()
            for sentence in _SENTENCE.split(content)
            if 40 <= len(sentence.strip()) <= 400
        ]
        if not sentences:
            return []
        scored = sorted(
            ((_overlap_score(question, sentence), index, sentence)
             for index, sentence in enumerate(sentences)),
            key=lambda item: (-item[0], item[1]),
        )
        # Keep only sentences with real overlap, then restore document order.
        chosen = [item for item in scored[:limit] if item[0] > 0.05]
        return [sentence for _, _, sentence in sorted(chosen, key=lambda item: item[1])]

    # -- conflict detection --------------------------------------------------

    _STALENESS_MARKERS = (
        "out of date", "outdated", "superseded", "potentially outdated", "draft",
        "conflict note", "may have changed", "no longer", "at the time",
    )

    def _detect_conflicts(self, prompt: str) -> ConflictAssessment:
        blocks = parse_evidence_blocks(prompt)
        internal = [b for b in blocks if b.get("source_type") == "internal_document"]
        official = [
            b for b in blocks if str(b.get("source_type", "")).startswith("official")
        ]
        if not internal or not official:
            return ConflictAssessment(conflicts=[])

        conflicts: list[SourceConflict] = []
        for internal_block in internal:
            content = internal_block["content"].lower()
            if not any(marker in content for marker in self._STALENESS_MARKERS):
                continue
            # Only a conflict if the two sources are talking about the same thing.
            partner = max(
                official,
                key=lambda block: _overlap_score(internal_block["content"], block["content"]),
            )
            if _overlap_score(internal_block["content"], partner["content"]) < 0.15:
                continue
            conflicts.append(
                SourceConflict(
                    topic=internal_block.get("title", "internal note vs official documentation")[
                        :200
                    ],
                    internal_evidence_id=internal_block.get("id"),
                    external_evidence_id=partner.get("id"),
                    internal_claim=internal_block["content"][:400],
                    external_claim=partner["content"][:400],
                    preferred_evidence_id=partner.get("id"),
                    resolution=(
                        "The internal note flags itself as potentially out of date. "
                        "Official vendor documentation is authoritative for current "
                        "product behaviour."
                    ),
                    affects_business_decision=False,
                )
            )
        return ConflictAssessment(conflicts=conflicts[:5])

    # -- ticket classification ----------------------------------------------

    # Weighted keyword signals per intent. Hindi/Hinglish terms included inline
    # because code-mixed tickets are the norm, not an edge case.
    _INTENT_SIGNALS: dict[Intent, tuple[tuple[str, float], ...]] = {
        Intent.SECURITY_ISSUE: (
            ("hacked", 3.0), ("hack", 2.0), ("unauthorized", 3.0), ("unauthorised", 3.0),
            ("breach", 3.0), ("compromised", 3.0), ("phishing", 3.0), ("stolen", 2.0),
            ("someone else", 2.0), ("suspicious activity", 2.5), ("fraud", 2.0),
        ),
        Intent.REFUND_REQUEST: (
            ("refund", 3.0), ("money back", 3.0), ("reimburse", 2.5), ("paisa wapas", 3.0),
            ("refund chahiye", 3.5), ("want my money", 2.5), ("return my money", 2.5),
        ),
        Intent.PAYMENT_ISSUE: (
            ("charged twice", 3.5), ("deducted twice", 3.5), ("double charge", 3.5),
            ("duplicate charge", 3.5), ("payment", 1.5), ("charged", 1.5), ("billing", 1.5),
            ("invoice", 1.5), ("card", 1.0), ("deducted", 2.0), ("paisa cut", 3.0),
            ("paise cut", 3.0), ("overcharged", 3.0), ("transaction failed", 2.5),
        ),
        Intent.LOGIN_ISSUE: (
            ("cannot log in", 3.0), ("can't log in", 3.0), ("cannot login", 3.0),
            ("login", 2.5), ("log in", 2.5), ("sign in", 2.5), ("locked out", 3.0),
            ("login nahi", 3.5), ("login nhi", 3.5), ("access my account", 2.0),
            ("2fa", 2.0), ("mfa", 2.0), ("otp", 1.5),
        ),
        Intent.PASSWORD_RESET: (
            ("password reset", 3.5), ("reset my password", 3.5), ("forgot password", 3.5),
            ("forgot my password", 3.5), ("change password", 2.5), ("password bhool", 3.0),
        ),
        Intent.CANCELLATION: (
            ("cancel", 3.0), ("cancel my subscription", 3.5), ("unsubscribe", 3.0),
            ("terminate", 2.0), ("stop billing", 2.5), ("band karo", 2.5),
        ),
        Intent.BUG_REPORT: (
            ("bug", 3.0), ("crash", 3.0), ("error message", 2.5), ("broken", 2.0),
            ("not working", 1.5), ("exception", 2.0), ("stack trace", 3.0), ("500 error", 3.0),
        ),
        Intent.TECHNICAL_ISSUE: (
            ("api", 2.0), ("integration", 2.5), ("timeout", 2.5), ("slow", 1.5),
            ("connection", 1.5), ("webhook", 2.5), ("sdk", 2.0), ("configuration", 1.5),
        ),
        Intent.ACCOUNT_ISSUE: (
            ("account", 1.0), ("profile", 1.5), ("update my email", 2.5),
            ("delete my account", 3.0), ("close my account", 3.0), ("account settings", 2.0),
        ),
        Intent.FEATURE_REQUEST: (
            ("feature request", 3.5), ("would be nice", 2.5), ("please add", 3.0),
            ("suggestion", 2.5), ("enhancement", 2.5), ("can you add", 2.5),
        ),
    }

    _URGENCY_CRITICAL = (
        "hacked", "breach", "compromised", "unauthorized access", "unauthorised access",
        "production down", "outage", "data leak", "everything is down", "cannot operate",
    )
    _URGENCY_HIGH = (
        "urgent", "urgently", "immediately", "asap", "emergency", "critical", "blocked",
        "cannot work", "twice", "double", "turant", "jaldi", "bahut zaroori", "very important",
        "losing money", "still waiting",
    )

    def _classify(self, prompt: str) -> TicketClassification:
        blocks = parse_evidence_blocks(prompt)
        ticket = next(
            (block["content"] for block in blocks if block.get("id") == "TICKET"),
            prompt,
        )
        lowered = ticket.lower()

        scores: dict[Intent, float] = {}
        signals: list[str] = []
        for intent, keywords in self._INTENT_SIGNALS.items():
            for phrase, weight in keywords:
                if phrase in lowered:
                    scores[intent] = scores.get(intent, 0.0) + weight
                    signals.append(phrase)

        if scores:
            ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            intent, top_score = ordered[0]
            secondary = ordered[1][0] if len(ordered) > 1 and ordered[1][1] >= 2.0 else None
            # Confidence grows with signal strength and with the margin over the runner-up.
            margin = top_score - (ordered[1][1] if len(ordered) > 1 else 0.0)
            confidence = min(0.5 + 0.08 * top_score + 0.04 * margin, 0.95)
        else:
            intent, secondary, confidence = Intent.GENERAL_QUERY, None, 0.35

        return TicketClassification(
            intent=intent,
            urgency=self._detect_urgency(lowered, intent),
            language=self._detect_language(ticket),
            confidence=round(confidence, 2),
            secondary_intent=secondary,
            rationale=(
                f"Matched {len(signals)} keyword signal(s) for {intent.value}."
                if signals
                else "No strong intent signals; classified as a general query."
            ),
            key_signals=list(dict.fromkeys(signals))[:6],
            customer_sentiment="frustrated" if self._is_frustrated(lowered) else "neutral",
            suggested_actions=[],
        )

    def _detect_urgency(self, lowered: str, intent: Intent) -> Urgency:
        if any(marker in lowered for marker in self._URGENCY_CRITICAL):
            return Urgency.CRITICAL
        if any(marker in lowered for marker in self._URGENCY_HIGH):
            return Urgency.HIGH
        if intent in {Intent.PAYMENT_ISSUE, Intent.REFUND_REQUEST, Intent.LOGIN_ISSUE}:
            return Urgency.MEDIUM
        if intent in {Intent.FEATURE_REQUEST, Intent.GENERAL_QUERY}:
            return Urgency.LOW
        return Urgency.MEDIUM

    @staticmethod
    def _detect_language(text: str) -> str:
        if _DEVANAGARI.search(text):
            return "Hindi"
        if _ARABIC.search(text):
            return "Arabic"
        if _CJK.search(text):
            return "Chinese"
        if _CYRILLIC.search(text):
            return "Russian"

        words = set(_WORD.findall(text.lower()))
        hinglish_hits = len(words & _HINGLISH_MARKERS)
        # Two markers is a reliable signal and avoids firing on stray words.
        if hinglish_hits >= 2:
            return "Hinglish"
        return "English"

    @staticmethod
    def _is_frustrated(lowered: str) -> bool:
        return any(
            marker in lowered
            for marker in (
                "frustrated", "unacceptable", "ridiculous", "angry", "worst",
                "still not", "third time", "again and again", "pathetic",
            )
        )
