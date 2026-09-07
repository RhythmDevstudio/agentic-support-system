"""Deterministic ticket routing.

The model classifies; this module decides. Queue names, escalation and approval
requirements come only from `routing.yaml`, so a wrong or manipulated
classification can at worst pick a wrong *intent* - never an arbitrary queue,
and never a skipped approval.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.config.policies import RoutingPolicy, get_routing_policy
from app.observability.logging import get_logger
from app.schemas.agent import TicketClassification

logger = get_logger(__name__)


class ValidatedClassification(BaseModel):
    """A classification after deterministic validation and normalisation."""

    model_config = ConfigDict(frozen=True)

    intent: str
    urgency: str
    language: str
    language_code: str
    confidence: float
    secondary_intent: str | None = None
    below_confidence_floor: bool = False
    adjustments: tuple[str, ...] = ()


class RoutingDecision(BaseModel):
    """The final, authoritative routing outcome."""

    model_config = ConfigDict(frozen=True)

    queue: str
    queue_display_name: str
    sla_hours: int
    intent: str
    urgency: str
    language: str
    language_code: str
    confidence: float
    requires_escalation: bool = False
    requires_human_approval: bool = False
    approval_reasons: tuple[str, ...] = ()
    applied_rules: tuple[str, ...] = ()
    language_supported: bool = True
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def summary(self) -> str:
        parts = [f"{self.intent} -> {self.queue} ({self.urgency}, {self.confidence:.2f})"]
        if self.requires_escalation:
            parts.append("escalated")
        if self.requires_human_approval:
            parts.append("awaiting human approval")
        return "; ".join(parts)


class RoutingEngine:
    def __init__(
        self, policy: RoutingPolicy | None = None, *, confidence_floor: float = 0.55
    ) -> None:
        self._policy = policy or get_routing_policy()
        self._confidence_floor = confidence_floor

    @property
    def policy(self) -> RoutingPolicy:
        return self._policy

    # -- validation ----------------------------------------------------------

    def validate(
        self, classification: TicketClassification, ticket_text: str = ""
    ) -> ValidatedClassification:
        """Normalise and sanity-check the model's output before it is acted on."""
        adjustments: list[str] = []

        intent = classification.intent.value
        if not self._policy.is_valid_intent(intent):
            adjustments.append(f"unknown intent '{intent}' replaced with general_query")
            intent = "general_query"

        urgency = classification.urgency.value
        if not self._policy.is_valid_urgency(urgency):
            adjustments.append(f"unknown urgency '{urgency}' replaced with medium")
            urgency = "medium"

        # Policy floors override a model that under-rates a sensitive intent.
        floor = self._policy.urgency_floors.get(intent)
        if floor is not None and self._policy.urgency_rank(urgency) < self._policy.urgency_rank(
            floor
        ):
            adjustments.append(f"urgency raised to {floor} by policy floor for {intent}")
            urgency = floor

        # Explicit urgency language enforces a FLOOR of "high", it does not
        # increment. Incrementing would double-count: the classifier has already
        # read the same words, so "urgently" would push high to critical and
        # reserve-for-real-emergencies would stop meaning anything.
        lowered = ticket_text.lower()
        if any(
            word in lowered for word in self._policy.urgency_escalation_keywords
        ) and self._policy.urgency_rank(urgency) < self._policy.urgency_rank("high"):
            adjustments.append("urgency raised to high by explicit urgency language")
            urgency = "high"

        language_code = self._policy.languages.normalise(classification.language)

        confidence = max(0.0, min(1.0, classification.confidence))
        below_floor = confidence < self._confidence_floor
        if below_floor:
            adjustments.append(
                f"confidence {confidence:.2f} below floor {self._confidence_floor:.2f}"
            )

        secondary = (
            classification.secondary_intent.value if classification.secondary_intent else None
        )
        if secondary and not self._policy.is_valid_intent(secondary):
            secondary = None

        return ValidatedClassification(
            intent=intent,
            urgency=urgency,
            language=classification.language,
            language_code=language_code,
            confidence=confidence,
            secondary_intent=secondary,
            below_confidence_floor=below_floor,
            adjustments=tuple(adjustments),
        )

    def _raise_urgency(self, urgency: str) -> str:
        ordered = list(reversed(self._policy.urgency_levels))  # low -> critical
        try:
            index = ordered.index(urgency)
        except ValueError:
            return urgency
        return ordered[min(index + 1, len(ordered) - 1)]

    # -- routing -------------------------------------------------------------

    def route(
        self,
        validated: ValidatedClassification,
        *,
        ticket_text: str = "",
        customer_tier: str | None = None,
        requested_actions: tuple[str, ...] = (),
    ) -> RoutingDecision:
        """Select the queue and decide escalation and approval requirements."""
        applied: list[str] = []
        intent = validated.intent
        urgency = validated.urgency

        # Low confidence goes to humans rather than to a plausible-looking queue.
        if validated.below_confidence_floor:
            applied.append("low_confidence_to_human_triage")
            queue = self._policy.fallback_queue
        else:
            queue = self._policy.queue_for_intent(intent)
            applied.append(f"intent_rule:{intent}")

        requires_escalation = False
        lowered = ticket_text.lower()

        for override in self._policy.overrides:
            if not self._override_matches(override.when, intent, urgency, lowered):
                continue
            applied.append(f"override:{override.name}")
            if override.then.queue:
                queue = override.then.queue
            if override.then.urgency:
                urgency = override.then.urgency
            if override.then.min_urgency and self._policy.urgency_rank(
                urgency
            ) < self._policy.urgency_rank(override.then.min_urgency):
                urgency = override.then.min_urgency
            if override.then.create_escalation:
                requires_escalation = True

        # Re-check the critical-urgency escalation rule after overrides raised urgency.
        if urgency == "critical":
            requires_escalation = True

        if not self._policy.is_valid_queue(queue):
            logger.error("routing_produced_invalid_queue", queue=queue, intent=intent)
            queue = self._policy.fallback_queue
            applied.append("invalid_queue_fallback")

        definition = self._policy.queues[queue]
        approval_reasons = self._approval_reasons(
            urgency=urgency,
            confidence=validated.confidence,
            customer_tier=customer_tier,
            requested_actions=requested_actions,
            queue=queue,
        )

        return RoutingDecision(
            queue=queue,
            queue_display_name=definition.display_name,
            sla_hours=definition.sla_hours,
            intent=intent,
            urgency=urgency,
            language=validated.language,
            language_code=validated.language_code,
            confidence=validated.confidence,
            requires_escalation=requires_escalation,
            requires_human_approval=bool(approval_reasons),
            approval_reasons=approval_reasons,
            applied_rules=tuple(applied),
            language_supported=validated.language_code
            in self._policy.languages.supported_by_agents,
        )

    @staticmethod
    def _override_matches(
        condition: object, intent: str, urgency: str, lowered_text: str
    ) -> bool:
        want_intent = getattr(condition, "intent", None)
        want_urgency = getattr(condition, "urgency", None)
        keywords = getattr(condition, "any_keyword", ())

        if want_intent and want_intent != intent:
            return False
        if want_urgency and want_urgency != urgency:
            return False
        if keywords and not any(keyword in lowered_text for keyword in keywords):
            return False
        # An override with no conditions at all must not match everything.
        return bool(want_intent or want_urgency or keywords)

    def _approval_reasons(
        self,
        *,
        urgency: str,
        confidence: float,
        customer_tier: str | None,
        requested_actions: tuple[str, ...],
        queue: str,
    ) -> tuple[str, ...]:
        approval = self._policy.approval_policy
        reasons: list[str] = []

        for action in requested_actions:
            if action in approval.required_for_actions:
                reasons.append(f"action '{action}' requires human approval")

        if queue == "security_escalation":
            reasons.append("security escalations require human confirmation")

        for condition in approval.required_for_conditions:
            if (condition.urgency_in and urgency in condition.urgency_in) or (
                customer_tier
                and condition.customer_tier_in
                and customer_tier in condition.customer_tier_in
                and requested_actions
            ) or (condition.max_confidence is not None and confidence < condition.max_confidence):
                reasons.append(condition.description or condition.name)

        return tuple(dict.fromkeys(reasons))

    def requires_approval(self, action: str) -> bool:
        approval = self._policy.approval_policy
        if action in approval.auto_approved_actions:
            return False
        return action in approval.required_for_actions
