"""Typed ticket tools with permission checks.

Each tool declares whether it mutates state. Mutating tools whose action appears
in the approval policy are refused unless an approval has been recorded - the
check lives here, in code, not in the prompt, so a persuaded model still cannot
issue a refund on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config.policies import RoutingPolicy, get_routing_policy
from app.observability.logging import get_logger
from app.tools.tickets.adapters import (
    Customer,
    Escalation,
    Ticket,
    TicketAdapter,
    TicketNotFoundError,
)

logger = get_logger(__name__)


class ToolPermissionError(PermissionError):
    """Raised when a tool call is blocked by policy."""


@dataclass
class ToolResult:
    """Uniform tool outcome. Failures are values, not exceptions, so the agent
    can observe and report them."""

    tool: str
    ok: bool
    data: Any = None
    error: str | None = None
    requires_approval: bool = False
    approval_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        if self.requires_approval:
            return f"{self.tool}: awaiting human approval ({self.approval_reason})"
        return f"{self.tool}: {'ok' if self.ok else f'failed - {self.error}'}"


class TicketTools:
    """The ticket tool surface exposed to the agent."""

    READ_ONLY = frozenset(
        {"get_ticket", "get_customer", "get_customer_history"}
    )

    def __init__(
        self,
        adapter: TicketAdapter,
        policy: RoutingPolicy | None = None,
        *,
        approvals: set[str] | None = None,
    ) -> None:
        self._adapter = adapter
        self._policy = policy or get_routing_policy()
        # Actions a human has explicitly approved for this run.
        self._approvals = approvals if approvals is not None else set()

    def grant_approval(self, action: str) -> None:
        self._approvals.add(action)

    def has_approval(self, action: str) -> bool:
        return action in self._approvals

    def _check_permission(self, action: str) -> ToolResult | None:
        approval = self._policy.approval_policy
        if action in approval.auto_approved_actions or action in self.READ_ONLY:
            return None
        if action in approval.required_for_actions and action not in self._approvals:
            logger.info("tool_blocked_pending_approval", tool=action)
            return ToolResult(
                tool=action,
                ok=False,
                requires_approval=True,
                approval_reason=f"'{action}' requires recorded human approval before execution",
            )
        return None

    # -- read-only -----------------------------------------------------------

    async def get_ticket(self, ticket_id: str) -> ToolResult:
        try:
            ticket = await self._adapter.get_ticket(ticket_id)
            return ToolResult(tool="get_ticket", ok=True, data=ticket)
        except TicketNotFoundError as exc:
            return ToolResult(tool="get_ticket", ok=False, error=str(exc))

    async def get_customer(self, customer_id: str) -> ToolResult:
        customer = await self._adapter.get_customer(customer_id)
        if customer is None:
            return ToolResult(
                tool="get_customer", ok=False, error=f"Customer '{customer_id}' not found"
            )
        return ToolResult(tool="get_customer", ok=True, data=customer)

    async def get_customer_history(self, customer_id: str, *, limit: int = 10) -> ToolResult:
        history = await self._adapter.get_customer_history(customer_id, limit=limit)
        return ToolResult(tool="get_customer_history", ok=True, data=history)

    # -- mutating ------------------------------------------------------------

    async def assign_queue(self, ticket_id: str, queue: str) -> ToolResult:
        # A queue that is not in the policy is never written, whatever produced it.
        if not self._policy.is_valid_queue(queue):
            return ToolResult(
                tool="assign_queue", ok=False, error=f"'{queue}' is not a defined queue"
            )
        blocked = self._check_permission("assign_queue")
        if blocked:
            return blocked
        try:
            ticket = await self._adapter.update_ticket(ticket_id, queue=queue, status="open")
        except TicketNotFoundError as exc:
            return ToolResult(tool="assign_queue", ok=False, error=str(exc))
        return ToolResult(tool="assign_queue", ok=True, data=ticket)

    async def update_priority(self, ticket_id: str, priority: str) -> ToolResult:
        if not self._policy.is_valid_urgency(priority):
            return ToolResult(
                tool="update_priority", ok=False, error=f"'{priority}' is not a valid priority"
            )
        blocked = self._check_permission("update_priority")
        if blocked:
            return blocked
        try:
            ticket = await self._adapter.update_ticket(ticket_id, priority=priority)
        except TicketNotFoundError as exc:
            return ToolResult(tool="update_priority", ok=False, error=str(exc))
        return ToolResult(tool="update_priority", ok=True, data=ticket)

    async def add_ticket_comment(
        self, ticket_id: str, body: str, *, internal: bool = True
    ) -> ToolResult:
        blocked = self._check_permission("add_ticket_comment")
        if blocked:
            return blocked
        try:
            ticket = await self._adapter.add_comment(ticket_id, body, internal=internal)
        except TicketNotFoundError as exc:
            return ToolResult(tool="add_ticket_comment", ok=False, error=str(exc))
        return ToolResult(tool="add_ticket_comment", ok=True, data=ticket)

    async def update_ticket(self, ticket_id: str, **fields: Any) -> ToolResult:
        blocked = self._check_permission("update_ticket")
        if blocked:
            return blocked
        try:
            ticket = await self._adapter.update_ticket(ticket_id, **fields)
        except TicketNotFoundError as exc:
            return ToolResult(tool="update_ticket", ok=False, error=str(exc))
        return ToolResult(tool="update_ticket", ok=True, data=ticket)

    async def create_escalation(
        self, ticket_id: str, reason: str, *, severity: str = "high"
    ) -> ToolResult:
        blocked = self._check_permission("security_escalation")
        if blocked:
            return blocked
        try:
            escalation = await self._adapter.create_escalation(
                ticket_id, reason, severity=severity
            )
        except TicketNotFoundError as exc:
            return ToolResult(tool="create_escalation", ok=False, error=str(exc))
        return ToolResult(tool="create_escalation", ok=True, data=escalation)

    async def issue_refund(self, ticket_id: str, amount_usd: float) -> ToolResult:
        """High-risk action. Always gated - there is no unattended path."""
        blocked = self._check_permission("issue_refund")
        if blocked:
            return blocked
        ticket = await self._adapter.add_comment(
            ticket_id,
            f"Refund of {amount_usd:.2f} USD submitted after recorded human approval.",
            internal=True,
        )
        return ToolResult(
            tool="issue_refund", ok=True, data=ticket, metadata={"amount_usd": amount_usd}
        )

    async def request_human_approval(self, action: str, reason: str) -> ToolResult:
        """Record that the agent is asking for approval. Never self-approves."""
        return ToolResult(
            tool="request_human_approval",
            ok=True,
            data={"action": action, "reason": reason, "granted": False},
            requires_approval=True,
            approval_reason=reason,
        )


__all__ = [
    "Customer",
    "Escalation",
    "Ticket",
    "TicketTools",
    "ToolPermissionError",
    "ToolResult",
]
