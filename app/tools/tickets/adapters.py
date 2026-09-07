"""Ticket system adapters.

`TicketAdapter` is the vendor boundary. The agent and routing engine talk only to
this interface, so connecting ServiceNow, Zendesk or Jira Service Management later
means writing one adapter, not touching the agent.

`MockTicketAdapter` is an in-memory implementation seeded with the specification's
example tickets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


class TicketStatus(StrEnum):
    NEW = "new"
    OPEN = "open"
    PENDING = "pending"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class CustomerTier(StrEnum):
    FREE = "free"
    STANDARD = "standard"
    PREMIUM = "premium"
    PLATINUM = "platinum"
    ENTERPRISE = "enterprise"


class Ticket(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticket_id: str
    subject: str = ""
    body: str
    customer_id: str | None = None
    status: TicketStatus = TicketStatus.NEW
    queue: str | None = None
    priority: str | None = None
    language: str | None = None
    tags: tuple[str, ...] = ()
    comments: tuple[dict[str, Any], ...] = ()
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def text(self) -> str:
        return f"{self.subject}\n{self.body}".strip()


class Customer(BaseModel):
    model_config = ConfigDict(frozen=True)

    customer_id: str
    name: str = ""
    email: str = ""
    tier: CustomerTier = CustomerTier.STANDARD
    lifetime_value_usd: float = 0.0
    account_status: str = "active"
    locale: str = "en"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Escalation(BaseModel):
    model_config = ConfigDict(frozen=True)

    escalation_id: str
    ticket_id: str
    reason: str
    severity: str = "high"
    created_at: datetime = Field(default_factory=_utcnow)
    acknowledged: bool = False


class TicketNotFoundError(KeyError):
    """Raised when a ticket ID does not exist."""


class TicketAdapter(ABC):
    """Vendor-neutral ticket system interface."""

    name: str

    @abstractmethod
    async def get_ticket(self, ticket_id: str) -> Ticket: ...

    @abstractmethod
    async def update_ticket(self, ticket_id: str, **fields: Any) -> Ticket: ...

    @abstractmethod
    async def add_comment(
        self, ticket_id: str, body: str, *, author: str = "agent", internal: bool = True
    ) -> Ticket: ...

    @abstractmethod
    async def get_customer(self, customer_id: str) -> Customer | None: ...

    @abstractmethod
    async def get_customer_history(self, customer_id: str, *, limit: int = 10) -> list[Ticket]: ...

    @abstractmethod
    async def create_escalation(
        self, ticket_id: str, reason: str, *, severity: str = "high"
    ) -> Escalation: ...

    @abstractmethod
    async def list_tickets(self) -> list[Ticket]: ...


class MockTicketAdapter(TicketAdapter):
    """In-memory ticket store for the prototype."""

    name = "mock"

    def __init__(self, *, seed: bool = True) -> None:
        self._tickets: dict[str, Ticket] = {}
        self._customers: dict[str, Customer] = {}
        self._escalations: dict[str, Escalation] = {}
        self._escalation_counter = 0
        if seed:
            self._seed()

    def _seed(self) -> None:
        customers = [
            Customer(
                customer_id="CUST-1001",
                name="Anita Sharma",
                email="anita.sharma@example.com",
                tier=CustomerTier.PREMIUM,
                lifetime_value_usd=2400.0,
                locale="en",
            ),
            Customer(
                customer_id="CUST-1002",
                name="Rahul Verma",
                email="rahul.verma@example.com",
                tier=CustomerTier.STANDARD,
                lifetime_value_usd=180.0,
                locale="hi",
            ),
            Customer(
                customer_id="CUST-1003",
                name="Northwind Trading",
                email="ops@northwind.example.com",
                tier=CustomerTier.ENTERPRISE,
                lifetime_value_usd=145000.0,
                locale="en",
            ),
        ]
        for customer in customers:
            self._customers[customer.customer_id] = customer

        tickets = [
            Ticket(
                ticket_id="TICK-001",
                subject="Charged twice for my subscription",
                body="My payment was deducted twice and I urgently need a refund.",
                customer_id="CUST-1001",
            ),
            Ticket(
                ticket_id="TICK-002",
                subject="Login problem",
                body="Mera account login nahi ho raha.",
                customer_id="CUST-1002",
            ),
            Ticket(
                ticket_id="TICK-003",
                subject="Suspicious activity on our account",
                body=(
                    "Someone else logged in to our admin account last night and changed "
                    "the billing email. We think the account is hacked."
                ),
                customer_id="CUST-1003",
            ),
            Ticket(
                ticket_id="TICK-004",
                subject="Feature idea",
                body="It would be nice if you could add dark mode to the dashboard.",
                customer_id="CUST-1001",
            ),
            Ticket(
                ticket_id="TICK-005",
                subject="API returning 500",
                body=(
                    "Our webhook integration started returning a 500 error this morning. "
                    "The stack trace mentions a timeout connecting to your API."
                ),
                customer_id="CUST-1003",
            ),
        ]
        for ticket in tickets:
            self._tickets[ticket.ticket_id] = ticket

    async def get_ticket(self, ticket_id: str) -> Ticket:
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            raise TicketNotFoundError(f"Ticket '{ticket_id}' not found")
        return ticket

    async def update_ticket(self, ticket_id: str, **fields: Any) -> Ticket:
        ticket = await self.get_ticket(ticket_id)
        allowed = {
            key: value
            for key, value in fields.items()
            if key in Ticket.model_fields and key != "ticket_id"
        }
        updated = ticket.model_copy(update={**allowed, "updated_at": _utcnow()})
        self._tickets[ticket_id] = updated
        return updated

    async def add_comment(
        self, ticket_id: str, body: str, *, author: str = "agent", internal: bool = True
    ) -> Ticket:
        ticket = await self.get_ticket(ticket_id)
        comment = {
            "author": author,
            "body": body,
            "internal": internal,
            "created_at": _utcnow().isoformat(),
        }
        updated = ticket.model_copy(
            update={"comments": (*ticket.comments, comment), "updated_at": _utcnow()}
        )
        self._tickets[ticket_id] = updated
        return updated

    async def get_customer(self, customer_id: str) -> Customer | None:
        return self._customers.get(customer_id)

    async def get_customer_history(self, customer_id: str, *, limit: int = 10) -> list[Ticket]:
        history = [
            ticket for ticket in self._tickets.values() if ticket.customer_id == customer_id
        ]
        history.sort(key=lambda ticket: ticket.created_at, reverse=True)
        return history[:limit]

    async def create_escalation(
        self, ticket_id: str, reason: str, *, severity: str = "high"
    ) -> Escalation:
        await self.get_ticket(ticket_id)
        self._escalation_counter += 1
        escalation = Escalation(
            escalation_id=f"ESC-{self._escalation_counter:04d}",
            ticket_id=ticket_id,
            reason=reason,
            severity=severity,
        )
        self._escalations[escalation.escalation_id] = escalation
        await self.update_ticket(ticket_id, status=TicketStatus.ESCALATED)
        return escalation

    async def list_tickets(self) -> list[Ticket]:
        return list(self._tickets.values())

    def create_ticket(self, ticket: Ticket) -> Ticket:
        """Test/demo helper for injecting an ad-hoc ticket."""
        self._tickets[ticket.ticket_id] = ticket
        return ticket
