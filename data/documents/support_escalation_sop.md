# Support Escalation Standard Operating Procedure

**Document owner:** Support Operations
**Version:** 2.1
**Last reviewed:** 2026-02-02

## Scope

This SOP defines when and how a support ticket is escalated, which queue owns it,
and what response commitments apply.

## Queue Ownership

| Queue | Owns | First response target |
| --- | --- | --- |
| Billing Support | Payments, invoices, refunds, chargebacks | 8 hours |
| Account Support | Login, authentication, profile, account lifecycle | 12 hours |
| Technical Support | Product defects, integrations, API failures | 24 hours |
| Security Escalation | Compromise, breach, vulnerability reports | 1 hour |
| Retention | Cancellation and win-back | 24 hours |
| Product Feedback | Feature requests and enhancements | 5 business days |
| Human Triage | Ambiguous or low-confidence tickets | 4 hours |

## Urgency Definitions

- **Critical** — Complete loss of service, confirmed security compromise, or
  financial loss in progress. Paged immediately, 24x7.
- **High** — Major function unavailable with no workaround, or a disputed charge.
  Handled within the business day.
- **Medium** — Degraded function with a workaround available.
- **Low** — Question, cosmetic issue, or enhancement request.

## Mandatory Escalation Triggers

A ticket must be escalated immediately, regardless of its original queue, when any
of the following is present:

1. Any indication of unauthorized account access or credential compromise.
2. Any report of customer data being visible to another customer.
3. A regulatory or legal threat, including a data protection authority complaint.
4. A payment failure affecting more than 10 customers within one hour.
5. Media or social-media amplification of a customer complaint.

## Security Escalation Procedure

Security escalations bypass the normal queue ladder.

1. Move the ticket to Security Escalation immediately.
2. Do not ask the customer to send credentials, tokens or session cookies. Ever.
3. Record the affected account identifiers only. Do not paste raw session data into
   the ticket.
4. Notify the on-call security engineer through the incident channel.
5. A human security engineer must confirm receipt. Automated systems may create a
   security escalation but may not close one.

## Human Approval Requirements

The following actions may never be executed by an automated agent without a recorded
human approval:

- Issuing any refund.
- Deleting or closing a customer account.
- Overriding a ticket priority set by a human.
- Waiving a contractual SLA.
- Any mutating action on an enterprise or platinum-tier account.
- Confirming or closing a security escalation.

An automated agent may recommend any of these actions and may gather the evidence
needed to decide, but the decision itself is reserved for a human.

## Language Handling

Tickets are accepted in any language. Tickets in English, Hindi and Hinglish
(romanized Hindi) are handled directly by the support team. Other languages are
routed to the same functional queue with a translation flag set, and the first
response is provided in the customer's original language where possible.

Romanized Hindi is common in Indian consumer tickets. Phrases such as
"login nahi ho raha" (cannot log in), "paisa cut gaya" (money was deducted) and
"refund chahiye" (need a refund) should be treated with the same intent mapping as
their English equivalents.

## Response Content Rules

- Never state a refund has been issued until the billing system confirms submission.
- Never quote an internal approval threshold to a customer.
- Never speculate about the cause of a security incident in customer-facing text.
