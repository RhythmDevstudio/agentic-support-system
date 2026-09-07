"""Run the five specification test cases end to end.

    python -m scripts.demo

Works with no API key (offline deterministic providers). Set OPENAI_API_KEY in
.env and the same code paths run against a real model.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from app.guardrails.citations import render_citations
from app.services.container import Container, build_container

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m",
)


def rule(title: str) -> None:
    print(f"\n{BOLD}{'=' * 78}\n{title}\n{'=' * 78}{RESET}")


def check(label: str, passed: bool, detail: str = "") -> bool:
    mark = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {label}{f' - {detail}' if detail else ''}")
    return passed


async def research_case(
    container: Container, number: int, question: str, expectations: dict[str, Any]
) -> bool:
    rule(f"TEST {number}: {question}")

    state = await container.agent.run(container.new_state(question))
    result = dict(state)

    print(f"\n{BOLD}Answer:{RESET}\n{result.get('answer', '')}\n")
    citations = result.get("citations", [])
    if citations:
        print(render_citations(citations))
    print(f"\n{DIM}Decisions: {' | '.join(result.get('decision_log', []))}{RESET}")
    print(f"{DIM}Tools: {', '.join(i.tool for i in result.get('tool_invocations', []))}{RESET}")

    passed = True
    if "vendor" in expectations:
        passed &= check(
            f"identified vendor = {expectations['vendor']}",
            result.get("vendor_key") == expectations["vendor"],
            f"got {result.get('vendor_key')}",
        )
    if "domain" in expectations:
        urls = [c.url or "" for c in citations]
        passed &= check(
            f"cited {expectations['domain']}",
            any(expectations["domain"] in url for url in urls),
            f"urls: {urls}",
        )
    if expectations.get("internal_source"):
        passed &= check(
            "cited an internal document",
            any(str(c.source_type) == "internal_document" for c in citations),
        )
    passed &= check("returned at least one verified citation", bool(citations))
    passed &= check(
        "answer is grounded (not an insufficient-evidence fallback)",
        not result.get("insufficient_evidence", False),
    )
    return passed


async def ticket_case(
    container: Container, number: int, text: str, expectations: dict[str, Any]
) -> bool:
    rule(f"TEST {number}: {text}")

    state = container.new_state(text)
    state["workflow"] = "triage"
    result = dict(await container.agent.run(state))

    decision = result.get("routing_decision")
    if decision is None:
        check("produced a routing decision", False)
        return False

    print(f"\n{BOLD}Classification:{RESET}")
    print(f"  intent     : {decision.intent}")
    print(f"  urgency    : {decision.urgency}")
    print(f"  language   : {decision.language} ({decision.language_code})")
    print(f"  queue      : {decision.queue}")
    print(f"  confidence : {decision.confidence:.2f}")
    if decision.requires_escalation:
        print(f"  {YELLOW}escalation required{RESET}")
    if decision.requires_human_approval:
        print(f"  {YELLOW}human approval: {', '.join(decision.approval_reasons)}{RESET}")
    print(f"\n{DIM}Rules applied: {', '.join(decision.applied_rules)}{RESET}")

    passed = True
    passed &= check(
        f"intent in {expectations['intent']}",
        decision.intent in expectations["intent"],
        f"got {decision.intent}",
    )
    passed &= check(
        f"urgency = {expectations['urgency']}",
        decision.urgency == expectations["urgency"],
        f"got {decision.urgency}",
    )
    passed &= check(
        f"language in {expectations['language']}",
        decision.language in expectations["language"]
        or decision.language_code in expectations["language"],
        f"got {decision.language} / {decision.language_code}",
    )
    passed &= check(
        f"queue = {expectations['queue']}",
        decision.queue == expectations["queue"],
        f"got {decision.queue}",
    )
    return passed


async def main() -> int:
    container = build_container()
    await container.initialize()
    ingested = await container.ingest_seed_documents()

    print(f"{BOLD}Agentic AI Support & Knowledge Research System - MVP demo{RESET}")
    for key, value in container.health().items():
        print(f"  {key:<22}: {value}")
    chunk_count = await container.store.count_chunks()
    print(f"  {'knowledge base chunks':<22}: {chunk_count} ({ingested} new)")
    if container.settings.is_offline_mode():
        print(
            f"\n{YELLOW}Running in OFFLINE mode: rule-based provider, recorded doc fixtures.\n"
            f"Control flow, guardrails and citations are real; answer fluency is not.\n"
            f"Set OPENAI_API_KEY in .env to run the same paths against a live model.{RESET}"
        )

    results: list[tuple[str, bool]] = []
    results.append((
        "TEST 1 Azure AI Search",
        await research_case(
            container, 1, "What is Azure AI Search agentic retrieval?",
            {"vendor": "microsoft_azure", "domain": "learn.microsoft.com"},
        ),
    ))
    results.append((
        "TEST 2 OpenAI Agents SDK",
        await research_case(
            container, 2, "How do I use OpenAI Agents SDK tools?",
            {"vendor": "openai", "domain": "openai"},
        ),
    ))
    results.append((
        "TEST 3 Internal refund policy",
        await research_case(
            container, 3, "How do we process refunds according to our company policy?",
            {"internal_source": True},
        ),
    ))
    results.append((
        "TEST 4 Duplicate charge ticket",
        await ticket_case(
            container, 4, "My payment was deducted twice and I urgently need a refund.",
            {
                "intent": {"payment_issue", "refund_request"},
                "urgency": "high",
                "language": {"English", "en"},
                "queue": "billing_support",
            },
        ),
    ))
    results.append((
        "TEST 5 Hinglish login ticket",
        await ticket_case(
            container, 5, "Mera account login nahi ho raha.",
            {
                "intent": {"login_issue"},
                "urgency": "medium",
                "language": {"Hindi", "Hinglish", "hi", "hi-Latn"},
                "queue": "account_support",
            },
        ),
    ))

    rule("SUMMARY")
    for name, passed in results:
        print(f"  {GREEN + 'PASS' + RESET if passed else RED + 'FAIL' + RESET}  {name}")
    failures = sum(1 for _, passed in results if not passed)
    print(f"\n{len(results) - failures}/{len(results)} specification test cases passed.")

    await container.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
