"""Prompt-injection detection for retrieved content.

Retrieved documents and web pages are untrusted input. The primary defence is
structural - content is fenced in `<untrusted_document>` tags with explicit
instructions that it is data - and this scanner is the second layer: it flags
passages that are actively trying to give the model instructions, so they can be
quarantined before they reach generation.

Detection is conservative by design. Documentation legitimately contains phrases
like "ignore the previous step", so patterns target the *combination* of an
override verb with an instruction target.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from app.models.evidence import Evidence, QuarantineReason


class InjectionSeverity(StrEnum):
    NONE = "none"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"


@dataclass(frozen=True)
class InjectionFinding:
    pattern_name: str
    severity: InjectionSeverity
    excerpt: str


@dataclass(frozen=True)
class InjectionScanResult:
    severity: InjectionSeverity
    findings: tuple[InjectionFinding, ...] = ()

    @property
    def is_malicious(self) -> bool:
        return self.severity is InjectionSeverity.MALICIOUS

    @property
    def is_clean(self) -> bool:
        return self.severity is InjectionSeverity.NONE

    def describe(self) -> str:
        if self.is_clean:
            return "no injection indicators"
        names = ", ".join(sorted({finding.pattern_name for finding in self.findings}))
        return f"{self.severity.value}: {names}"


_MALICIOUS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b"
            r"(previous|prior|above|earlier|all)\b[^.\n]{0,30}\b"
            r"(instruction|prompt|direction|rule|system)\w*",
            re.IGNORECASE,
        ),
    ),
    (
        "role_reassignment",
        # No trailing \b: the "new instructions:" branch ends in a colon, and a
        # word boundary can never match between ':' and a following space.
        re.compile(
            r"\b(you are now|from now on you|act as|pretend to be|your new (role|task)|"
            r"new instructions?\s*:)",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt_exfiltration",
        re.compile(
            r"\b(reveal|show|print|output|repeat|disclose)\b[^.\n]{0,30}\b"
            r"(system prompt|your instructions|initial prompt|configuration)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "credential_exfiltration",
        re.compile(
            r"\b(send|post|email|upload|transmit|forward|exfiltrate)\b[^.\n]{0,40}\b"
            r"(api[_ -]?keys?|passwords?|tokens?|credentials?|secrets?|"
            r"customer (data|database|records?|list|details))",
            re.IGNORECASE,
        ),
    ),
    (
        "tool_invocation_lure",
        re.compile(
            r"\b(call|invoke|execute|run|use)\b[^.\n]{0,30}\b"
            r"(issue_refund|delete_account|account_deletion|update_ticket|assign_queue|"
            r"create_escalation|approve)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "fence_escape",
        re.compile(r"</?\s*(untrusted_document|system|instructions?)\s*>", re.IGNORECASE),
    ),
)

_SUSPICIOUS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "authority_claim",
        re.compile(
            r"\b(this is (a|an)? ?(system|admin|operator) (message|instruction|override)|"
            r"as the system administrator|priority override)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "urgency_manipulation",
        re.compile(
            r"\b(you must (immediately|now)|it is critical that you|do not tell the user|"
            r"without asking (for )?(approval|permission)|bypass)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "hidden_directive_marker",
        re.compile(r"\[\[\s*(system|instruction|prompt)\s*\]\]|<!--\s*ai:", re.IGNORECASE),
    ),
)


def scan_for_injection(text: str) -> InjectionScanResult:
    """Scan a passage for prompt-injection indicators."""
    if not text.strip():
        return InjectionScanResult(severity=InjectionSeverity.NONE)

    findings: list[InjectionFinding] = []

    for name, pattern in _MALICIOUS_PATTERNS:
        match = pattern.search(text)
        if match:
            findings.append(
                InjectionFinding(
                    pattern_name=name,
                    severity=InjectionSeverity.MALICIOUS,
                    excerpt=_excerpt(text, match),
                )
            )

    for name, pattern in _SUSPICIOUS_PATTERNS:
        match = pattern.search(text)
        if match:
            findings.append(
                InjectionFinding(
                    pattern_name=name,
                    severity=InjectionSeverity.SUSPICIOUS,
                    excerpt=_excerpt(text, match),
                )
            )

    if any(finding.severity is InjectionSeverity.MALICIOUS for finding in findings):
        severity = InjectionSeverity.MALICIOUS
    elif findings:
        severity = InjectionSeverity.SUSPICIOUS
    else:
        severity = InjectionSeverity.NONE

    return InjectionScanResult(severity=severity, findings=tuple(findings))


def _excerpt(text: str, match: re.Match[str], width: int = 60) -> str:
    start = max(0, match.start() - width // 2)
    end = min(len(text), match.end() + width // 2)
    return text[start:end].replace("\n", " ").strip()


def screen_evidence(items: list[Evidence]) -> tuple[list[Evidence], list[Evidence]]:
    """Split evidence into usable and quarantined.

    Only MALICIOUS findings quarantine. SUSPICIOUS content is kept but annotated,
    because over-quarantining silently removes legitimate documentation.
    """
    usable: list[Evidence] = []
    quarantined: list[Evidence] = []

    for item in items:
        result = scan_for_injection(item.content)
        if result.is_malicious:
            quarantined.append(
                item.quarantine(QuarantineReason.PROMPT_INJECTION, result.describe())
            )
        elif result.severity is InjectionSeverity.SUSPICIOUS:
            usable.append(
                item.model_copy(
                    update={"metadata": {**item.metadata, "injection_scan": result.describe()}}
                )
            )
        else:
            usable.append(item)

    return usable, quarantined
