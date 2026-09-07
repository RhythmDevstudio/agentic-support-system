"""Structured logging with secret and PII redaction.

Two rules this module enforces mechanically, so no call site has to remember them:

1. Credentials never reach the log stream. Values that look like API keys, bearer
   tokens or DSNs are replaced before rendering.
2. PII in ticket text is masked when `REDACT_PII_IN_LOGS` is on. Support tickets
   routinely carry emails, phone numbers and card fragments; those must not end
   up in a log aggregator.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

import structlog

from app.config.settings import Settings, get_settings

REDACTED = "[REDACTED]"

# Keys whose values are always dropped, regardless of content.
_SECRET_KEY_PATTERN = re.compile(
    r"(api[_-]?key|secret|password|passwd|token|authorization|auth|credential|dsn|database_url)",
    re.IGNORECASE,
)

# Value-level patterns for credentials that arrive inside free text.
_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),  # OpenAI-style keys
    re.compile(r"tvly-[A-Za-z0-9_\-]{10,}"),  # Tavily keys
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{12,}", re.IGNORECASE),
    re.compile(r"\b(?:postgresql|postgres|mysql)://[^\s\"']+", re.IGNORECASE),
)

# PII patterns. Deliberately conservative - masking a false positive in a log
# line is harmless, leaking a real one is not.
_PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){9,13}\d\b"), "[PHONE]"),
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[CARD]"),
)


def _redact_text(value: str, redact_pii: bool) -> str:
    for pattern in _SECRET_VALUE_PATTERNS:
        value = pattern.sub(REDACTED, value)
    if redact_pii:
        for pattern, placeholder in _PII_PATTERNS:
            value = pattern.sub(placeholder, value)
    return value


def _redact_value(value: Any, redact_pii: bool, depth: int = 0) -> Any:
    if depth > 6:  # guard against pathological nesting
        return value
    if isinstance(value, str):
        return _redact_text(value, redact_pii)
    if isinstance(value, dict):
        return {
            key: (
                REDACTED
                if isinstance(key, str) and _SECRET_KEY_PATTERN.search(key)
                else _redact_value(item, redact_pii, depth + 1)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        rebuilt = [_redact_value(item, redact_pii, depth + 1) for item in value]
        return type(value)(rebuilt) if isinstance(value, tuple) else rebuilt
    return value


def make_redaction_processor(redact_pii: bool) -> Any:
    """structlog processor that scrubs secrets (always) and PII (when enabled)."""

    def processor(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        return {
            key: (
                REDACTED
                if isinstance(key, str) and _SECRET_KEY_PATTERN.search(key)
                else _redact_value(value, redact_pii)
            )
            for key, value in event_dict.items()
        }

    return processor


def configure_logging(settings: Settings | None = None) -> None:
    """Idempotently configure structlog for the process."""
    settings = settings or get_settings()

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level),
        force=True,
    )
    # Third-party loggers are noisy at DEBUG and can echo request bodies.
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if settings.log_format == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            make_redaction_processor(settings.redact_pii_in_logs),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """Return a bound structlog logger for a module."""
    return structlog.get_logger(name)
