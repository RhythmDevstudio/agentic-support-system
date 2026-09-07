"""Phase 1: log redaction guardrail.

Credentials must never reach the log stream, and ticket PII must be masked when
redaction is enabled. These are asserted at the processor level so the guarantee
holds for every call site.
"""

from __future__ import annotations

from app.observability.logging import REDACTED, make_redaction_processor


def _process(event: dict[str, object], *, redact_pii: bool = True) -> dict[str, object]:
    processor = make_redaction_processor(redact_pii)
    return processor(None, "info", event)


class TestSecretRedaction:
    def test_redacts_by_key_name(self) -> None:
        result = _process({"openai_api_key": "sk-live-abcdef1234567890", "event": "call"})
        assert result["openai_api_key"] == REDACTED

    def test_redacts_assorted_secret_key_names(self) -> None:
        result = _process(
            {
                "password": "hunter2",
                "authorization": "Bearer xyz",
                "database_url": "postgresql://u:p@host/db",
                "access_token": "abc123",
            }
        )
        assert all(value == REDACTED for value in result.values())

    def test_redacts_openai_key_embedded_in_free_text(self) -> None:
        result = _process({"message": "failed using sk-proj-AbCdEf0123456789xyz for auth"})
        assert "sk-proj-AbCdEf0123456789xyz" not in str(result["message"])
        assert REDACTED in str(result["message"])

    def test_redacts_tavily_key_in_free_text(self) -> None:
        result = _process({"message": "key tvly-abc123def456 rejected"})
        assert "tvly-abc123def456" not in str(result["message"])

    def test_redacts_connection_string_in_free_text(self) -> None:
        result = _process({"error": "could not connect to postgresql://user:pw@db:5432/app"})
        assert "user:pw" not in str(result["error"])

    def test_redacts_secrets_nested_in_structures(self) -> None:
        result = _process({"tool_args": {"config": {"api_key": "sk-nested-1234567890abc"}}})
        assert result["tool_args"] == {"config": {"api_key": REDACTED}}

    def test_redacts_secrets_inside_lists(self) -> None:
        result = _process({"items": ["harmless", "sk-inlist-1234567890abcdef"]})
        assert "sk-inlist-1234567890abcdef" not in str(result["items"])


class TestPIIRedaction:
    def test_masks_email(self) -> None:
        result = _process({"ticket_body": "contact me at jane.doe@example.com please"})
        assert "jane.doe@example.com" not in str(result["ticket_body"])
        assert "[EMAIL]" in str(result["ticket_body"])

    def test_masks_phone_number(self) -> None:
        result = _process({"ticket_body": "call me on +91 98765 43210"})
        assert "9876543210" not in str(result["ticket_body"]).replace(" ", "")

    def test_masks_card_number(self) -> None:
        result = _process({"ticket_body": "charged card 4111 1111 1111 1111 twice"})
        assert "4111" not in str(result["ticket_body"])

    def test_pii_preserved_when_redaction_disabled(self) -> None:
        result = _process({"ticket_body": "reach me at jane@example.com"}, redact_pii=False)
        assert "jane@example.com" in str(result["ticket_body"])

    def test_secrets_still_redacted_when_pii_redaction_disabled(self) -> None:
        """Secret redaction is unconditional - it is not a privacy toggle."""
        result = _process({"message": "token sk-still-secret-1234567890"}, redact_pii=False)
        assert "sk-still-secret-1234567890" not in str(result["message"])

    def test_ordinary_content_is_untouched(self) -> None:
        result = _process({"event": "retrieval_complete", "chunks": 8, "latency_ms": 142})
        assert result["event"] == "retrieval_complete"
        assert result["chunks"] == 8
        assert result["latency_ms"] == 142
