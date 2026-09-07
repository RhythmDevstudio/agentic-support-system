"""Phase 1: configuration and policy validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app.config.policies import (
    PolicyError,
    RoutingPolicy,
    VendorRegistry,
    load_routing_policy,
    load_vendor_registry,
)
from app.config.settings import (
    DomainMode,
    EmbeddingProviderName,
    LLMProviderName,
    Settings,
    VectorStoreName,
    WebSearchProviderName,
)


class TestSettings:
    def test_boots_with_empty_environment(self, settings: Settings) -> None:
        assert settings.app_env == "development"
        assert settings.llm_synthesis_model
        assert settings.embedding_dimensions > 0

    def test_blank_env_values_are_treated_as_unset(self) -> None:
        loaded = Settings(_env_file=None, OPENAI_API_KEY="   ")  # type: ignore[call-arg]
        assert loaded.openai_api_key is None

    def test_auto_degrades_to_offline_without_credentials(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        loaded = Settings(_env_file=None)  # type: ignore[call-arg]
        assert loaded.resolve_llm_provider() is LLMProviderName.DETERMINISTIC
        assert loaded.resolve_embedding_provider() is EmbeddingProviderName.HASHING
        assert loaded.resolve_vector_store() is VectorStoreName.SQLITE
        assert loaded.resolve_web_search_provider() is WebSearchProviderName.FIXTURE
        assert loaded.is_offline_mode() is True

    def test_auto_selects_openai_when_key_present(self) -> None:
        loaded = Settings(  # type: ignore[call-arg]
            _env_file=None,
            OPENAI_API_KEY="sk-test-key",
            LLM_PROVIDER="auto",
            EMBEDDING_PROVIDER="auto",
        )
        assert loaded.resolve_llm_provider() is LLMProviderName.OPENAI
        assert loaded.resolve_embedding_provider() is EmbeddingProviderName.OPENAI
        assert loaded.is_offline_mode() is False

    def test_explicit_provider_overrides_auto_detection(self) -> None:
        loaded = Settings(  # type: ignore[call-arg]
            _env_file=None,
            OPENAI_API_KEY="sk-test-key",
            LLM_PROVIDER="deterministic",
        )
        assert loaded.resolve_llm_provider() is LLMProviderName.DETERMINISTIC

    def test_secrets_are_not_exposed_by_repr(self) -> None:
        loaded = Settings(_env_file=None, OPENAI_API_KEY="sk-super-secret")  # type: ignore[call-arg]
        assert "sk-super-secret" not in repr(loaded)
        assert loaded.openai_api_key is not None
        assert loaded.openai_api_key.get_secret_value() == "sk-super-secret"

    def test_rejects_out_of_range_hybrid_alpha(self) -> None:
        with pytest.raises(ValueError, match="retrieval_hybrid_alpha"):
            Settings(_env_file=None, RETRIEVAL_HYBRID_ALPHA=1.5)  # type: ignore[call-arg]

    def test_cors_origins_parse_to_list(self) -> None:
        loaded = Settings(  # type: ignore[call-arg]
            _env_file=None, API_CORS_ORIGINS="http://a.com, http://b.com"
        )
        assert loaded.cors_origin_list() == ["http://a.com", "http://b.com"]


class TestVendorRegistry:
    def test_loads_expected_vendors(self, vendor_registry: VendorRegistry) -> None:
        for key in ("microsoft_azure", "openai", "langchain", "aws", "google_cloud"):
            assert vendor_registry.get(key) is not None

    def test_official_domains_match_specification(self, vendor_registry: VendorRegistry) -> None:
        azure = vendor_registry.get("microsoft_azure")
        assert azure is not None
        assert "learn.microsoft.com" in azure.official_domains

        openai_vendor = vendor_registry.get("openai")
        assert openai_vendor is not None
        assert "openai.github.io" in openai_vendor.official_domains

        langchain = vendor_registry.get("langchain")
        assert langchain is not None
        assert "docs.langchain.com" in langchain.official_domains

        aws = vendor_registry.get("aws")
        assert aws is not None
        assert "docs.aws.amazon.com" in aws.official_domains

        gcp = vendor_registry.get("google_cloud")
        assert gcp is not None
        assert "cloud.google.com" in gcp.official_domains

    def test_domain_tiers_rank_primary_docs_highest(
        self, vendor_registry: VendorRegistry
    ) -> None:
        azure = vendor_registry.get("microsoft_azure")
        assert azure is not None
        assert azure.tier_for_domain("learn.microsoft.com") == 1

    def test_tier_lookup_matches_subdomains(self, vendor_registry: VendorRegistry) -> None:
        aws = vendor_registry.get("aws")
        assert aws is not None
        # A doc subdomain should inherit its parent's tier rather than falling through.
        assert aws.tier_for_domain("docs.aws.amazon.com") == 1

    def test_content_farms_are_blocked(self, vendor_registry: VendorRegistry) -> None:
        assert vendor_registry.is_blocked("w3schools.com") is True
        assert vendor_registry.is_blocked("learn.microsoft.com") is False

    def test_default_domain_mode_is_strict(self, vendor_registry: VendorRegistry) -> None:
        assert vendor_registry.default_domain_mode is DomainMode.STRICT

    def test_rejects_vendor_with_no_official_domains(self, tmp_path: Path) -> None:
        path = tmp_path / "vendors.yaml"
        path.write_text(
            textwrap.dedent(
                """
                version: 1
                vendors:
                  broken:
                    display_name: Broken
                    official_domains: []
                """
            )
        )
        with pytest.raises(PolicyError, match="no official_domains"):
            load_vendor_registry(path)

    def test_rejects_tier_for_undeclared_domain(self, tmp_path: Path) -> None:
        path = tmp_path / "vendors.yaml"
        path.write_text(
            textwrap.dedent(
                """
                version: 1
                vendors:
                  acme:
                    display_name: Acme
                    official_domains: [docs.acme.com]
                    domain_tiers:
                      blog.acme.com: 1
                """
            )
        )
        with pytest.raises(PolicyError, match="not in official_domains"):
            load_vendor_registry(path)

    def test_missing_file_raises_policy_error(self, tmp_path: Path) -> None:
        with pytest.raises(PolicyError, match="not found"):
            load_vendor_registry(tmp_path / "absent.yaml")


class TestRoutingPolicy:
    def test_specification_intents_are_present(self, routing_policy: RoutingPolicy) -> None:
        expected = {
            "payment_issue",
            "refund_request",
            "login_issue",
            "password_reset",
            "technical_issue",
            "bug_report",
            "account_issue",
            "cancellation",
            "feature_request",
            "security_issue",
            "general_query",
        }
        assert expected.issubset(set(routing_policy.intents))

    def test_urgency_levels_match_specification(self, routing_policy: RoutingPolicy) -> None:
        assert routing_policy.urgency_levels == ("critical", "high", "medium", "low")

    @pytest.mark.parametrize(
        ("intent", "queue"),
        [
            ("payment_issue", "billing_support"),
            ("refund_request", "billing_support"),
            ("login_issue", "account_support"),
            ("password_reset", "account_support"),
            ("technical_issue", "technical_support"),
            ("bug_report", "technical_support"),
            ("security_issue", "security_escalation"),
        ],
    )
    def test_specified_routing_rules(
        self, routing_policy: RoutingPolicy, intent: str, queue: str
    ) -> None:
        assert routing_policy.queue_for_intent(intent) == queue

    def test_every_intent_routes_somewhere(self, routing_policy: RoutingPolicy) -> None:
        for intent in routing_policy.intents:
            queue = routing_policy.queue_for_intent(intent)
            assert routing_policy.is_valid_queue(queue), f"{intent} -> unknown queue {queue}"

    def test_urgency_ranking_is_ordered(self, routing_policy: RoutingPolicy) -> None:
        assert routing_policy.urgency_rank("critical") > routing_policy.urgency_rank("high")
        assert routing_policy.urgency_rank("high") > routing_policy.urgency_rank("medium")
        assert routing_policy.urgency_rank("medium") > routing_policy.urgency_rank("low")
        assert routing_policy.urgency_rank("nonsense") == -1

    def test_high_risk_actions_require_approval(self, routing_policy: RoutingPolicy) -> None:
        required = routing_policy.approval_policy.required_for_actions
        assert "issue_refund" in required
        assert "account_deletion" in required
        assert "security_escalation" in required

    def test_read_only_actions_are_auto_approved(self, routing_policy: RoutingPolicy) -> None:
        auto = routing_policy.approval_policy.auto_approved_actions
        assert "get_ticket" in auto
        assert "issue_refund" not in auto

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("English", "en"),
            ("english", "en"),
            ("en", "en"),
            ("en-US", "en"),
            ("Hindi", "hi"),
            ("hi", "hi"),
            ("Hinglish", "hi-Latn"),
            ("hi-Latn", "hi-Latn"),
            ("Klingon", "en"),  # unsupported falls back to default
            (None, "en"),
            ("", "en"),
        ],
    )
    def test_language_normalisation(
        self, routing_policy: RoutingPolicy, raw: str | None, expected: str
    ) -> None:
        assert routing_policy.languages.normalise(raw) == expected

    def test_rejects_intent_without_routing_rule(self, tmp_path: Path) -> None:
        path = tmp_path / "routing.yaml"
        path.write_text(
            textwrap.dedent(
                """
                version: 1
                intents: [alpha, beta]
                urgency_levels: [critical, high, medium, low]
                queues:
                  q1: {display_name: Q1, sla_hours: 1}
                routing_rules:
                  alpha: q1
                fallback_queue: q1
                languages:
                  supported: [{code: en, name: English}]
                  default: en
                """
            )
        )
        with pytest.raises(PolicyError, match="no routing rule"):
            load_routing_policy(path)

    def test_rejects_rule_pointing_at_undefined_queue(self, tmp_path: Path) -> None:
        path = tmp_path / "routing.yaml"
        path.write_text(
            textwrap.dedent(
                """
                version: 1
                intents: [alpha]
                urgency_levels: [critical, high, medium, low]
                queues:
                  q1: {display_name: Q1, sla_hours: 1}
                routing_rules:
                  alpha: ghost_queue
                fallback_queue: q1
                languages:
                  supported: [{code: en, name: English}]
                  default: en
                """
            )
        )
        with pytest.raises(PolicyError, match="undefined queues"):
            load_routing_policy(path)

    def test_rejects_undefined_fallback_queue(self, tmp_path: Path) -> None:
        path = tmp_path / "routing.yaml"
        path.write_text(
            textwrap.dedent(
                """
                version: 1
                intents: [alpha]
                urgency_levels: [critical, high, medium, low]
                queues:
                  q1: {display_name: Q1, sla_hours: 1}
                routing_rules:
                  alpha: q1
                fallback_queue: nowhere
                languages:
                  supported: [{code: en, name: English}]
                  default: en
                """
            )
        )
        with pytest.raises(PolicyError, match="fallback_queue"):
            load_routing_policy(path)
