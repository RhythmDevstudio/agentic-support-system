"""Phase 3: official-source policy - vendor detection, allowlisting, authority ranking."""

from __future__ import annotations

import pytest

from app.config.policies import VendorRegistry
from app.config.settings import DomainMode
from app.models.documents import SourceType
from app.tools.web_research.domain_policy import (
    OfficialSourcePolicy,
    VendorDetector,
    domain_matches,
    normalise_domain,
    repository_matches,
)


@pytest.fixture
def detector(vendor_registry: VendorRegistry) -> VendorDetector:
    return VendorDetector(vendor_registry)


@pytest.fixture
def policy(vendor_registry: VendorRegistry) -> OfficialSourcePolicy:
    return OfficialSourcePolicy(vendor_registry)


class TestDomainNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("https://learn.microsoft.com/en-us/azure/search", "learn.microsoft.com"),
            ("http://www.postgresql.org/docs", "postgresql.org"),
            ("learn.microsoft.com", "learn.microsoft.com"),
            ("HTTPS://LEARN.MICROSOFT.COM/x", "learn.microsoft.com"),
            ("https://docs.aws.amazon.com:443/lambda", "docs.aws.amazon.com"),
        ],
    )
    def test_normalises_urls_and_bare_domains(self, raw: str, expected: str) -> None:
        assert normalise_domain(raw) == expected

    def test_exact_match(self) -> None:
        assert domain_matches("learn.microsoft.com", "learn.microsoft.com") is True

    def test_subdomain_matches_parent(self) -> None:
        assert domain_matches("docs.aws.amazon.com", "aws.amazon.com") is True

    def test_lookalike_domain_is_rejected(self) -> None:
        """The check must be anchored on a dot, or typosquats slip through."""
        assert domain_matches("notmicrosoft.com", "microsoft.com") is False
        assert domain_matches("evil-learn.microsoft.com.attacker.io", "microsoft.com") is False

    def test_parent_does_not_match_child(self) -> None:
        assert domain_matches("microsoft.com", "learn.microsoft.com") is False


class TestRepositoryMatching:
    """Allowing a vendor's GitHub org must not allow all of GitHub."""

    def test_url_under_the_owner_matches(self) -> None:
        url = "https://github.com/Azure/azure-sdk-for-python"
        assert repository_matches(url, "github.com/Azure") is True

    def test_owner_root_matches(self) -> None:
        assert repository_matches("https://github.com/pgvector", "github.com/pgvector") is True

    def test_different_owner_is_rejected(self) -> None:
        url = "https://github.com/attacker/malware"
        assert repository_matches(url, "github.com/Azure") is False

    def test_owner_prefix_collision_is_rejected(self) -> None:
        """'github.com/Azure' must not match 'github.com/AzureEvil'."""
        assert repository_matches("https://github.com/AzureEvil/x", "github.com/Azure") is False

    def test_matching_is_case_insensitive(self) -> None:
        assert repository_matches("https://github.com/azure/repo", "github.com/Azure") is True

    def test_different_host_is_rejected(self) -> None:
        assert repository_matches("https://gitlab.com/Azure/x", "github.com/Azure") is False


class TestVendorDetection:
    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("What is Azure AI Search agentic retrieval?", "microsoft_azure"),
            ("How do I use OpenAI Agents SDK tools?", "openai"),
            ("How do I build a LangGraph StateGraph?", "langchain"),
            ("How does AWS Bedrock handle streaming?", "aws"),
            ("How do I deploy to Google Cloud Run?", "google_cloud"),
            ("How do I create an HNSW index in pgvector?", "postgresql"),
            ("What is the Model Context Protocol?", "anthropic"),
        ],
    )
    def test_detects_vendor_from_question(
        self, detector: VendorDetector, question: str, expected: str
    ) -> None:
        match = detector.detect(question)
        assert match is not None
        assert match.key == expected

    def test_returns_none_for_vendor_neutral_question(self, detector: VendorDetector) -> None:
        assert detector.detect("What is the capital of France?") is None

    def test_longer_alias_outweighs_shorter_one(self, detector: VendorDetector) -> None:
        """'azure ai search' is a far stronger signal than a bare 'gpt' mention."""
        match = detector.detect("Configure Azure AI Search for my gpt application")
        assert match is not None
        assert match.key == "microsoft_azure"

    def test_alias_must_match_on_a_word_boundary(self, detector: VendorDetector) -> None:
        """A substring hit inside an unrelated word must not select a vendor."""
        assert detector.detect("I am building a gpteam collaboration product") is None

    def test_model_hint_can_surface_a_vendor_keywords_missed(
        self, detector: VendorDetector
    ) -> None:
        match = detector.detect("How do I use the Responses API?", model_hint="openai")
        assert match is not None
        assert match.key == "openai"
        assert match.from_model_hint is True

    def test_model_hint_accepts_display_name_form(self, detector: VendorDetector) -> None:
        match = detector.detect("some question", model_hint="Microsoft Azure")
        assert match is not None
        assert match.key == "microsoft_azure"

    def test_unknown_model_hint_is_ignored(self, detector: VendorDetector) -> None:
        """A hallucinated vendor must not create one."""
        assert detector.detect("What is the capital of France?", model_hint="acme_corp") is None

    def test_hint_cannot_override_a_strong_keyword_signal(
        self, detector: VendorDetector
    ) -> None:
        match = detector.detect(
            "How do I configure Azure AI Search agentic retrieval with knowledge agents?",
            model_hint="aws",
        )
        assert match is not None
        assert match.key == "microsoft_azure"


class TestPolicyResolution:
    def test_microsoft_question_restricts_to_learn_microsoft(
        self, policy: OfficialSourcePolicy
    ) -> None:
        """TEST 1 requirement."""
        resolved = policy.resolve("What is Azure AI Search agentic retrieval?")
        assert resolved.vendor_key == "microsoft_azure"
        assert "learn.microsoft.com" in resolved.allowed_domains
        assert resolved.mode is DomainMode.STRICT
        assert resolved.is_restricted is True

    def test_openai_question_restricts_to_openai_domains(
        self, policy: OfficialSourcePolicy
    ) -> None:
        """TEST 2 requirement."""
        resolved = policy.resolve("How do I use OpenAI Agents SDK tools?")
        assert resolved.vendor_key == "openai"
        assert "openai.github.io" in resolved.allowed_domains
        assert "openai.com" in resolved.allowed_domains

    def test_langgraph_question_restricts_to_langchain_docs(
        self, policy: OfficialSourcePolicy
    ) -> None:
        resolved = policy.resolve("How do I add a conditional edge in LangGraph?")
        assert "docs.langchain.com" in resolved.allowed_domains

    def test_vendor_neutral_question_is_unrestricted(
        self, policy: OfficialSourcePolicy
    ) -> None:
        resolved = policy.resolve("What are good practices for writing tests?")
        assert resolved.vendor_key is None
        assert resolved.is_restricted is False

    def test_explicit_override_domains_are_honoured(
        self, policy: OfficialSourcePolicy
    ) -> None:
        resolved = policy.resolve("anything", override_domains=("example.org",))
        assert resolved.allowed_domains == ("example.org",)

    def test_override_cannot_reach_a_blocked_domain(
        self, policy: OfficialSourcePolicy
    ) -> None:
        """An override is a convenience, not an escape hatch from the blocklist."""
        resolved = policy.resolve(
            "anything", override_domains=("w3schools.com", "example.org")
        )
        assert "w3schools.com" not in resolved.allowed_domains
        assert "example.org" in resolved.allowed_domains

    def test_describe_is_human_readable(self, policy: OfficialSourcePolicy) -> None:
        resolved = policy.resolve("Azure AI Search agentic retrieval")
        assert "restricted to" in resolved.describe()
        assert "learn.microsoft.com" in resolved.describe()


class TestPermissionChecks:
    def test_official_url_is_permitted(self, policy: OfficialSourcePolicy) -> None:
        resolved = policy.resolve("Azure AI Search agentic retrieval")
        permitted, reason = policy.is_permitted(
            "https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-overview", resolved
        )
        assert permitted is True
        assert reason is None

    def test_blog_is_rejected_under_strict_mode(self, policy: OfficialSourcePolicy) -> None:
        resolved = policy.resolve("Azure AI Search agentic retrieval")
        permitted, reason = policy.is_permitted("https://medium.com/@a/azure-post", resolved)
        assert permitted is False
        assert reason is not None
        assert "official source" in reason

    def test_blocked_domain_is_rejected_even_when_unrestricted(
        self, policy: OfficialSourcePolicy
    ) -> None:
        resolved = policy.resolve("what are good testing practices")
        permitted, reason = policy.is_permitted("https://w3schools.com/x", resolved)
        assert permitted is False
        assert reason is not None
        assert "blocked" in reason

    def test_lookalike_domain_is_rejected(self, policy: OfficialSourcePolicy) -> None:
        resolved = policy.resolve("Azure AI Search agentic retrieval")
        permitted, _ = policy.is_permitted("https://learn.microsoft.com.evil.io/x", resolved)
        assert permitted is False

    def test_prefer_mode_permits_non_official_sources(
        self, policy: OfficialSourcePolicy
    ) -> None:
        resolved = policy.resolve(
            "Azure AI Search agentic retrieval", mode=DomainMode.PREFER
        )
        permitted, _ = policy.is_permitted("https://medium.com/@a/azure-post", resolved)
        assert permitted is True


class TestAuthorityClassification:
    def test_primary_docs_are_tier_one(self, policy: OfficialSourcePolicy) -> None:
        resolved = policy.resolve("Azure AI Search agentic retrieval")
        source_type, tier = policy.classify(
            "https://learn.microsoft.com/en-us/azure/search/x", resolved
        )
        assert tier == 1
        assert source_type is SourceType.OFFICIAL_DOCUMENTATION
        assert source_type.is_official is True

    def test_community_blog_is_lowest_tier(self, policy: OfficialSourcePolicy) -> None:
        resolved = policy.resolve("Azure AI Search agentic retrieval", mode=DomainMode.PREFER)
        source_type, tier = policy.classify("https://medium.com/@a/post", resolved)
        assert tier == 5
        assert source_type is SourceType.COMMUNITY
        assert source_type.is_official is False

    def test_recognised_secondary_source_is_tier_four(
        self, policy: OfficialSourcePolicy
    ) -> None:
        resolved = policy.resolve("what is an RFC", mode=DomainMode.PREFER)
        source_type, tier = policy.classify("https://www.ietf.org/rfc/rfc2616.txt", resolved)
        assert tier == 4
        assert source_type is SourceType.AUTHORITATIVE_SECONDARY

    def test_another_vendors_official_docs_still_rank_as_official(
        self, policy: OfficialSourcePolicy
    ) -> None:
        """A GCP doc surfacing on an AWS query is still first-party material."""
        resolved = policy.resolve("AWS Lambda cold starts", mode=DomainMode.PREFER)
        _, tier = policy.classify("https://cloud.google.com/run/docs", resolved)
        assert tier == 1

    def test_official_docs_outrank_community_for_the_same_topic(
        self, policy: OfficialSourcePolicy
    ) -> None:
        resolved = policy.resolve("Azure AI Search agentic retrieval", mode=DomainMode.PREFER)
        _, official_tier = policy.classify("https://learn.microsoft.com/x", resolved)
        _, community_tier = policy.classify("https://medium.com/x", resolved)
        assert official_tier < community_tier
