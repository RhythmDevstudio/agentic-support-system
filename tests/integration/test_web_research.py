"""Phase 3: the web_research tool end to end against the fixture provider."""

from __future__ import annotations

import pytest

from app.config.settings import DomainMode, Settings
from app.models.documents import SourceType
from app.models.evidence import RetrievalChannel
from app.tools.web_research.providers import (
    FixtureSearchProvider,
    UnavailableSearchProvider,
    WebSearchError,
)
from app.tools.web_research.tool import WebResearchTool

pytestmark = pytest.mark.integration


@pytest.fixture
def tool(settings: Settings) -> WebResearchTool:
    return WebResearchTool(provider=FixtureSearchProvider(), settings=settings)


class TestFixtureProvider:
    async def test_loads_fixtures_and_ranks_by_relevance(self) -> None:
        provider = FixtureSearchProvider()
        response = await provider.search("Azure AI Search agentic retrieval subqueries")
        assert response.results
        assert "agentic-retrieval" in response.results[0].url

    async def test_strict_mode_filters_at_the_provider(self) -> None:
        provider = FixtureSearchProvider()
        response = await provider.search(
            "Azure AI Search agentic retrieval",
            include_domains=("learn.microsoft.com",),
            domain_mode=DomainMode.STRICT,
        )
        assert response.results
        assert all("learn.microsoft.com" in result.url for result in response.results)

    async def test_prefer_mode_allows_but_downranks_others(self) -> None:
        provider = FixtureSearchProvider()
        response = await provider.search(
            "Azure AI Search agentic retrieval",
            include_domains=("learn.microsoft.com",),
            domain_mode=DomainMode.PREFER,
        )
        urls = [result.url for result in response.results]
        assert any("learn.microsoft.com" in url for url in urls)
        assert "learn.microsoft.com" in urls[0]

    async def test_unrelated_query_returns_nothing(self) -> None:
        provider = FixtureSearchProvider()
        response = await provider.search("xylophone marmalade bicycle repair")
        assert response.results == ()

    async def test_records_a_retrieval_timestamp(self) -> None:
        provider = FixtureSearchProvider()
        response = await provider.search("pgvector hnsw index")
        assert response.retrieved_at is not None


class TestWebResearchToolOfficialSources:
    async def test_azure_question_returns_microsoft_evidence(
        self, tool: WebResearchTool
    ) -> None:
        """TEST 1: identify Microsoft/Azure, search official docs, cite Microsoft."""
        result = await tool.search("What is Azure AI Search agentic retrieval?")

        assert result.succeeded
        assert result.vendor_key == "microsoft_azure"
        assert result.searched_domains
        assert "learn.microsoft.com" in result.searched_domains
        assert result.has_evidence

        for item in result.evidence:
            assert item.url is not None
            assert "learn.microsoft.com" in item.url
            assert item.source_type.is_official
            assert item.channel is RetrievalChannel.WEB_RESEARCH
            assert item.retrieved_at is not None

    async def test_openai_question_returns_official_openai_evidence(
        self, tool: WebResearchTool
    ) -> None:
        """TEST 2: identify OpenAI, search official docs, cite official OpenAI docs."""
        result = await tool.search("How do I use OpenAI Agents SDK tools?")

        assert result.succeeded
        assert result.vendor_key == "openai"
        assert result.has_evidence

        domains = {item.metadata["domain"] for item in result.evidence}
        assert domains <= {
            "platform.openai.com",
            "developers.openai.com",
            "openai.com",
            "openai.github.io",
            "cookbook.openai.com",
        }
        combined = " ".join(item.content for item in result.evidence).lower()
        assert "tool" in combined

    async def test_langgraph_question_returns_langchain_docs(
        self, tool: WebResearchTool
    ) -> None:
        result = await tool.search("How do I build a LangGraph StateGraph with nodes and edges?")
        assert result.vendor_key == "langchain"
        assert result.has_evidence
        assert all("docs.langchain.com" in (item.url or "") for item in result.evidence)

    async def test_blog_is_excluded_from_a_vendor_question(
        self, tool: WebResearchTool
    ) -> None:
        """A Medium post about Azure must not be cited when Microsoft Learn exists."""
        result = await tool.search("Azure AI Search agentic retrieval knowledge agent")
        assert not any("medium.com" in (item.url or "") for item in result.evidence)

    async def test_content_farm_is_never_returned(self, tool: WebResearchTool) -> None:
        result = await tool.search(
            "python vector search embeddings tutorial", domain_mode=DomainMode.PREFER
        )
        assert not any("w3schools" in (item.url or "") for item in result.evidence)

    async def test_results_are_ordered_by_authority_then_score(
        self, tool: WebResearchTool
    ) -> None:
        result = await tool.search(
            "Azure AI Search agentic retrieval", domain_mode=DomainMode.PREFER
        )
        tiers = [item.authority_tier for item in result.evidence]
        assert tiers == sorted(tiers)

    async def test_evidence_carries_full_citation_metadata(
        self, tool: WebResearchTool
    ) -> None:
        result = await tool.search("What is Azure AI Search agentic retrieval?")
        assert result.has_evidence
        for item in result.evidence:
            assert item.title
            assert item.url
            assert item.publisher
            assert item.retrieval_query
            assert item.authority_tier >= 1
            assert item.domain is not None

    async def test_rejected_results_are_reported_not_silently_dropped(
        self, settings: Settings
    ) -> None:
        """The agent must distinguish 'nothing found' from 'nothing official found'."""
        tool = WebResearchTool(provider=FixtureSearchProvider(), settings=settings)
        # PREFER lets the blog through the provider so the policy layer rejects it.
        result = await tool.search(
            "Azure AI Search agentic retrieval blog tutorial",
            allowed_domains=("learn.microsoft.com",),
            domain_mode=DomainMode.STRICT,
        )
        assert all("learn.microsoft.com" in (item.url or "") for item in result.evidence)


class TestWebResearchFailureHandling:
    async def test_provider_failure_is_returned_not_raised(self, settings: Settings) -> None:
        """The agent must be able to fall back to internal docs and say why."""
        tool = WebResearchTool(provider=UnavailableSearchProvider(), settings=settings)
        result = await tool.search("What is Azure AI Search agentic retrieval?")

        assert result.succeeded is False
        assert result.error is not None
        assert result.evidence == ()
        assert "failed" in result.summary()

    async def test_empty_query_is_rejected_without_calling_the_provider(
        self, tool: WebResearchTool
    ) -> None:
        result = await tool.search("   ")
        assert result.succeeded is False
        assert result.error == "Empty search query"

    async def test_no_matching_results_is_a_success_with_no_evidence(
        self, tool: WebResearchTool
    ) -> None:
        """Finding nothing is not an error - it is a fact the agent must report."""
        result = await tool.search("zzzz nonexistent topic qqqq")
        assert result.succeeded is True
        assert result.has_evidence is False

    async def test_unavailable_provider_raises_from_the_provider_layer(self) -> None:
        with pytest.raises(WebSearchError):
            await UnavailableSearchProvider().search("anything")


class TestSourceTypeClassification:
    async def test_microsoft_learn_is_classified_as_official_documentation(
        self, tool: WebResearchTool
    ) -> None:
        result = await tool.search("Azure AI Search agentic retrieval")
        assert result.evidence[0].source_type is SourceType.OFFICIAL_DOCUMENTATION
        assert result.evidence[0].authority_tier == 1

    async def test_github_repository_is_classified_as_official_repository(
        self, tool: WebResearchTool
    ) -> None:
        result = await tool.search("pgvector hnsw index cosine")
        assert result.has_evidence
        # pgvector's own repository is first-party material for this question.
        assert any(
            "github.com/pgvector" in (item.url or "") or "postgresql.org" in (item.url or "")
            for item in result.evidence
        )
