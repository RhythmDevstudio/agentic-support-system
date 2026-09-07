"""The `web_research` tool.

Wraps a search provider with the official-source policy and converts results into
`Evidence` records. Three properties matter:

* **Every URL in the output was actually returned by the provider.** Nothing here
  constructs or guesses a URL, so a citation built from this evidence points at a
  page that genuinely exists in the result set.
* **Off-policy results are dropped before content is read**, not filtered out
  afterwards. Rejected results are still reported, so the agent can tell the
  difference between "nothing found" and "found, but not from an official source".
* **Results are ordered by source authority first**, score second, so first-party
  documentation outranks a higher-scoring blog post.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.config.settings import DomainMode, Settings, get_settings
from app.models.evidence import Evidence, RetrievalChannel
from app.observability.logging import get_logger
from app.tools.web_research.domain_policy import (
    DomainPolicy,
    OfficialSourcePolicy,
    normalise_domain,
)
from app.tools.web_research.providers import (
    RawSearchResult,
    SearchResponse,
    WebSearchError,
    WebSearchProvider,
    build_web_search_provider,
)

logger = get_logger(__name__)

# Cap per-result content so a single long page cannot crowd out other evidence
# or blow the model's context budget.
MAX_CONTENT_CHARS = 6000


class RejectedResult(BaseModel):
    """A result the policy refused, kept for observability and explanation."""

    model_config = ConfigDict(frozen=True)

    url: str
    title: str
    reason: str


class WebResearchResult(BaseModel):
    """Outcome of one `web_research` call."""

    model_config = ConfigDict(frozen=True)

    query: str
    evidence: tuple[Evidence, ...] = ()
    rejected: tuple[RejectedResult, ...] = ()
    policy_description: str = ""
    vendor_key: str | None = None
    vendor_display_name: str | None = None
    searched_domains: tuple[str, ...] = ()
    domain_mode: DomainMode = DomainMode.STRICT
    provider: str = ""
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence)

    def summary(self) -> str:
        """One-line status for the agent's decision log."""
        if self.error:
            return f"web_research failed: {self.error}"
        parts = [f"{len(self.evidence)} results {self.policy_description}"]
        if self.rejected:
            parts.append(f"{len(self.rejected)} rejected as non-official")
        return "; ".join(parts)


class WebResearchTool:
    """Official-source-aware web research."""

    def __init__(
        self,
        provider: WebSearchProvider | None = None,
        policy: OfficialSourcePolicy | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._provider = provider or build_web_search_provider(self._settings)
        self._policy = policy or OfficialSourcePolicy(
            default_mode=self._settings.web_search_domain_mode
        )

    @property
    def provider_name(self) -> str:
        return self._provider.name

    async def search(
        self,
        query: str,
        *,
        vendor_hint: str | None = None,
        allowed_domains: tuple[str, ...] | None = None,
        domain_mode: DomainMode | None = None,
        max_results: int | None = None,
    ) -> WebResearchResult:
        """Search official documentation for `query`.

        A provider failure is returned as a populated `error` rather than raised:
        the agent should be able to fall back to internal documentation and say
        that the external source was unavailable.
        """
        if not query.strip():
            return WebResearchResult(query=query, error="Empty search query")

        policy = self._policy.resolve(
            query,
            model_hint=vendor_hint,
            override_domains=allowed_domains,
            mode=domain_mode,
        )
        limit = max_results or self._settings.web_search_max_results

        try:
            response = await self._provider.search(
                query,
                # Repository hosts are included here so official repos are
                # reachable; `is_permitted` then applies the owner-path check.
                include_domains=policy.search_domains(),
                exclude_domains=tuple(sorted(policy.blocked_domains)),
                domain_mode=policy.mode,
                max_results=limit,
            )
        except WebSearchError as exc:
            logger.warning("web_research_failed", error=str(exc), query_length=len(query))
            return WebResearchResult(
                query=query,
                policy_description=policy.describe(),
                vendor_key=policy.vendor_key,
                vendor_display_name=policy.vendor_display_name,
                searched_domains=policy.allowed_domains,
                domain_mode=policy.mode,
                provider=self._provider.name,
                error=str(exc),
            )

        evidence, rejected = self._to_evidence(response.results, policy, query, response)

        logger.info(
            "web_research_complete",
            vendor=policy.vendor_key,
            accepted=len(evidence),
            rejected=len(rejected),
            provider=self._provider.name,
        )

        return WebResearchResult(
            query=query,
            evidence=evidence,
            rejected=rejected,
            policy_description=policy.describe(),
            vendor_key=policy.vendor_key,
            vendor_display_name=policy.vendor_display_name,
            searched_domains=policy.allowed_domains,
            domain_mode=policy.mode,
            provider=response.provider,
            retrieved_at=response.retrieved_at,
        )

    def _to_evidence(
        self,
        results: tuple[RawSearchResult, ...],
        policy: DomainPolicy,
        query: str,
        response: SearchResponse,
    ) -> tuple[tuple[Evidence, ...], tuple[RejectedResult, ...]]:
        accepted: list[Evidence] = []
        rejected: list[RejectedResult] = []

        for item in results:
            permitted, reason = self._policy.is_permitted(item.url, policy)
            if not permitted:
                rejected.append(
                    RejectedResult(url=item.url, title=item.title, reason=reason or "off-policy")
                )
                continue

            content = item.best_content.strip()
            if not content:
                rejected.append(
                    RejectedResult(
                        url=item.url, title=item.title, reason="no extractable content"
                    )
                )
                continue

            source_type, tier = self._policy.classify(item.url, policy)
            domain = normalise_domain(item.url)

            accepted.append(
                Evidence(
                    # The handle is assigned by the run's EvidenceRegistry.
                    evidence_id="",
                    content=content[:MAX_CONTENT_CHARS],
                    channel=RetrievalChannel.WEB_RESEARCH,
                    source_type=source_type,
                    title=item.title or domain or item.url,
                    url=item.url,
                    publisher=policy.vendor_display_name or domain,
                    score=item.score,
                    authority_tier=tier,
                    vendor_key=policy.vendor_key,
                    retrieved_at=response.retrieved_at,
                    retrieval_query=query,
                    metadata={
                        "domain": domain,
                        "provider": response.provider,
                        "truncated": len(content) > MAX_CONTENT_CHARS,
                        "published_date": item.published_date,
                    },
                )
            )

        # Authority first, then relevance: an official page outranks a
        # higher-scoring community post.
        accepted.sort(key=lambda item: (item.authority_tier, -item.score))
        return tuple(accepted), tuple(rejected)
