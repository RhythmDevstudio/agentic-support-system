"""Web search providers.

`TavilySearchProvider` is the live implementation. Tavily's `include_domains_mode`
maps exactly onto the two modes of the official-source policy: `filter` is a hard
allowlist (STRICT) and `boost` is a strong preference (PREFER).

`FixtureSearchProvider` replays recorded official-documentation pages from
`data/fixtures/web/`. It exists so the research workflow, its guardrails and its
citation checks can be exercised deterministically with no API key and no network,
which is what makes the test suite reproducible.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.config.settings import DomainMode, Settings, WebSearchProviderName, get_settings
from app.observability.logging import get_logger

logger = get_logger(__name__)


class WebSearchError(RuntimeError):
    """Raised when a web search cannot be completed."""


class RawSearchResult(BaseModel):
    """A single result as returned by the provider, before policy is applied."""

    model_config = ConfigDict(frozen=True)

    title: str
    url: str
    content: str
    raw_content: str | None = None
    score: float = 0.0
    published_date: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def best_content(self) -> str:
        """Prefer full page content when the provider supplied it."""
        if self.raw_content and len(self.raw_content) > len(self.content):
            return self.raw_content
        return self.content


class SearchResponse(BaseModel):
    """A provider's complete response, including the retrieval timestamp."""

    model_config = ConfigDict(frozen=True)

    query: str
    results: tuple[RawSearchResult, ...]
    provider: str
    retrieved_at: datetime
    requested_domains: tuple[str, ...] = ()
    domain_mode: DomainMode = DomainMode.STRICT
    elapsed_seconds: float = 0.0


class WebSearchProvider(ABC):
    name: str

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        include_domains: tuple[str, ...] = (),
        exclude_domains: tuple[str, ...] = (),
        domain_mode: DomainMode = DomainMode.STRICT,
        max_results: int = 6,
    ) -> SearchResponse: ...

    def is_offline(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Tavily
# ---------------------------------------------------------------------------


class TavilySearchProvider(WebSearchProvider):
    name = "tavily"

    def __init__(self, settings: Settings, client: Any = None) -> None:
        self._settings = settings
        self._timeout = settings.web_search_timeout_seconds
        self._depth = settings.web_search_depth
        self._client = client or self._build_client(settings)

    @staticmethod
    def _build_client(settings: Settings) -> Any:
        if settings.tavily_api_key is None:
            raise WebSearchError("TAVILY_API_KEY is required for the Tavily provider")
        try:
            from tavily import AsyncTavilyClient
        except ImportError as exc:
            raise WebSearchError(
                "Tavily support requires the 'web' extra: uv pip install -e '.[web]'"
            ) from exc
        return AsyncTavilyClient(api_key=settings.tavily_api_key.get_secret_value())

    async def search(
        self,
        query: str,
        *,
        include_domains: tuple[str, ...] = (),
        exclude_domains: tuple[str, ...] = (),
        domain_mode: DomainMode = DomainMode.STRICT,
        max_results: int = 6,
    ) -> SearchResponse:
        started = datetime.now(UTC)
        # filter = hard allowlist; boost = rank official domains first but allow others.
        include_mode = "filter" if domain_mode is DomainMode.STRICT else "boost"

        payload: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "search_depth": self._depth,
            "include_raw_content": "markdown",
        }
        if include_domains:
            payload["include_domains"] = list(include_domains)
            payload["include_domains_mode"] = include_mode
        if exclude_domains:
            payload["exclude_domains"] = list(exclude_domains)

        try:
            response = await asyncio.wait_for(
                self._client.search(**payload), timeout=self._timeout
            )
        except TimeoutError as exc:
            raise WebSearchError(
                f"Web search timed out after {self._timeout:.0f}s"
            ) from exc
        except Exception as exc:
            raise WebSearchError(f"Web search failed: {exc}") from exc

        results = tuple(
            RawSearchResult(
                title=item.get("title") or "",
                url=item.get("url") or "",
                content=item.get("content") or "",
                raw_content=item.get("raw_content"),
                score=float(item.get("score") or 0.0),
                published_date=item.get("published_date"),
            )
            for item in (response.get("results") or [])
            if item.get("url")
        )
        finished = datetime.now(UTC)

        return SearchResponse(
            query=query,
            results=results,
            provider=self.name,
            retrieved_at=finished,
            requested_domains=include_domains,
            domain_mode=domain_mode,
            elapsed_seconds=(finished - started).total_seconds(),
        )


# ---------------------------------------------------------------------------
# Offline fixtures
# ---------------------------------------------------------------------------


class FixtureSearchProvider(WebSearchProvider):
    """Replays recorded official-documentation pages.

    Matching is lexical overlap between the query and each fixture's keywords,
    title and content, so it behaves like a plausible (if simple) search engine
    rather than returning a fixed list regardless of the question.
    """

    name = "fixture"

    def __init__(self, fixture_dir: Path | None = None) -> None:
        self._fixture_dir = fixture_dir or (
            Path(__file__).resolve().parents[3] / "data" / "fixtures" / "web"
        )
        self._fixtures: list[dict[str, Any]] | None = None

    def is_offline(self) -> bool:
        return True

    def _load(self) -> list[dict[str, Any]]:
        if self._fixtures is not None:
            return self._fixtures

        fixtures: list[dict[str, Any]] = []
        if self._fixture_dir.exists():
            for path in sorted(self._fixture_dir.glob("*.json")):
                if path.name.startswith("._"):
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("fixture_load_failed", path=str(path), error=str(exc))
                    continue
                fixtures.extend(payload if isinstance(payload, list) else [payload])
        else:
            logger.warning("fixture_dir_missing", path=str(self._fixture_dir))

        self._fixtures = fixtures
        return fixtures

    @staticmethod
    def _score(query: str, fixture: dict[str, Any]) -> float:
        terms = {term for term in query.lower().split() if len(term) > 2}
        if not terms:
            return 0.0

        keywords = {str(k).lower() for k in fixture.get("keywords", [])}
        title_terms = set(str(fixture.get("title", "")).lower().split())
        body = str(fixture.get("content", "")).lower()

        # Keyword hits are weighted highest: they are the curated signal.
        keyword_hits = sum(1 for term in terms if any(term in k for k in keywords))
        title_hits = len(terms & title_terms)
        body_hits = sum(1 for term in terms if term in body)

        return (3.0 * keyword_hits + 2.0 * title_hits + body_hits) / (4.0 * len(terms))

    async def search(
        self,
        query: str,
        *,
        include_domains: tuple[str, ...] = (),
        exclude_domains: tuple[str, ...] = (),
        domain_mode: DomainMode = DomainMode.STRICT,
        max_results: int = 6,
    ) -> SearchResponse:
        from app.tools.web_research.domain_policy import domain_matches, normalise_domain

        scored: list[tuple[float, dict[str, Any]]] = []
        for fixture in self._load():
            url = fixture.get("url", "")
            domain = normalise_domain(url)

            if exclude_domains and any(domain_matches(domain, d) for d in exclude_domains):
                continue
            # Mirror the live provider: STRICT filters at the source, PREFER does not.
            if (
                include_domains
                and domain_mode is DomainMode.STRICT
                and not any(domain_matches(domain, d) for d in include_domains)
            ):
                continue

            score = self._score(query, fixture)
            if score <= 0.0:
                continue
            if (
                include_domains
                and domain_mode is DomainMode.PREFER
                and any(domain_matches(domain, d) for d in include_domains)
            ):
                score *= 2.0
            scored.append((score, fixture))

        scored.sort(key=lambda item: item[0], reverse=True)
        now = datetime.now(UTC)

        results = tuple(
            RawSearchResult(
                title=fixture.get("title", ""),
                url=fixture.get("url", ""),
                content=fixture.get("content", ""),
                raw_content=fixture.get("raw_content"),
                score=round(min(score, 1.0), 4),
                published_date=fixture.get("published_date"),
                metadata={"fixture": True},
            )
            for score, fixture in scored[:max_results]
        )

        return SearchResponse(
            query=query,
            results=results,
            provider=self.name,
            retrieved_at=now,
            requested_domains=include_domains,
            domain_mode=domain_mode,
        )


class UnavailableSearchProvider(WebSearchProvider):
    """Provider that always fails.

    Used to configure web research off entirely, and to exercise the
    "external source unavailable" path in the evaluation suite.
    """

    name = "unavailable"

    def is_offline(self) -> bool:
        return True

    async def search(
        self,
        query: str,
        *,
        include_domains: tuple[str, ...] = (),
        exclude_domains: tuple[str, ...] = (),
        domain_mode: DomainMode = DomainMode.STRICT,
        max_results: int = 6,
    ) -> SearchResponse:
        raise WebSearchError("Web research is disabled by configuration")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_web_search_provider(settings: Settings | None = None) -> WebSearchProvider:
    settings = settings or get_settings()
    resolved = settings.resolve_web_search_provider()

    if resolved is WebSearchProviderName.NONE:
        return UnavailableSearchProvider()
    if resolved is WebSearchProviderName.FIXTURE:
        return FixtureSearchProvider()

    try:
        return TavilySearchProvider(settings)
    except WebSearchError as exc:
        logger.warning(
            "web_search_provider_fallback",
            requested=str(resolved),
            reason=str(exc),
            using="fixture",
        )
        return FixtureSearchProvider()
