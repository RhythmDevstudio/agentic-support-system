"""Official source policy: vendor detection, domain allowlisting and authority ranking.

This module is the enforcement point for the rule that vendor questions are
answered from vendor documentation.

The division of labour matters. The agent may *propose* a vendor key (it is often
better than keyword matching at reading intent - "how do I use the Responses API"
names no vendor explicitly). But the agent can never propose a **domain**: domains
come only from `vendors.yaml`. So the worst a confused or manipulated model can do
is select the wrong vendor's official documentation, never an arbitrary site.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from app.config.policies import VendorPolicy, VendorRegistry, get_vendor_registry
from app.config.settings import DomainMode
from app.models.documents import SourceType
from app.observability.logging import get_logger

logger = get_logger(__name__)

# An alias only counts when it appears as a whole word/phrase, so "gpt" does not
# match "gpteam" and ".net" does not match "example.network".
_WORD_BOUNDARY = r"(?<![a-z0-9]){alias}(?![a-z0-9])"


@dataclass(frozen=True)
class VendorMatch:
    """A detected vendor plus why it was detected."""

    vendor: VendorPolicy
    score: float
    matched_aliases: tuple[str, ...] = ()
    from_model_hint: bool = False

    @property
    def key(self) -> str:
        return self.vendor.key


@dataclass(frozen=True)
class DomainPolicy:
    """The resolved search constraint for one research call."""

    allowed_domains: tuple[str, ...]
    mode: DomainMode
    vendor_key: str | None = None
    vendor_display_name: str | None = None
    blocked_domains: frozenset[str] = field(default_factory=frozenset)
    # Owner-scoped repository prefixes such as "github.com/Azure". Matched on host
    # *and* path, so allowing a vendor's org does not allow all of github.com.
    repository_prefixes: tuple[str, ...] = ()

    @property
    def is_restricted(self) -> bool:
        return self.mode is DomainMode.STRICT and bool(self.allowed_domains)

    def search_domains(self) -> tuple[str, ...]:
        """Domains to hand the search provider.

        Repository hosts are included so official repos are reachable at all; the
        precise owner-path check then happens in `is_permitted`.
        """
        hosts = list(self.allowed_domains)
        for prefix in self.repository_prefixes:
            host = prefix.partition("/")[0]
            if host and host not in hosts:
                hosts.append(host)
        return tuple(hosts)

    def describe(self) -> str:
        if not self.allowed_domains:
            return "unrestricted web search"
        verb = "restricted to" if self.mode is DomainMode.STRICT else "prioritising"
        listed = ", ".join(self.allowed_domains)
        if self.repository_prefixes:
            listed += f" and {', '.join(self.repository_prefixes)}"
        return f"{verb} {listed}"


def normalise_domain(value: str) -> str:
    """Extract a comparable host from a URL or bare domain."""
    candidate = value.strip().lower()
    if "://" in candidate:
        candidate = urlparse(candidate).netloc
    candidate = candidate.split("/")[0].split("@")[-1].split(":")[0]
    return candidate.removeprefix("www.")


def domain_matches(domain: str, allowed: str) -> bool:
    """True when `domain` is `allowed` or a subdomain of it.

    Subdomain matching is anchored on a dot so that `notmicrosoft.com` does not
    match an allowlist entry of `microsoft.com`.
    """
    left = normalise_domain(domain)
    right = normalise_domain(allowed)
    return bool(left) and (left == right or left.endswith(f".{right}"))


def repository_matches(url: str, prefix: str) -> bool:
    """True when `url` sits under an owner-scoped repository prefix.

    Repository entries look like `github.com/Azure`. Matching on the host alone
    would allow the whole of github.com, so the owner path segment is compared
    too - `github.com/Azure/x` matches, `github.com/attacker/x` does not.
    """
    parsed = urlparse(url if "://" in url else f"https://{url}")
    prefix_host, _, prefix_path = prefix.partition("/")
    if not domain_matches(parsed.netloc, prefix_host):
        return False
    if not prefix_path:
        return True
    path = parsed.path.strip("/").lower()
    wanted = prefix_path.strip("/").lower()
    return path == wanted or path.startswith(f"{wanted}/")


class VendorDetector:
    """Deterministic vendor detection from question text, with an optional model hint."""

    def __init__(self, registry: VendorRegistry | None = None) -> None:
        self._registry = registry or get_vendor_registry()
        self._patterns: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
        for key, vendor in self._registry.vendors.items():
            compiled: list[tuple[str, re.Pattern[str]]] = []
            for alias in vendor.aliases:
                pattern = _WORD_BOUNDARY.format(alias=re.escape(alias.lower()))
                compiled.append((alias, re.compile(pattern)))
            self._patterns[key] = compiled

    def detect(self, text: str, *, model_hint: str | None = None) -> VendorMatch | None:
        """Return the best-matching vendor, or None when the question is vendor-neutral.

        Alias scoring weights longer aliases more heavily: "azure ai search" is a far
        stronger signal than "azure", and "agents sdk" than "gpt".
        """
        lowered = text.lower()
        scores: dict[str, float] = {}
        matched: dict[str, list[str]] = {}

        for key, patterns in self._patterns.items():
            for alias, pattern in patterns:
                if pattern.search(lowered):
                    # Multi-word aliases are much more discriminating.
                    weight = 1.0 + 0.75 * alias.count(" ") + 0.02 * len(alias)
                    scores[key] = scores.get(key, 0.0) + weight
                    matched.setdefault(key, []).append(alias)

        hint_key = self._resolve_hint(model_hint)
        if hint_key is not None:
            # The hint breaks ties and can surface a vendor the keywords missed,
            # but it is bounded: it can only select a vendor that already exists
            # in the registry.
            scores[hint_key] = scores.get(hint_key, 0.0) + 1.5

        if not scores:
            return None

        best_key = max(scores, key=lambda key: scores[key])
        vendor = self._registry.vendors[best_key]
        return VendorMatch(
            vendor=vendor,
            score=scores[best_key],
            matched_aliases=tuple(matched.get(best_key, ())),
            from_model_hint=best_key == hint_key,
        )

    def _resolve_hint(self, model_hint: str | None) -> str | None:
        """Map a model's vendor guess onto a registry key, ignoring anything unknown."""
        if not model_hint:
            return None
        candidate = model_hint.strip().lower().replace(" ", "_").replace("-", "_")
        if candidate in self._registry.vendors:
            return candidate
        # Tolerate display names and near-misses like "microsoft" for "microsoft_azure".
        for key, vendor in self._registry.vendors.items():
            if candidate == vendor.display_name.lower().replace(" ", "_"):
                return key
            if candidate and (candidate in key or key.startswith(candidate)):
                return key
        logger.debug("vendor_hint_unrecognised", hint=model_hint)
        return None


class OfficialSourcePolicy:
    """Resolves the domain constraint for a query and classifies retrieved results."""

    def __init__(
        self,
        registry: VendorRegistry | None = None,
        *,
        default_mode: DomainMode = DomainMode.STRICT,
    ) -> None:
        self._registry = registry or get_vendor_registry()
        self._detector = VendorDetector(self._registry)
        self._default_mode = default_mode

    @property
    def registry(self) -> VendorRegistry:
        return self._registry

    def detect_vendor(self, text: str, *, model_hint: str | None = None) -> VendorMatch | None:
        return self._detector.detect(text, model_hint=model_hint)

    def resolve(
        self,
        query: str,
        *,
        model_hint: str | None = None,
        override_domains: tuple[str, ...] | None = None,
        mode: DomainMode | None = None,
    ) -> DomainPolicy:
        """Decide which domains this search may draw from.

        `override_domains` lets a caller pin domains explicitly (the API exposes
        this); it is still filtered against the blocklist, so an override cannot
        be used to reach a banned content farm.
        """
        blocked = self._registry.blocked_domains

        if override_domains:
            permitted = tuple(
                domain
                for domain in override_domains
                if not self._registry.is_blocked(domain)
            )
            rejected = set(override_domains) - set(permitted)
            if rejected:
                logger.warning("override_domains_blocked", domains=sorted(rejected))
            return DomainPolicy(
                allowed_domains=permitted,
                mode=mode or self._default_mode,
                blocked_domains=blocked,
            )

        match = self.detect_vendor(query, model_hint=model_hint)
        if match is None:
            # Vendor-neutral question: search broadly but still exclude the blocklist.
            return DomainPolicy(
                allowed_domains=self._registry.fallback_domains,
                mode=mode or self._registry.fallback_domain_mode,
                blocked_domains=blocked,
            )

        return DomainPolicy(
            allowed_domains=match.vendor.official_domains,
            mode=mode or match.vendor.domain_mode,
            vendor_key=match.key,
            vendor_display_name=match.vendor.display_name,
            blocked_domains=blocked,
            repository_prefixes=match.vendor.repositories,
        )

    def is_permitted(self, url: str, policy: DomainPolicy) -> tuple[bool, str | None]:
        """Check a result URL against the policy. Returns (permitted, reason_if_not)."""
        domain = normalise_domain(url)
        if not domain:
            return False, "unparseable URL"
        if self._registry.is_blocked(domain):
            return False, f"{domain} is on the blocked domain list"
        if policy.mode is DomainMode.STRICT and policy.allowed_domains:
            on_official_domain = any(
                domain_matches(domain, allowed) for allowed in policy.allowed_domains
            )
            in_official_repository = any(
                repository_matches(url, prefix) for prefix in policy.repository_prefixes
            )
            if not on_official_domain and not in_official_repository:
                return False, (
                    f"{domain} is not an official source for "
                    f"{policy.vendor_display_name or 'this query'}"
                )
        return True, None

    def classify(self, url: str, policy: DomainPolicy) -> tuple[SourceType, int]:
        """Assign a source type and authority tier (1 = most authoritative)."""
        domain = normalise_domain(url)

        vendor = self._registry.get(policy.vendor_key) if policy.vendor_key else None
        if vendor is not None:
            tier = vendor.tier_for_domain(domain)
            if tier is not None:
                return self._source_type_for_tier(tier), tier
            if any(domain_matches(domain, allowed) for allowed in vendor.official_domains):
                return SourceType.OFFICIAL_DOCUMENTATION, 2
            if any(repository_matches(url, repo) for repo in vendor.repositories):
                return SourceType.OFFICIAL_REPOSITORY, 3

        # Not this vendor's domain, but possibly another vendor's official docs.
        for other in self._registry.vendors.values():
            tier = other.tier_for_domain(domain)
            if tier is not None:
                return self._source_type_for_tier(tier), tier

        if self._registry.is_authoritative_secondary(domain):
            return SourceType.AUTHORITATIVE_SECONDARY, 4

        return SourceType.COMMUNITY, 5

    @staticmethod
    def _source_type_for_tier(tier: int) -> SourceType:
        return {
            1: SourceType.OFFICIAL_DOCUMENTATION,
            2: SourceType.OFFICIAL_API_REFERENCE,
            3: SourceType.OFFICIAL_REPOSITORY,
            4: SourceType.AUTHORITATIVE_SECONDARY,
        }.get(tier, SourceType.COMMUNITY)
