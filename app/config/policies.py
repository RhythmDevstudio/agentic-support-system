"""Typed loaders for the YAML policy files.

These files are the single source of truth for the two things the LLM is never
permitted to invent: **official source domains** and **ticket queues**.

Both policies are validated at load time. A routing rule that points at an
unknown queue, or an intent with no rule at all, raises `PolicyError` on startup
rather than silently degrading into a mis-route at request time.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.config.settings import POLICY_DIR, DomainMode


class PolicyError(RuntimeError):
    """Raised when a policy file is structurally invalid or internally inconsistent."""


# ---------------------------------------------------------------------------
# Vendor / official-source policy
# ---------------------------------------------------------------------------


class AuthorityTier(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""


class VendorPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    display_name: str
    aliases: tuple[str, ...] = ()
    official_domains: tuple[str, ...] = ()
    domain_tiers: dict[str, int] = Field(default_factory=dict)
    repositories: tuple[str, ...] = ()
    domain_mode: DomainMode = DomainMode.STRICT

    def tier_for_domain(self, domain: str) -> int | None:
        """Authority tier for a domain, honouring subdomain matches."""
        normalised = domain.lower().removeprefix("www.")
        for configured, tier in self.domain_tiers.items():
            candidate = configured.lower().removeprefix("www.")
            if normalised == candidate or normalised.endswith(f".{candidate}"):
                return tier
        return None


class VendorRegistry(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int
    vendors: dict[str, VendorPolicy]
    authority_tiers: dict[int, AuthorityTier]
    blocked_domains: frozenset[str]
    authoritative_secondary_domains: frozenset[str]
    default_domain_mode: DomainMode
    fallback_domain_mode: DomainMode
    fallback_domains: tuple[str, ...]

    def get(self, key: str) -> VendorPolicy | None:
        return self.vendors.get(key)

    def all_official_domains(self) -> frozenset[str]:
        return frozenset(
            domain for vendor in self.vendors.values() for domain in vendor.official_domains
        )

    def is_blocked(self, domain: str) -> bool:
        normalised = domain.lower().removeprefix("www.")
        return any(
            normalised == blocked or normalised.endswith(f".{blocked}")
            for blocked in self.blocked_domains
        )

    def is_authoritative_secondary(self, domain: str) -> bool:
        normalised = domain.lower().removeprefix("www.")
        return any(
            normalised == known or normalised.endswith(f".{known}")
            for known in self.authoritative_secondary_domains
        )


# ---------------------------------------------------------------------------
# Ticket routing policy
# ---------------------------------------------------------------------------


class QueueDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    display_name: str
    sla_hours: int
    description: str = ""


class OverrideCondition(BaseModel):
    model_config = ConfigDict(frozen=True)

    intent: str | None = None
    urgency: str | None = None
    any_keyword: tuple[str, ...] = ()


class OverrideEffect(BaseModel):
    model_config = ConfigDict(frozen=True)

    queue: str | None = None
    urgency: str | None = None
    min_urgency: str | None = None
    create_escalation: bool = False


class RoutingOverride(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    when: OverrideCondition
    then: OverrideEffect


class ApprovalCondition(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    customer_tier_in: tuple[str, ...] = ()
    urgency_in: tuple[str, ...] = ()
    max_confidence: float | None = None


class ApprovalPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    required_for_actions: frozenset[str]
    required_for_conditions: tuple[ApprovalCondition, ...]
    auto_approved_actions: frozenset[str]


class LanguageDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    name: str


class LanguagePolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    supported: tuple[LanguageDefinition, ...]
    default: str
    supported_by_agents: frozenset[str]

    @functools.cached_property
    def by_code(self) -> dict[str, LanguageDefinition]:
        return {lang.code: lang for lang in self.supported}

    @functools.cached_property
    def _lookup(self) -> dict[str, str]:
        """Case-insensitive lookup from both code and display name to canonical code."""
        table: dict[str, str] = {}
        for lang in self.supported:
            table[lang.code.lower()] = lang.code
            table[lang.name.lower()] = lang.code
        return table

    def normalise(self, raw: str | None) -> str:
        """Map a model-produced language string onto a supported code.

        Falls back to the configured default rather than propagating an
        unrecognised value into downstream systems.
        """
        if not raw:
            return self.default
        key = raw.strip().lower()
        if key in self._lookup:
            return self._lookup[key]
        # Tolerate regional variants such as "en-US" or "hi-IN".
        base = key.split("-")[0].split("_")[0]
        if base in self._lookup:
            return self._lookup[base]
        return self.default


class RoutingPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int
    intents: tuple[str, ...]
    urgency_levels: tuple[str, ...]
    queues: dict[str, QueueDefinition]
    routing_rules: dict[str, str]
    fallback_queue: str
    overrides: tuple[RoutingOverride, ...]
    urgency_floors: dict[str, str]
    urgency_escalation_keywords: tuple[str, ...]
    approval_policy: ApprovalPolicy
    languages: LanguagePolicy

    def queue_for_intent(self, intent: str) -> str:
        return self.routing_rules.get(intent, self.fallback_queue)

    def is_valid_intent(self, intent: str) -> bool:
        return intent in self.intents

    def is_valid_queue(self, queue: str) -> bool:
        return queue in self.queues

    def is_valid_urgency(self, urgency: str) -> bool:
        return urgency in self.urgency_levels

    def urgency_rank(self, urgency: str) -> int:
        """Higher number = more urgent. Unknown values rank lowest."""
        ordered = list(reversed(self.urgency_levels))  # low -> critical
        try:
            return ordered.index(urgency)
        except ValueError:
            return -1


# ---------------------------------------------------------------------------
# Loading + validation
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PolicyError(f"Policy file not found: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PolicyError(f"Policy file {path.name} is not valid YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise PolicyError(f"Policy file {path.name} must contain a YAML mapping at the top level")
    return loaded


def load_vendor_registry(path: Path | None = None) -> VendorRegistry:
    raw = _read_yaml(path or POLICY_DIR / "vendors.yaml")
    defaults = raw.get("defaults", {}) or {}

    vendors: dict[str, VendorPolicy] = {}
    for key, body in (raw.get("vendors") or {}).items():
        body = dict(body or {})
        vendors[key] = VendorPolicy(
            key=key,
            display_name=body.get("display_name", key),
            aliases=tuple(body.get("aliases") or ()),
            official_domains=tuple(body.get("official_domains") or ()),
            domain_tiers=dict(body.get("domain_tiers") or {}),
            repositories=tuple(body.get("repositories") or ()),
            domain_mode=DomainMode(
                body.get("domain_mode", defaults.get("domain_mode", DomainMode.STRICT))
            ),
        )

    if not vendors:
        raise PolicyError("vendors.yaml defines no vendors")

    registry = VendorRegistry(
        version=int(raw.get("version", 1)),
        vendors=vendors,
        authority_tiers={
            int(tier): AuthorityTier(**(body or {}))
            for tier, body in (raw.get("authority_tiers") or {}).items()
        },
        blocked_domains=frozenset(
            d.lower().removeprefix("www.") for d in (raw.get("blocked_domains") or ())
        ),
        authoritative_secondary_domains=frozenset(
            d.lower().removeprefix("www.")
            for d in (raw.get("authoritative_secondary_domains") or ())
        ),
        default_domain_mode=DomainMode(defaults.get("domain_mode", DomainMode.STRICT)),
        fallback_domain_mode=DomainMode(defaults.get("fallback_domain_mode", DomainMode.PREFER)),
        fallback_domains=tuple(defaults.get("fallback_domains") or ()),
    )

    _validate_vendor_registry(registry)
    return registry


def _validate_vendor_registry(registry: VendorRegistry) -> None:
    for vendor in registry.vendors.values():
        if not vendor.official_domains:
            raise PolicyError(f"Vendor '{vendor.key}' declares no official_domains")
        unknown = set(vendor.domain_tiers) - set(vendor.official_domains)
        if unknown:
            raise PolicyError(
                f"Vendor '{vendor.key}' assigns tiers to domains that are not in "
                f"official_domains: {sorted(unknown)}"
            )
        for domain in vendor.official_domains:
            if registry.is_blocked(domain):
                raise PolicyError(
                    f"Vendor '{vendor.key}' lists '{domain}' as official but it is "
                    "also in blocked_domains"
                )


def load_routing_policy(path: Path | None = None) -> RoutingPolicy:
    raw = _read_yaml(path or POLICY_DIR / "routing.yaml")

    queues = {
        key: QueueDefinition(
            key=key,
            display_name=(body or {}).get("display_name", key),
            sla_hours=int((body or {}).get("sla_hours", 24)),
            description=(body or {}).get("description", ""),
        )
        for key, body in (raw.get("queues") or {}).items()
    }

    approval_raw = raw.get("approval_policy") or {}
    languages_raw = raw.get("languages") or {}

    policy = RoutingPolicy(
        version=int(raw.get("version", 1)),
        intents=tuple(raw.get("intents") or ()),
        urgency_levels=tuple(raw.get("urgency_levels") or ()),
        queues=queues,
        routing_rules=dict(raw.get("routing_rules") or {}),
        fallback_queue=raw.get("fallback_queue", ""),
        overrides=tuple(
            RoutingOverride(
                name=item.get("name", f"override_{index}"),
                description=item.get("description", ""),
                when=OverrideCondition(
                    intent=(item.get("when") or {}).get("intent"),
                    urgency=(item.get("when") or {}).get("urgency"),
                    any_keyword=tuple((item.get("when") or {}).get("any_keyword") or ()),
                ),
                then=OverrideEffect(**(item.get("then") or {})),
            )
            for index, item in enumerate(raw.get("overrides") or ())
        ),
        urgency_floors=dict(raw.get("urgency_floors") or {}),
        urgency_escalation_keywords=tuple(raw.get("urgency_escalation_keywords") or ()),
        approval_policy=ApprovalPolicy(
            required_for_actions=frozenset(approval_raw.get("required_for_actions") or ()),
            required_for_conditions=tuple(
                ApprovalCondition(
                    name=c.get("name", "unnamed"),
                    description=c.get("description", ""),
                    customer_tier_in=tuple(c.get("customer_tier_in") or ()),
                    urgency_in=tuple(c.get("urgency_in") or ()),
                    max_confidence=c.get("max_confidence"),
                )
                for c in (approval_raw.get("required_for_conditions") or ())
            ),
            auto_approved_actions=frozenset(approval_raw.get("auto_approved_actions") or ()),
        ),
        languages=LanguagePolicy(
            supported=tuple(
                LanguageDefinition(**lang) for lang in (languages_raw.get("supported") or ())
            ),
            default=languages_raw.get("default", "en"),
            supported_by_agents=frozenset(languages_raw.get("supported_by_agents") or ()),
        ),
    )

    _validate_routing_policy(policy)
    return policy


def _validate_routing_policy(policy: RoutingPolicy) -> None:
    """Fail loudly on any inconsistency that could produce a wrong queue."""
    if not policy.intents:
        raise PolicyError("routing.yaml defines no intents")
    if not policy.queues:
        raise PolicyError("routing.yaml defines no queues")

    # Every intent must route somewhere. This is the check that prevents a new
    # intent being added to the vocabulary without a corresponding rule.
    missing = [intent for intent in policy.intents if intent not in policy.routing_rules]
    if missing:
        raise PolicyError(f"Intents with no routing rule: {missing}")

    unknown_intents = set(policy.routing_rules) - set(policy.intents)
    if unknown_intents:
        raise PolicyError(f"routing_rules reference undeclared intents: {sorted(unknown_intents)}")

    bad_targets = {
        intent: queue
        for intent, queue in policy.routing_rules.items()
        if queue not in policy.queues
    }
    if bad_targets:
        raise PolicyError(f"routing_rules point at undefined queues: {bad_targets}")

    if policy.fallback_queue not in policy.queues:
        raise PolicyError(f"fallback_queue '{policy.fallback_queue}' is not a defined queue")

    for override in policy.overrides:
        if override.then.queue and override.then.queue not in policy.queues:
            raise PolicyError(
                f"Override '{override.name}' targets undefined queue '{override.then.queue}'"
            )
        for field_name in ("urgency", "min_urgency"):
            value = getattr(override.then, field_name)
            if value and value not in policy.urgency_levels:
                raise PolicyError(
                    f"Override '{override.name}' sets {field_name} to undefined "
                    f"urgency '{value}'"
                )
        if override.when.intent and override.when.intent not in policy.intents:
            raise PolicyError(
                f"Override '{override.name}' matches undefined intent '{override.when.intent}'"
            )
        if override.when.urgency and override.when.urgency not in policy.urgency_levels:
            raise PolicyError(
                f"Override '{override.name}' matches undefined urgency '{override.when.urgency}'"
            )

    for intent, urgency in policy.urgency_floors.items():
        if intent not in policy.intents:
            raise PolicyError(f"urgency_floors references undefined intent '{intent}'")
        if urgency not in policy.urgency_levels:
            raise PolicyError(f"urgency_floors sets undefined urgency '{urgency}'")

    if policy.languages.default not in {lang.code for lang in policy.languages.supported}:
        raise PolicyError(
            f"languages.default '{policy.languages.default}' is not in languages.supported"
        )


@functools.lru_cache(maxsize=1)
def get_vendor_registry() -> VendorRegistry:
    return load_vendor_registry()


@functools.lru_cache(maxsize=1)
def get_routing_policy() -> RoutingPolicy:
    return load_routing_policy()


def reset_policies() -> None:
    get_vendor_registry.cache_clear()
    get_routing_policy.cache_clear()
