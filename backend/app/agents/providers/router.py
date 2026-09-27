"""Three-layer provider routing.

1. Role binding picks the preferred vendor for this contract.
2. Health score picks among vendors of the same capability when several exist.
3. An open breaker, or a missing binding, falls through to the contract's fallback role.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.agents.contracts.contracts import AgentProfile, profile_for
from app.agents.providers.governance import ProviderGovernor
from app.models import ProviderConfig, RoleBinding


@dataclass(frozen=True)
class RouteChoice:
    """The vendor a call should use, plus why the other layers did not win."""

    role: str
    provider: ProviderConfig | None
    binding: RoleBinding | None
    reason: str
    score: float
    degraded: bool


def choose_provider(db: Session, role: str) -> RouteChoice:
    """Resolve a role to a vendor. Never raises — callers still have the env fallback."""
    profile = profile_for(role)
    primary = _candidate(db, profile)
    if primary is not None and _usable(db, primary[0]):
        provider, binding, score = primary
        return RouteChoice(profile.role, provider, binding, "role+health", score, False)

    if profile.fallback_role:
        fallback = _candidate(db, profile_for(profile.fallback_role))
        if fallback is not None and _usable(db, fallback[0]):
            provider, binding, score = fallback
            return RouteChoice(
                profile.fallback_role,
                provider,
                binding,
                f"degraded:{profile.role}->{profile.fallback_role}",
                score,
                True,
            )

    # Last resort: oldest LLM row whose breaker is not open.
    provider = next((row for row in _llm_pool(db) if _usable(db, row)), None)
    return RouteChoice(profile.role, provider, None, "oldest-llm", 0.0, True)


def _candidate(
    db: Session, profile: AgentProfile
) -> tuple[ProviderConfig, RoleBinding | None, float] | None:
    binding = (
        db.query(RoleBinding).filter(RoleBinding.role == profile.role).one_or_none()
        if _binds(db, profile.role)
        else None
    )
    bound = db.get(ProviderConfig, binding.provider_id) if binding and binding.provider_id else None
    pool = [bound] if bound is not None else _llm_pool(db)
    best: tuple[ProviderConfig, RoleBinding | None, float] | None = None
    for provider in pool:
        if provider is None or provider.capability not in {"llm", ""}:
            continue
        score = ProviderGovernor(db, provider.id, provider.name).score()
        if best is None or score > best[2]:
            best = (provider, binding if bound is not None else None, score)
    return best


def _binds(db: Session, role: str) -> bool:
    """The unit double stores one binding list. Production rows are filtered in SQL."""
    rows = getattr(db, "bindings", None)
    if not isinstance(rows, list):
        return True
    return any(getattr(row, "role", None) == role for row in rows)


def _usable(db: Session, provider: ProviderConfig) -> bool:
    """Only the gateway may claim the recovery probe when a call actually starts."""
    return ProviderGovernor(db, provider.id, provider.name).eligible()


def _llm_pool(db: Session) -> list[ProviderConfig]:
    return (
        db.query(ProviderConfig)
        .filter(ProviderConfig.capability == "llm")
        .order_by(ProviderConfig.created_at.asc())
        .all()
    )
