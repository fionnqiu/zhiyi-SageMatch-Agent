"""Tool governance stored in Postgres, not process memory.

A provider that fails repeatedly opens its breaker. Routing reads the same row,
so a second worker does not keep sending traffic to a dead vendor.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any

from sqlalchemy import or_, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.models.platform.runtime import ProviderHealth, ToolCacheEntry

FAILURE_THRESHOLD = 3
# Closed again after this many seconds without another failure.
RECOVERY_S = 60.0
DEFAULT_CACHE_TTL_S = 120.0


def _now() -> float:
    return time.time()


def cache_key(tool: str, params: dict[str, Any], *, scope: str = "", index_version: str = "") -> str:
    raw = json.dumps({"tool": tool, "params": params, "scope": scope, "index_version": index_version}, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cache_get(db: Session, tool: str, params: dict[str, Any], *, scope: str = "", index_version: str = "") -> dict[str, Any] | None:
    # An unspecified scope must never expose a prior user's retrieval result.
    if not scope or not index_version:
        return None
    row = db.get(ToolCacheEntry, cache_key(tool, params, scope=scope, index_version=index_version))
    if row is None or row.scope != scope or row.index_version != index_version or row.expires_at <= _now():
        return None
    return dict(row.payload or {})


def cache_put(
    db: Session, tool: str, params: dict[str, Any], payload: dict[str, Any], ttl_s: float = DEFAULT_CACHE_TTL_S,
    *, scope: str = "", index_version: str = "",
) -> None:
    if not scope or not index_version:
        return
    table = ToolCacheEntry.__table__
    values = dict(key=cache_key(tool, params, scope=scope, index_version=index_version), tool_name=tool,
                  scope=scope, index_version=index_version, payload=payload, expires_at=_now() + ttl_s)
    dialect = db.get_bind().dialect.name
    if dialect not in {"postgresql", "sqlite"}:
        raise RuntimeError(f"Unsupported cache UPSERT dialect: {dialect}")
    statement = (pg_insert(table) if dialect == "postgresql" else sqlite_insert(table)).values(**values)
    db.execute(statement.on_conflict_do_update(index_elements=[table.c.key], set_={
        "payload": statement.excluded.payload, "expires_at": statement.excluded.expires_at,
    }))
    # Core UPSERT bypasses ORM instances already loaded by cache_get.
    db.expire_all()


class ProviderGovernor:
    """Sliding success/latency plus a three-state breaker for one vendor."""

    def __init__(self, db: Session, provider_id: str | None, provider_name: str) -> None:
        self.db = db
        self.provider_id = provider_id or ""
        self.provider_name = provider_name
        self._probe_owner: str | None = None

    def row(self) -> ProviderHealth | None:
        if not self.provider_id:
            return None
        found = db_get(self.db, self.provider_id)
        if found is None:
            found = ProviderHealth(
                provider_id=self.provider_id,
                provider_name=self.provider_name,
                state="closed",
                consecutive_fails=0,
                total=0,
                success=0,
                total_ms=0,
            )
            self.db.add(found)
            self.db.flush()
        return found

    def allow(self) -> bool:
        row = self.row()
        if row is None or row.state == "closed":
            return True
        current = _now()
        if row.state == "open" and (row.opened_at is None or current - row.opened_at < RECOVERY_S):
            return False
        token = str(uuid.uuid4())
        # Compare-and-swap grants one probe across workers; an abandoned probe
        # becomes available after its short lease expires.
        claimed = self.db.execute(
            update(ProviderHealth).where(
                ProviderHealth.provider_id == self.provider_id,
                or_(ProviderHealth.state == "open", ProviderHealth.state == "half_open"),
                or_(ProviderHealth.probe_lease_until.is_(None), ProviderHealth.probe_lease_until <= current),
            ).values(state="half_open", probe_owner=token, probe_lease_until=current + RECOVERY_S)
        )
        if claimed.rowcount != 1:
            return False
        self._probe_owner = token
        self.db.expire(row)
        self.db.flush()
        return True

    def eligible(self) -> bool:
        """Inspect routing eligibility without consuming a recovery probe."""
        row = self.row()
        if row is None or row.state == "closed":
            return True
        current = _now()
        if row.state == "open":
            return row.opened_at is not None and current - row.opened_at >= RECOVERY_S and (
                row.probe_lease_until is None or row.probe_lease_until <= current
            )
        # Pre-lease half-open rows have NULL here after an additive migration;
        # the atomic allow() update can safely grant their first probe.
        return row.state == "half_open" and (
            row.probe_lease_until is None or row.probe_lease_until <= current
        )

    def record_success(self, latency_ms: int) -> None:
        row = self.row()
        if row is None:
            return
        # Lock the current database state before recording a provider result.
        # A prior probe may have expired and been claimed by a different worker.
        if hasattr(self.db, "refresh"):
            self.db.refresh(row, with_for_update=True)
        if self._probe_owner is not None and row.probe_owner != self._probe_owner:
            return
        if row.state == "half_open" and row.probe_owner != self._probe_owner:
            return
        row.total += 1
        row.success += 1
        row.total_ms += max(0, latency_ms)
        row.consecutive_fails = 0
        row.state = "closed"
        row.opened_at = None
        row.probe_owner = None
        row.probe_lease_until = None
        self.db.flush()

    def record_failure(self, latency_ms: int) -> None:
        row = self.row()
        if row is None:
            return
        if hasattr(self.db, "refresh"):
            self.db.refresh(row, with_for_update=True)
        if self._probe_owner is not None and row.probe_owner != self._probe_owner:
            return
        if row.state == "half_open" and row.probe_owner != self._probe_owner:
            return
        probing = row.state == "half_open"
        row.total += 1
        row.total_ms += max(0, latency_ms)
        row.consecutive_fails += 1
        if probing or row.consecutive_fails >= FAILURE_THRESHOLD:
            row.state = "open"
            row.opened_at = _now()
            row.probe_owner = None
            row.probe_lease_until = None
        self.db.flush()

    def score(self) -> float:
        """Higher is healthier. An open breaker scores 0 so routing skips it."""
        row = self.row()
        if row is None or row.state == "open":
            return 0.0
        rate = (row.success / row.total) if row.total else 1.0
        avg = (row.total_ms / row.total) if row.total else 0.0
        latency = 1.0 / (1.0 + avg / 1000.0)
        return rate * 0.7 + latency * 0.3


def db_get(db: Session, provider_id: str) -> ProviderHealth | None:
    return db.get(ProviderHealth, provider_id)
