"""Shared ids, timestamps, and the audit row writer used by every business service."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import AuditLog

ANON = get_settings().anonymous_user_id


def new_id() -> str:
    return str(uuid.uuid4())


def now() -> datetime:
    return datetime.now(timezone.utc)


def audit(db: Session, action: str, target: str, detail: dict[str, Any]) -> None:
    """Append one mutation row. Callers commit; this must not commit on its own."""
    db.add(
        AuditLog(
            id=new_id(),
            actor="admin",
            action=action,
            target=target,
            detail=detail,
        )
    )


def upload_dir() -> Path:
    # services/common.py → app → backend → repo root, then data/uploads.
    return Path(__file__).resolve().parents[4] / "data" / "uploads"


def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "****"
    return key[:3] + "****" + key[-4:]
