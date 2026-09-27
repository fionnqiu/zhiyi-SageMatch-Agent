"""Read-side queries for the admin audit and model-call screens."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import AuditLog, LlmCallLog


def list_call_logs(db: Session, limit: int | None = 50, status: str | None = None) -> list[LlmCallLog]:
    query = db.query(LlmCallLog).order_by(LlmCallLog.created_at.desc())
    if status is not None:
        query = query.filter(LlmCallLog.status == status)
    return query.all() if limit is None else query.limit(limit).all()


def list_audit_logs(db: Session, limit: int | None = 50) -> list[AuditLog]:
    query = db.query(AuditLog).order_by(AuditLog.created_at.desc())
    rows = query.all() if limit is None else query.limit(limit).all()
    # 旧的删除记录把详情写成了空对象，页面只能显示空白。读出来时用对象名补上。
    for row in rows:
        if not row.detail and row.target:
            row.detail = {"对象": row.target}
    return rows
