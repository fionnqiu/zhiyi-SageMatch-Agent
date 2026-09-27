"""Admin audit and model-call log routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app import schemas, services
from app.core.db import get_db

router = APIRouter(prefix="/api/admin")


@router.get("/logs/calls", response_model=list[schemas.LlmCallLogOut])
def admin_call_logs(db: Session = Depends(get_db)) -> list[schemas.LlmCallLogOut]:
    return [schemas.LlmCallLogOut.model_validate(r, from_attributes=True) for r in services.list_call_logs(db)]


@router.get("/logs/audit", response_model=list[schemas.AuditLogOut])
def admin_audit_logs(db: Session = Depends(get_db)) -> list[schemas.AuditLogOut]:
    return [schemas.AuditLogOut.model_validate(r, from_attributes=True) for r in services.list_audit_logs(db)]


@router.get("/logs/export-errors")
def admin_export_logs(db: Session = Depends(get_db)) -> Response:
    """Export failed model calls as a human-readable log file."""
    calls = [
        schemas.LlmCallLogOut.model_validate(row, from_attributes=True).model_dump(mode="json")
        for row in services.list_call_logs(db, limit=None, status="error")
    ]
    lines = ["SageMatch error logs", f"records: {len(calls)}", ""]
    for index, call in enumerate(calls, start=1):
        lines.extend(
            [
                f"[{index}] {call.get('created_at') or '-'}",
                f"role: {call.get('role') or '-'}",
                f"provider: {call.get('provider_name') or '-'}",
                f"model: {call.get('model') or '-'}",
                f"latency_ms: {call.get('latency_ms') or 0}",
                "error:",
                str(call.get("error") or "-"),
                "-" * 72,
            ]
        )
    body = "\n".join(lines) + "\n"
    return Response(
        content=body,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="sagematch-error-logs.log"'},
    )
