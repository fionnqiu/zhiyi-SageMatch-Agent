"""Admin vendor and role-binding routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import schemas, services
from app.api.shared.deps import provider_out
from app.core.db import get_db

router = APIRouter(prefix="/api/admin")


@router.get("/providers", response_model=list[schemas.ProviderOut])
def admin_providers(db: Session = Depends(get_db)) -> list[schemas.ProviderOut]:
    return [provider_out(p) for p in services.list_providers(db)]


@router.post("/providers", response_model=schemas.ProviderOut)
def admin_create_provider(payload: schemas.ProviderIn, db: Session = Depends(get_db)) -> schemas.ProviderOut:
    row = services.create_provider(db, payload.model_dump())
    return provider_out(row)


@router.post("/provider-models")
async def admin_probe_models(payload: schemas.ProviderProbeIn, db: Session = Depends(get_db)) -> dict[str, list[str]]:
    try:
        models = await services.probe_models(db, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"models": models}


@router.patch("/providers/{provider_id}", response_model=schemas.ProviderOut)
def admin_update_provider(
    provider_id: str, payload: schemas.ProviderIn, db: Session = Depends(get_db)
) -> schemas.ProviderOut:
    try:
        row = services.update_provider(db, provider_id, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return provider_out(row)


@router.post("/providers/{provider_id}/ping", response_model=schemas.ProviderOut)
async def admin_ping_provider(provider_id: str, db: Session = Depends(get_db)) -> schemas.ProviderOut:
    try:
        row = await services.ping_provider(db, provider_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return provider_out(row)


@router.delete("/providers/{provider_id}")
@router.post("/providers/{provider_id}/delete")
def admin_delete_provider(provider_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    # POST /delete is the stable path: some proxies 405 a DELETE with JSON headers.
    try:
        services.delete_provider(db, provider_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": "deleted"}


@router.get("/roles", response_model=list[schemas.RoleBindingOut])
def admin_roles(db: Session = Depends(get_db)) -> list[schemas.RoleBindingOut]:
    return [schemas.RoleBindingOut.model_validate(r, from_attributes=True) for r in services.list_roles(db)]


@router.patch("/roles/{role}", response_model=schemas.RoleBindingOut)
def admin_update_role(
    role: str, payload: schemas.RoleBindingIn, db: Session = Depends(get_db)
) -> schemas.RoleBindingOut:
    try:
        # 只提交前端实际改的字段，避免改 ASR 时把 TTS 槽位写成空。
        row = services.update_role(db, role, payload.model_dump(exclude_unset=True))
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return schemas.RoleBindingOut.model_validate(row, from_attributes=True)
