"""Knowledge routes: material upload, chunk detail, and recall probes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app import schemas, services
from app.core.db import get_db

router = APIRouter(prefix="/api/admin")


@router.get("/materials", response_model=list[schemas.MaterialOut])
def admin_materials(db: Session = Depends(get_db)) -> list[schemas.MaterialOut]:
    return [schemas.MaterialOut.model_validate(m, from_attributes=True) for m in services.list_materials(db)]


@router.get("/materials/{material_id}", response_model=schemas.MaterialDetail)
def admin_material(material_id: str, db: Session = Depends(get_db)) -> schemas.MaterialDetail:
    row = services.get_material(db, material_id)
    if not row:
        raise HTTPException(404, "material not found")
    return schemas.MaterialDetail(
        **schemas.MaterialOut.model_validate(row, from_attributes=True).model_dump(),
        chunks=[schemas.ChunkOut.model_validate(c, from_attributes=True) for c in row.chunks],
    )


@router.post("/materials", response_model=schemas.MaterialDetail)
async def admin_upload_material(file: UploadFile = File(...), db: Session = Depends(get_db)) -> schemas.MaterialDetail:
    data = await file.read()
    # 先落成 pending 再排队。解析和向量不占用这条请求，切换页面也不会取消。
    row = services.accept_material(db, file.filename or "upload.txt", data, file.content_type or "text/plain")
    services.enqueue_material_id(row.id)
    return schemas.MaterialDetail(
        **schemas.MaterialOut.model_validate(row, from_attributes=True).model_dump(),
        chunks=[],
    )


@router.delete("/materials/{material_id}")
def admin_delete_material(material_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        services.delete_material(db, material_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": "deleted"}


@router.post("/recall", response_model=schemas.RecallOut)
async def admin_recall(payload: dict, db: Session = Depends(get_db)) -> schemas.RecallOut:
    query = str(payload.get("query") or "").strip()
    if not query:
        raise HTTPException(400, "query required")
    data = await services.recall(db, query)
    return schemas.RecallOut(
        query=data["query"],
        index_status=data["index_status"],
        hits=[schemas.RecallHit.model_validate(h) for h in data["hits"]],
    )
