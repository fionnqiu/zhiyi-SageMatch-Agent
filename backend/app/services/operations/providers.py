"""Vendor catalog and role bindings shown on the admin providers screen."""

from __future__ import annotations

from typing import Any

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.integrations import llm
from app.core.config import get_settings
from app.integrations.llm import CHAT_PROTOCOL
from app.models import ProviderConfig, RoleBinding
from app.services.shared.common import audit, new_id
from app.services.operations.llm_gateway import log_call

# Role table shown on the admin providers screen. Speech stays deferred.
ROLES = [
    ("analyst", "岗位分析师", 0.2),
    ("author", "出题官", 0.4),
    ("critic", "质检员", 0.0),
    ("interviewer", "面试官", 0.5),
    ("scorer", "评分员", 0.0),
    ("coach", "复盘教练", 0.3),
    ("judge", "质量审计员", 0.0),
    # 一个角色同时绑 ASR（provider_id/model）和 TTS（tts_provider_id/tts_model）。
    ("speech", "语音", 0.0),
]
# 列表顺序以 ROLES 为准。更新绑定会改行，不能靠数据库默认返回顺序。
_ROLE_RANK = {role: index for index, (role, _, _) in enumerate(ROLES)}


def ordered_roles(rows: list[RoleBinding]) -> list[RoleBinding]:
    # 不在 ROLES 里的旧角色（例如已移走的嵌入模型）不再出现在路由表。
    kept = [row for row in rows if row.role in _ROLE_RANK]
    return sorted(kept, key=lambda row: _ROLE_RANK[row.role])


def seed_providers(db: Session) -> None:
    settings = get_settings()
    if not db.query(ProviderConfig).count():
        rows = [
            ProviderConfig(
                id=new_id(),
                name="OpenAI Chat Completions 兼容端点",
                protocol=CHAT_PROTOCOL,
                base_url=settings.llm_base_url or "https://movoapi.top",
                capability="llm",
                status="configured" if settings.llm_api_key else "missing_key",
                api_key=settings.llm_api_key or "",
                models=[settings.llm_model] if settings.llm_model else [],
            ),
            ProviderConfig(
                id=new_id(),
                name="ASR · 流式识别",
                protocol="websocket_audio",
                base_url="",
                capability="asr",
                status="deferred",
                notes="用户语音转写。云端网关未接前保持后置",
            ),
            ProviderConfig(
                id=new_id(),
                name="TTS · 流式合成",
                protocol="websocket_audio",
                base_url="",
                capability="tts",
                status="deferred",
                notes="面试官语音播报。云端网关未接前保持后置",
            ),
        ]
        db.add_all(rows)
        db.flush()
    else:
        # Columns added after first seed; copy env key onto the default LLM row if empty.
        default_llm = (
            db.query(ProviderConfig)
            .filter(ProviderConfig.capability == "llm")
            .order_by(ProviderConfig.created_at.asc())
            .first()
        )
        if default_llm and not default_llm.api_key and settings.llm_api_key:
            default_llm.api_key = settings.llm_api_key
            default_llm.base_url = default_llm.base_url or settings.llm_base_url
            default_llm.status = "configured"
            if not default_llm.models and settings.llm_model:
                default_llm.models = [settings.llm_model]
    # 旧的「语音」能力拆成 ASR / TTS：原行改为识别，并补一条合成占位，避免角色没有 TTS 可选。
    split_legacy_speech_capability(db)
    ensure_roles(db)
    db.commit()


def split_legacy_speech_capability(db: Session) -> None:
    """Turn pre-split capability='speech' rows into ASR, and add one TTS sibling if missing."""
    legacy = (
        db.query(ProviderConfig)
        .filter(ProviderConfig.capability == "speech")
        .order_by(ProviderConfig.created_at.asc())
        .all()
    )
    if not legacy:
        return
    for row in legacy:
        row.capability = "asr"
        row.name = row.name.replace("ASR / TTS", "ASR").replace("ASR/TTS", "ASR")
    if db.query(ProviderConfig).filter(ProviderConfig.capability == "tts").first():
        return
    src = legacy[0]
    db.add(
        ProviderConfig(
            id=new_id(),
            name="TTS · 流式合成",
            protocol=src.protocol,
            base_url=src.base_url,
            capability="tts",
            status=src.status,
            api_key=src.api_key or "",
            models=[],
            notes=src.notes or "由原「语音」供应商拆出，请单独选择合成模型",
        )
    )
    db.flush()


def first_provider(db: Session, capability: str) -> ProviderConfig | None:
    return (
        db.query(ProviderConfig)
        .filter(ProviderConfig.capability == capability)
        .order_by(ProviderConfig.created_at.asc())
        .first()
    )


def ensure_roles(db: Session) -> None:
    existing = {r.role: r for r in db.query(RoleBinding).all()}
    for role, row in list(existing.items()):
        if role not in _ROLE_RANK:
            db.delete(row)
            existing.pop(role)
    default_llm = first_provider(db, "llm")
    asr = first_provider(db, "asr")
    tts = first_provider(db, "tts")
    settings = get_settings()
    for role, label, temp in ROLES:
        if role in existing:
            # 角色文案以 ROLES 为准，避免库里还留着拆分前的旧标签。
            if existing[role].label != label:
                existing[role].label = label
            continue
        provider = asr if role == "speech" else default_llm
        db.add(
            RoleBinding(
                id=new_id(),
                role=role,
                label=label,
                provider_id=provider.id if provider else None,
                model="" if role == "speech" else settings.llm_model,
                tts_provider_id=tts.id if role == "speech" and tts else None,
                tts_model="",
                temperature=temp,
            )
        )


def list_providers(db: Session) -> list[ProviderConfig]:
    seed_providers(db)
    return db.query(ProviderConfig).order_by(ProviderConfig.created_at.asc()).all()


def create_provider(db: Session, payload: dict[str, Any]) -> ProviderConfig:
    name = (payload.get("name") or "新供应商").strip() or "新供应商"
    row = ProviderConfig(
        id=new_id(),
        name=name,
        protocol=_chat_protocol(payload.get("protocol")),
        base_url=payload.get("base_url") or "",
        capability=payload.get("capability") or "llm",
        api_key=payload.get("api_key") or "",
        models=payload.get("models") or [],
        notes=payload.get("notes") or "",
        status="configured" if payload.get("api_key") else "missing_key",
    )
    db.add(row)
    audit(db, "provider.create", row.name, {"protocol": row.protocol})
    db.commit()
    db.refresh(row)
    return row


def update_provider(db: Session, provider_id: str, payload: dict[str, Any]) -> ProviderConfig:
    row = db.get(ProviderConfig, provider_id)
    if not row:
        raise ValueError("供应商不存在")
    for key in ("name", "base_url", "capability", "notes"):
        if key in payload and payload[key] is not None:
            setattr(row, key, payload[key])
    if payload.get("protocol") is not None:
        row.protocol = _chat_protocol(payload.get("protocol"), row.capability)
    if payload.get("models") is not None:
        row.models = payload["models"]
    if payload.get("api_key"):
        row.api_key = payload["api_key"]
    # 没密钥时：WebSocket 音频占位保持「后置」，其余能力标缺密钥。
    row.status = (
        "configured"
        if row.api_key
        else ("deferred" if (row.protocol or "").startswith("websocket") else "missing_key")
    )
    audit(db, "provider.update", row.name, {"protocol": row.protocol})
    db.commit()
    db.refresh(row)
    return row


def delete_provider(db: Session, provider_id: str) -> None:
    row = db.get(ProviderConfig, provider_id)
    if not row:
        raise ValueError("供应商不存在")
    # Drop role bindings first so a delete cannot leave dangling provider_id FKs.
    # ASR 槽和 TTS 槽都可能指向这条供应商，删之前先解开，避免外键悬挂。
    bound = (
        db.query(RoleBinding)
        .filter(or_(RoleBinding.provider_id == provider_id, RoleBinding.tts_provider_id == provider_id))
        .all()
    )
    for binding in bound:
        if binding.provider_id == provider_id:
            binding.provider_id = None
            binding.model = ""
        if binding.tts_provider_id == provider_id:
            binding.tts_provider_id = None
            binding.tts_model = ""
    name = row.name
    protocol = row.protocol
    db.delete(row)
    audit(db, "provider.delete", name, {"protocol": protocol})
    db.commit()


async def probe_models(db: Session, payload: dict[str, Any]) -> list[str]:
    """List remote models using the form key, or the stored key when editing."""
    protocol = _chat_protocol(payload.get("protocol"))
    base_url = (payload.get("base_url") or "").strip()
    api_key = (payload.get("api_key") or "").strip()
    provider_id = payload.get("provider_id")
    if provider_id:
        row = db.get(ProviderConfig, provider_id)
        if not row:
            raise ValueError("供应商不存在")
        protocol = protocol or row.protocol
        base_url = base_url or row.base_url
        if not api_key:
            api_key = row.api_key or ""
    if not api_key:
        api_key = get_settings().llm_api_key or ""
    if not base_url:
        base_url = get_settings().llm_base_url or ""
    return await llm.list_models(protocol, api_key, base_url)


async def ping_provider(db: Session, provider_id: str) -> ProviderConfig:
    row = db.get(ProviderConfig, provider_id)
    if not row:
        raise ValueError("供应商不存在")
    if (row.protocol or "").startswith("websocket"):
        row.status = "deferred"
        db.commit()
        return row
    model = (row.models or [get_settings().llm_model] or [""])[0]
    ok, ms, err = await llm.ping_provider(
        row.protocol, row.api_key, row.base_url, model, capability=row.capability
    )
    row.latency_ms = ms
    row.status = "configured" if ok else "error"
    log_call(db, "ping", row.name, model, "ok" if ok else "error", ms, None if ok else err)
    db.commit()
    db.refresh(row)
    if not ok:
        raise ValueError(err or "连通失败")
    return row


def list_roles(db: Session) -> list[RoleBinding]:
    seed_providers(db)
    heal_role_bindings(db)
    return ordered_roles(db.query(RoleBinding).all())


def heal_role_bindings(db: Session) -> None:
    """Reattach roles that lost their provider after a delete / empty seed."""
    default_llm = first_provider(db, "llm")
    asr = first_provider(db, "asr")
    tts = first_provider(db, "tts")
    changed = False
    for row in db.query(RoleBinding).all():
        if row.role == "speech":
            # 语音角色的两槽各自回填：空的 ASR 找 ASR 供应商，空的 TTS 找 TTS 供应商。
            if not row.provider_id and asr:
                row.provider_id = asr.id
                changed = True
            if not row.tts_provider_id and tts:
                row.tts_provider_id = tts.id
                changed = True
            changed = fill_model_from_provider(db, row, "model", row.provider_id) or changed
            changed = fill_model_from_provider(db, row, "tts_model", row.tts_provider_id) or changed
            continue
        if not row.provider_id and default_llm:
            row.provider_id = default_llm.id
            changed = True
        changed = fill_model_from_provider(db, row, "model", row.provider_id) or changed
    if changed:
        db.commit()


def fill_model_from_provider(db: Session, row: RoleBinding, field: str, provider_id: str | None) -> bool:
    """If the bound model is empty or no longer in that vendor's catalog, take the first catalog id."""
    bound = db.get(ProviderConfig, provider_id) if provider_id else None
    models = list(bound.models or []) if bound else []
    current = getattr(row, field) or ""
    if models and current not in models:
        setattr(row, field, models[0])
        return True
    return False


def update_role(db: Session, role: str, payload: dict[str, Any]) -> RoleBinding:
    row = db.query(RoleBinding).filter(RoleBinding.role == role).one_or_none()
    if not row:
        raise ValueError("角色不存在")
    if "provider_id" in payload:
        row.provider_id = payload["provider_id"]
    if "model" in payload and payload["model"] is not None:
        row.model = payload["model"]
    if "tts_provider_id" in payload:
        row.tts_provider_id = payload["tts_provider_id"]
    if "tts_model" in payload and payload["tts_model"] is not None:
        row.tts_model = payload["tts_model"]
    if "temperature" in payload and payload["temperature"] is not None:
        row.temperature = float(payload["temperature"])
    audit(
        db,
        "role.bind",
        role,
        {
            "provider_id": row.provider_id,
            "model": row.model,
            "tts_provider_id": row.tts_provider_id,
            "tts_model": row.tts_model,
        },
    )
    db.commit()
    db.refresh(row)
    return row


def _chat_protocol(value: object, capability: object = "llm") -> str:
    """文本对话固定 Chat Completions。语音占位仍可保留 WebSocket，不改成聊天协议。"""
    protocol = str(value or "").strip()
    if str(capability or "") in {"asr", "tts"} and protocol.startswith("websocket"):
        return protocol
    if protocol in {"openai_embeddings", "openai_embed"}:
        return protocol
    return CHAT_PROTOCOL
