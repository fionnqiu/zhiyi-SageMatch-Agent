"""Admin home numbers: generation volume, index health, and 24h call quality."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy.orm import Session

from app.models import Interview, LlmCallLog, Material, MaterialChunk, ProviderConfig, Question, Report, RoleBinding
from app.models.platform.stream import StreamRun
from app.models.platform.runtime import GraphRun
from app.services.chat.stream_events import stream_completion_outcomes
from app.services.shared.common import now
from app.services.operations.providers import ordered_roles, seed_providers
from app.services.materials.recall import embedding_ready


def admin_overview(db: Session) -> dict[str, Any]:
    seed_providers(db)
    providers = db.query(ProviderConfig).all()
    roles = ordered_roles(db.query(RoleBinding).all())
    q_count = db.query(Question).count()
    i_count = db.query(Interview).count()
    r_count = db.query(Report).count()
    m_count = db.query(Material).count()
    c_count = db.query(MaterialChunk).count()
    since = now() - timedelta(hours=24)
    logs = db.query(LlmCallLog).filter(LlmCallLog.created_at >= since).all()
    streams = db.query(StreamRun).filter(StreamRun.created_at >= since).all()
    completed = list(stream_completion_outcomes(db, streams).values())
    graph_runs = db.query(GraphRun).filter(GraphRun.created_at >= since).all()
    recoveries = [row.diagnostics["checkpoint_recovered"] for row in graph_runs
                  if isinstance(row.diagnostics, dict)
                  and type(row.diagnostics.get("checkpoint_recovered")) is bool]
    fails = sum(1 for x in logs if x.status != "ok")
    p95 = 0
    if logs:
        lat = sorted(x.latency_ms for x in logs)
        p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
    return {
        "providers": providers,
        "roles": roles,
        "metrics": {
            "已生成题目": str(q_count),
            "面试场次": str(i_count),
            "复盘报告": str(r_count),
            "知识物料": str(m_count),
        },
        "material_count": m_count,
        "chunk_count": c_count,
        "index_status": "就绪" if c_count else "空索引",
        "recall_score": "—" if not c_count else ("混合召回" if embedding_ready() else "词法召回"),
        "calls_24h": len(logs),
        "fail_rate": f"{(fails / len(logs) * 100):.1f}%" if logs else "0%",
        "p95_ms": p95,
        # Exclude still-running streams; a closed error frame is a failure.
        "sse_completion_rate_24h": round(sum(completed) / len(completed), 4) if completed else None,
        "checkpoint_recovery_rate_24h": round(sum(recoveries) / len(recoveries), 4) if recoveries else None,
    }
