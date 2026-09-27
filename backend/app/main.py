"""FastAPI entry: wire the business routers and resume unfinished ingest on boot."""

from __future__ import annotations

import os
import logging
import asyncio

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.engine import make_url

from app import schemas, services
from app.integrations import llm
from app.api.admin import audit, eval as eval_api, knowledge, providers
from app.api.chat import session
from app.api.interviews import interview
from app.api.shared.deps import provider_out
from app.core.db import Base, SessionLocal, engine, ensure_schema, get_db
from app.agents.orchestration.checkpoint import CheckpointRunActive, CheckpointValidationError, PostgresCheckpointer
from app.agents.orchestration.workflows import GraphExecutionError
from app.core.config import get_settings
from app.services.operations.jobs import heartbeat, live_check, readiness_check, startup_reclaim
from app.services.interviews.interview import report_worker_running
from app.services.materials.knowledge import material_worker_running, stop_material_worker


def _cors_origins() -> list[str]:
    """默认只放行本地 5173。重启脚本改了前端端口时，用环境变量带上新地址。"""
    configured = os.environ.get("SAGEMATCH_CORS_ORIGIN", "")
    origins = [item.strip() for item in configured.split(",") if item.strip()]
    return origins or ["http://127.0.0.1:5173", "http://localhost:5173"]


app = FastAPI(title="SageMatch", version="0.2.0")
logger = logging.getLogger(__name__)


@app.exception_handler(GraphExecutionError)
async def graph_execution_error(_request: Request, exc: GraphExecutionError) -> JSONResponse:
    """Expose a stable graph error without leaking provider or prompt details."""

    budget = exc.error_code == "model_budget_exceeded"
    # Precommit command rejections are client state errors, not service outages.
    invalid_command = exc.error_code in {"interview_command_invalid", "report_regeneration_invalid"}
    detail = "报告状态已变化或不可用" if exc.error_code == "report_regeneration_invalid" else "面试状态已变化或不可用"
    return JSONResponse(status_code=429 if budget else 400 if invalid_command else 503, content={
        "detail": "本次请求已达到模型预算上限" if budget else detail if invalid_command else "请求暂时无法完成，请稍后重试",
        "error_code": exc.error_code,
        "retryable": exc.retryable,
    })


@app.exception_handler(llm.UsageBudgetError)
async def usage_budget_error(_request: Request, _exc: llm.UsageBudgetError) -> JSONResponse:
    """A configured budget denial is a bounded request failure."""

    return JSONResponse(status_code=429, content={
        "detail": "本次请求已达到模型预算上限",
        "error_code": "model_budget_exceeded",
        "retryable": False,
    })


@app.exception_handler(CheckpointValidationError)
async def checkpoint_validation_error(_request: Request, _exc: CheckpointValidationError) -> JSONResponse:
    """Reject a mismatched or expired resume without revealing thread state."""

    if isinstance(_exc, CheckpointRunActive):
        # A concurrent worker holds this run; the same command may be retried.
        return JSONResponse(status_code=409, content={
            "detail": "请求正在处理中，请稍后重试",
            "error_code": "checkpoint_run_active",
            "retryable": True,
        })
    return JSONResponse(status_code=409, content={
        "detail": "请求状态已变化，请刷新后重试",
        "error_code": "checkpoint_conflict",
        "retryable": False,
    })

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session.router)
app.include_router(interview.router)
app.include_router(knowledge.router)
app.include_router(providers.router)
app.include_router(eval_api.router)
app.include_router(audit.router)


def _runtime_worker_id() -> str:
    """Use a stable process-local identity so readiness can detect stale boots."""
    return os.environ.get("SAGEMATCH_WORKER_ID", f"api-{os.getpid()}")[:120]


async def _heartbeat_loop() -> None:
    """Keep readiness tied to a currently running worker, not a startup row."""
    while True:
        await asyncio.sleep(30)
        try:
            with SessionLocal() as db:
                heartbeat(db, _runtime_worker_id())
                db.commit()
        except Exception:
            logger.exception("worker heartbeat update failed")


async def _release_runtime_resources() -> None:
    """Unwind a partial startup and normal shutdown in reverse dependency order."""
    for stop in (session.stop_chat_recovery_worker, interview.stop_generation_worker,
                 services.stop_report_worker, stop_material_worker):
        try:
            await stop()
        except Exception as exc:
            logger.error("runtime worker cleanup failed: %s", type(exc).__name__)
    task = getattr(app.state, "heartbeat_task", None)
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("heartbeat cleanup failed: %s", type(exc).__name__)
    context = getattr(app.state, "checkpointer_context", None)
    try:
        if context is not None:
            await context.__aexit__(None, None, None)
    except Exception as exc:
        logger.error("checkpointer cleanup failed: %s", type(exc).__name__)
    finally:
        app.state.heartbeat_task = None
        app.state.checkpointer = None
        app.state.checkpointer_context = None


@app.on_event("startup")
async def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    ensure_schema(engine)
    # A graph-backed worker must never claim a durable job without its saver.
    # Open it before reclaiming or starting any poller so failed startup leaves
    # pending work available to the next healthy process.
    dsn = make_url(get_settings().database_url).set(drivername="postgresql").render_as_string(hide_password=False)
    context = PostgresCheckpointer(connection_string=dsn).open()
    try:
        app.state.checkpointer = await context.__aenter__()
        app.state.checkpointer_context = context
    except Exception as exc:
        app.state.checkpointer = None
        app.state.checkpointer_context = None
        logger.error("PostgreSQL graph checkpointer unavailable: %s", type(exc).__name__)
        raise RuntimeError("PostgreSQL graph checkpointer unavailable") from None
    try:
        db = next(get_db())
        try:
            services.seed_providers(db)
            pending = services.pending_material_ids(db)
            startup_reclaim(db, _runtime_worker_id())
            db.commit()
        finally:
            db.close()
        # 重启后把没跑完的入库接着做，不靠浏览器还连着。
        services.start_material_worker()
        for material_id in pending:
            services.enqueue_material_id(material_id)
        # All pollers begin only after the saver is available to their graph runs.
        services.start_report_worker(app.state.checkpointer)
        interview.start_generation_worker(app.state.checkpointer)
        session.start_chat_recovery_worker(app.state.checkpointer)
        app.state.heartbeat_task = asyncio.create_task(_heartbeat_loop())
    except Exception:
        await _release_runtime_resources()
        raise


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Release the saver connection after all request work has stopped."""
    await _release_runtime_resources()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/health/live")
def health_live() -> dict:
    """Liveness must stay independent from database and worker dependencies."""
    return live_check()


@app.get("/api/health/ready")
async def health_ready(db: Session = Depends(get_db)) -> JSONResponse:
    """Readiness exposes each dependency instead of returning a fixed success."""
    saver = getattr(app.state, "checkpointer", None)
    saver_ok = False
    if saver is not None:
        try:
            # The guard connection is separate from saver pipeline writes.
            await saver.ping()
            saver_ok = True
        except Exception:
            logger.exception("PostgreSQL graph checkpointer readiness probe failed")
    result = readiness_check(
        db,
        worker_id=_runtime_worker_id(),
        checkpointer=(lambda: saver_ok),
    )
    # The persisted heartbeat can remain fresh after a poller task crashes.
    # Inspect task handles on this process before advertising it as ready.
    pollers = {
        "material_worker": material_worker_running(),
        "report_worker": report_worker_running(),
        "generation_worker": interview.generation_worker_running(),
        "chat_worker": session.chat_worker_running(),
        "heartbeat_task": (
            (task := getattr(app.state, "heartbeat_task", None)) is not None
            and not task.done()
        ),
    }
    result["checks"].update({
        name: {"status": "ok"} if running else {"status": "error", "error_code": "worker_unavailable"}
        for name, running in pollers.items()
    })
    if not all(pollers.values()):
        result["status"] = "not_ready"
    status_code = 200 if result["status"] == "ready" else 503
    return JSONResponse(status_code=status_code, content=result)


@app.get("/api/admin/overview")
def admin_overview(db: Session = Depends(get_db)) -> dict:
    data = services.admin_overview(db)
    data["providers"] = [provider_out(p).model_dump() for p in data["providers"]]
    data["roles"] = [
        schemas.RoleBindingOut.model_validate(r, from_attributes=True).model_dump() for r in data["roles"]
    ]
    return data
