"""SQLAlchemy engine and session factory."""

from collections.abc import Generator
from contextlib import contextmanager
import time

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for first-slice tables."""


engine = create_engine(get_settings().database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

_GRAPH_DEADLINE_KEY = "graph_deadline_at"


@event.listens_for(Session, "before_commit")
def _reject_expired_graph_commit(db: Session) -> None:
    """Reject a late synchronous graph write before SQLAlchemy commits it."""
    deadline = db.info.get(_GRAPH_DEADLINE_KEY)
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("graph deadline exceeded")


@contextmanager
def graph_commit_deadline(db: Session, deadline_at: float) -> Generator[None, None, None]:
    """Limit commit enforcement to this graph invocation and preserve nested scopes."""
    previous = db.info.get(_GRAPH_DEADLINE_KEY)
    db.info[_GRAPH_DEADLINE_KEY] = min(previous, deadline_at) if previous is not None else deadline_at
    try:
        yield
    finally:
        if previous is None:
            db.info.pop(_GRAPH_DEADLINE_KEY, None)
        else:
            db.info[_GRAPH_DEADLINE_KEY] = previous


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_schema(eng: Engine) -> None:
    """Add post-first-slice columns and register the runtime schema version.

    Production uses PostgreSQL's ``IF NOT EXISTS`` form.  The test and local
    recovery path also supports SQLite, whose ALTER TABLE grammar differs, so
    only missing columns are emitted there.
    """
    # Importing models here registers new runtime tables even when callers use
    # ensure_schema directly instead of going through FastAPI startup.
    from app import models as _models  # noqa: F401

    Base.metadata.create_all(bind=eng)
    patches = [
        "ALTER TABLE provider_configs ADD COLUMN IF NOT EXISTS api_key TEXT DEFAULT ''",
        "ALTER TABLE provider_configs ADD COLUMN IF NOT EXISTS models JSONB",
        "ALTER TABLE provider_configs ADD COLUMN IF NOT EXISTS notes TEXT DEFAULT ''",
        "ALTER TABLE provider_configs ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW()",
        "ALTER TABLE provider_configs ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW()",
        "ALTER TABLE material_chunks ADD COLUMN IF NOT EXISTS embedding JSONB",
        "ALTER TABLE material_chunks ADD COLUMN IF NOT EXISTS embedding_model TEXT DEFAULT ''",
        # 语音角色要同时绑 ASR 与 TTS，后加的 TTS 槽位补在已有表上。
        "ALTER TABLE role_bindings ADD COLUMN IF NOT EXISTS tts_provider_id VARCHAR(36)",
        "ALTER TABLE role_bindings ADD COLUMN IF NOT EXISTS tts_model VARCHAR(120) DEFAULT ''",
        # 开放题和场景题的参考答案是要点，不是单个选项字母。
        "ALTER TABLE questions ALTER COLUMN answer TYPE VARCHAR(200)",
        # 创建面试复用会话表存岗位，但不能因此出现在历史对话里。
        "ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS origin VARCHAR(20) DEFAULT 'chat'",
        # 分项证据作为 JSON 单独保存，旧报告保持 NULL 以便 API 区分未评分与零分。
        "ALTER TABLE reports ADD COLUMN IF NOT EXISTS dimensions JSONB",
        # Persist scorer validity directly; inferring it from localized evidence text is brittle.
        "ALTER TABLE reports ADD COLUMN IF NOT EXISTS scoring_status VARCHAR(20)",
        # The active question and follow-up count share one durable workflow state.
        "ALTER TABLE interviews ADD COLUMN IF NOT EXISTS followups_on_question INTEGER NOT NULL DEFAULT 0",
        # Governance metadata must be present on an upgraded database before
        # cache reads and half-open probes use the ORM models.
        "ALTER TABLE provider_health ADD COLUMN IF NOT EXISTS probe_lease_until DOUBLE PRECISION",
        "ALTER TABLE provider_health ADD COLUMN IF NOT EXISTS probe_owner VARCHAR(36)",
        "ALTER TABLE tool_cache ADD COLUMN IF NOT EXISTS scope VARCHAR(120) NOT NULL DEFAULT 'legacy'",
        "ALTER TABLE tool_cache ADD COLUMN IF NOT EXISTS index_version VARCHAR(120) NOT NULL DEFAULT ''",
        "ALTER TABLE llm_call_logs ADD COLUMN IF NOT EXISTS run_id VARCHAR(36)",
        "ALTER TABLE llm_call_logs ADD COLUMN IF NOT EXISTS prompt_tokens INTEGER",
        "ALTER TABLE llm_call_logs ADD COLUMN IF NOT EXISTS completion_tokens INTEGER",
        "ALTER TABLE llm_call_logs ADD COLUMN IF NOT EXISTS total_tokens INTEGER",
        "ALTER TABLE llm_call_logs ADD COLUMN IF NOT EXISTS estimated_cost DOUBLE PRECISION",
        "ALTER TABLE node_runs ADD COLUMN IF NOT EXISTS duration_ms DOUBLE PRECISION",
        "ALTER TABLE episode_briefs ADD COLUMN IF NOT EXISTS owner_id VARCHAR(120) DEFAULT 'local-user'",
        "ALTER TABLE episode_briefs ADD COLUMN IF NOT EXISTS source VARCHAR(120) DEFAULT 'legacy'",
        "ALTER TABLE episode_briefs ADD COLUMN IF NOT EXISTS provenance JSONB",
        "ALTER TABLE episode_briefs ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1",
        "ALTER TABLE episode_briefs ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ",
        "ALTER TABLE episode_briefs ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION DEFAULT 1.0",
        "ALTER TABLE user_profile_memory ADD COLUMN IF NOT EXISTS owner_id VARCHAR(120) DEFAULT 'local-user'",
        "ALTER TABLE user_profile_memory ADD COLUMN IF NOT EXISTS scope VARCHAR(20) DEFAULT 'profile'",
        "ALTER TABLE user_profile_memory ADD COLUMN IF NOT EXISTS source VARCHAR(120) DEFAULT 'legacy'",
        "ALTER TABLE user_profile_memory ADD COLUMN IF NOT EXISTS provenance JSONB",
        "ALTER TABLE user_profile_memory ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1",
        "ALTER TABLE user_profile_memory ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ",
        "ALTER TABLE user_profile_memory ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION DEFAULT 1.0",
        "ALTER TABLE stream_runs ADD COLUMN IF NOT EXISTS lease_owner VARCHAR(120)",
        "ALTER TABLE stream_runs ADD COLUMN IF NOT EXISTS lease_until TIMESTAMPTZ",
        "ALTER TABLE stream_runs ADD COLUMN IF NOT EXISTS recovery JSONB",
    ]
    sqlite_columns = {
        "provider_configs": {
            "api_key": "TEXT DEFAULT ''",
            "models": "TEXT",
            "notes": "TEXT DEFAULT ''",
            "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        },
        "material_chunks": {
            "embedding": "TEXT",
            "embedding_model": "TEXT DEFAULT ''",
        },
        "role_bindings": {
            "tts_provider_id": "VARCHAR(36)",
            "tts_model": "VARCHAR(120) DEFAULT ''",
        },
        "chat_sessions": {"origin": "VARCHAR(20) DEFAULT 'chat'"},
        "reports": {"dimensions": "TEXT", "scoring_status": "VARCHAR(20)"},
        "interviews": {"followups_on_question": "INTEGER NOT NULL DEFAULT 0"},
        "provider_health": {"probe_lease_until": "FLOAT", "probe_owner": "VARCHAR(36)"},
        "tool_cache": {"scope": "VARCHAR(120) NOT NULL DEFAULT 'legacy'", "index_version": "VARCHAR(120) NOT NULL DEFAULT ''"},
        "llm_call_logs": {
            "run_id": "VARCHAR(36)", "prompt_tokens": "INTEGER",
            "completion_tokens": "INTEGER", "total_tokens": "INTEGER", "estimated_cost": "FLOAT",
        },
        "node_runs": {"duration_ms": "FLOAT"},
        "episode_briefs": {
            "owner_id": "VARCHAR(120) DEFAULT 'local-user'",
            "source": "VARCHAR(120) DEFAULT 'legacy'",
            "provenance": "TEXT",
            "version": "INTEGER DEFAULT 1",
            "expires_at": "TIMESTAMP",
            "confidence": "FLOAT DEFAULT 1.0",
        },
        "user_profile_memory": {
            "owner_id": "VARCHAR(120) DEFAULT 'local-user'",
            "scope": "VARCHAR(20) DEFAULT 'profile'",
            "source": "VARCHAR(120) DEFAULT 'legacy'",
            "provenance": "TEXT",
            "version": "INTEGER DEFAULT 1",
            "expires_at": "TIMESTAMP",
            "confidence": "FLOAT DEFAULT 1.0",
        },
        "stream_runs": {"lease_owner": "VARCHAR(120)", "lease_until": "TIMESTAMP", "recovery": "TEXT"},
    }
    with eng.begin() as conn:
        if eng.dialect.name == "sqlite":
            # SQLite has no ADD COLUMN IF NOT EXISTS; inspect each table before
            # issuing a plain ADD COLUMN and skip the PostgreSQL-only type edit.
            from sqlalchemy import inspect

            inspector = inspect(conn)
            for table, columns in sqlite_columns.items():
                existing = {item["name"] for item in inspector.get_columns(table)} if inspector.has_table(table) else set()
                for column, definition in columns.items():
                    if column not in existing:
                        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {definition}"))
        else:
            for sql in patches:
                conn.execute(text(sql))
        # 旧的创建面试会话没有 origin。只有一条用户岗位描述、没有助手回复的，就是这类记录。
        conn.execute(
            text(
                """
                UPDATE chat_sessions
                SET origin = 'interview'
                WHERE COALESCE(origin, 'chat') = 'chat'
                  AND NOT EXISTS (
                    SELECT 1 FROM chat_messages AS reply
                    WHERE reply.session_id = chat_sessions.id AND reply.role = 'assistant'
                  )
                  AND EXISTS (
                    SELECT 1 FROM chat_messages AS ask
                    WHERE ask.session_id = chat_sessions.id AND ask.role = 'user'
                  )
                """
            )
        )
        # A marker makes readiness distinguish a migrated runtime from a DB that
        # merely happened to answer SELECT 1.
        conn.execute(
            text(
                """
                INSERT INTO runtime_schema_versions (name, version, updated_at)
                VALUES ('runtime', 2, CURRENT_TIMESTAMP)
                ON CONFLICT (name) DO UPDATE SET version = EXCLUDED.version, updated_at = EXCLUDED.updated_at
                """
            )
        )
