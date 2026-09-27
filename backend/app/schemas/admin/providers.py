"""Request and response shapes for vendors, role bindings, and the admin overview."""

from pydantic import BaseModel, Field


class ProviderOut(BaseModel):
    id: str
    name: str
    protocol: str
    base_url: str
    capability: str
    status: str
    latency_ms: int | None = None
    models: list[str] = Field(default_factory=list)
    notes: str = ""
    key_masked: str = ""
    has_key: bool = False


class ProviderProbeIn(BaseModel):
    protocol: str = "openai_chat"
    base_url: str = ""
    api_key: str | None = None
    provider_id: str | None = None


class ProviderIn(BaseModel):
    name: str | None = None
    protocol: str | None = None
    base_url: str | None = None
    capability: str | None = None
    api_key: str | None = None
    models: list[str] | None = None
    notes: str | None = None


class RoleBindingOut(BaseModel):
    id: str
    role: str
    label: str
    provider_id: str | None = None
    model: str = ""
    # 仅语音角色使用：与 provider_id/model（ASR）成对的 TTS 绑定。
    tts_provider_id: str | None = None
    tts_model: str = ""
    temperature: float = 0.4


class RoleBindingIn(BaseModel):
    """Partial update. Unset fields must stay unset so a one-slot edit cannot wipe the other."""

    provider_id: str | None = None
    model: str | None = None
    tts_provider_id: str | None = None
    tts_model: str | None = None
    temperature: float | None = None


class AdminOverview(BaseModel):
    providers: list[ProviderOut]
    roles: list[RoleBindingOut] = Field(default_factory=list)
    metrics: dict[str, str]
    material_count: int = 0
    chunk_count: int = 0
    index_status: str = "就绪"
    recall_score: str = "—"
    calls_24h: int = 0
    fail_rate: str = "0%"
    p95_ms: int = 0
    sse_completion_rate_24h: float | None = None
    checkpoint_recovery_rate_24h: float | None = None
