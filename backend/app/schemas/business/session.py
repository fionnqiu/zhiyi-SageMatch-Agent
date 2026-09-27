"""Request and response shapes for the session cockpit."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatSessionOut(BaseModel):
    id: str
    title: str
    job_title: str | None = None
    created_at: datetime
    updated_at: datetime


class ChatMessageOut(BaseModel):
    id: str
    role: Literal["user", "assistant", "system"]
    content: str
    extra: dict[str, Any] | None = None
    created_at: datetime


class ChatSessionDetail(ChatSessionOut):
    messages: list[ChatMessageOut] = Field(default_factory=list)


class ClarificationAnswerIn(BaseModel):
    id: str
    option_id: str
    label: str = ""
    prompt: str = ""


class ChatAttachmentIn(BaseModel):
    """随这条消息一起提交的附件。正文已在前端读出，气泡只展示文件名。"""

    name: str
    size: int = 0
    text: str = ""


class ChatSendIn(BaseModel):
    content: str = ""
    session_id: str | None = None
    # 仅用于回复上一轮澄清题。普通消息留空。
    answers: list[ClarificationAnswerIn] = Field(default_factory=list)
    attachments: list[ChatAttachmentIn] = Field(default_factory=list)
