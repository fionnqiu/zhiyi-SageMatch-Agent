"""Three memory layers for a long session.

Working memory is the latest turns. Episodic memory is a structured brief of
what already happened, not a vector search — quoting the wrong earlier sentence
is worse than missing a vaguely similar one. The profile is durable preference
and job context for this user.
"""

from __future__ import annotations

import uuid
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.models import ChatMessage, ChatSession
from app.models.platform.runtime import EpisodeBrief, UserProfileMemory

WORKING_LIMIT = 5
BRIEF_MAX = 800


@dataclass(frozen=True)
class MemoryProposal:
    """Untrusted candidate; only commit_proposal may persist its contents."""

    layer: str
    owner_id: str
    scope: str
    source: str
    provenance: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    expected_version: int | None = None
    expires_at: datetime | None = None
    patch: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


def validate_memory_proposal(proposal: MemoryProposal, *, owner_id: str, scope: str) -> None:
    """Reject foreign, untraceable, stale, or malformed memory before mutation."""
    if proposal.layer not in {"profile", "episode"} or proposal.scope != scope:
        raise ValueError("invalid memory layer or scope")
    if not owner_id or proposal.owner_id != owner_id:
        raise PermissionError("memory owner mismatch")
    if not proposal.source.strip() or not isinstance(proposal.provenance, dict):
        raise ValueError("memory source and provenance are required")
    if not 0 <= proposal.confidence <= 1:
        raise ValueError("memory confidence must be between zero and one")
    if proposal.expected_version is not None and proposal.expected_version < 0:
        raise ValueError("memory version must be nonnegative")
    if proposal.expires_at is not None:
        if proposal.expires_at.tzinfo is None or proposal.expires_at <= datetime.now(timezone.utc):
            raise ValueError("memory expiry must be a future aware datetime")
    if not isinstance(proposal.patch, dict) or any(not isinstance(key, str) for key in proposal.patch):
        raise ValueError("memory patch must have string keys")
    if proposal.layer == "profile" and (not proposal.patch or proposal.summary):
        raise ValueError("profile proposal requires a patch only")
    if proposal.layer == "episode" and not (proposal.summary or proposal.patch):
        raise ValueError("episode proposal requires content")


class MemoryManager:
    """Read and update the three layers. Callers inject the rendered text, not raw rows."""

    def __init__(self, db: Session, *, session_id: str | None = None, interview_id: str | None = None, owner_id: str = "local-user") -> None:
        self.db = db
        self.session_id = session_id
        self.interview_id = interview_id
        self.scope = "interview" if interview_id else "session"
        self.scope_id = interview_id or session_id or ""
        self.owner_id = owner_id

    def working(self, turns: list[Any] | None = None) -> list[dict[str, str]]:
        """Latest utterances. Pass interview turns, or leave empty to read the chat."""
        if turns is not None:
            rows = list(turns)[-WORKING_LIMIT:]
            return [{"role": str(getattr(row, "role", "")), "content": str(getattr(row, "content", ""))[:400]} for row in rows]
        if not self.session_id:
            return []
        messages = (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == self.session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(WORKING_LIMIT)
            .all()
        )
        messages.reverse()
        return [{"role": row.role, "content": row.content[:400]} for row in messages]

    def episode(self) -> dict[str, Any]:
        row = self._episode_row()
        if row is None or self._expired(row) or row.owner_id != self.owner_id:
            return {"summary": "", "slots": {}}
        return {"summary": row.summary or "", "slots": dict(row.slots or {})}

    def profile(self, user_id: str) -> dict[str, Any]:
        row = self.db.get(UserProfileMemory, user_id)
        if row is None or self._expired(row) or row.owner_id != user_id or user_id != self.owner_id:
            return {}
        return dict(row.profile or {})

    def remember_profile(self, user_id: str, patch: dict[str, Any]) -> None:
        """Merge durable facts. Later interviews can read them without re-parsing the JD."""
        self.commit_proposal(MemoryProposal(layer="profile", owner_id=user_id, scope="profile", source="application", patch=patch))

    def remember_user_statement(self, text: str, *, message_id: str | None = None) -> dict[str, str]:
        """Persist only explicit, stable self-descriptions from ordinary chat.

        This intentionally uses a small deterministic allowlist instead of
        treating every conversational sentence as durable identity data.
        The resulting profile is then available to later sessions through
        ``render()``.
        """
        patterns = {
            "name": r"(?:我叫|我的名字是)\s*([^，。！？\n]{1,40})",
            "role": r"(?:我是|我的职业是|我从事)\s*([^，。！？\n]{1,60})",
            "goal": r"(?:我的目标是|我想成为|我希望)\s*([^，。！？\n]{1,80})",
            "preference": r"(?:我喜欢|我偏好|我更喜欢)\s*([^，。！？\n]{1,80})",
        }
        patch = {key: match.group(1).strip() for key, pattern in patterns.items() if (match := re.search(pattern, text))}
        if "goal" in patch:
            patch["goal"] = patch["goal"].removeprefix("成为").strip()
        if patch:
            self.commit_proposal(MemoryProposal(
                layer="profile", owner_id=self.owner_id, scope="profile", source="user_statement",
                provenance={"message_id": message_id} if message_id else {}, patch=patch,
            ))
        return patch

    def update_episode(self, *, summary: str = "", slots: dict[str, Any] | None = None) -> None:
        if not self.scope_id:
            return
        if summary or slots:
            self.commit_proposal(MemoryProposal(layer="episode", owner_id=self.owner_id, scope=self.scope, source="application", summary=summary, patch=slots or {}))

    def commit_proposal(self, proposal: MemoryProposal) -> int:
        """Validate a proposal and atomically stage one versioned memory update."""
        validate_memory_proposal(proposal, owner_id=self.owner_id, scope="profile" if proposal.layer == "profile" else self.scope)
        row = self.db.get(UserProfileMemory, self.owner_id) if proposal.layer == "profile" else self._episode_row()
        if row is not None and row.owner_id != self.owner_id:
            raise PermissionError("memory owner mismatch")
        current_version = (row.version or 1) if row is not None else 0
        if proposal.expected_version is not None and proposal.expected_version != current_version:
            raise ValueError("memory version conflict")
        if row is None:
            if proposal.layer == "profile":
                row = UserProfileMemory(user_id=self.owner_id, owner_id=self.owner_id, scope="profile", profile={})
            else:
                if not self.scope_id:
                    raise ValueError("episode scope id is required")
                row = EpisodeBrief(id=uuid.uuid4().hex, owner_id=self.owner_id, scope=self.scope, scope_id=self.scope_id, summary="", slots={})
            self.db.add(row)
        # Expired content is replaced rather than silently resurrected by a merge.
        prior = {} if self._expired(row) else (row.profile if proposal.layer == "profile" else row.slots) or {}
        if proposal.layer == "profile":
            row.profile = {**prior, **proposal.patch}
        else:
            row.slots = {**prior, **proposal.patch}
            if proposal.summary:
                row.summary = proposal.summary[:BRIEF_MAX]
        row.source = proposal.source
        row.provenance = proposal.provenance
        row.confidence = proposal.confidence
        row.expires_at = proposal.expires_at
        row.version = current_version + 1
        row.updated_at = datetime.now(timezone.utc)
        self.db.flush()
        return row.version

    @staticmethod
    def _expired(row: EpisodeBrief | UserProfileMemory) -> bool:
        expiry = row.expires_at
        return expiry is not None and expiry.replace(tzinfo=expiry.tzinfo or timezone.utc) <= datetime.now(timezone.utc)

    def render(self, turns: list[Any] | None = None, user_id: str = "local-user") -> str:
        """Text injected into a role prompt. Full history is never included."""
        episode = self.episode()
        parts: list[str] = []
        if episode["summary"]:
            parts.append(f"[情景纪要]\n{episode['summary']}")
        if episode["slots"]:
            parts.append(f"[状态槽]\n{episode['slots']}")
        profile = self.profile(user_id)
        if profile:
            parts.append(f"[用户画像]\n{profile}")
        recent = self.working(turns)
        if recent:
            lines = "\n".join(f"{item['role']}: {item['content']}" for item in recent)
            parts.append(f"[工作记忆]\n{lines}")
        return "\n\n".join(parts)

    def advance_interview_slot(self, *, question_index: int, quote: str, followups_on_question: int) -> dict[str, Any]:
        """Stay on a question until it has been followed up once, then move on."""
        slots = {
            "question_index": question_index,
            "followups_on_question": followups_on_question,
            "last_quote": quote[:180],
        }
        self.update_episode(summary=f"进行到第 {question_index + 1} 题，本题已追问 {followups_on_question} 次。", slots=slots)
        return slots

    def _episode_row(self) -> EpisodeBrief | None:
        if not self.scope_id:
            return None
        return (
            self.db.query(EpisodeBrief)
            .filter(EpisodeBrief.scope == self.scope, EpisodeBrief.scope_id == self.scope_id)
            .one_or_none()
        )


def session_user_id(db: Session, session_id: str | None, default: str = "local-user") -> str:
    if not session_id:
        return default
    row = db.get(ChatSession, session_id)
    return row.user_id if row else default
