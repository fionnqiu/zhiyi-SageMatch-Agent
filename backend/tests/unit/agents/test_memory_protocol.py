"""Persistent memory proposals are checked before any durable mutation."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.agents.roles.memory import MemoryManager, MemoryProposal
from app.models.platform.runtime import EpisodeBrief, UserProfileMemory


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    # The application registry includes PostgreSQL-only tables unrelated to memory.
    UserProfileMemory.__table__.create(engine)
    EpisodeBrief.__table__.create(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def test_profile_proposal_commits_metadata_and_version(db):
    memory = MemoryManager(db, owner_id="alice")
    proposal = MemoryProposal(
        layer="profile", owner_id="alice", scope="profile", source="user_answer",
        provenance={"message_id": "m1"}, confidence=0.9, expected_version=0,
        patch={"role": "developer"},
    )
    assert memory.commit_proposal(proposal) == 1
    db.commit()
    row = db.get(UserProfileMemory, "alice")
    assert (row.owner_id, row.source, row.provenance, row.confidence) == (
        "alice", "user_answer", {"message_id": "m1"}, 0.9,
    )
    assert memory.profile("alice") == {"role": "developer"}
    with pytest.raises(ValueError, match="version conflict"):
        memory.commit_proposal(proposal)
    assert memory.profile("alice") == {"role": "developer"}


def test_episode_rejects_foreign_owner_and_expires(db):
    memory = MemoryManager(db, interview_id="iv1", owner_id="alice")
    foreign = MemoryProposal(layer="episode", owner_id="bob", scope="interview", source="summary", summary="foreign")
    with pytest.raises(PermissionError, match="owner"):
        memory.commit_proposal(foreign)
    assert memory.episode() == {"summary": "", "slots": {}}

    valid = MemoryProposal(layer="episode", owner_id="alice", scope="interview", source="summary", summary="first", patch={"step": 1})
    assert memory.commit_proposal(valid) == 1
    assert memory.episode() == {"summary": "first", "slots": {"step": 1}}
    row = memory._episode_row()
    row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    assert memory.episode() == {"summary": "", "slots": {}}
    assert memory.commit_proposal(MemoryProposal(layer="episode", owner_id="alice", scope="interview", source="summary", summary="second")) == 2
    assert memory.episode() == {"summary": "second", "slots": {}}


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_invalid_confidence_never_writes(db, confidence):
    memory = MemoryManager(db, owner_id="alice")
    with pytest.raises(ValueError, match="confidence"):
        memory.commit_proposal(MemoryProposal(layer="profile", owner_id="alice", scope="profile", source="user", confidence=confidence, patch={"role": "dev"}))
    assert memory.profile("alice") == {}
