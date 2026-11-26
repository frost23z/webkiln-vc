"""What a long-term memory is, and where it lives.

Every record is one sentence of text (the only field that is embedded) plus filter fields. The
namespace always starts with the user id, project memory included, so a search can never see
another user's records: nothing is ever searched without that prefix.
"""

import hashlib
import re
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

Kind = Literal[
    "brand", "preference", "decision", "feedback", "business_fact", "seo", "architecture"
]
Source = Literal["user_said", "agent_inferred", "approved"]
Status = Literal["current", "superseded"]

MAX_TEXT = 600


class MemoryRecord(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT)
    kind: Kind
    source: Source
    # A fixed place in a project, like "brand": a new record in a slot replaces the one before it.
    slot: str | None = None
    version: int = 1
    status: Status = "current"
    supersedes: str | None = None  # key of the record this one replaced
    superseded_by: str | None = None  # key of the record that replaced this one
    project_id: str | None = None
    hash: str
    created_at: str


class Recalled(BaseModel):
    """A record found by a search, with the score that ranked it."""

    key: str
    score: float
    record: MemoryRecord


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def text_hash(text: str) -> str:
    """Identical text (ignoring case and spacing) has the same hash, so it is not embedded twice."""
    return hashlib.sha256(normalise(text).casefold().encode()).hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat()


def preferences_namespace(user_id: str) -> tuple[str, ...]:
    return ("user", user_id, "preferences")


def project_namespace(user_id: str, project_id: str) -> tuple[str, ...]:
    """The prefix of one project's memory. Searching it covers its decisions and feedback."""
    return ("user", user_id, "project", project_id)


def namespace_for(user_id: str, project_id: str, kind: Kind) -> tuple[str, ...]:
    if kind == "preference":
        return preferences_namespace(user_id)
    section = "feedback" if kind == "feedback" else "decisions"
    return (*project_namespace(user_id, project_id), section)
