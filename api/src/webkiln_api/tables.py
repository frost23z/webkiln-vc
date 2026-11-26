from datetime import datetime
from typing import Any
from uuid import UUID, uuid7

from sqlalchemy import DateTime, ForeignKey, ForeignKeyConstraint, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid7)
    brand_name: Mapped[str]
    niche: Mapped[str]
    prompt: Mapped[str]
    status: Mapped[str] = mapped_column(default="active")
    # Who the project belongs to, as sent in `X-User-Id`. It scopes long-term memory, and a run
    # with no user has no memory. Not a login: the API has no accounts yet.
    user_id: Mapped[str | None] = mapped_column(index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid7)
    project_id: Mapped[UUID] = mapped_column(ForeignKey("projects.id"))
    thread_id: Mapped[str] = mapped_column(unique=True)
    status: Mapped[str]
    # The key of the request that last resumed the run, so a repeat of it is told apart from a race.
    resume_key: Mapped[str | None]
    # Model calls the run has used and the most it may use. The user can allow more.
    model_calls: Mapped[int] = mapped_column(default=0, server_default="0")
    call_limit: Mapped[int] = mapped_column(default=60, server_default="60")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class TaskRow(Base):
    """One planned task. Named TaskRow so it does not clash with the agents' `Task` model."""

    __tablename__ = "tasks"

    # Task ids such as "niche" repeat in every run, so a task is identified by run and id.
    run_id: Mapped[UUID] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    id: Mapped[str] = mapped_column(primary_key=True)
    agent: Mapped[str]
    instruction: Mapped[str] = mapped_column(default="")
    status: Mapped[str] = mapped_column(default="pending")
    depends_on: Mapped[list[str]] = mapped_column(JSONB, default=list)
    attempts: Mapped[int] = mapped_column(default=0)
    error: Mapped[str | None]


class TaskOutput(Base):
    """The full validated output of a finished task. It shares the task's key."""

    __tablename__ = "task_outputs"
    __table_args__ = (ForeignKeyConstraint(["run_id", "task_id"], ["tasks.run_id", "tasks.id"]),)

    run_id: Mapped[UUID] = mapped_column(primary_key=True)
    task_id: Mapped[str] = mapped_column(primary_key=True)
    summary: Mapped[str]
    data: Mapped[dict[str, Any]] = mapped_column(JSONB)


class SearchCacheRow(Base):
    """Web search results by a hash of the query, so a re-run spends no search credits."""

    __tablename__ = "search_cache"

    key: Mapped[str] = mapped_column(primary_key=True)  # sha256 of the normalised query
    query: Mapped[str]
    results: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class BrandVersion(Base):
    """One brand guideline the user decided on, with the decision. Together they are the audit
    trail of the review: what was shown, what the user answered and what they asked for."""

    __tablename__ = "brand_versions"

    run_id: Mapped[UUID] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    version: Mapped[int] = mapped_column(primary_key=True)
    guideline: Mapped[dict[str, Any]] = mapped_column(JSONB)
    decision: Mapped[str]  # "approve" or "revise"
    feedback: Mapped[str | None]
    edits: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SiteVersion(Base):
    """One version of the website. Version 1 is what `generate` built and holds every file. A
    later version holds only the files it changed and points to its parent for the rest.

    The wireframe, the page copy and the picture notes are saved with the version, so an edit can
    render the site again without parsing HTML."""

    __tablename__ = "site_versions"

    run_id: Mapped[UUID] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    version: Mapped[int] = mapped_column(primary_key=True)
    parent_version: Mapped[int | None]
    cause: Mapped[str]  # what made this version, such as "generate"
    summary: Mapped[str]
    # The brand the pages were rendered with. A design edit changes it, so a restore needs it.
    # Null on a version 1 saved before this column existed: the run's brand output is the same.
    brand: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    wireframe: Mapped[dict[str, Any]] = mapped_column(JSONB)
    copies: Mapped[dict[str, Any]] = mapped_column(JSONB)
    images: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SiteFile(Base):
    """One file of a site version, as text."""

    __tablename__ = "site_files"
    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "version"], ["site_versions.run_id", "site_versions.version"]
        ),
    )

    run_id: Mapped[UUID] = mapped_column(primary_key=True)
    version: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(primary_key=True)
    content: Mapped[str] = mapped_column(Text)


class ChatMessage(Base):
    """One message of the chat about a finished site, as the user and the UI see it.

    The row is a copy of the graph's `messages` (its position in that list is the key), made on
    every sync. The model reads only the last turns and what long-term memory recalls, so this
    table is the full text for the UI and the audit.
    """

    __tablename__ = "chat_messages"

    run_id: Mapped[UUID] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    position: Mapped[int] = mapped_column(primary_key=True)
    role: Mapped[str]  # "user" or "assistant"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
