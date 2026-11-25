from datetime import datetime
from uuid import UUID, uuid7

from sqlalchemy import DateTime, ForeignKey, func
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
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid7)
    project_id: Mapped[UUID] = mapped_column(ForeignKey("projects.id"))
    thread_id: Mapped[str] = mapped_column(unique=True)
    phase: Mapped[str] = mapped_column(default="planning")
    status: Mapped[str]


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
    output_ref: Mapped[str | None]
