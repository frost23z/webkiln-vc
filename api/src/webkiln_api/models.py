from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from webkiln_agents.core.schemas import Input

RunStatus = Literal["running", "waiting", "completed", "failed"]


class RunResponse(BaseModel):
    thread_id: str
    status: RunStatus


class RunDetail(RunResponse):
    brief: Input
    state: dict[str, Any] | None  # None until the run has its first checkpoint
    review: dict[str, Any] | None  # what the user is asked to decide, while the run is `waiting`
    model_calls: int  # model calls the run has used
    call_limit: int  # the most it may use. At the limit it stops until the user allows more


class RunSummary(RunResponse):
    brief: Input
    updated_at: str  # ISO 8601 time of the run's last change


class MessageRequest(BaseModel):
    """A chat message about the finished site."""

    message: str = Field(min_length=1, max_length=2000)
    # The version the user is looking at. The edit is refused when the site has moved on.
    base_version: int | None = Field(default=None, ge=1)


class ChatEntry(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class VersionInfo(BaseModel):
    version: int
    parent_version: int | None
    cause: str  # what made it: "generate", an edit intent such as "content_edit", or "restore"
    summary: str
    created_at: str  # ISO 8601


class RestoreResponse(BaseModel):
    version: int  # the new version, a copy of the one that was restored
    restored_from: int


class TaskReport(BaseModel):
    """One task as saved in Postgres. `output` is the worker's full result, once it is done."""

    task_id: str
    agent: str
    status: str
    attempts: int
    error: str | None
    summary: str | None
    output: dict[str, Any] | None


def thread_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


class ErrorDetail(BaseModel):
    field: str
    message: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
