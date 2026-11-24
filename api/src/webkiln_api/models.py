from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from webkiln_agents.core.schemas import Input

RunStatus = Literal["created", "running", "completed", "failed"]


class RunRecord(BaseModel):
    brief: Input
    status: RunStatus = "created"


class RunResponse(BaseModel):
    thread_id: str
    status: RunStatus


class RunDetail(RunResponse):
    brief: Input
    state: dict[str, Any] | None  # None until the run has started


class RunSummary(RunResponse):
    brief: Input
    updated_at: str  # ISO 8601 time of the run's latest checkpoint


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
