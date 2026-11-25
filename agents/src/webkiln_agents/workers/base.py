from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypedDict

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from webkiln_agents.manager.context import Context


class WorkerInput(TypedDict):
    """What `dispatch` sends to one worker. A worker sees nothing else of the run's state."""

    task: dict[str, Any]  # the `Task`, as a dict so the packet is plain data in the checkpoint
    brief: dict[str, Any]  # the user's `Input`
    analysis: dict[str, Any] | None  # the Manager's `BriefAnalysis`, when there is one
    context: dict[str, str]  # summaries of the tasks this one depends on, by task id


class WorkerResult(BaseModel):
    summary: str = Field(min_length=1, max_length=1500)  # about 150 words, for dependent workers
    data: dict[str, Any]  # the full output, already validated against the agent's own model


type Worker = Callable[[WorkerInput, Context], Awaitable[WorkerResult]]
