from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AgentName = Literal["niche", "audience", "competitor", "seo", "brand", "wireframe", "generate"]
Status = Literal[
    "pending",  # not started
    "running",  # claimed by `dispatch`
    "done",  # finished, output validated
    "failed",  # retries used up, waiting for the Manager's decision
    "needs_rework",  # sent back to run again, with a new instruction
    "skipped",  # the Manager dropped it, and the run goes on without it
    "needs_user",  # the Manager could not fix it, so it waits for the user
]


class PlannedTask(BaseModel):
    id: str = Field(min_length=1)
    agent: AgentName
    instruction: str = ""
    depends_on: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class Plan(BaseModel):
    rationale: str
    tasks: list[PlannedTask] = Field(min_length=1, max_length=12)

    model_config = ConfigDict(extra="forbid")


class Task(PlannedTask):
    status: Status = "pending"
    attempts: int = 0
    error: str | None = None

    @classmethod
    def from_planned(cls, planned: PlannedTask) -> Task:
        return cls(**planned.model_dump())
