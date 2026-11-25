from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AgentName = Literal["niche", "audience", "competitor", "seo", "brand", "wireframe", "generate"]
Status = Literal["pending", "running", "done", "failed", "needs_rework"]


class PlannedTask(BaseModel):
    id: str
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
