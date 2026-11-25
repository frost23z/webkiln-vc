from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.plan import Plan, Task
from webkiln_agents.manager.scheduler import merge_dicts


class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    initial_input: Input
    plan: Plan | None = None
    # Parallel workers write these keys in the same step, so each one merges by task id.
    tasks: Annotated[dict[str, Task], merge_dicts] = Field(default_factory=dict)
    # A short summary per finished task. It is all a dependent worker gets to read.
    summaries: Annotated[dict[str, str], merge_dicts] = Field(default_factory=dict)
    # The full validated output per finished task. The API saves it in Postgres.
    outputs: Annotated[dict[str, dict[str, Any]], merge_dicts] = Field(default_factory=dict)
