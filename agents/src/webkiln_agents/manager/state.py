from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.plan import Plan, Task


class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    initial_input: Input
    plan: Plan | None = None
    tasks: dict[str, Task] = Field(default_factory=dict)
