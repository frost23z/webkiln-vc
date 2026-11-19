from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel

from webkiln_agents.core.schemas import Input


class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    initial_input: Input
