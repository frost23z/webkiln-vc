from typing import Protocol

from langchain_core.messages import BaseMessage
from pydantic import BaseModel


class StructuredCall[T: BaseModel](Protocol):
    """One model call whose answer is parsed into `T`, or `None` when the model did not comply."""

    async def ainvoke(self, input: list[BaseMessage]) -> T | None: ...


class LLMProvider(Protocol):
    """Where agents get their models. Tests pass a `FakeLLM`, the API passes the real chains."""

    def structured[T: BaseModel](self, role: str, schema: type[T]) -> StructuredCall[T]:
        """The model for `role` (`manager` or an agent name), set up to answer as `schema`."""
        ...
