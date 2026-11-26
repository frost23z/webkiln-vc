"""A cap on the model calls of one run, so a bug or a loop cannot drain a daily quota.

Every model call goes through `LLMProvider.structured(...).ainvoke`, so wrapping the provider
counts them all: the workers, the Manager and the edit nodes. One `ainvoke` is one call, even when
the model chain falls back to a second model inside it.
"""

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from webkiln_agents.llm import LLMProvider, StructuredCall

# A full build is 8 to 13 calls and a chat edit is 2, so this is room for many edits.
DEFAULT_CALL_LIMIT = 60
# What the user adds each time they let a stopped run continue.
CALL_ALLOWANCE = 60


class CallCapReached(Exception):
    """The run used its allowance of model calls. It is never retried: only the user's answer
    (more calls) lets the run go on."""


class CallBudget:
    """The calls a run has used and the most it may use. Counts live for one process, and the
    API saves `used` with the run so a restart goes on from it."""

    def __init__(self, used: int = 0, limit: int = DEFAULT_CALL_LIMIT) -> None:
        self.used = used
        self.limit = limit

    @property
    def spent(self) -> bool:
        return self.used >= self.limit

    def spend(self) -> None:
        """Count one call, or raise `CallCapReached` when the allowance is gone."""
        if self.spent:
            msg = f"The run used its {self.limit} model calls. Allow more calls to continue."
            raise CallCapReached(msg)
        self.used += 1


class _CountedCall[T: BaseModel]:
    def __init__(self, inner: StructuredCall[T], budget: CallBudget) -> None:
        self._inner = inner
        self._budget = budget

    async def ainvoke(self, input: list[BaseMessage]) -> T | None:
        self._budget.spend()
        return await self._inner.ainvoke(input)


class BudgetedLLM:
    """An `LLMProvider` that counts each call against a `CallBudget`."""

    def __init__(self, inner: LLMProvider, budget: CallBudget) -> None:
        self._inner = inner
        self.budget = budget

    def structured[T: BaseModel](self, role: str, schema: type[T]) -> StructuredCall[T]:
        return _CountedCall(self._inner.structured(role, schema), self.budget)


CAP_REPLY = (
    "This site has used all the model calls its run is allowed, so nothing was changed. "
    "Allow more calls to keep editing."
)


def cap_reached(llm: LLMProvider | None) -> bool:
    """Whether `llm` is a budgeted provider with no calls left."""
    return isinstance(llm, BudgetedLLM) and llm.budget.spent
