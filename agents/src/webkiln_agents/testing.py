"""Helpers shared by the tests of both packages. Nothing here calls a real model."""

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from webkiln_agents.llm import LLMProvider
from webkiln_agents.manager.catalog import CATALOG
from webkiln_agents.manager.context import Context
from webkiln_agents.workers.base import Worker, WorkerInput, WorkerResult

SAMPLE_BRIEF = {
    "brand_name": "Cedar Stay",
    "prompt": "A boutique hotel website with booking",
    "niche": "Hospitality",
}


class FakeStructured[T: BaseModel]:
    def __init__(self, llm: FakeLLM, role: str, schema: type[T]) -> None:
        self._llm = llm
        self._role = role
        self._schema = schema

    async def ainvoke(self, input: list[BaseMessage]) -> T | None:
        self._llm.prompts.append((self._role, list(input)))
        item = self._llm.next_response(self._role)
        if isinstance(item, BaseException):
            raise item
        if item is None or isinstance(item, self._schema):
            return item
        return self._schema.model_validate(
            item
        )  # a bad dict raises ValidationError, as a model would


class FakeLLM:
    """A model that answers from a script, one list of responses per role.

    A response is a schema instance, a dict (validated against the schema, so a bad one raises
    `ValidationError`), `None` (a model that did not comply) or an exception (raised). When a
    role runs out of responses, its last one repeats. Every call is recorded in `prompts`.
    """

    def __init__(self, script: Mapping[str, Sequence[Any]] | None = None) -> None:
        self._script = {role: list(items) for role, items in (script or {}).items()}
        self.prompts: list[tuple[str, list[BaseMessage]]] = []

    def next_response(self, role: str) -> Any:
        items = self._script.get(role)
        if not items:
            msg = f"FakeLLM has no response for role {role!r}"
            raise AssertionError(msg)
        return items.pop(0) if len(items) > 1 else items[0]

    def structured[T: BaseModel](self, role: str, schema: type[T]) -> FakeStructured[T]:
        return FakeStructured(self, role, schema)

    def prompts_for(self, role: str) -> list[list[BaseMessage]]:
        return [messages for name, messages in self.prompts if name == role]


async def fake_worker(packet: WorkerInput, _: Context) -> WorkerResult:
    agent = packet["task"]["agent"]
    return WorkerResult(summary=f"{agent} summary", data={"agent": agent})


def fake_workers() -> dict[str, Worker]:
    """A fake worker for every agent in the catalog."""
    return dict.fromkeys(CATALOG, fake_worker)


def fake_context(
    *,
    workers: Mapping[str, Worker] | None = None,
    llm: LLMProvider | None = None,
    max_concurrency: int = 4,
) -> Context:
    """A context where every agent has a fake worker, unless `workers` says otherwise."""
    return Context(
        workers=fake_workers() if workers is None else workers,
        llm=llm,
        max_concurrency=max_concurrency,
    )
