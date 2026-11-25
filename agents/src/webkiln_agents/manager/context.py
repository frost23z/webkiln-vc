from collections.abc import Mapping
from dataclasses import dataclass, field

from langchain_core.runnables import RunnableConfig

from webkiln_agents.llm import LLMProvider
from webkiln_agents.search import SearchProvider
from webkiln_agents.workers.base import Worker


@dataclass(kw_only=True)
class Context:
    """Per-run settings that are not graph state. They are never checkpointed.

    `workers` maps an agent name to its worker. An agent with no entry is not run: its tasks stay
    `pending`. The cap is applied through `run_config`, because LangGraph reads it from the config.
    """

    workers: Mapping[str, Worker] = field(default_factory=dict)
    llm: LLMProvider | None = None
    search: SearchProvider | None = None  # web search, for the agents that use it
    max_concurrency: int = 4


def run_config(context: Context, thread_id: str | None = None) -> RunnableConfig:
    """The config to run the graph with. LangGraph reads the concurrency cap from here."""
    config: RunnableConfig = {"max_concurrency": context.max_concurrency}
    if thread_id is not None:
        config["configurable"] = {"thread_id": thread_id}
    return config
