from collections.abc import Mapping
from dataclasses import dataclass, field

from webkiln_agents.llm import LLMProvider
from webkiln_agents.workers.base import Worker


@dataclass(kw_only=True)
class Context:
    """Per-run settings that are not graph state. They are never checkpointed.

    `workers` maps an agent name to its worker. An agent with no entry is not run: its tasks stay
    `pending`. The cap is applied through `run_config`, because LangGraph reads it from the config.
    """

    workers: Mapping[str, Worker] = field(default_factory=dict)
    llm: LLMProvider | None = None
    max_concurrency: int = 4
