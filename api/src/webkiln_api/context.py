from webkiln_agents.core.config import get_settings
from webkiln_agents.manager.context import Context
from webkiln_agents.models import ChainLLM
from webkiln_agents.workers.registry import WORKERS


def build_context() -> Context:
    """The context of a real run: the registered workers and the real model chains.

    Nothing is called here. A missing API key shows up when a worker asks for its model, and
    then the task fails with a message that names the missing key. Tests pass their own context
    to `create_app`.
    """
    return Context(workers=dict(WORKERS), llm=ChainLLM(get_settings()))
