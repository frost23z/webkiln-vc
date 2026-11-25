from webkiln_agents.manager.context import Context
from webkiln_agents.workers.registry import WORKERS


def build_context() -> Context:
    """The context of a real run: the registered workers. Tests pass their own to `create_app`."""
    return Context(workers=dict(WORKERS))
