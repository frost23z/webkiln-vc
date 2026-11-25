from webkiln_agents.workers.base import Worker
from webkiln_agents.workers.research import SPECS, research_worker

# The real worker of each agent. An agent that is not listed here is not run: its task stays
# `pending`. Workers are added as they are built.
WORKERS: dict[str, Worker] = {agent: research_worker(spec) for agent, spec in SPECS.items()}
