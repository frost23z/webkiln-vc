from webkiln_agents.workers.base import Worker

# The real worker of each agent. An agent that is not listed here is not run: its task stays
# `pending`. The research workers are added as they are built, one per commit.
WORKERS: dict[str, Worker] = {}
