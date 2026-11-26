from webkiln_agents.workers.base import Worker
from webkiln_agents.workers.brand import brand_worker
from webkiln_agents.workers.competitor import competitor_worker
from webkiln_agents.workers.research import SPECS, research_worker
from webkiln_agents.workers.wireframe import wireframe_worker

# The real worker of each agent. An agent that is not listed here is not run: its task stays
# `pending`. Workers are added as they are built.
WORKERS: dict[str, Worker] = {
    **{agent: research_worker(spec) for agent, spec in SPECS.items()},
    "competitor": competitor_worker,
    "brand": brand_worker,
    "wireframe": wireframe_worker,
}
