from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

ALLOWED_TYPES = [
    ("webkiln_agents.core.schemas", "Input"),
    ("webkiln_agents.manager.analysis", "BriefAnalysis"),
    ("webkiln_agents.manager.plan", "Plan"),
    ("webkiln_agents.manager.plan", "PlannedTask"),
    ("webkiln_agents.manager.plan", "Task"),
    # The state of the `generate` subgraph, which is checkpointed with its parent.
    ("webkiln_agents.workers.generate", "GenState"),
    ("webkiln_agents.workers.generate", "PageJob"),
]

serde = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_TYPES)
