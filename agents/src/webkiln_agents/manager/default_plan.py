from webkiln_agents.manager.catalog import CATALOG
from webkiln_agents.manager.plan import Plan, PlannedTask


def default_plan() -> Plan:
    """The fixed plan used when the model cannot produce a valid one: one task per agent."""
    return Plan(
        rationale="Default plan: parallel research, then brand, wireframe, and generation.",
        tasks=[
            PlannedTask(
                id=spec.name,
                agent=spec.name,
                depends_on=list(spec.after),
            )
            for spec in CATALOG.values()
        ],
    )
