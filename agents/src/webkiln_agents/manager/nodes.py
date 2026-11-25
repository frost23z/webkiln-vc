from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.default_plan import default_plan
from webkiln_agents.manager.plan import Task
from webkiln_agents.manager.state import State


def process_initial_input(state: Input) -> dict:
    return {"initial_input": state}


def seed_tasks(state: State) -> dict:
    """Start every run from the default plan and turn it into trackable task state.

    The Manager model does not plan the first pass: the catalog's `after` rules already fix the
    order, so a model call here would only risk a bad plan. The Manager plans when there is
    something to decide, which is feedback on the brand and edits to the finished site.
    """
    plan = default_plan()
    return {"plan": plan, "tasks": {t.id: Task.from_planned(t) for t in plan.tasks}}
