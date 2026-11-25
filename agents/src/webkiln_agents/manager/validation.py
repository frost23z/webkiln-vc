from webkiln_agents.manager.catalog import CATALOG
from webkiln_agents.manager.plan import Plan


def find_cycle(plan: Plan) -> list[str] | None:
    """Return the ids along one dependency cycle (first id repeated at the end), or None."""
    deps = {task.id: task.depends_on for task in plan.tasks}
    done: set[str] = set()
    path: list[str] = []

    def visit(node: str) -> list[str] | None:
        if node in path:
            return [*path[path.index(node) :], node]
        if node in done or node not in deps:  # unknown ids are reported elsewhere
            return None
        path.append(node)
        for dep in deps[node]:
            if cycle := visit(dep):
                return cycle
        path.pop()
        done.add(node)
        return None

    for task_id in deps:
        if cycle := visit(task_id):
            return cycle
    return None


def validate_plan(plan: Plan) -> list[str]:
    """Return every problem found, so a retry can paste them into the next promptask."""
    errors: list[str] = []
    ids = [task.id for task in plan.tasks]
    if len(ids) != len(set(ids)):
        errors.append("task ids must be unique")
    for task in plan.tasks:
        if task.agent not in CATALOG:
            errors.append(f"{task.id} uses unknown agent {task.agent}")
        errors.extend(
            f"{task.id} depends on unknown task {dep}" for dep in task.depends_on if dep not in ids
        )
        if len(task.depends_on) != len(set(task.depends_on)):
            errors.append(f"{task.id} lists the same dependency more than once")
    if cycle := find_cycle(plan):
        errors.append(f"dependency cycle: {' -> '.join(cycle)}")
    return errors
