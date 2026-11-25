from webkiln_agents.manager.catalog import CATALOG
from webkiln_agents.manager.plan import AgentName, Plan


def find_cycle(plan: Plan) -> list[str] | None:
    """Return one dependency cycle, with the first ID repeated at the end."""
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


def _depends_on_agent(plan: Plan, task_id: str, agent: AgentName) -> bool:
    """Check whether a task transitively depends on a task of another agent."""
    by_id = {task.id: task for task in plan.tasks}
    seen: set[str] = set()
    stack = list(by_id[task_id].depends_on) if task_id in by_id else []

    while stack:
        current = stack.pop()

        if current in seen or current not in by_id:
            continue

        seen.add(current)

        if by_id[current].agent == agent:
            return True

        stack.extend(by_id[current].depends_on)

    return False


def validate_plan(plan: Plan) -> list[str]:
    """Return every problem found, so a retry can paste them into the next prompt."""
    errors: list[str] = []
    ids = [task.id for task in plan.tasks]
    used = {task.agent for task in plan.tasks}
    # The graph checks look tasks up by id, so they mean nothing while ids repeat.
    ids_unique = len(ids) == len(set(ids))
    if not ids_unique:
        errors.append("task ids must be unique")
    for task in plan.tasks:
        if task.agent not in CATALOG:
            errors.append(f"{task.id} uses unknown agent {task.agent}")
        errors.extend(
            f"{task.id} depends on unknown task {dep}" for dep in task.depends_on if dep not in ids
        )
        if len(task.depends_on) != len(set(task.depends_on)):
            errors.append(f"{task.id} lists the same dependency more than once")
    if ids_unique and (cycle := find_cycle(plan)):
        errors.append(f"dependency cycle: {' -> '.join(cycle)}")

    errors.extend(f"plan is missing agent {name}" for name in CATALOG if name not in used)

    if not ids_unique:
        return errors

    # Enforce catalog dependencies for every agent.
    # Missing prerequisite agents are already reported above.
    for task in plan.tasks:
        if task.agent not in CATALOG:
            continue

        for required_agent in CATALOG[task.agent].after:
            if required_agent not in used:
                continue

            if not _depends_on_agent(plan, task.id, required_agent):
                errors.append(f"{task.id} ({task.agent}) must depend on a {required_agent} task")

    return errors
