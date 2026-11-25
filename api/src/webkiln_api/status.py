from langgraph.types import StateSnapshot

from webkiln_api.models import RunStatus

# Task statuses that stop a run until the user retries.
STUCK = frozenset({"failed", "needs_user"})


def status_of(snapshot: StateSnapshot) -> RunStatus:
    """The run status a saved checkpoint stands for. One function, so every caller agrees.

    A step still waiting means the run is unfinished. With nothing left to run, a task that
    failed or needs the user makes the run `failed` (the user can retry it), and otherwise it is
    `completed`. An approval wait (`waiting`) comes with the interrupt in M7.
    """
    if snapshot.next:
        return "running"
    tasks = snapshot.values.get("tasks", {})
    stuck = any(task.status in STUCK for task in tasks.values())
    return "failed" if stuck else "completed"
