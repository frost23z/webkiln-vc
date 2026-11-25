from langgraph.types import StateSnapshot

from webkiln_api.models import RunStatus


def status_of(snapshot: StateSnapshot) -> RunStatus:
    """The run status a saved checkpoint stands for. One function, so every caller agrees.

    A step still waiting means the run is unfinished. With nothing left to run, a failed task
    makes the run `failed`, and otherwise it is `completed`. An approval wait (`waiting`) comes
    with the interrupt in M7.
    """
    if snapshot.next:
        return "running"
    tasks = snapshot.values.get("tasks", {})
    return "failed" if any(task.status == "failed" for task in tasks.values()) else "completed"
