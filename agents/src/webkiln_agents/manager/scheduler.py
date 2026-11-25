from webkiln_agents.manager.plan import Status, Task

# A task can be started from these statuses. `needs_rework` is a task the Manager sent back.
RUNNABLE: frozenset[Status] = frozenset({"pending", "needs_rework"})

# A dependency that is finished for good. A skipped task counts: the Manager decided the run
# goes on without it, and its dependents are told in their context.
SATISFIED: frozenset[Status] = frozenset({"done", "skipped"})


def merge_dicts[V](current: dict[str, V], update: dict[str, V]) -> dict[str, V]:
    """Reducer for state keyed by task id.

    Parallel workers each return the entries of their own task, so merging by key keeps every
    worker's result. A later update to the same key replaces the earlier one.
    """
    return {**current, **update}


def ready_tasks(tasks: dict[str, Task]) -> list[Task]:
    """The tasks that can start now: runnable, with every dependency finished.

    Sorted by id, so a wave always starts in the same order.
    """
    ready = [
        task
        for task in tasks.values()
        if task.status in RUNNABLE
        and all(dep in tasks and tasks[dep].status in SATISFIED for dep in task.depends_on)
    ]
    return sorted(ready, key=lambda task: task.id)
