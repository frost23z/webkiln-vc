import pytest
from webkiln_agents.manager.plan import Status, Task
from webkiln_agents.manager.scheduler import merge_dicts, ready_tasks


def tasks_of(*specs: tuple[str, Status, list[str]]) -> dict[str, Task]:
    return {
        id_: Task(id=id_, agent="niche", status=status, depends_on=deps)
        for id_, status, deps in specs
    }


def ready_ids(tasks: dict[str, Task]) -> list[str]:
    return [task.id for task in ready_tasks(tasks)]


@pytest.mark.parametrize(
    ("specs", "expected"),
    [
        # No dependencies: everything pending starts, in id order.
        ([("b", "pending", []), ("a", "pending", [])], ["a", "b"]),
        # A dependency that is not finished blocks its dependent.
        ([("a", "pending", []), ("b", "pending", ["a"])], ["a"]),
        ([("a", "running", []), ("b", "pending", ["a"])], []),
        ([("a", "failed", []), ("b", "pending", ["a"])], []),
        ([("a", "needs_user", []), ("b", "pending", ["a"])], []),
        # Done and skipped dependencies both let the dependent start.
        ([("a", "done", []), ("b", "pending", ["a"])], ["b"]),
        ([("a", "skipped", []), ("b", "pending", ["a"])], ["b"]),
        # Every dependency has to be finished, not just one.
        ([("a", "done", []), ("b", "running", []), ("c", "pending", ["a", "b"])], []),
        # A task sent back for rework starts again. Finished and running tasks never do.
        ([("a", "needs_rework", [])], ["a"]),
        ([("a", "done", []), ("b", "running", []), ("c", "skipped", [])], []),
        # An unknown dependency never finishes, so the task never starts.
        ([("a", "pending", ["ghost"])], []),
        ([], []),
    ],
)
def test_ready_tasks(specs: list[tuple[str, Status, list[str]]], expected: list[str]) -> None:
    assert ready_ids(tasks_of(*specs)) == expected


def test_merge_dicts_keeps_both_sides_and_the_later_value() -> None:
    assert merge_dicts({"a": 1, "b": 2}, {"b": 3, "c": 4}) == {"a": 1, "b": 3, "c": 4}


def test_merge_dicts_does_not_change_its_arguments() -> None:
    current = {"a": 1}
    merge_dicts(current, {"b": 2})
    assert current == {"a": 1}
