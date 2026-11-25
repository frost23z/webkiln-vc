from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.types import StateSnapshot
from webkiln_agents.manager.plan import Task
from webkiln_api.status import status_of


def snapshot(statuses: list[str], *, next_steps: tuple[str, ...] = ()) -> StateSnapshot:
    tasks = {str(i): Task(id=str(i), agent="niche", status=s) for i, s in enumerate(statuses)}  # pyrefly: ignore[bad-argument-type]
    fake: Any = SimpleNamespace(values={"tasks": tasks}, next=next_steps)
    return fake


@pytest.mark.parametrize(
    ("statuses", "next_steps", "expected"),
    [
        (["done", "pending"], ("dispatch",), "running"),
        (["done", "failed"], ("run_worker",), "running"),  # siblings still running
        (["done", "failed", "pending"], (), "failed"),
        (["done", "done"], (), "completed"),
        (["done", "pending"], (), "completed"),  # nothing left that can run
    ],
)
def test_status_comes_from_the_snapshot(
    statuses: list[str], next_steps: tuple[str, ...], expected: str
) -> None:
    assert status_of(snapshot(statuses, next_steps=next_steps)) == expected


def test_a_snapshot_with_no_tasks_is_completed() -> None:
    fake: Any = SimpleNamespace(values={}, next=())
    assert status_of(fake) == "completed"
