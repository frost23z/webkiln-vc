from collections import Counter
from typing import Any

import pytest
from webkiln_agents.manager.context import Context, run_config
from webkiln_agents.manager.failures import (
    MAX_REWORKS,
    FailureDecision,
    FailureReview,
    apply_decision,
    can_retry,
    enforce,
)
from webkiln_agents.manager.graph import build_graph
from webkiln_agents.manager.plan import Task
from webkiln_agents.models import ConfigError
from webkiln_agents.testing import SAMPLE_BRIEF, FakeLLM, fake_context, fake_worker, fake_workers
from webkiln_agents.workers.base import WorkerInput, WorkerResult
from webkiln_agents.workers.errors import WorkerOutputError


def decision(task_id: str, action: str, reason: str = "because", instruction: str = "") -> dict:
    return {"task_id": task_id, "action": action, "reason": reason, "instruction": instruction}


def review(*decisions: dict) -> dict[str, Any]:
    return {"decisions": list(decisions)}


def task(agent: str = "seo", **kw: Any) -> Task:
    return Task(id=agent, agent=agent, **kw)  # pyrefly: ignore[bad-argument-type]


def failing(
    agents: dict[str, Exception | int],
) -> tuple[dict[str, Any], Counter[str], list[WorkerInput]]:
    """Workers that raise: an exception always, or an int for that many first calls."""
    calls = Counter[str]()
    packets: list[WorkerInput] = []

    async def worker(packet: WorkerInput, context: Context) -> WorkerResult:
        name = packet["task"]["agent"]
        calls[name] += 1
        packets.append(packet)
        spec = agents.get(name)
        if isinstance(spec, Exception):
            raise spec
        if isinstance(spec, int) and calls[name] <= spec:
            msg = "bad output"
            raise WorkerOutputError(msg)
        return await fake_worker(packet, context)

    return dict.fromkeys(fake_workers(), worker), calls, packets


async def run(workers: dict[str, Any], llm: FakeLLM | None) -> dict[str, Any]:
    context = fake_context(workers=workers, llm=llm)
    return await build_graph().ainvoke(SAMPLE_BRIEF, run_config(context), context=context)


async def test_a_retry_with_a_new_instruction_runs_only_that_worker_again() -> None:
    workers, calls, packets = failing({"seo": 1})
    llm = FakeLLM({"manager": [review(decision("seo", "retry", "too long", "Keep it short"))]})

    result = await run(workers, llm)

    assert calls["seo"] == 2
    assert calls["niche"] == calls["audience"] == calls["competitor"] == 1
    assert result["tasks"]["seo"].status == "done"
    assert result["tasks"]["seo"].attempts == 2
    assert result["tasks"]["seo"].instruction == "Keep it short"
    second = [p for p in packets if p["task"]["agent"] == "seo"][1]
    assert second["task"]["instruction"] == "Keep it short"  # the worker got the new instruction
    assert result["tasks"]["brand"].status == "done"  # the run went on to its dependents
    assert len(llm.prompts_for("manager")) == 1


async def test_skipping_a_research_task_lets_the_run_go_on() -> None:
    workers, calls, packets = failing({"competitor": ValueError("boom")})
    llm = FakeLLM({"manager": [review(decision("competitor", "skip", "no sites found"))]})

    result = await run(workers, llm)

    assert result["tasks"]["competitor"].status == "skipped"
    assert result["tasks"]["competitor"].error.startswith("Skipped: no sites found")
    assert "ValueError: boom" in result["tasks"]["competitor"].error
    assert result["tasks"]["brand"].status == "done"
    brand = next(p for p in packets if p["task"]["agent"] == "brand")
    assert brand["context"]["competitor"] == "(skipped: no result)"
    assert calls["competitor"] == 1


async def test_ask_user_stops_the_dependents_and_keeps_the_reason() -> None:
    workers, _, _ = failing({"seo": ValueError("boom")})
    llm = FakeLLM({"manager": [review(decision("seo", "ask_user", "The niche is unclear"))]})

    result = await run(workers, llm)

    assert result["tasks"]["seo"].status == "needs_user"
    assert "The niche is unclear" in result["tasks"]["seo"].error
    assert result["tasks"]["brand"].status == "pending"


async def test_one_manager_call_covers_every_failure_of_the_wave() -> None:
    workers, _, _ = failing({"seo": ValueError("a"), "niche": ValueError("b")})
    llm = FakeLLM(
        {"manager": [review(decision("seo", "ask_user"), decision("niche", "skip", "not needed"))]}
    )

    result = await run(workers, llm)

    assert len(llm.prompts_for("manager")) == 1
    prompt = "\n".join(str(m.content) for m in llm.prompts_for("manager")[0])
    assert '"task_id": "seo"' in prompt
    assert '"task_id": "niche"' in prompt
    assert '"task_id": "audience"' not in prompt  # only the failed tasks are listed
    assert result["tasks"]["niche"].status == "skipped"
    assert result["tasks"]["seo"].status == "needs_user"


async def test_no_manager_call_when_nothing_failed() -> None:
    llm = FakeLLM({"manager": [review()]})
    result = await run(fake_workers(), llm)
    assert llm.prompts_for("manager") == []
    assert {task.status for task in result["tasks"].values()} == {"done"}


async def test_the_manager_cannot_retry_forever() -> None:
    workers, calls, _ = failing({"seo": ValueError("boom")})
    llm = FakeLLM({"manager": [review(decision("seo", "retry", "again", "Try again"))]})

    result = await run(workers, llm)

    assert calls["seo"] == 1 + MAX_REWORKS  # the first run and two reworks
    assert len(llm.prompts_for("manager")) == 1 + MAX_REWORKS
    assert result["tasks"]["seo"].status == "needs_user"
    assert "retries are used up" in result["tasks"]["seo"].error


@pytest.mark.parametrize(
    "answer",
    [RuntimeError("provider down"), TimeoutError(), ConfigError("no key"), None, review()],
    ids=["error", "timeout", "config", "no answer", "no decisions"],
)
async def test_without_a_usable_manager_answer_the_user_decides(answer: Any) -> None:
    workers, calls, _ = failing({"seo": ValueError("boom")})

    result = await run(workers, FakeLLM({"manager": [answer]}))

    assert result["tasks"]["seo"].status == "needs_user"
    assert calls["seo"] == 1
    assert result["tasks"]["niche"].status == "done"


async def test_a_decision_for_an_unknown_or_finished_task_is_ignored() -> None:
    workers, _, _ = failing({"seo": ValueError("boom")})
    llm = FakeLLM({"manager": [review(decision("niche", "skip"), decision("nope", "retry"))]})

    result = await run(workers, llm)

    assert result["tasks"]["niche"].status == "done"
    assert result["tasks"]["seo"].status == "needs_user"


def test_a_setup_error_cannot_be_retried_or_skipped_into_a_loop() -> None:
    broken = task(status="failed", attempts=1, error="ConfigError: no key for seo")
    assert not can_retry(broken)
    outcome = enforce(broken, FailureDecision(task_id="seo", action="retry", reason="try"))
    assert outcome.action == "ask_user"
    assert "setup problem" in outcome.reason


def test_a_task_the_run_needs_cannot_be_skipped() -> None:
    brand = task("brand", status="failed", attempts=1, error="x")
    outcome = enforce(brand, FailureDecision(task_id="brand", action="skip", reason="meh"))
    assert outcome.action == "ask_user"
    assert "cannot go on without brand" in outcome.reason


def test_apply_decision_keeps_the_old_instruction_when_the_new_one_is_blank() -> None:
    failed = task(status="failed", attempts=1, error="x", instruction="Focus on eco hotels")
    retry = FailureDecision(task_id="seo", action="retry", reason="r", instruction="  ")
    assert apply_decision(failed, retry).instruction == "Focus on eco hotels"
    assert apply_decision(failed, retry).status == "needs_rework"


def test_the_manager_may_not_write_an_oversized_decision() -> None:
    with pytest.raises(ValueError, match="reason"):
        FailureReview.model_validate(review(decision("seo", "skip", "x" * 301)))
