"""The brand approval gate: the run stops here until the user decides.

`dispatch` sends the run to `brand_review` once the brand guideline is done and not yet approved.
The node pauses with `interrupt`, and the payload holds the whole guideline, so the card can show
and edit every section. The answer is one of three things:

- Approve: the wireframe and generation are unblocked.
- Revise with direct edits only: code patches the guideline, no model is called, and the user
  reviews the new version.
- Revise with feedback: `manager_feedback` asks the Manager which tasks must run again.

Every decision is added to `brand_reviews`. The graph has no database, so the API copies those
rows into `brand_versions`. A malformed answer, or a revision past the cap, asks again with the
reason in the payload, so the run keeps waiting.
"""

from typing import Any, Literal

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from webkiln_agents.manager.context import Context
from webkiln_agents.manager.feedback import BRAND, ask_order, plan_rework
from webkiln_agents.manager.remember import announce, feedback_candidates, save
from webkiln_agents.manager.review import MAX_REVISIONS, ReviewDecision
from webkiln_agents.manager.state import State
from webkiln_agents.memory.prompt import memory_block
from webkiln_agents.workers.brand import BrandGuideline, apply_edits


def awaiting_brand_review(state: State) -> bool:
    """True when the brand is finished and the user has not approved it yet."""
    brand = state.tasks.get(BRAND)
    return brand is not None and brand.status == "done" and not state.brand_approved


def revisions_used(state: State) -> int:
    return sum(1 for entry in state.brand_reviews if entry["decision"] != "approve")


def review_payload(state: State, error: str | None) -> dict[str, Any]:
    """What the user sees. It is built from the state, so it is the same after a restart."""
    return {
        "type": "brand_review",
        "version": len(state.brand_reviews) + 1,
        "revisions_left": MAX_REVISIONS - revisions_used(state),
        "guideline": state.outputs[BRAND],
        "error": error,
    }


def _check(state: State, answer: Any) -> tuple[ReviewDecision, BrandGuideline | None]:
    """Validate an answer. Raises `ValueError` (a `ValidationError` too) with the reason."""
    decision = ReviewDecision.model_validate(answer)
    if decision.action == "approve":
        return decision, None
    if revisions_used(state) >= MAX_REVISIONS:
        msg = f"the limit of {MAX_REVISIONS} rounds of changes is used up, so only approve is left"
        raise ValueError(msg)
    edited = None
    if decision.edits and not decision.edits.is_empty():
        edited = apply_edits(BrandGuideline.model_validate(state.outputs[BRAND]), decision.edits)
    return decision, edited


def _reason(error: ValueError) -> str:
    if isinstance(error, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
    return str(error)[:300]


def brand_review(
    state: State,
) -> Command[Literal["remember_approved", "manager_feedback", "brand_review"]]:
    """Pause for the user's decision on the brand guideline.

    A node that calls `interrupt` runs again from the top on resume, and a second `interrupt` in
    the same node takes the next resume value. The loop relies on that to ask again.
    """
    error: str | None = None
    while True:
        answer = interrupt(review_payload(state, error))
        try:
            decision, edited = _check(state, answer)
        except ValueError as problem:
            error = _reason(problem)
            continue
        break

    entry = {
        "version": len(state.brand_reviews) + 1,
        "guideline": state.outputs[BRAND],
        "decision": decision.action if decision.action == "approve" else "revise",
        "feedback": decision.feedback or None,
        "edits": decision.edits.model_dump(exclude_none=True) if decision.edits else None,
    }
    update: dict[str, Any] = {"brand_reviews": [entry]}
    get_stream_writer()({"event": "review_decision", "decision": entry["decision"]})
    if decision.action == "approve":
        update["brand_approved"] = True
        return Command(update=update, goto="remember_approved")
    if edited is not None:
        update["outputs"] = {BRAND: edited.model_dump()}
        update["summaries"] = {BRAND: edited.summary}
    # Edits alone are done in code. Feedback goes on to the Manager, who sees the edited guideline.
    goto = "manager_feedback" if decision.feedback else "brand_review"
    if goto == "manager_feedback":
        update["pending_feedback"] = decision.feedback
    return Command(update=update, goto=goto)


async def manager_feedback(state: State, runtime: Runtime[Context]) -> Command[Literal["dispatch"]]:
    """One Manager call decides which tasks run again. Code applies only what the rules allow."""
    feedback = state.pending_feedback or ""
    context = runtime.context
    llm = context.llm if context else None
    recalled = await context.memory.recall(feedback) if context and context.memory else []
    order = await ask_order(
        llm,
        state.initial_input,
        feedback,
        state.outputs.get(BRAND),
        state.tasks,
        memory_block(recalled),
    )
    rework = plan_rework(state.tasks, order, feedback)
    if context and context.memory:
        preferences = order.preferences if order else []
        announce("feedback", await save(context.memory, feedback_candidates(feedback, preferences)))
    get_stream_writer()(
        {
            "event": "rework_planned",
            "tasks": sorted(rework.tasks),
            "rationale": rework.rationale,
            "refused": rework.refused,
        }
    )
    return Command(update={"tasks": rework.tasks, "pending_feedback": None}, goto="dispatch")
