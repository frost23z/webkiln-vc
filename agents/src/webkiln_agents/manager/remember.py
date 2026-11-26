"""What the run writes to long-term memory, and when.

Capture is by code, with no model call, except for lasting preferences, which the Manager lists
inside the `ReworkOrder` it already produces. Three moments write:

- Approval (`remember_approved`): the approved brand, the research the brand was built on, and
  the changes the user made by hand.
- Feedback (`manager_feedback`): the user's words, and any lasting preference the Manager found.
- The site (`remember_site`, then each edit and each restore): the architecture, as one record in
  the `wireframe` slot, and the brand again after a design edit. A slot holds one current record,
  so memory always describes the site as it is now, never a colour or a page that an edit removed.

Memory is best effort. A store or embedding failure is logged and the run goes on.
"""

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from webkiln_agents.manager.context import Context
from webkiln_agents.manager.state import State
from webkiln_agents.memory.records import MAX_TEXT, Kind, Source, normalise
from webkiln_agents.memory.service import Memory
from webkiln_agents.workers.brand import BrandGuideline
from webkiln_agents.workers.wireframe import Wireframe

logger = logging.getLogger(__name__)

BRAND = "brand"
ARCHITECTURE = "wireframe"  # the slot of the site's architecture
# The edits that change the sitemap or a page's sections, and the one that changes the brand.
STRUCTURE_CAUSES = frozenset({"add_section", "remove_section", "add_page"})
DESIGN_CAUSE = "design_edit"
RESTORE_CAUSE = "restore"
# The research whose summary is kept, and the kind each one is filed under.
RESEARCH_KINDS: dict[str, Kind] = {
    "niche": "business_fact",
    "audience": "business_fact",
    "seo": "seo",
}
EDIT_LABELS = {
    "primary": "the primary colour",
    "secondary": "the secondary colour",
    "accent": "the accent colour",
    "background": "the background colour",
    "heading_font": "the heading font",
    "body_font": "the body font",
    "positioning": "the positioning",
    "tagline": "the tagline",
}


@dataclass(frozen=True)
class Candidate:
    """One sentence that may become a memory."""

    text: str
    kind: Kind
    source: Source
    slot: str | None = None


def clip(text: str, limit: int = MAX_TEXT) -> str:
    text = normalise(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def brand_sentence(brand_name: str, guideline: BrandGuideline) -> str:
    palette, fonts = guideline.palette, guideline.typography
    traits = ", ".join(guideline.personality_and_tone.traits)
    return clip(
        f'Approved brand for {brand_name}: tagline "{guideline.messaging.tagline}", '
        f"{traits} tone, primary {palette.primary} on {palette.background}, "
        f"{fonts.heading_font} headings with {fonts.body_font} text. "
        f"Positioning: {guideline.positioning}"
    )


def wireframe_sentence(brand_name: str, wireframe: Wireframe) -> str:
    pages = "; ".join(
        f"{page.name} ({', '.join(section.type for section in page.sections)})"
        for page in wireframe.pages
    )
    return clip(
        f"Site architecture for {brand_name}: {pages}. Menu: {', '.join(wireframe.navigation)}"
    )


def site_candidates(
    brand_name: str,
    cause: str,
    brand: Mapping[str, Any],
    wireframe: Mapping[str, Any],
) -> list[Candidate]:
    """What memory must say about the site after a version is saved. A design edit changes the
    brand, a structure edit changes the architecture, and a restore can change both. A content
    edit changes neither."""
    found: list[Candidate] = []
    if cause in {DESIGN_CAUSE, RESTORE_CAUSE}:
        sentence = brand_sentence(brand_name, BrandGuideline.model_validate(brand))
        found.append(Candidate(sentence, "brand", "user_said", BRAND))
    if cause in STRUCTURE_CAUSES | {RESTORE_CAUSE}:
        sentence = wireframe_sentence(brand_name, Wireframe.model_validate(wireframe))
        found.append(Candidate(sentence, "architecture", "user_said", ARCHITECTURE))
    return found


def edit_sentence(edits: dict[str, str]) -> str:
    changed = ", ".join(f"{EDIT_LABELS[name]} to {value}" for name, value in edits.items())
    return clip(f"The user set {changed} by hand in the brand guideline")


def approval_candidates(state: State) -> list[Candidate]:
    """Everything worth keeping once the user approves the brand."""
    found: list[Candidate] = []
    brief = state.initial_input
    if raw := state.outputs.get(BRAND):
        guideline = BrandGuideline.model_validate(raw)
        found.append(
            Candidate(brand_sentence(brief.brand_name, guideline), "brand", "approved", BRAND)
        )
    for task_id, kind in RESEARCH_KINDS.items():
        task = state.tasks.get(task_id)
        if task is not None and task.status == "done" and (summary := state.summaries.get(task_id)):
            text = (
                f"{task_id.capitalize()} research for {brief.brand_name} ({brief.niche}): {summary}"
            )
            found.append(Candidate(clip(text), kind, "approved", task_id))
    found.extend(
        Candidate(edit_sentence(review["edits"]), "decision", "user_said")
        for review in state.brand_reviews
        if review.get("edits")
    )
    return found


def feedback_candidates(feedback: str, preferences: Iterable[str]) -> list[Candidate]:
    found = [
        Candidate(
            clip(f"Asked for changes to the brand guideline: {feedback}"), "feedback", "user_said"
        )
    ]
    found.extend(Candidate(clip(text), "preference", "user_said") for text in preferences)
    return found


async def save(memory: Memory, candidates: Sequence[Candidate]) -> list[Kind]:
    """Write each candidate. One that fails is logged and skipped. Returns the kinds that were
    saved as something new (a duplicate adds nothing)."""
    saved: list[Kind] = []
    for item in candidates:
        try:
            outcome = await memory.remember(item.text, item.kind, item.source, slot=item.slot)
        except Exception:
            logger.exception("could not save a %s memory", item.kind)
            continue
        if outcome != "duplicate":
            saved.append(item.kind)
    return saved


def announce(stage: str, kinds: Sequence[Kind]) -> None:
    """Tell the stream what was kept. Nothing is sent when nothing new was saved."""
    if kinds:
        get_stream_writer()({"event": "memory_saved", "stage": stage, "kinds": sorted(kinds)})


async def remember_approved(state: State, runtime: Runtime[Context]) -> None:
    """The step between approval and `dispatch`: keep what the user just signed off."""
    memory = runtime.context.memory if runtime.context else None
    if memory is None:
        return
    announce("approval", await save(memory, approval_candidates(state)))


async def remember_site(state: State, runtime: Runtime[Context]) -> None:
    """After the site is built: keep its architecture. An edit keeps it up to date after that,
    so a run that has been edited is left alone. Best effort, like all memory."""
    memory = runtime.context.memory if runtime.context else None
    raw = state.outputs.get("wireframe")
    if memory is None or raw is None or state.site_changes:
        return
    try:
        wireframe = Wireframe.model_validate(raw)
    except ValueError:
        logger.exception("the wireframe could not be read for memory")
        return
    sentence = wireframe_sentence(state.initial_input.brand_name, wireframe)
    candidate = Candidate(sentence, "architecture", "agent_inferred", ARCHITECTURE)
    announce("build", await save(memory, [candidate]))
