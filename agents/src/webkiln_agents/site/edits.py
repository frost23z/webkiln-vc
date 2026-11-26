"""What a chat message can ask of the finished site, and the plain-code parts of doing it.

The Manager's `EditIntent` is a proposal. `resolve` checks it against the site and turns it into a
`Target` that points at real pages and sections, or into a short reply for the user when nothing
should change (a question, an unclear request, a section that does not exist). The structural
edits (add or remove a section, add a page) are pure functions on the wireframe and the copy, so
they are tested without a model and keep every section id stable: an id is never reused or
renumbered, because the saved copy and a later edit point at it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from webkiln_agents.site.copy import PageCopy, SectionCopy
from webkiln_agents.site.head import EditRefused
from webkiln_agents.site.vocabulary import (
    MAX_PAGES,
    MAX_SECTIONS_PER_PAGE,
    RULES,
    SectionType,
)
from webkiln_agents.workers.wireframe import (
    HOME,
    MAX_NAVIGATION,
    Page,
    PagePlan,
    Section,
    SectionPlan,
    Wireframe,
    WireframeDraft,
    structure_problems,
)

EditKind = Literal[
    "content_edit",
    "design_edit",
    "add_section",
    "remove_section",
    "add_page",
    "question",
    "clarify",
    "unsupported",
]
# Below this the Manager is guessing, so the user is asked instead.
MIN_CONFIDENCE = 0.5
MAX_MESSAGE = 1000  # characters in one chat message


class EditIntent(BaseModel):
    """The Manager's reading of one chat message."""

    intent: EditKind
    confidence: float = Field(ge=0, le=1, description="How sure you are, from 0 to 1.")
    instruction: str = Field(
        min_length=1,
        max_length=600,
        description="The request in full, so a writer who never saw the chat can follow it. Fold "
        "in what an earlier message said when this one refers to it ('the same', 'that').",
    )
    page: str | None = Field(default=None, description="A page slug from <site>, if one is meant.")
    section: str | None = Field(
        default=None, description="A section id from <site>, for a content edit or a removal."
    )
    section_type: SectionType | None = Field(
        default=None, description="The type of section to add, for add_section."
    )
    reply: str | None = Field(
        default=None,
        max_length=600,
        description="For question, clarify and unsupported: the answer or the one short question "
        "to ask. Leave it empty for every other intent.",
    )
    preferences: list[Annotated[str, StringConstraints(min_length=5, max_length=200)]] = Field(
        default_factory=list,
        max_length=3,
        description="Lasting preferences the user states for every site they build, each as one "
        "short sentence. Empty for a request about this site only.",
    )


@dataclass(frozen=True)
class Target:
    """An edit that can go ahead: what to do and to which page or section."""

    intent: EditKind
    instruction: str
    page: Page | None = None
    section: Section | None = None
    section_type: SectionType | None = None


def find_page(wireframe: Wireframe, reference: str | None) -> Page | None:
    """A page by its slug, or by its name in any case."""
    if not reference:
        return None
    wanted = reference.strip().casefold()
    for page in wireframe.pages:
        if wanted in (page.slug.casefold(), page.name.casefold()):
            return page
    return None


def find_section(wireframe: Wireframe, section_id: str | None) -> tuple[Page, Section] | None:
    if not section_id:
        return None
    for page in wireframe.pages:
        for section in page.sections:
            if section.id == section_id.strip():
                return page, section
    return None


def listing(wireframe: Wireframe, copies: dict[str, PageCopy], *, words: int = 6) -> list[dict]:
    """The site as compact rows for the Manager: page, section id, type and a few opening words."""
    rows = []
    for page in wireframe.pages:
        words_of = (
            {s.id: s.heading for s in copies[page.slug].sections} if page.slug in copies else {}
        )
        rows.append(
            {
                "page": page.slug,
                "name": page.name,
                "sections": [
                    {
                        "id": s.id,
                        "type": s.type,
                        "starts": " ".join(words_of.get(s.id, s.title).split()[:words]),
                    }
                    for s in page.sections
                ],
            }
        )
    return rows


def _where(wireframe: Wireframe) -> str:
    return "; ".join(
        f"{page.slug}: {', '.join(s.id for s in page.sections)}" for page in wireframe.pages
    )


def resolve(intent: EditIntent, wireframe: Wireframe) -> Target | str:
    """A `Target` when the edit can go ahead, or the text to answer the user with when not."""
    kind = intent.intent
    if kind in ("question", "clarify", "unsupported"):
        fallback = {
            "question": "I could not find that in the site's details.",
            "clarify": "Which page or section do you mean?",
            "unsupported": "I cannot do that yet.",
        }[kind]
        return intent.reply or fallback
    if intent.confidence < MIN_CONFIDENCE:
        return intent.reply or (
            "I am not sure what to change. Tell me the page and the section, and what you want."
        )
    if kind == "design_edit":
        return Target(kind, intent.instruction)
    if kind in ("content_edit", "remove_section"):
        found = find_section(wireframe, intent.section)
        if found is None:
            page = find_page(wireframe, intent.page)
            if page is not None and len(page.sections) == 1:
                found = (page, page.sections[0])
        if found is None:
            return (
                f"I could not tell which section you mean. Sections by page: {_where(wireframe)}."
            )
        return Target(kind, intent.instruction, *found)
    if kind == "add_section":
        page = find_page(wireframe, intent.page)
        if page is None:
            names = ", ".join(p.slug for p in wireframe.pages)
            return f"Which page should the new section go on? The pages are: {names}."
        if intent.section_type is None or intent.section_type == "hero":
            return (
                "Which kind of section do you want? One of: "
                + ", ".join(t for t in RULES if t != "hero")
                + "."
            )
        if len(page.sections) >= MAX_SECTIONS_PER_PAGE:
            return (
                f"The {page.name} page already has {MAX_SECTIONS_PER_PAGE} sections, "
                "the most one page holds."
            )
        return Target(kind, intent.instruction, page, None, intent.section_type)
    # add_page
    if len(wireframe.pages) >= MAX_PAGES:
        return f"The site already has {MAX_PAGES} pages, the most it can have. Remove one first."
    if len(wireframe.navigation) >= MAX_NAVIGATION:
        return "The menu is full, so a new page could not be reached from it."
    return Target(kind, intent.instruction)


def to_draft(wireframe: Wireframe) -> WireframeDraft:
    """The wireframe as a draft without ids, so the rules for drafts can check it again."""
    return WireframeDraft(
        summary=wireframe.summary,
        pages=[
            PagePlan(
                **page.model_dump(exclude={"sections"}),
                sections=[SectionPlan(**s.model_dump(exclude={"id"})) for s in page.sections],
            )
            for page in wireframe.pages
        ],
        navigation=wireframe.navigation,
        functional_requirements=wireframe.functional_requirements,
    )


def wireframe_problems(wireframe: Wireframe) -> list[str]:
    return structure_problems(to_draft(wireframe))


def next_section_id(page: Page, kind: str) -> str:
    """`<page>-<type>`, with the lowest number that no section of the page has used."""
    base = f"{page.slug}-{kind.replace('_', '-')}"
    taken = {s.id for s in page.sections}
    if base not in taken:
        return base
    number = 2
    while f"{base}-{number}" in taken:
        number += 1
    return f"{base}-{number}"


def _insert_index(page: Page) -> int:
    """New sections go before a closing call to action, and otherwise at the end."""
    if page.sections and page.sections[-1].type == "cta_band":
        return len(page.sections) - 1
    return len(page.sections)


def _replace_page(wireframe: Wireframe, page: Page) -> Wireframe:
    pages = [page if p.slug == page.slug else p for p in wireframe.pages]
    return wireframe.model_copy(update={"pages": pages})


def with_section(
    wireframe: Wireframe, copy: PageCopy, page: Page, section: Section, words: SectionCopy
) -> tuple[Wireframe, PageCopy]:
    """The wireframe and the page copy with one new section in the same place in both."""
    index = _insert_index(page)
    sections = [*page.sections[:index], section, *page.sections[index:]]
    texts = [*copy.sections[:index], words, *copy.sections[index:]]
    new_wireframe = _replace_page(wireframe, page.model_copy(update={"sections": sections}))
    return new_wireframe, PageCopy(sections=texts)


def without_section(
    wireframe: Wireframe, copy: PageCopy, page: Page, section_id: str
) -> tuple[Wireframe, PageCopy]:
    """The wireframe and the page copy without one section. A page keeps at least one."""
    if len(page.sections) == 1:
        msg = f"{page.name} has only that section, and a page needs at least one."
        raise EditRefused(msg)
    sections = [s for s in page.sections if s.id != section_id]
    texts = [s for s in copy.sections if s.id != section_id]
    new_wireframe = _replace_page(wireframe, page.model_copy(update={"sections": sections}))
    return new_wireframe, PageCopy(sections=texts)


def with_replaced_copy(copy: PageCopy, words: SectionCopy) -> PageCopy:
    return PageCopy(sections=[words if s.id == words.id else s for s in copy.sections])


def with_page(wireframe: Wireframe, page: Page, requirements: Sequence = ()) -> Wireframe:
    """The wireframe with a new page, in the menu before the contact page when there is one."""
    navigation = list(wireframe.navigation)
    at = navigation.index("contact") if "contact" in navigation else len(navigation)
    navigation.insert(at, page.slug)
    return wireframe.model_copy(
        update={
            "pages": [*wireframe.pages, page],
            "navigation": navigation,
            "functional_requirements": [*wireframe.functional_requirements, *requirements],
        }
    )


def page_from_plan(plan: PagePlan) -> Page:
    """A planned page with section ids, by the same rule that numbers a generated site."""
    seen: dict[str, int] = {}
    sections = []
    for section in plan.sections:
        seen[section.type] = seen.get(section.type, 0) + 1
        suffix = "" if seen[section.type] == 1 else f"-{seen[section.type]}"
        section_id = f"{plan.slug}-{section.type.replace('_', '-')}{suffix}"
        sections.append(Section(id=section_id, **section.model_dump()))
    return Page(**plan.model_dump(exclude={"sections"}), sections=sections)


def new_page_problems(wireframe: Wireframe, plan: PagePlan, requirements: Sequence) -> list[str]:
    """What is wrong with adding `plan` to the site, in words the model can act on."""
    if any(p.slug == plan.slug for p in wireframe.pages):
        return [f"the slug {plan.slug!r} is already a page: choose another"]
    if plan.slug == HOME or plan.parent == plan.slug:
        return [f"the slug {plan.slug!r} cannot be used for a new page"]
    page = page_from_plan(plan)
    return wireframe_problems(with_page(wireframe, page, requirements))
