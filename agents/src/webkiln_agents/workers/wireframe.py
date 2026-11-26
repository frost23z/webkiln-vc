"""The wireframe worker and the wireframe it writes.

The wireframe is the plan of the site: pages, the sections on each page and what every section
must say. It is data, not prose. The model writes a `WireframeDraft` without section ids, because
ids must stay the same across versions and a model cannot promise that. Code gives each section
its id (`finalize`), and the saved `Wireframe` is what everything after it reads.

The checks tie the draft to earlier work. They are plain code, so a wireframe that drops an SEO
landing page or leaves an audience need unanswered is sent back to the model with the reasons.
"""

import json
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from webkiln_agents.manager.context import Context
from webkiln_agents.models import ConfigError
from webkiln_agents.site.vocabulary import (
    MAX_PAGES,
    MAX_SECTIONS_PER_PAGE,
    describe_vocabulary,
)
from webkiln_agents.site.vocabulary import SectionType as SectionType
from webkiln_agents.workers.base import WorkerInput, WorkerResult
from webkiln_agents.workers.outputs import Report
from webkiln_agents.workers.research import ResearchSpec, block, research_messages
from webkiln_agents.workers.structured import generate

HOME = "index"  # the slug of the home page, whose file is `index.html`
MAX_NAVIGATION = 6
MAX_DEPTH = 3  # levels of the page tree, the home page being the first

Slug = Annotated[
    str, StringConstraints(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$", min_length=1, max_length=40)
]


class Cta(BaseModel):
    label: str = Field(min_length=2, max_length=30)
    target: Slug = Field(description="The slug of the page the button opens.")


class SectionPlan(BaseModel):
    type: SectionType
    title: str = Field(min_length=3, max_length=80, description="The working heading.")
    content_brief: str = Field(
        min_length=20,
        max_length=400,
        description="What the section must say, written in the brand's tone.",
    )
    audience_needs: list[str] = Field(
        min_length=1,
        max_length=3,
        description="Needs from the audience report that this section answers, copied word for "
        "word from the report's list of needs.",
    )
    cta_primary: Cta | None = None
    cta_secondary: Cta | None = None

    model_config = ConfigDict(extra="forbid")


class PageSeo(BaseModel):
    target_keyword: str = Field(min_length=3, max_length=60)
    title: str = Field(min_length=10, max_length=60, description="The page's title tag.")
    meta_description: str = Field(min_length=50, max_length=160)

    model_config = ConfigDict(extra="forbid")


class PagePlan(BaseModel):
    slug: Slug = Field(description=f"Lowercase words with hyphens. The home page is '{HOME}'.")
    name: str = Field(min_length=2, max_length=40, description="The name used in the navigation.")
    purpose: str = Field(min_length=10, max_length=200)
    parent: Slug | None = Field(default=None, description="The slug of the page above this one.")
    seo: PageSeo
    sections: list[SectionPlan] = Field(min_length=1, max_length=MAX_SECTIONS_PER_PAGE)

    model_config = ConfigDict(extra="forbid")


FunctionKind = Literal["form", "booking_link", "newsletter", "map", "search"]


class FunctionalRequirement(BaseModel):
    kind: FunctionKind
    page: Slug
    description: str = Field(min_length=10, max_length=200)

    model_config = ConfigDict(extra="forbid")


class WireframeDraft(Report):
    """What the model writes."""

    pages: list[PagePlan] = Field(min_length=1, max_length=MAX_PAGES)
    navigation: list[Slug] = Field(
        min_length=1, max_length=MAX_NAVIGATION, description="Page slugs in menu order."
    )
    functional_requirements: list[FunctionalRequirement] = Field(default_factory=list, max_length=6)

    model_config = ConfigDict(extra="forbid")


class Section(SectionPlan):
    id: str  # `<page>-<type>`, with a number when a page has the type twice


class Page(BaseModel):
    slug: str
    name: str
    purpose: str
    parent: str | None
    seo: PageSeo
    sections: list[Section]


class Wireframe(Report):
    """What is saved and what generation reads."""

    pages: list[Page]
    navigation: list[str]
    functional_requirements: list[FunctionalRequirement]


def finalize(draft: WireframeDraft) -> Wireframe:
    """Give every section its id. The same draft always gets the same ids."""
    pages = []
    for plan in draft.pages:
        seen: dict[str, int] = {}
        sections = []
        for section in plan.sections:
            seen[section.type] = seen.get(section.type, 0) + 1
            suffix = "" if seen[section.type] == 1 else f"-{seen[section.type]}"
            section_id = f"{plan.slug}-{section.type.replace('_', '-')}{suffix}"
            sections.append(Section(id=section_id, **section.model_dump()))
        pages.append(Page(**plan.model_dump(exclude={"sections"}), sections=sections))
    return Wireframe(
        summary=draft.summary,
        pages=pages,
        navigation=draft.navigation,
        functional_requirements=draft.functional_requirements,
    )


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def structure_problems(draft: WireframeDraft) -> list[str]:
    """Rules about the draft on its own: slugs, parents, navigation, buttons."""
    problems: list[str] = []
    slugs = [page.slug for page in draft.pages]
    known = set(slugs)
    problems.extend(
        f"the slug {slug!r} is used by more than one page"
        for slug in sorted({s for s in slugs if slugs.count(s) > 1})
    )
    if HOME not in known:
        problems.append(f"there is no home page: one page must have the slug {HOME!r}")
    parents = {page.slug: page.parent for page in draft.pages}
    problems.extend(
        f"page {page.slug!r} has the unknown parent {page.parent!r}"
        for page in draft.pages
        if page.parent is not None and page.parent not in known
    )
    if parents.get(HOME) is not None:
        problems.append(f"the home page {HOME!r} cannot have a parent")
    for slug in known:
        depth, current = 1, slug
        while (up := parents.get(current)) is not None and depth <= MAX_DEPTH + 1:
            depth, current = depth + 1, up
        if depth > MAX_DEPTH:
            problems.append(f"page {slug!r} is nested too deep or in a loop (limit {MAX_DEPTH})")
    problems.extend(
        f"the navigation lists {slug!r}, which is not a page"
        for slug in draft.navigation
        if slug not in known
    )
    if len(set(draft.navigation)) != len(draft.navigation):
        problems.append("the navigation lists a page twice")
    if HOME in known and HOME not in draft.navigation:
        problems.append(f"the navigation must include the home page {HOME!r}")
    for page in draft.pages:
        for index, section in enumerate(page.sections):
            where = f"page {page.slug!r}, section {index + 1} ({section.type})"
            if section.type == "hero" and index != 0:
                problems.append(f"{where}: a hero can only be the first section")
            problems.extend(
                f"{where}: the button {cta.label!r} opens {cta.target!r}"
                for cta in (section.cta_primary, section.cta_secondary)
                if cta is not None and cta.target not in known
            )
    for need in draft.functional_requirements:
        if need.page not in known:
            problems.append(f"the {need.kind} requirement is for the unknown page {need.page!r}")
            continue
        page = next(p for p in draft.pages if p.slug == need.page)
        if need.kind == "form" and not any(s.type == "contact" for s in page.sections):
            problems.append(f"the form on {need.page!r} needs a contact section on that page")
    return problems


def _landing_page_problems(draft: WireframeDraft, seo: dict[str, Any]) -> list[str]:
    """Every approved landing page is a page, or a section when the page limit forces it."""
    places = [_words(f"{page.name} {page.slug}") for page in draft.pages] + [
        _words(section.title) for page in draft.pages for section in page.sections
    ]
    problems = []
    for landing in seo.get("landing_pages", []):
        wanted = _words(landing["page"])
        if wanted and not any(wanted <= place for place in places):
            problems.append(
                f"the SEO landing page {landing['page']!r} is not in the site: make it a page "
                f"or, if the {MAX_PAGES}-page limit is reached, a section whose title names it"
            )
    return problems


def _need_problems(draft: WireframeDraft, audience: dict[str, Any]) -> list[str]:
    claimed = {
        need.strip().casefold()
        for page in draft.pages
        for section in page.sections
        for need in section.audience_needs
    }
    return [
        f"the audience need {need!r} is not answered by any section: list it, word for word, "
        "in the audience_needs of a section that answers it"
        for need in audience.get("needs", [])
        if need.strip().casefold() not in claimed
    ]


def _tone_problems(draft: WireframeDraft, brand: dict[str, Any]) -> list[str]:
    traits = brand.get("personality_and_tone", {}).get("traits", [])
    if not traits:
        return []
    briefs = " ".join(s.content_brief for page in draft.pages for s in page.sections).casefold()
    missing = [trait for trait in traits if trait.casefold() not in briefs]
    needed = (len(traits) + 1) // 2
    if len(traits) - len(missing) >= needed:
        return []
    return [
        f"the content briefs should use the brand's tone words, at least {needed} of "
        f"{', '.join(traits)}, and these are missing: {', '.join(missing)}"
    ]


def plan_problems(draft: WireframeDraft, upstream: dict[str, dict[str, Any]]) -> list[str]:
    """Every problem of a draft, for the model to fix. A skipped agent has no output, and its
    check is left out."""
    problems = structure_problems(draft)
    if seo := upstream.get("seo"):
        problems += _landing_page_problems(draft, seo)
    if audience := upstream.get("audience"):
        problems += _need_problems(draft, audience)
    if brand := upstream.get("brand"):
        problems += _tone_problems(draft, brand)
    return problems


def outline(wireframe: Wireframe) -> list[str]:
    """The sitemap and its sections as indented lines, for the chat."""
    by_parent: dict[str | None, list[Page]] = {}
    for page in wireframe.pages:
        by_parent.setdefault(page.parent, []).append(page)
    lines: list[str] = []

    def walk(parent: str | None, depth: int) -> None:
        for page in by_parent.get(parent, []):
            lines.append("  " * depth + f"{page.name} ({page.slug})")
            lines.extend("  " * (depth + 1) + f"{s.type}: {s.title}" for s in page.sections)
            walk(page.slug, depth + 1)

    walk(None, 0)
    return lines


WIREFRAME_SPEC = ResearchSpec(
    "wireframe",
    "UX planner",
    "Plan the site as a wireframe. Give 1 to "
    f"{MAX_PAGES} pages, one of them the home page with the slug '{HOME}', and for each page its "
    "slug, name, purpose, parent, SEO (a target keyword, a title of at most 60 characters and a "
    f"meta description of 50 to 160 characters) and 1 to {MAX_SECTIONS_PER_PAGE} sections. Every "
    "section has a type from the list below, a working title, a content brief that says what the "
    "section must say in the brand's tone of voice (use the tone words of the guideline), the "
    "audience needs it answers (copied word for word from the audience report) and at most two "
    "buttons, each with a label and the slug of the page it opens. Also give the navigation (page "
    "slugs in menu order) and the functional requirements the brief implies (a form, a booking "
    "link, a newsletter, a map or search), each tied to a page. The SEO landing pages must each "
    "be a page, or a section titled with their name when the page limit leaves no room. A hero "
    "may only be the first section of a page.\n\nSection types:\n" + describe_vocabulary(),
    WireframeDraft,
)

# What the model reads of the earlier work. The saved reports are long, so only the parts a
# wireframe needs are copied, and the model is told to use them word for word where it must.
_ABOUT_BRAND = ("positioning", "personality_and_tone", "messaging", "ui_ux_direction")


def upstream_block(upstream: dict[str, dict[str, Any]]) -> str:
    parts: list[str] = []
    if brand := upstream.get("brand"):
        parts.append(block("approved_brand", json.dumps({k: brand.get(k) for k in _ABOUT_BRAND})))
    if audience := upstream.get("audience"):
        needs = {
            "needs": audience.get("needs"),
            "personas": audience.get("personas"),
            "primary_audiences": audience.get("primary_audiences"),
        }
        parts.append(block("audience", json.dumps(needs)))
    if seo := upstream.get("seo"):
        keywords = [k["term"] for k in seo.get("primary_keywords", [])]
        landing = seo.get("landing_pages")
        parts.append(block("seo", json.dumps({"landing_pages": landing, "keywords": keywords})))
    return "\n\n".join(parts)


async def wireframe_worker(packet: WorkerInput, context: Context) -> WorkerResult:
    if context.llm is None:
        msg = "No model provider is configured for the wireframe agent"
        raise ConfigError(msg)
    upstream = packet["upstream"]
    call = context.llm.structured("wireframe", WireframeDraft)
    messages = research_messages(WIREFRAME_SPEC, packet, upstream_block(upstream))
    draft = await generate(call, messages, extra_check=lambda d: plan_problems(d, upstream))
    wireframe = finalize(draft)
    return WorkerResult(summary=wireframe.summary, data=wireframe.model_dump(mode="json"))
