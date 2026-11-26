"""The words of a page, as structured data.

The copy worker writes one `PageCopy` per page. It is data, not HTML: the templates draw it, so
the model never writes markup and cannot break a page. `copy_problems` ties the copy to the
wireframe, so a section that is missing, out of order or does not hold what its type needs is sent
back to the model with the reasons.
"""

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from webkiln_agents.site.vocabulary import rule_for

MAX_SECTION_WORDS = 220  # all the text of one section
MARKUP = re.compile(r"</?[A-Za-z!][^>]*>")
WORDS = re.compile(r"[a-z0-9]+")


class Item(BaseModel):
    """A card, a question, a plan, a quote or a detail, depending on the section type."""

    title: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=300)
    meta: str | None = Field(default=None, max_length=60)

    model_config = ConfigDict(extra="forbid")


class SectionCopy(BaseModel):
    id: str = Field(description="The id of the section in the wireframe, copied exactly.")
    heading: str = Field(min_length=3, max_length=80)
    subheading: str | None = Field(default=None, max_length=200)
    body: str | None = Field(
        default=None, max_length=1200, description="Paragraphs, separated by a blank line."
    )
    items: list[Item] = Field(default_factory=list, max_length=8)

    model_config = ConfigDict(extra="forbid")

    @property
    def paragraphs(self) -> list[str]:
        """The body split into paragraphs, without empty ones."""
        return [p.strip() for p in re.split(r"\n\s*\n", self.body or "") if p.strip()]

    def words(self) -> int:
        parts = [self.heading, self.subheading or "", self.body or ""]
        for item in self.items:
            parts += [item.title, item.text, item.meta or ""]
        return len(" ".join(parts).split())


class PageCopy(BaseModel):
    """The copy of one page: its sections, in the order of the wireframe."""

    sections: list[SectionCopy] = Field(min_length=1, max_length=8)

    model_config = ConfigDict(extra="forbid")


def _texts(section: SectionCopy) -> list[str]:
    parts = [section.heading, section.subheading or "", section.body or ""]
    for item in section.items:
        parts += [item.title, item.text, item.meta or ""]
    return parts


def copy_problems(
    section_ids_and_types: Sequence[tuple[str, str]],
    copy: PageCopy,
    keyword: str | None = None,
    elsewhere: str = "",
) -> list[str]:
    """What is wrong with a page's copy, given the page's `(section id, type)` pairs.

    `keyword` is the page's target keyword. The site check looks for its words on the whole page,
    so `elsewhere` is the text the page shows besides the copy (its title and meta description),
    and the copy only has to supply the words that text lacks. Asking for more made models repeat
    the keyword, and the brand name inside it, in several headings.
    """
    expected = [section_id for section_id, _ in section_ids_and_types]
    got = [section.id for section in copy.sections]
    if got != expected:
        return [f"the sections must be exactly {expected} in that order, but they are {got}"]
    problems: list[str] = []
    for (section_id, kind), section in zip(section_ids_and_types, copy.sections, strict=True):
        rule = rule_for(kind)
        count = len(section.items)
        if count < rule.min_items or count > rule.max_items:
            problems.append(
                f"{section_id} ({kind}) needs {rule.min_items} to {rule.max_items} items, "
                f"and has {count}"
            )
        if rule.body and not section.paragraphs:
            problems.append(f"{section_id} ({kind}) needs a body of one or more paragraphs")
        if rule.meta:
            problems.extend(
                f"{section_id}: the item {item.title!r} needs its `meta` line"
                for item in section.items
                if not item.meta
            )
        if section.words() > MAX_SECTION_WORDS:
            problems.append(
                f"{section_id} has {section.words()} words, and the limit is {MAX_SECTION_WORDS}"
            )
        if any(MARKUP.search(text) for text in _texts(section)):
            problems.append(f"{section_id} has markup in it: write plain text, no HTML tags")
    if keyword:
        wanted = set(WORDS.findall(keyword.casefold()))
        written = set(
            WORDS.findall(
                " ".join([elsewhere, *(t for s in copy.sections for t in _texts(s))]).casefold()
            )
        )
        if wanted and not wanted <= written:
            problems.append(f"the target keyword {keyword!r} is not used in the copy of the page")
    return problems
