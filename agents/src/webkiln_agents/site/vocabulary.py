"""The fixed vocabulary of section types.

The wireframe picks types from this list, the copy worker writes what each type needs, and the
templates render it. One table drives all three, so a section type cannot be planned that no
template draws, and an edit such as "change the pricing section" has a type to point at.
"""

from dataclasses import dataclass
from typing import Literal

SectionType = Literal[
    "hero", "about", "features", "pricing", "testimonials", "faq", "gallery", "contact",
    "cta_band", "text",
]  # fmt: skip


@dataclass(frozen=True)
class SectionRule:
    """What a section of one type holds. `items` are the cards, questions or rows inside it."""

    description: str
    min_items: int = 0
    max_items: int = 0
    body: bool = False  # needs paragraphs of text
    meta: bool = False  # every item needs a short extra line (a price, a name)
    images: int = 0  # image slots the section draws


RULES: dict[SectionType, SectionRule] = {
    "hero": SectionRule(
        "The opening banner: a heading, one line under it, one image.", images=1
    ),
    "about": SectionRule("A short story about the business, with one image.", body=True, images=1),
    "features": SectionRule(
        "Three to six cards: what is offered or why choose it. An item is a card title and a "
        "sentence.",
        min_items=3,
        max_items=6,
    ),
    "pricing": SectionRule(
        "Two to four plans or rates. An item is a plan name, what it includes, and its price in "
        "`meta`.",
        min_items=2,
        max_items=4,
        meta=True,
    ),
    "testimonials": SectionRule(
        "Two to four short quotes. An item's title is a headline, its text is the quote, and "
        "`meta` is who said it.",
        min_items=2,
        max_items=4,
        meta=True,
    ),
    "faq": SectionRule(
        "Three to eight questions. An item's title is the question and its text the answer.",
        min_items=3,
        max_items=8,
    ),
    "gallery": SectionRule("Three pictures with a short introduction.", images=3),
    "contact": SectionRule(
        "How to reach the business: a short text, and up to four details such as the address or "
        "opening hours (the item's title is the label, its text the value). The page also gets a "
        "contact form when the wireframe asks for one.",
        body=True,
        max_items=4,
    ),
    "cta_band": SectionRule("A full-width call to action: a heading and one short line."),
    "text": SectionRule(
        "Plain paragraphs for a page that needs reading, such as a policy.", body=True
    ),
}  # fmt: skip

MAX_SECTIONS_PER_PAGE = 8
MAX_PAGES = 5


def describe_vocabulary() -> str:
    """The section types as lines for a prompt."""
    return "\n".join(f"- {name}: {rule.description}" for name, rule in RULES.items())
