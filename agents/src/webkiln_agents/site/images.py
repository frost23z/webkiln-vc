"""Image slots: where a section draws a picture, and what the image worker says about each.

Slots come from the wireframe, in code, so the worker cannot invent one or forget one. The
pictures themselves are drawn as placeholders from the brand's colours. The prompt, the alt text
and the aspect ratio of every slot are saved with the site, so a real image can fill a slot later.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from webkiln_agents.site.vocabulary import rule_for

ASPECTS = {"hero": "16:9", "about": "4:3", "gallery": "1:1"}
# A site has at most 5 pages, and a gallery alone holds 3 slots, so 12 was too few.
MAX_NOTES = 60


@dataclass(frozen=True)
class Slot:
    id: str  # `<section id>-<n>`
    section: str
    page: str
    aspect: str


class ImageNote(BaseModel):
    slot: str = Field(description="The id of the slot, copied exactly.")
    prompt: str = Field(
        min_length=10, max_length=300, description="What to draw, in the brand's style."
    )
    alt: str = Field(
        min_length=5, max_length=125, description="What a person who cannot see it should know."
    )

    model_config = ConfigDict(extra="forbid")


class ImagePlan(BaseModel):
    images: list[ImageNote] = Field(max_length=MAX_NOTES)

    model_config = ConfigDict(extra="forbid")


def slots_for(pages: Sequence[tuple[str, Sequence[tuple[str, str]]]]) -> list[Slot]:
    """The slots of a site, from `(page slug, [(section id, type)])`, in page order."""
    slots = []
    for page, sections in pages:
        for section_id, kind in sections:
            rule = rule_for(kind)
            slots.extend(
                Slot(f"{section_id}-{n}", section_id, page, ASPECTS.get(kind, "4:3"))
                for n in range(1, rule.images + 1)
            )
    return slots
