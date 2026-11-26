"""The brand worker and the brand guideline it writes.

The guideline is the one output a person reads and signs off before anything is built from it, so
its colours and fonts are checked in code: a model can write any text, but these values later go
into CSS, so they must be exactly what a stylesheet can take.
"""

import re
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

from webkiln_agents.workers.outputs import Report, Strings
from webkiln_agents.workers.research import ResearchSpec, research_worker

_HEX = re.compile(r"#[0-9a-fA-F]{6}")
# The least contrast between the main colour and the background that keeps the site readable.
MIN_CONTRAST = 3.0


def _hex_color(value: str) -> str:
    value = value.strip()
    if not _HEX.fullmatch(value):
        msg = f"{value!r} is not a colour: write it as #RRGGBB"
        raise ValueError(msg)
    return value.lower()


HexColor = Annotated[str, AfterValidator(_hex_color)]
# Letters, digits, spaces and hyphens only, so a font name can never break out of a CSS value.
FontName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=2, max_length=60, pattern=r"^[A-Za-z0-9][A-Za-z0-9 \-]*$"
    ),
]


class Palette(BaseModel):
    primary: HexColor
    secondary: HexColor
    accent: HexColor
    background: HexColor


class Typography(BaseModel):
    heading_font: FontName = Field(description="A Google Fonts family, like 'Playfair Display'.")
    body_font: FontName = Field(description="A Google Fonts family, like 'Inter'.")
    notes: str = Field(min_length=10, max_length=300, description="How the two fonts are used.")


class PersonalityAndTone(BaseModel):
    traits: list[Annotated[str, StringConstraints(max_length=40)]] = Field(
        min_length=3, max_length=6, description="Single words or short phrases, like 'warm'."
    )
    voice: str = Field(min_length=10, max_length=400, description="How the brand sounds.")
    do: Strings = Field(min_length=2, max_length=5)
    dont: Strings = Field(min_length=2, max_length=5)


class Messaging(BaseModel):
    tagline: str = Field(min_length=3, max_length=80)
    value_proposition: str = Field(min_length=10, max_length=300)
    key_messages: Strings = Field(min_length=3, max_length=5)


class BrandGuideline(Report):
    positioning: str = Field(min_length=20, max_length=600)
    personality_and_tone: PersonalityAndTone
    messaging: Messaging
    palette: Palette
    typography: Typography
    logo_direction: str = Field(min_length=10, max_length=400)
    image_guidelines: Strings = Field(min_length=3, max_length=6)
    ui_ux_direction: Strings = Field(min_length=3, max_length=8)
    inspiration: Strings = Field(min_length=1, max_length=5)

    model_config = ConfigDict(extra="forbid")


def _luminance(color: str) -> float:
    """WCAG relative luminance of a `#rrggbb` colour."""
    channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(first: str, second: str) -> float:
    """WCAG contrast ratio of two colours, from 1 (identical) to 21 (black on white)."""
    light, dark = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def palette_problems(output: Report) -> list[str]:
    """Readability rules for a guideline's palette, as problems to show the model or the user."""
    if not isinstance(output, BrandGuideline):
        return []
    palette = output.palette
    ratio = contrast_ratio(palette.primary, palette.background)
    if ratio < MIN_CONTRAST:
        return [
            f"palette.primary {palette.primary} on palette.background {palette.background} has a "
            f"contrast of {ratio:.1f}:1, and at least {MIN_CONTRAST:.0f}:1 is needed to read it"
        ]
    return []


BrandRole = Literal["primary", "secondary", "accent", "background"]


class DirectEdits(BaseModel):
    """Exact changes the user makes by hand. Code applies them, so no model call is spent."""

    primary: HexColor | None = None
    secondary: HexColor | None = None
    accent: HexColor | None = None
    background: HexColor | None = None
    heading_font: FontName | None = None
    body_font: FontName | None = None
    positioning: str | None = Field(default=None, min_length=20, max_length=600)
    tagline: str | None = Field(default=None, min_length=3, max_length=80)

    model_config = ConfigDict(extra="forbid")

    def is_empty(self) -> bool:
        return not self.model_dump(exclude_none=True)


def apply_edits(guideline: BrandGuideline, edits: DirectEdits) -> BrandGuideline:
    """The guideline with the edits applied, checked as a whole. Raises `ValueError` (which
    includes a Pydantic `ValidationError`) when the result breaks a rule."""
    changes = edits.model_dump(exclude_none=True)
    data = guideline.model_dump()
    for name in ("primary", "secondary", "accent", "background"):
        if name in changes:
            data["palette"][name] = changes[name]
    for name in ("heading_font", "body_font"):
        if name in changes:
            data["typography"][name] = changes[name]
    if "positioning" in changes:
        data["positioning"] = changes["positioning"]
    if "tagline" in changes:
        data["messaging"]["tagline"] = changes["tagline"]
    edited = BrandGuideline.model_validate(data)
    if problems := palette_problems(edited):
        raise ValueError("; ".join(problems))
    if changes:
        # The summary is what teammates read in place of the guideline, so it must not go stale.
        edited = edited.model_copy(update={"summary": summarise(edited)})
    return edited


def summarise(guideline: BrandGuideline) -> str:
    """A summary built by code, for a guideline the user changed by hand."""
    palette = guideline.palette
    typography = guideline.typography
    text = (
        f"Positioning: {guideline.positioning} Tagline: {guideline.messaging.tagline}. "
        f"Voice: {guideline.personality_and_tone.voice} "
        f"Colours: primary {palette.primary}, secondary {palette.secondary}, "
        f"accent {palette.accent}, background {palette.background}. "
        f"Fonts: {typography.heading_font} for headings and {typography.body_font} for text."
    )
    return text[:1500]


BRAND_SPEC = ResearchSpec(
    "brand",
    "brand designer",
    "Write the brand guideline for the site. Use the research summaries in the context: the niche, "
    "the audience, the competitors' gaps and the SEO plan. Give the positioning, the personality "
    "and tone of voice, the messaging, a palette of four colours (primary, secondary, accent and "
    "background) as #RRGGBB values where the primary colour reads clearly on the background, one "
    "heading font and one body font from Google Fonts, the logo direction, image guidelines, the "
    "UI and UX direction and two or three inspiration references. When a <memory> block is given, "
    "follow the user's lasting preferences in it.",
    BrandGuideline,
    extra_check=palette_problems,
    uses_memory=True,
)

brand_worker = research_worker(BRAND_SPEC)
