"""What each research agent must return. The fields come from the assignment's own lists."""

from typing import Literal

from pydantic import BaseModel, Field

Strings = list[str]


class Report(BaseModel):
    """Shared by every research output."""

    summary: str = Field(
        min_length=40,
        max_length=1500,
        description="At most 150 words: the findings a teammate needs, no filler.",
    )


class Term(BaseModel):
    term: str
    meaning: str


class NicheReport(Report):
    characteristics: Strings = Field(min_length=3, max_length=8)
    market_trends: Strings = Field(min_length=3, max_length=8)
    opportunities: Strings = Field(min_length=3, max_length=8)
    challenges: Strings = Field(min_length=3, max_length=8)
    typical_services: Strings = Field(min_length=3, max_length=10)
    terminology: list[Term] = Field(min_length=4, max_length=12)


class Segment(BaseModel):
    name: str
    description: str


class Persona(BaseModel):
    name: str
    background: str
    goals: Strings = Field(min_length=1, max_length=4)
    frustrations: Strings = Field(min_length=1, max_length=4)


class AudienceReport(Report):
    primary_audiences: list[Segment] = Field(min_length=1, max_length=3)
    secondary_audiences: list[Segment] = Field(min_length=1, max_length=3)
    demographics_and_interests: Strings = Field(min_length=3, max_length=8)
    needs: Strings = Field(min_length=3, max_length=8)
    pain_points: Strings = Field(min_length=3, max_length=8)
    motivations: Strings = Field(min_length=3, max_length=8)
    buying_behavior: str
    personas: list[Persona] = Field(min_length=2, max_length=3)


class Keyword(BaseModel):
    term: str
    intent: Literal["informational", "navigational", "commercial", "transactional"]


class LandingPage(BaseModel):
    page: str = Field(description="A page name, like 'Rooms' or 'Contact'.")
    purpose: str
    target_keyword: str


class SeoStrategy(Report):
    primary_keywords: list[Keyword] = Field(min_length=3, max_length=8)
    secondary_keywords: list[Keyword] = Field(min_length=3, max_length=10)
    long_tail_keywords: Strings = Field(min_length=5, max_length=15)
    recommended_site_structure: Strings = Field(min_length=3, max_length=10)
    landing_pages: list[LandingPage] = Field(min_length=2, max_length=8)
    content_topics: Strings = Field(min_length=3, max_length=10)
    initial_strategy: str


class Competitor(BaseModel):
    name: str
    url: str | None = Field(
        default=None, description="The competitor's own site, a full https:// address."
    )
    positioning: str = Field(description="How the competitor presents itself, in one sentence.")


class CompetitorAnalysis(Report):
    """What the model writes. Code adds `unsourced` and `sources` (see `CompetitorReport`)."""

    competitors: list[Competitor] = Field(min_length=2, max_length=6)
    common_site_structures: Strings = Field(min_length=3, max_length=8)
    common_features: Strings = Field(min_length=3, max_length=8)
    content_and_messaging: Strings = Field(min_length=3, max_length=8)
    differentiation_opportunities: Strings = Field(min_length=3, max_length=8)


class CompetitorReport(CompetitorAnalysis):
    """The saved report. `unsourced` is true when no web search backed it."""

    unsourced: bool = False
    sources: list[str] = Field(default_factory=list, description="Search result URLs used.")
