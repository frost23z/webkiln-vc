"""What the user can answer when the run pauses at the brand review."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from webkiln_agents.workers.brand import DirectEdits

MAX_REVISIONS = 4  # rounds of changes the user can ask for before only Approve is left


class ReviewDecision(BaseModel):
    """The value a resume carries. Anything that does not fit is refused, and the run waits."""

    action: Literal["approve", "revise"]
    feedback: str | None = Field(default=None, max_length=2000)
    edits: DirectEdits | None = None

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @model_validator(mode="after")
    def _revise_needs_a_change(self) -> ReviewDecision:
        if self.action == "approve":
            return self
        if not (self.feedback or (self.edits and not self.edits.is_empty())):
            msg = "revise needs feedback or at least one direct edit"
            raise ValueError(msg)
        return self
