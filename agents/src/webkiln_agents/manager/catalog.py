from dataclasses import dataclass

from webkiln_agents.manager.plan import AgentName


@dataclass(frozen=True)
class AgentSpec:
    name: AgentName
    description: str
    after: tuple[AgentName, ...] = ()
    # The run can go on without this agent's result. Its dependents are told it was skipped.
    skippable: bool = False
    # The agent starts only after the user has approved the brand guideline.
    needs_approval: bool = False
    # Agents whose full output (not just the summary) this agent needs to read, and check against.
    reads: tuple[AgentName, ...] = ()
    # Tries for transient errors (a rate limit, a timeout, a network error), the first one
    # included, and how long one try may run. A worker also fixes a rejected answer by itself,
    # at most twice (`MAX_FIXES`), and the Manager may send a failed task back twice.
    max_tries: int = 3
    attempt_timeout: float = 300.0


CATALOG: dict[AgentName, AgentSpec] = {
    spec.name: spec
    for spec in [
        AgentSpec(
            "niche", "Researches the niche: market, trends and typical customers.", skippable=True
        ),
        AgentSpec("audience", "Defines the target audience and their needs.", skippable=True),
        AgentSpec(
            "competitor", "Looks at competitor sites and finds gaps to exploit.", skippable=True
        ),
        AgentSpec("seo", "Finds keywords and search intent for the site.", skippable=True),
        AgentSpec(
            "brand",
            "Writes the brand guideline: tone, colours, typography.",
            after=("niche", "audience", "competitor", "seo"),
        ),
        AgentSpec(
            "wireframe",
            "Plans the pages and sections of the site.",
            after=("brand",),
            needs_approval=True,
            reads=("brand", "audience", "seo"),
        ),
        AgentSpec(
            "generate",
            "Writes the website files from the wireframe and brand.",
            after=("wireframe",),
            needs_approval=True,
            reads=("brand", "wireframe"),
            # A new try writes every page again, so it gets fewer tries and more time.
            max_tries=2,
            attempt_timeout=900.0,
        ),
    ]
}
