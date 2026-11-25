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
        ),
        AgentSpec(
            "generate",
            "Writes the website files from the wireframe and brand.",
            after=("wireframe",),
            needs_approval=True,
        ),
    ]
}
