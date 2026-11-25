from dataclasses import dataclass

from webkiln_agents.manager.plan import AgentName


@dataclass(frozen=True)
class AgentSpec:
    name: AgentName
    description: str
    after: tuple[AgentName, ...] = ()


CATALOG: dict[AgentName, AgentSpec] = {
    spec.name: spec
    for spec in [
        AgentSpec("niche", "Researches the niche: market, trends and typical customers."),
        AgentSpec("audience", "Defines the target audience and their needs."),
        AgentSpec("competitor", "Looks at competitor sites and finds gaps to exploit."),
        AgentSpec("seo", "Finds keywords and search intent for the site."),
        AgentSpec(
            "brand",
            "Writes the brand guideline: tone, colours, typography.",
            after=("niche", "audience", "competitor", "seo"),
        ),
        AgentSpec("wireframe", "Plans the pages and sections of the site.", after=("brand",)),
        AgentSpec(
            "generate",
            "Writes the website files from the wireframe and brand.",
            after=("wireframe",),
        ),
    ]
}
