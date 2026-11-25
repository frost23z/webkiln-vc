from typing import get_args

from webkiln_agents.manager.catalog import CATALOG
from webkiln_agents.manager.plan import AgentName


def test_every_catalog_name_is_a_valid_agent_name() -> None:
    assert set(CATALOG) <= set(get_args(AgentName))


def test_every_agent_name_has_a_catalog_entry() -> None:
    assert set(CATALOG) == set(get_args(AgentName))


def test_catalog_key_matches_spec_name() -> None:
    assert all(key == spec.name for key, spec in CATALOG.items())


def test_every_after_name_is_a_catalog_agent() -> None:
    assert all(set(spec.after) <= set(CATALOG) for spec in CATALOG.values())


def test_research_agents_have_no_dependencies() -> None:
    research_agents: set[AgentName] = {
        "niche",
        "audience",
        "seo",
        "competitor",
    }

    assert all(CATALOG[name].after == () for name in research_agents)


def test_brand_depends_on_all_research_agents() -> None:
    assert set(CATALOG["brand"].after) == {
        "niche",
        "audience",
        "seo",
        "competitor",
    }


def test_wireframe_depends_on_brand() -> None:
    assert CATALOG["wireframe"].after == ("brand",)


def test_generate_depends_on_wireframe() -> None:
    assert CATALOG["generate"].after == ("wireframe",)
