from typing import get_args

from webkiln_agents.manager.catalog import CATALOG
from webkiln_agents.manager.plan import AgentName


def test_every_catalog_name_is_a_valid_agent_name() -> None:
    assert set(CATALOG) <= set(get_args(AgentName))


def test_every_agent_name_has_a_catalog_entry() -> None:
    assert set(CATALOG) == set(get_args(AgentName))


def test_catalog_key_matches_spec_name() -> None:
    assert all(key == spec.name for key, spec in CATALOG.items())
