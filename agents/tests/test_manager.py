import pytest
from pydantic import ValidationError
from webkiln_agents.core.schemas import Input
from webkiln_agents.manager.graph import graph

VALID = {
    "brand_name": "Cedar Stay",
    "prompt": "A boutique hotel website with booking",
    "niche": "Hospitality",
}


def test_valid_input_is_stored_in_state() -> None:
    result = graph.invoke(VALID)
    assert result["initial_input"] == Input(**VALID)


def test_input_is_stripped_before_it_is_stored() -> None:
    result = graph.invoke({**VALID, "brand_name": "  Cedar Stay  "})
    assert result["initial_input"].brand_name == "Cedar Stay"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("brand_name", "x"),  # too short
        ("brand_name", "   "),  # blank after stripping
        ("prompt", "too short"),
        ("niche", "ab"),
        ("niche", "n" * 81),  # too long
    ],
)
def test_invalid_input_raises_validation_error(field, value) -> None:
    with pytest.raises(ValidationError):
        graph.invoke({**VALID, field: value})
