import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ValidationError
from webkiln_agents.testing import FakeLLM


class Answer(BaseModel):
    text: str


async def test_responses_come_in_order_and_the_last_one_repeats() -> None:
    llm = FakeLLM({"niche": [Answer(text="one"), {"text": "two"}]})
    call = llm.structured("niche", Answer)

    answers = [await call.ainvoke([HumanMessage("q")]) for _ in range(3)]

    assert answers == [Answer(text="one"), Answer(text="two"), Answer(text="two")]


async def test_a_bad_dict_raises_validation_error_and_an_exception_is_raised() -> None:
    llm = FakeLLM({"niche": [{"wrong": 1}, TimeoutError("slow"), None]})
    call = llm.structured("niche", Answer)

    with pytest.raises(ValidationError):
        await call.ainvoke([HumanMessage("q")])
    with pytest.raises(TimeoutError):
        await call.ainvoke([HumanMessage("q")])
    assert await call.ainvoke([HumanMessage("q")]) is None


async def test_prompts_are_recorded_per_role() -> None:
    llm = FakeLLM({"niche": [{"text": "a"}], "seo": [{"text": "b"}]})

    await llm.structured("niche", Answer).ainvoke([HumanMessage("about hotels")])
    await llm.structured("seo", Answer).ainvoke([HumanMessage("about keywords")])

    assert [m.content for m in llm.prompts_for("niche")[0]] == ["about hotels"]
    assert len(llm.prompts) == 2


async def test_a_role_without_a_script_is_a_test_error() -> None:
    with pytest.raises(AssertionError, match="no response for role 'brand'"):
        await FakeLLM().structured("brand", Answer).ainvoke([HumanMessage("q")])
