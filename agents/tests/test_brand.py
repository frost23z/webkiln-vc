import json

import pytest
from pydantic import ValidationError
from samples import AUDIENCE, BRAND, NICHE
from test_research import packet, text_of
from webkiln_agents.manager.context import Context
from webkiln_agents.testing import FakeLLM
from webkiln_agents.workers.base import WorkerInput
from webkiln_agents.workers.brand import (
    BrandGuideline,
    DirectEdits,
    apply_edits,
    contrast_ratio,
    palette_problems,
)
from webkiln_agents.workers.errors import WorkerOutputError
from webkiln_agents.workers.registry import WORKERS


def with_palette(**colors: str) -> dict:
    return {**BRAND, "palette": {**BRAND["palette"], **colors}}


def guideline() -> BrandGuideline:
    return BrandGuideline.model_validate(BRAND)


def brand_packet(previous: dict | None = None, instruction: str = "") -> WorkerInput:
    base = packet("brand", instruction=instruction)
    base["context"] = {"niche": NICHE["summary"], "audience": AUDIENCE["summary"]}
    base["previous"] = previous
    return base


async def run(llm: FakeLLM):
    return await WORKERS["brand"](brand_packet(), Context(llm=llm))


def test_a_valid_guideline_lowercases_its_colours() -> None:
    parsed = BrandGuideline.model_validate(with_palette(primary="#1F4D3A"))
    assert parsed.palette.primary == "#1f4d3a"


@pytest.mark.parametrize("bad", ["green", "#12345", "#12345g", "123456", "#1234567", "url(x)"])
def test_a_colour_must_be_a_six_digit_hex_value(bad: str) -> None:
    with pytest.raises(ValidationError, match="palette"):
        BrandGuideline.model_validate(with_palette(accent=bad))


@pytest.mark.parametrize("bad", ["Inter; color: red", "a", "Foo}", 'Quote"s', "../x", "x" * 61])
def test_a_font_name_cannot_carry_css(bad: str) -> None:
    typography = {**BRAND["typography"], "heading_font": bad}
    with pytest.raises(ValidationError, match="heading_font"):
        BrandGuideline.model_validate({**BRAND, "typography": typography})


def test_contrast_matches_the_wcag_extremes() -> None:
    assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21)
    assert contrast_ratio("#777777", "#777777") == pytest.approx(1)


def test_a_primary_colour_that_cannot_be_read_on_the_background_is_a_problem() -> None:
    low = BrandGuideline.model_validate(with_palette(primary="#f5f0e6"))
    assert "contrast" in palette_problems(low)[0]
    assert palette_problems(guideline()) == []


async def test_the_worker_returns_the_guideline_and_reads_only_its_context() -> None:
    llm = FakeLLM({"brand": [BRAND]})
    result = await run(llm)
    assert result.summary == BRAND["summary"]
    assert result.data["palette"]["primary"] == "#1f4d3a"
    prompt = text_of(llm.prompts_for("brand")[0])
    assert f"<context from niche>\n{NICHE['summary']}" in prompt
    assert f"<context from audience>\n{AUDIENCE['summary']}" in prompt
    assert "<previous_output>\n" not in prompt  # a first run has nothing to rework


async def test_a_low_contrast_answer_is_sent_back_with_the_problem() -> None:
    llm = FakeLLM({"brand": [with_palette(primary="#f5f0e6"), BRAND]})
    await run(llm)
    assert "contrast" in text_of(llm.prompts_for("brand")[1])


async def test_a_guideline_that_never_passes_fails_the_worker() -> None:
    llm = FakeLLM({"brand": [with_palette(primary="#fafafa")]})
    with pytest.raises(WorkerOutputError, match="contrast"):
        await run(llm)


async def test_a_rework_shows_the_model_its_last_answer_and_the_new_instruction() -> None:
    llm = FakeLLM({"brand": [BRAND]})
    rework = brand_packet(previous=BRAND, instruction="Make it more premium")
    await WORKERS["brand"](rework, Context(llm=llm))
    prompt = text_of(llm.prompts_for("brand")[0])
    assert "<previous_output>\n{" in prompt
    assert json.dumps(BRAND["messaging"]["tagline"]) in prompt
    assert "This is a rework" in prompt
    assert "<instruction>\nMake it more premium" in prompt


def test_direct_edits_change_only_the_named_fields() -> None:
    edited = apply_edits(
        guideline(), DirectEdits(accent="#AA5500", heading_font="Lora", tagline="Home, away")
    )
    assert edited.palette.accent == "#aa5500"
    assert edited.palette.primary == "#1f4d3a"
    assert edited.typography.heading_font == "Lora"
    assert edited.typography.body_font == "Inter"
    assert edited.messaging.tagline == "Home, away"
    assert "Home, away" in edited.summary  # rebuilt, so it does not go stale


def test_an_empty_edit_is_empty_and_leaves_the_guideline_alone() -> None:
    assert DirectEdits().is_empty()
    assert apply_edits(guideline(), DirectEdits()) == guideline()


def test_an_edit_that_hurts_readability_is_refused() -> None:
    with pytest.raises(ValueError, match="contrast"):
        apply_edits(guideline(), DirectEdits(background="#2a5c45"))


def test_an_edit_with_a_bad_value_is_refused() -> None:
    with pytest.raises(ValidationError):
        DirectEdits(primary="dark green")
    with pytest.raises(ValidationError):
        DirectEdits.model_validate({"primary": "#112233", "logo": "x"})
