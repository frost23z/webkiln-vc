import copy
import json
from typing import Any

import pytest
from samples import AUDIENCE, SEO
from test_research import packet, text_of
from webkiln_agents.manager.context import Context
from webkiln_agents.testing import SAMPLE_BRAND, SAMPLE_WIREFRAME_DRAFT, FakeLLM
from webkiln_agents.workers.errors import WorkerOutputError
from webkiln_agents.workers.registry import WORKERS
from webkiln_agents.workers.wireframe import (
    Wireframe,
    WireframeDraft,
    finalize,
    outline,
    plan_problems,
    structure_problems,
)

UPSTREAM = {"brand": SAMPLE_BRAND, "audience": AUDIENCE, "seo": SEO}


def draft(**changes: Any) -> dict[str, Any]:
    return {**copy.deepcopy(SAMPLE_WIREFRAME_DRAFT), **changes}


def parsed(data: dict[str, Any] | None = None) -> WireframeDraft:
    return WireframeDraft.model_validate(data or SAMPLE_WIREFRAME_DRAFT)


def page(data: dict[str, Any], slug: str) -> dict[str, Any]:
    return next(p for p in data["pages"] if p["slug"] == slug)


def wireframe_packet() -> Any:
    base = packet("wireframe")
    base["context"] = {"brand": SAMPLE_BRAND["summary"]}
    base["upstream"] = UPSTREAM
    return base


def test_the_sample_wireframe_has_no_problems() -> None:
    assert plan_problems(parsed(), UPSTREAM) == []


def test_finalize_gives_every_section_a_stable_id() -> None:
    wireframe = finalize(parsed())
    ids = [s.id for p in wireframe.pages for s in p.sections]
    assert ids[:4] == ["index-hero", "index-features", "index-testimonials", "index-cta-band"]
    assert len(ids) == len(set(ids))
    assert ids == [s.id for p in finalize(parsed()).pages for s in p.sections]


def test_a_type_used_twice_on_a_page_gets_a_number() -> None:
    data = draft()
    index = page(data, "index")
    index["sections"].append(copy.deepcopy(index["sections"][3]))
    ids = [s.id for s in finalize(parsed(data)).pages[0].sections]
    assert ids[-2:] == ["index-cta-band", "index-cta-band-2"]


def test_the_outline_shows_pages_with_their_sections() -> None:
    lines = outline(finalize(parsed()))
    assert lines[0] == "Home (index)"
    assert "  hero: Stay like you live here" in lines
    assert "Contact (contact)" in lines


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda d: page(d, "rooms").update(slug="index"), "used by more than one page"),
        (lambda d: page(d, "index").update(slug="home"), "no home page"),
        (lambda d: page(d, "rooms").update(parent="nowhere"), "unknown parent"),
        (lambda d: page(d, "rooms").update(parent="rooms"), "nested too deep or in a loop"),
        (lambda d: d["navigation"].append("blog"), "which is not a page"),
        (lambda d: d["navigation"].append("rooms"), "lists a page twice"),
        (lambda d: d["navigation"].remove("index"), "must include the home page"),
        (lambda d: page(d, "rooms")["sections"].reverse(), "a hero can only be the first"),
        (
            lambda d: page(d, "index")["sections"][0]["cta_primary"].update(target="shop"),
            "opens 'shop'",
        ),
        (
            lambda d: d["functional_requirements"].append(
                {"kind": "search", "page": "blog", "description": "Search the whole blog"}
            ),
            "unknown page 'blog'",
        ),
        (
            lambda d: (
                page(d, "contact")["sections"].clear()
                or page(d, "contact")["sections"].append(
                    page(d, "rooms")["sections"][2]  # a faq, so the page has no contact section
                )
            ),
            "needs a contact section",
        ),
    ],
)
def test_structure_problems(change: Any, expected: str) -> None:
    data = draft()
    change(data)
    problems = structure_problems(parsed(data))
    assert any(expected in problem for problem in problems), problems


def test_an_unknown_section_type_is_refused_by_the_schema() -> None:
    data = draft()
    page(data, "index")["sections"][1]["type"] = "carousel"
    with pytest.raises(ValueError, match="type"):
        parsed(data)


def test_a_landing_page_must_be_a_page_or_a_section() -> None:
    data = draft()
    data["pages"].remove(page(data, "offers"))
    data["navigation"].remove("offers")
    data["pages"][0]["sections"][3]["cta_primary"] = None
    problems = plan_problems(parsed(data), UPSTREAM)
    assert any("landing page 'Offers'" in p for p in problems), problems
    # The same landing page as a section title is enough, for when the page limit is reached.
    page(data, "rooms")["sections"][1]["title"] = "Offers for long stays"
    assert not any("landing page" in p for p in plan_problems(parsed(data), UPSTREAM))


def test_every_audience_need_must_be_answered() -> None:
    data = draft()
    for p in data["pages"]:
        for section in p["sections"]:
            section["audience_needs"] = [
                n for n in section["audience_needs"] if n != "Clear photos"
            ]
            section["audience_needs"] = section["audience_needs"] or ["Easy booking"]
    problems = plan_problems(parsed(data), UPSTREAM)
    assert [p for p in problems if "audience need" in p] == [
        "the audience need 'Clear photos' is not answered by any section: list it, word for "
        "word, in the audience_needs of a section that answers it"
    ]


def test_the_briefs_must_use_the_brand_tone_words() -> None:
    data = draft()
    for p in data["pages"]:
        for section in p["sections"]:
            section["content_brief"] = "A plain description of what this section is about."
    problems = plan_problems(parsed(data), UPSTREAM)
    assert any("tone words" in p and "warm" in p for p in problems), problems


def test_a_skipped_agent_leaves_its_check_out() -> None:
    data = draft()
    data["pages"].remove(page(data, "offers"))
    data["navigation"].remove("offers")
    data["pages"][0]["sections"][3]["cta_primary"] = None
    assert plan_problems(parsed(data), {"brand": SAMPLE_BRAND}) == []


async def test_the_worker_returns_the_wireframe_with_ids() -> None:
    llm = FakeLLM({"wireframe": [SAMPLE_WIREFRAME_DRAFT]})
    result = await WORKERS["wireframe"](wireframe_packet(), Context(llm=llm))
    saved = Wireframe.model_validate(result.data)
    assert saved.pages[0].sections[0].id == "index-hero"
    assert result.summary == SAMPLE_WIREFRAME_DRAFT["summary"]


async def test_the_prompt_carries_the_earlier_work_and_the_vocabulary() -> None:
    llm = FakeLLM({"wireframe": [SAMPLE_WIREFRAME_DRAFT]})
    await WORKERS["wireframe"](wireframe_packet(), Context(llm=llm))
    (prompt,) = llm.prompts_for("wireframe")
    text = text_of(prompt)
    assert "<approved_brand>" in text
    assert "Easy booking" in text  # an audience need
    assert "Offers" in text  # a landing page
    assert "- pricing:" in text  # the vocabulary
    assert json.dumps(SAMPLE_BRAND["palette"]) not in text  # colours are not the wireframe's job


async def test_a_wireframe_with_problems_is_sent_back_with_the_reasons() -> None:
    bad = draft()
    bad["navigation"].append("blog")
    llm = FakeLLM({"wireframe": [bad, SAMPLE_WIREFRAME_DRAFT]})
    result = await WORKERS["wireframe"](wireframe_packet(), Context(llm=llm))
    assert result.data["navigation"] == SAMPLE_WIREFRAME_DRAFT["navigation"]
    retry = text_of(llm.prompts_for("wireframe")[1])
    assert "lists 'blog', which is not a page" in retry


async def test_a_wireframe_that_never_passes_fails_the_task() -> None:
    bad = draft()
    bad["navigation"].append("blog")
    with pytest.raises(WorkerOutputError, match="blog"):
        await WORKERS["wireframe"](wireframe_packet(), Context(llm=FakeLLM({"wireframe": [bad]})))
