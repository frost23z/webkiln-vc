"""The plain-code parts of an edit: reading the Manager's proposal against the site, and the
structural changes. No model and no graph."""

import pytest
from webkiln_agents.site.copy import PageCopy, SectionCopy
from webkiln_agents.site.edits import (
    EditIntent,
    Target,
    find_page,
    find_section,
    listing,
    next_section_id,
    resolve,
    to_draft,
    wireframe_problems,
    with_page,
    with_section,
    without_section,
)
from webkiln_agents.site.head import EditRefused, head_from_outputs, rebuild
from webkiln_agents.testing import sample_site_copy, sample_wireframe
from webkiln_agents.workers.wireframe import Section, finalize

WIREFRAME = sample_wireframe()
COPIES = {slug: PageCopy.model_validate(c) for slug, c in sample_site_copy().items()}


def intent(kind: str, **fields: object) -> EditIntent:
    base = {"intent": kind, "confidence": 0.9, "instruction": "Do what the user asked."}
    return EditIntent.model_validate({**base, **fields})


def test_a_content_edit_resolves_to_its_page_and_section() -> None:
    target = resolve(intent("content_edit", section="rooms-faq"), WIREFRAME)
    assert isinstance(target, Target)
    assert (target.page and target.page.slug, target.section and target.section.id) == (
        "rooms",
        "rooms-faq",
    )


def test_a_page_with_one_section_needs_no_section_id() -> None:
    target = resolve(intent("content_edit", page="Offers"), WIREFRAME)  # by name, any case
    assert isinstance(target, Target)
    assert target.section
    assert target.section.id == "offers-pricing"


@pytest.mark.parametrize("kind", ["question", "clarify", "unsupported"])
def test_questions_and_clarifications_answer_with_the_models_reply(kind: str) -> None:
    assert resolve(intent(kind, reply="Here you go."), WIREFRAME) == "Here you go."
    assert isinstance(resolve(intent(kind), WIREFRAME), str)  # and have a fallback


def test_a_low_confidence_edit_becomes_a_question() -> None:
    low = intent("content_edit", section="rooms-faq", confidence=0.2, reply="Which FAQ?")
    assert resolve(low, WIREFRAME) == "Which FAQ?"


def test_an_unknown_section_lists_the_ids_that_exist() -> None:
    answer = resolve(intent("content_edit", section="nope"), WIREFRAME)
    assert isinstance(answer, str)
    assert "index-hero" in answer
    assert "contact-contact" in answer


def test_add_section_needs_a_page_and_a_type_that_is_not_a_hero() -> None:
    assert isinstance(resolve(intent("add_section", section_type="faq"), WIREFRAME), str)
    assert isinstance(resolve(intent("add_section", page="rooms"), WIREFRAME), str)
    hero = resolve(intent("add_section", page="rooms", section_type="hero"), WIREFRAME)
    assert isinstance(hero, str)
    ok = resolve(intent("add_section", page="rooms", section_type="faq"), WIREFRAME)
    assert isinstance(ok, Target)
    assert ok.section_type == "faq"


def test_add_page_is_refused_at_the_page_limit_or_with_a_full_menu() -> None:
    assert isinstance(resolve(intent("add_page"), WIREFRAME), Target)
    full = WIREFRAME.model_copy(update={"pages": [*WIREFRAME.pages, WIREFRAME.pages[0]]})
    assert "5 pages" in str(resolve(intent("add_page"), full))
    crowded = WIREFRAME.model_copy(update={"navigation": ["index"] * 6})
    assert "menu is full" in str(resolve(intent("add_page"), crowded))


def test_find_helpers() -> None:
    assert find_page(WIREFRAME, "rooms") is find_page(WIREFRAME, " ROOMS ")
    assert find_page(WIREFRAME, None) is None
    assert find_page(WIREFRAME, "x") is None
    assert find_section(WIREFRAME, "index-hero") is not None
    assert find_section(WIREFRAME, "") is None


def test_the_listing_shows_each_section_with_its_first_words() -> None:
    rows = listing(WIREFRAME, COPIES, words=2)
    home = rows[0]
    assert home["page"] == "index"
    assert home["sections"][0] == {"id": "index-hero", "type": "hero", "starts": "Stay like"}


def test_section_ids_are_never_reused_or_renumbered() -> None:
    page = WIREFRAME.pages[0]
    assert next_section_id(page, "faq") == "index-faq"
    assert next_section_id(page, "features") == "index-features-2"
    twice = page.model_copy(
        update={
            "sections": [
                *page.sections,
                page.sections[1].model_copy(update={"id": "index-features-2"}),
            ]
        }
    )
    assert next_section_id(twice, "features") == "index-features-3"
    assert next_section_id(twice, "cta_band") == "index-cta-band-2"


def test_a_section_added_and_removed_again_leaves_the_wireframe_as_it_was() -> None:
    page = WIREFRAME.pages[1]
    fresh = Section(
        id="rooms-text",
        type="text",
        title="Good to know",
        content_brief="Something the guests should know before they arrive.",
        audience_needs=["Clear photos"],
    )
    words = SectionCopy(id="rooms-text", heading="Good to know", body="A paragraph.")
    added, copy = with_section(WIREFRAME, COPIES["rooms"], page, fresh, words)
    assert [s.id for s in added.pages[1].sections][-1] == "rooms-text"
    assert [s.id for s in copy.sections] == [s.id for s in added.pages[1].sections]
    back, back_copy = without_section(added, copy, added.pages[1], "rooms-text")
    assert back == WIREFRAME
    assert back_copy == COPIES["rooms"]


def test_the_last_section_of_a_page_cannot_be_removed() -> None:
    page = WIREFRAME.pages[2]
    with pytest.raises(EditRefused, match="at least one"):
        without_section(WIREFRAME, COPIES["offers"], page, "offers-pricing")


def test_removing_the_contact_section_is_caught_by_the_structure_rules() -> None:
    page = WIREFRAME.pages[3]
    extra = Section(
        id="contact-text",
        type="text",
        title="More",
        content_brief="Another paragraph for the guests to read.",
        audience_needs=["Easy booking"],
    )
    words = SectionCopy(id="contact-text", heading="More", body="Text.")
    added, copy = with_section(WIREFRAME, COPIES["contact"], page, extra, words)
    gone, _ = without_section(added, copy, added.pages[3], "contact-contact")
    assert any("contact section" in p for p in wireframe_problems(gone))  # the form needs it


def test_a_wireframe_survives_the_trip_to_a_draft_and_back() -> None:
    assert wireframe_problems(WIREFRAME) == []
    assert finalize(to_draft(WIREFRAME)) == WIREFRAME


def test_a_new_page_goes_before_contact_in_the_menu() -> None:
    page = WIREFRAME.pages[2].model_copy(update={"slug": "extra"})
    wireframe = with_page(WIREFRAME, page)
    assert wireframe.navigation == ["index", "rooms", "offers", "extra", "contact"]
    assert wireframe.pages[-1].slug == "extra"


def test_rebuilding_unchanged_inputs_is_refused_as_no_change() -> None:
    from webkiln_agents.testing import SAMPLE_BRAND

    head = full_head()
    assert head.brand == SAMPLE_BRAND
    with pytest.raises(EditRefused, match="exactly as it is"):
        rebuild(head, brand_name="Cedar Stay", cause="content_edit", summary="s", request="r")


def test_rebuilding_a_page_without_words_is_refused() -> None:
    head = full_head()
    with pytest.raises(EditRefused, match="no words"):
        rebuild(
            head,
            brand_name="Cedar Stay",
            cause="content_edit",
            summary="s",
            request="r",
            copies={"index": COPIES["index"]},
        )


def test_head_from_outputs_needs_the_build_the_wireframe_and_the_brand() -> None:
    assert head_from_outputs({}) is None
    assert head_from_outputs({"generate": {"files": {}}, "wireframe": {}, "brand": {}}) is None


def full_head():
    from webkiln_agents.site.images import ImageNote
    from webkiln_agents.site.render import build_site
    from webkiln_agents.testing import SAMPLE_BRAND, sample_image_plan

    notes = [ImageNote.model_validate(n) for n in sample_image_plan()["images"]]
    files = build_site(SAMPLE_BRAND, "Cedar Stay", WIREFRAME, COPIES, notes)
    outputs = {
        "generate": {
            "files": files,
            "copies": {k: v.model_dump(mode="json") for k, v in COPIES.items()},
            "images": [n.model_dump(mode="json") for n in notes],
        },
        "wireframe": WIREFRAME.model_dump(mode="json"),
        "brand": SAMPLE_BRAND,
    }
    head = head_from_outputs(outputs)
    assert head is not None
    return head
