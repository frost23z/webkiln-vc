from typing import Any

import pytest
from webkiln_agents.site.copy import PageCopy, copy_problems
from webkiln_agents.site.images import ImageNote
from webkiln_agents.site.render import (
    BASE_PATH,
    IMAGES_PATH,
    TOKENS_PATH,
    build_site,
    page_path,
)
from webkiln_agents.site.tokens import fonts_url, token_values, tokens_css
from webkiln_agents.testing import SAMPLE_BRAND, SAMPLE_WIREFRAME_DRAFT, sample_copy
from webkiln_agents.workers.wireframe import Page, Wireframe, WireframeDraft, finalize


def wireframe() -> Wireframe:
    return finalize(WireframeDraft.model_validate(SAMPLE_WIREFRAME_DRAFT))


def triples(page: Page) -> list[tuple[str, str, str]]:
    return [(s.id, s.type, s.title) for s in page.sections]


def copies(frame: Wireframe) -> dict[str, PageCopy]:
    return {p.slug: PageCopy.model_validate(sample_copy(triples(p))) for p in frame.pages}


def site(**overrides: Any) -> dict[str, str]:
    frame = wireframe()
    return build_site(SAMPLE_BRAND, "Cedar Stay", frame, overrides.pop("copies", copies(frame)))


def test_tokens_hold_the_brand_and_derived_colours() -> None:
    values = token_values(SAMPLE_BRAND)
    assert values["--color-primary"] == "#1F4D3A"
    assert values["--color-background"] == "#FAF7F0"
    assert values["--color-text"] == "#1a1a1a"  # dark text on a light background
    assert values["--color-on-primary"] == "#ffffff"  # light text on a deep green
    assert values["--font-heading"].startswith('"Playfair Display"')
    css = tokens_css(SAMPLE_BRAND)
    assert css.count("{") == 1
    assert "--color-surface:" in css


def test_text_stays_readable_on_a_dark_background() -> None:
    dark = {**SAMPLE_BRAND, "palette": {**SAMPLE_BRAND["palette"], "background": "#101820"}}
    assert token_values(dark)["--color-text"] == "#ffffff"


def test_fonts_come_from_one_google_fonts_url() -> None:
    url = fonts_url(SAMPLE_BRAND)
    assert url.startswith("https://fonts.googleapis.com/css2?family=Playfair+Display")
    assert "family=Inter" in url
    same = {
        **SAMPLE_BRAND,
        "typography": {**SAMPLE_BRAND["typography"], "body_font": "Inter", "heading_font": "Inter"},
    }
    assert fonts_url(same).count("family=") == 1


def test_valid_sample_copy_has_no_problems() -> None:
    for page in wireframe().pages:
        copy = PageCopy.model_validate(sample_copy(triples(page)))
        assert copy_problems([(i, t) for i, t, _ in triples(page)], copy) == []


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda c: c["sections"].reverse(), "in that order"),
        (lambda c: c["sections"][1].update(items=[]), "needs 3 to 6 items"),
        (lambda c: c["sections"][2]["items"][0].pop("meta"), "needs its `meta` line"),
    ],
)
def test_copy_that_does_not_fit_the_wireframe_is_named(change: Any, expected: str) -> None:
    page = wireframe().pages[0]
    raw = sample_copy(triples(page))
    change(raw)
    copy = PageCopy.model_validate(raw)
    problems = copy_problems([(i, t) for i, t, _ in triples(page)], copy)
    assert any(expected in problem for problem in problems), problems


def test_a_section_that_runs_long_is_named() -> None:
    page = wireframe().pages[0]
    raw = sample_copy(triples(page))
    raw["sections"][1]["items"][0]["text"] = "word " * 60
    raw["sections"][1]["items"][1]["text"] = "word " * 60
    raw["sections"][1]["items"][2]["text"] = "word " * 60
    raw["sections"][1]["body"] = "word " * 60
    copy = PageCopy.model_validate(raw)
    problems = copy_problems([(i, t) for i, t, _ in triples(page)], copy)
    assert any("index-features has" in p and "words" in p for p in problems), problems


def test_every_page_is_a_file_with_the_design_system_beside_it() -> None:
    files = site()
    assert set(files) == {
        "index.html", "rooms.html", "offers.html", "contact.html",
        TOKENS_PATH, BASE_PATH, IMAGES_PATH,
    }  # fmt: skip
    assert page_path("index") == "index.html"


def test_a_page_has_its_title_description_viewport_and_one_h1() -> None:
    html = site()["rooms.html"]
    assert "<title>Rooms at Cedar Stay</title>" in html
    assert 'name="description" content="See the boutique hotel rooms' in html
    assert 'name="viewport" content="width=device-width, initial-scale=1"' in html
    assert html.count("<h1>") == 1
    assert 'aria-current="page"' in html


def test_navigation_and_buttons_link_to_real_files() -> None:
    files = site()
    for slug in ("index", "rooms", "offers", "contact"):
        assert f'href="{page_path(slug)}"' in files["index.html"]
    assert 'class="btn btn-primary" href="contact.html">Book direct' in files["index.html"]


def test_the_contact_page_gets_its_form_and_map_from_the_wireframe() -> None:
    contact = site()["contact.html"]
    assert '<form class="form"' in contact
    assert "https://www.google.com/maps/search/?api=1&amp;query=" in contact
    assert "<form" not in site()["rooms.html"]


def test_model_copy_is_escaped() -> None:
    frame = wireframe()
    pages = copies(frame)
    raw = pages["index"].model_dump()
    raw["sections"][0]["heading"] = '<script>alert("x")</script> Welcome'
    pages["index"] = PageCopy.model_validate(raw)
    html = build_site(SAMPLE_BRAND, "Cedar Stay", frame, pages)["index.html"]
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_image_slots_use_the_workers_note_or_a_default() -> None:
    frame = wireframe()
    note = ImageNote(slot="index-hero-1", prompt="A calm room in warm light", alt="A calm room")
    files = build_site(SAMPLE_BRAND, "Cedar Stay", frame, copies(frame), [note])
    assert 'aria-label="A calm room"' in files["index.html"]
    assert 'aria-label="Illustration for Real rooms, real light"' in files["rooms.html"]
    assert files[IMAGES_PATH].count('"slot"') == 5  # two heroes and three gallery pictures


def test_the_same_input_gives_the_same_files() -> None:
    assert site() == site()


def test_a_page_without_copy_is_an_error() -> None:
    frame = wireframe()
    pages = copies(frame)
    del pages["offers"]
    with pytest.raises(KeyError):
        build_site(SAMPLE_BRAND, "Cedar Stay", frame, pages)
