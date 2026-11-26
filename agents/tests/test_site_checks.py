from collections.abc import Callable

import pytest
from webkiln_agents.site.checks import check_site
from webkiln_agents.site.copy import PageCopy
from webkiln_agents.site.render import BASE_PATH, TOKENS_PATH, build_site
from webkiln_agents.testing import SAMPLE_BRAND, SAMPLE_WIREFRAME_DRAFT, sample_copy
from webkiln_agents.workers.wireframe import Wireframe, WireframeDraft, finalize


def wireframe() -> Wireframe:
    return finalize(WireframeDraft.model_validate(SAMPLE_WIREFRAME_DRAFT))


def good_site() -> dict[str, str]:
    frame = wireframe()
    copies = {
        p.slug: PageCopy.model_validate(sample_copy([(s.id, s.type, s.title) for s in p.sections]))
        for p in frame.pages
    }
    # The keyword of every page is part of its title in the sample, so the check can pass.
    return build_site(SAMPLE_BRAND, "Cedar Stay", frame, copies)


def test_a_rendered_site_passes_every_check() -> None:
    assert check_site(good_site(), wireframe()) == []


def edit(path: str, old: str, new: str) -> Callable[[dict[str, str]], None]:
    def apply(files: dict[str, str]) -> None:
        assert old in files[path], old
        files[path] = files[path].replace(old, new, 1)

    return apply


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (
            edit("rooms.html", "<title>Rooms at Cedar Stay</title>", ""),
            "rooms.html: the page has no title",
        ),
        (
            edit("rooms.html", 'name="description"', 'name="other"'),
            "rooms.html: the page has no meta description",
        ),
        (
            edit("rooms.html", 'name="viewport"', 'name="port"'),
            "rooms.html: the viewport tag is missing",
        ),
        (
            edit("index.html", 'href="rooms.html"', 'href="shop.html"'),
            "index.html: the link 'shop.html' is not a page",
        ),
        (edit("index.html", 'href="#main"', 'href="#nowhere"'), "points to nothing on the page"),
        (edit("rooms.html", "<h1>Our rooms</h1>", "<h3>Our rooms</h3>"), "exactly one h1"),
        (
            edit(
                "rooms.html", "<h2>Real rooms, real light</h2>", "<h4>Real rooms, real light</h4>"
            ),
            "jumps from h1 to h4",
        ),
        (
            edit("rooms.html", 'class="section hero"', 'class="section hero fancy"'),
            "classes that base.css does not have: ['fancy']",
        ),
        (
            edit("rooms.html", '<main id="main">', '<main id="main" style="color: red">'),
            "inline style",
        ),
        (
            edit("rooms.html", '<main id="main">', '<main id="main" onclick="x()">'),
            "inline handler 'onclick'",
        ),
        (edit("rooms.html", "</head>", "<script>alert(1)</script></head>"), "<script> element"),
        (
            edit(
                "rooms.html",
                "</head>",
                '<link rel="stylesheet" href="https://evil.example/a.css"></head>',
            ),
            "external stylesheet",
        ),
        (
            edit(
                "rooms.html",
                'aria-label="Illustration for Real rooms, real light"',
                'aria-label=""',
            ),
            "no alt text",
        ),
        (edit("rooms.html", "</main>", "</div></main>"), "does not match"),
        (edit("rooms.html", "boutique hotel rooms", "x"), "target keyword"),
        (
            edit(BASE_PATH, ".lead {", ".lead { border-color: #ff0000;"),
            "colours must come from the tokens",
        ),
        (edit(BASE_PATH, ".container {", ".container { width: 980px;"), "not pixels"),
        (edit(TOKENS_PATH, "--color-muted", "--color-gone"), "variables it does not define"),
    ],
)
def test_a_broken_site_is_stopped_with_the_page_named(
    change: Callable[[dict[str, str]], None], expected: str
) -> None:
    files = good_site()
    change(files)
    problems = check_site(files, wireframe())
    assert any(expected in problem for problem in problems), problems


def test_a_missing_page_is_named() -> None:
    files = good_site()
    del files["offers.html"]
    assert "offers.html" in check_site(files, wireframe())[0]
