"""Pages from the wireframe, the copy and the brand. Plain code, no model.

`build_site` returns every file of the site as text, by path. Jinja2 runs with autoescape on,
because the copy is written by a model and goes into HTML: a `<script>` in a heading is shown as
text. The templates only use the classes of `base.css`, and `checks.py` verifies that.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from jinja2 import Environment, PackageLoader, StrictUndefined

from webkiln_agents.site.copy import PageCopy, SectionCopy
from webkiln_agents.site.images import ImageNote, Slot, slots_for
from webkiln_agents.site.tokens import fonts_url, tokens_css
from webkiln_agents.workers.wireframe import HOME, Page, Section, Wireframe

BASE_CSS = Path(__file__).parent / "static" / "base.css"
TOKENS_PATH = "assets/tokens.css"
BASE_PATH = "assets/base.css"
IMAGES_PATH = "assets/images.json"


def page_path(slug: str) -> str:
    """The file of a page. Every page is a file in one folder, so links stay relative."""
    return "index.html" if slug == HOME else f"{slug}.html"


def _environment() -> Environment:
    env = Environment(
        loader=PackageLoader("webkiln_agents.site", "templates"),
        autoescape=True,  # the copy comes from a model
        undefined=StrictUndefined,  # a typo in a template is an error, not an empty string
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals.update(href=page_path)  # pyrefly: ignore[bad-argument-type]
    return env


ENV = _environment()


@dataclass(frozen=True)
class Image:
    slot: str
    aspect: str
    alt: str
    prompt: str


@dataclass(frozen=True)
class Entry:
    """One section as its template sees it."""

    section: Section
    copy: SectionCopy
    images: list[Image]
    heading_tag: str
    item_tag: str


def page_images(wireframe: Wireframe, notes: Sequence[ImageNote]) -> dict[str, list[Image]]:
    """Every image slot of the site by section id, with the worker's note or a plain default."""
    by_slot = {note.slot: note for note in notes}
    pages = [(p.slug, [(s.id, s.type) for s in p.sections]) for p in wireframe.pages]
    titles = {s.id: s.title for p in wireframe.pages for s in p.sections}
    images: dict[str, list[Image]] = {}
    for slot in slots_for(pages):
        images.setdefault(slot.section, []).append(_image(slot, by_slot.get(slot.id), titles))
    return images


def _image(slot: Slot, note: ImageNote | None, titles: Mapping[str, str]) -> Image:
    if note is None:  # a missing note never fails the build
        label = titles[slot.section]
        return Image(slot.id, slot.aspect, f"Illustration for {label}", f"A picture for {label}")
    return Image(slot.id, slot.aspect, note.alt, note.prompt)


def _entries(page: Page, copy: PageCopy, images: Mapping[str, list[Image]]) -> list[Entry]:
    entries = []
    for index, (section, words) in enumerate(zip(page.sections, copy.sections, strict=True)):
        tag = "h1" if index == 0 else "h2"
        entries.append(
            Entry(section, words, images.get(section.id, []), tag, "h2" if tag == "h1" else "h3")
        )
    return entries


def _map_url(page: Page, copy: PageCopy, brand_name: str) -> str:
    query = brand_name
    for section in copy.sections:
        for item in section.items:
            if "address" in item.title.casefold():
                query = item.text
    return "https://www.google.com/maps/search/?api=1&query=" + quote(query)


def render_page(
    page: Page,
    wireframe: Wireframe,
    copy: PageCopy,
    images: Mapping[str, list[Image]],
    brand: dict,
    brand_name: str,
) -> str:
    names = {p.slug: p.name for p in wireframe.pages}
    return ENV.get_template("layout.html.j2").render(
        page=page,
        sections=_entries(page, copy, images),
        navigation=[{"slug": slug, "name": names[slug]} for slug in wireframe.navigation],
        needs={r.kind for r in wireframe.functional_requirements if r.page == page.slug},
        brand_name=brand_name,
        tagline=brand["messaging"]["tagline"],
        fonts_url=fonts_url(brand),
        map_url=_map_url(page, copy, brand_name),
    )


def build_site(
    brand: dict,
    brand_name: str,
    wireframe: Wireframe,
    copies: Mapping[str, PageCopy],
    notes: Sequence[ImageNote] = (),
) -> dict[str, str]:
    """All the files of the site, by path. Raises `KeyError` for a page without copy."""
    images = page_images(wireframe, notes)
    files = {
        page_path(page.slug): render_page(
            page, wireframe, copies[page.slug], images, brand, brand_name
        )
        for page in wireframe.pages
    }
    files[TOKENS_PATH] = tokens_css(brand)
    files[BASE_PATH] = BASE_CSS.read_text(encoding="utf-8")
    manifest = [
        vars(image) | {"section": section} for section, group in images.items() for image in group
    ]
    files[IMAGES_PATH] = json.dumps(manifest, indent=2) + "\n"
    return files
