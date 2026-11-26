"""Site checks in code, not an agent. They cost no model call and run at the end of `generate`.

`check_site` takes the files of a site (path to text) and the wireframe, and returns every
problem it finds, each one naming its file. A site with any problem is never saved: `assemble`
stops and says which page is wrong. The same function runs after every edit in M9.
"""

import re
from collections.abc import Mapping
from html.parser import HTMLParser

from webkiln_agents.site.render import BASE_PATH, TOKENS_PATH, page_path
from webkiln_agents.site.tokens import FONTS_ORIGIN
from webkiln_agents.workers.wireframe import Wireframe

HEX_COLOR = re.compile(r"#[0-9a-fA-F]{3,8}\b")
CSS_CLASS = re.compile(r"\.([A-Za-z_][\w-]*)")
PIXEL_WIDTH = re.compile(r"(?<![-\w])(?:min-|max-)?width\s*:\s*\d+px", re.IGNORECASE)
WORDS = re.compile(r"[a-z0-9]+")

VOID = frozenset({"meta", "link", "input", "br", "hr", "img"})
HEADINGS = ("h1", "h2", "h3", "h4", "h5", "h6")


class _Page(HTMLParser):
    """Reads one page and keeps what the checks need."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.problems: list[str] = []
        self.classes: set[str] = set()
        self.hrefs: list[str] = []
        self.ids: set[str] = set()
        self.headings: list[int] = []
        self.meta: dict[str, str] = {}
        self.title = ""
        self.text: list[str] = []
        self.stylesheets: list[str] = []
        self.has_script = False
        self.stack: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name: value or "" for name, value in attrs}
        if tag not in VOID:
            self.stack.append(tag)
        self.classes.update(values.get("class", "").split())
        if "id" in values:
            self.ids.add(values["id"])
        if tag == "a" and "href" in values:
            self.hrefs.append(values["href"])
        if tag in HEADINGS:
            self.headings.append(int(tag[1]))
        if tag == "title":
            self._in_title = True
        if tag == "meta" and "name" in values:
            self.meta[values["name"]] = values.get("content", "")
        if tag == "link" and values.get("rel") == "stylesheet":
            self.stylesheets.append(values.get("href", ""))
        if tag in {"script", "iframe", "object", "embed"}:
            self.has_script = True
            self.problems.append(f"it has a <{tag}> element")
        if "style" in values:
            self.problems.append("it has an inline style")
        for name in values:
            if name.startswith("on"):
                self.problems.append(f"it has the inline handler {name!r}")
        if tag == "img":
            self.problems.append("it has an <img> element, and pictures are slots (class media)")
        if values.get("role") == "img" and not values.get("aria-label", "").strip():
            self.problems.append("a picture has no alt text")

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID:
            return
        if tag == "title":
            self._in_title = False
        if not self.stack or self.stack[-1] != tag:
            self.problems.append(f"the tag </{tag}> does not match what is open")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
            return
        self.stack.pop()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        self.text.append(data)


def _words(text: str) -> set[str]:
    return set(WORDS.findall(text.casefold()))


def _page_problems(
    path: str, html: str, wireframe_page_keyword: str, files: Mapping[str, str], known: set[str]
) -> list[str]:
    page = _Page()
    page.feed(html)
    page.close()
    problems = [f"{path}: {p}" for p in dict.fromkeys(page.problems)]
    if page.stack:
        problems.append(f"{path}: the tag <{page.stack[-1]}> is never closed")
    if "viewport" not in page.meta or "width=device-width" not in page.meta["viewport"]:
        problems.append(f"{path}: the viewport tag is missing")
    if not page.title.strip():
        problems.append(f"{path}: the page has no title")
    if not page.meta.get("description", "").strip():
        problems.append(f"{path}: the page has no meta description")
    for stylesheet in page.stylesheets:
        local = not stylesheet.startswith(("http:", "https:", "//"))
        if local and stylesheet not in files:
            problems.append(f"{path}: the stylesheet {stylesheet!r} is not in the site")
        if not local and not stylesheet.startswith(FONTS_ORIGIN + "/"):
            problems.append(f"{path}: the external stylesheet {stylesheet!r} is not allowed")
    if page.headings.count(1) != 1:
        problems.append(f"{path}: it needs exactly one h1 and has {page.headings.count(1)}")
    for before, after in zip(page.headings, page.headings[1:], strict=False):
        if after > before + 1:
            problems.append(f"{path}: a heading jumps from h{before} to h{after}")
            break
    for href in page.hrefs:
        if href.startswith("https://"):
            continue
        if href.startswith("#"):
            if href[1:] not in page.ids:
                problems.append(f"{path}: the link {href!r} points to nothing on the page")
        elif href not in known:
            problems.append(f"{path}: the link {href!r} is not a page of the site")
    unknown = page.classes - set(CSS_CLASS.findall(files[BASE_PATH]))
    if unknown:
        problems.append(f"{path}: it uses classes that base.css does not have: {sorted(unknown)}")
    keyword = _words(wireframe_page_keyword)
    seen = " ".join([*page.text, page.title, page.meta.get("description", "")])
    if keyword and not keyword <= _words(seen):
        problems.append(f"{path}: the target keyword {wireframe_page_keyword!r} is not on the page")
    return problems


def _css_problems(files: Mapping[str, str]) -> list[str]:
    problems = []
    if HEX_COLOR.search(files[BASE_PATH]):
        problems.append(f"{BASE_PATH}: colours must come from the tokens, not be written in it")
    if PIXEL_WIDTH.search(files[BASE_PATH]):
        problems.append(f"{BASE_PATH}: widths are in rem or percent, not pixels")
    if "max-width: 100%" not in files[BASE_PATH]:
        problems.append(f"{BASE_PATH}: images have no max-width")
    used = set(re.findall(r"var\((--[\w-]+)", files[BASE_PATH]))
    defined = set(re.findall(r"(--[\w-]+)\s*:", files[TOKENS_PATH]))
    if missing := used - defined:
        problems.append(
            f"{TOKENS_PATH}: base.css uses variables it does not define: {sorted(missing)}"
        )
    return problems


def check_site(files: Mapping[str, str], wireframe: Wireframe) -> list[str]:
    """Every problem of a site, empty when it is sound. Each problem starts with its file."""
    required = [TOKENS_PATH, BASE_PATH, *(page_path(p.slug) for p in wireframe.pages)]
    if missing := [path for path in required if path not in files]:
        return [f"the site is missing the files {missing}"]
    known = {page_path(p.slug) for p in wireframe.pages}
    problems = _css_problems(files)
    for page in wireframe.pages:
        path = page_path(page.slug)
        problems += _page_problems(path, files[path], page.seo.target_keyword, files, known)
    return problems
