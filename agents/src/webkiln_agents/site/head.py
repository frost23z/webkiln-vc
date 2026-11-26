"""The site as an edit sees it: the newest version, and what one edit changes in it.

`SiteHead` holds everything needed to render the site again without parsing HTML: the brand (an
edit may change its colours), the wireframe, the copy of every page and the picture notes, plus the
files of that version. `rebuild` renders the site from changed inputs, runs the site checks and
returns the new head with a `SiteChange`. Rendering is plain, deterministic code, so a file whose
inputs did not change comes out byte for byte the same, and only files that differ are listed as
changed. A site that fails its checks is never returned: the edit is refused with the reasons.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from webkiln_agents.site.checks import check_site
from webkiln_agents.site.copy import PageCopy
from webkiln_agents.site.images import ImageNote
from webkiln_agents.site.render import build_site
from webkiln_agents.workers.wireframe import Wireframe

MAX_PROBLEMS_SHOWN = 4


class EditRefused(Exception):
    """An edit that cannot be applied. The message is written for the user and is shown in chat."""


class SiteHead(BaseModel):
    """One version of the site with its inputs and all its files. Kept in the graph state."""

    version: int = Field(ge=1)
    brand: dict[str, Any]
    wireframe: dict[str, Any]
    copies: dict[str, dict[str, Any]]  # page slug to `PageCopy`
    images: list[dict[str, Any]]  # `ImageNote`s
    files: dict[str, str]

    def wireframe_model(self) -> Wireframe:
        return Wireframe.model_validate(self.wireframe)

    def page_copies(self) -> dict[str, PageCopy]:
        return {slug: PageCopy.model_validate(copy) for slug, copy in self.copies.items()}

    def notes(self) -> list[ImageNote]:
        return [ImageNote.model_validate(note) for note in self.images]


class SiteChange(BaseModel):
    """One saved version made by an edit. It holds the inputs of the new version and only the
    files that differ from its parent, so the API saves just those."""

    version: int = Field(ge=2)
    parent_version: int = Field(ge=1)
    cause: str  # the intent that made it, such as `content_edit`
    summary: str
    request: str  # the instruction as the Manager wrote it
    brand: dict[str, Any]
    wireframe: dict[str, Any]
    copies: dict[str, dict[str, Any]]
    images: list[dict[str, Any]]
    changed: dict[str, str]  # path to content, for the files that differ from the parent
    sections: list[str]  # ids of the sections whose words were written or added


def head_from_outputs(outputs: Mapping[str, Any]) -> SiteHead | None:
    """Version 1, from what `generate`, `wireframe` and `brand` saved. `None` before they exist."""
    build, wireframe, brand = (
        outputs.get("generate"),
        outputs.get("wireframe"),
        outputs.get("brand"),
    )
    if not (build and wireframe and brand and build.get("files")):
        return None
    return SiteHead(
        version=1,
        brand=brand,
        wireframe=wireframe,
        copies=build["copies"],
        images=build["images"],
        files=build["files"],
    )


def current_head(site: Mapping[str, Any] | None, outputs: Mapping[str, Any]) -> SiteHead | None:
    """The newest version: the head an edit saved, or else version 1 from the build."""
    return SiteHead.model_validate(site) if site else head_from_outputs(outputs)


def rebuild(
    head: SiteHead,
    *,
    brand_name: str,
    cause: str,
    summary: str,
    request: str,
    sections: Sequence[str] = (),
    brand: Mapping[str, Any] | None = None,
    wireframe: Wireframe | None = None,
    copies: Mapping[str, PageCopy] | None = None,
    images: Sequence[ImageNote] | None = None,
) -> tuple[SiteHead, SiteChange]:
    """Render the site from the head's inputs with the given ones replaced, and check it.

    Raises `EditRefused` when the result fails a site check or when nothing changed.
    """
    new_brand = dict(brand) if brand is not None else head.brand
    new_wireframe = wireframe or head.wireframe_model()
    new_copies = dict(copies) if copies is not None else head.page_copies()
    new_notes = list(images) if images is not None else head.notes()
    try:
        files = build_site(new_brand, brand_name, new_wireframe, new_copies, new_notes)
    except KeyError as error:
        msg = f"The page {error} has no words yet, so the site could not be built."
        raise EditRefused(msg) from error
    if problems := check_site(files, new_wireframe):
        shown = "; ".join(problems[:MAX_PROBLEMS_SHOWN])
        more = len(problems) - MAX_PROBLEMS_SHOWN
        suffix = f"; and {more} more" if more > 0 else ""
        msg = f"That change would break the site, so nothing was saved: {shown}{suffix}"
        raise EditRefused(msg[:600])
    changed = {path: text for path, text in files.items() if head.files.get(path) != text}
    if not changed:
        msg = "That change would leave the site exactly as it is, so nothing was saved."
        raise EditRefused(msg)
    inputs: dict[str, Any] = {
        "brand": new_brand,
        "wireframe": new_wireframe.model_dump(mode="json"),
        "copies": {slug: copy.model_dump(mode="json") for slug, copy in new_copies.items()},
        "images": [note.model_dump(mode="json") for note in new_notes],
    }
    new_head = SiteHead(version=head.version + 1, files=files, **inputs)
    change = SiteChange(
        version=new_head.version,
        parent_version=head.version,
        cause=cause,
        summary=summary,
        request=request,
        changed=changed,
        sections=list(sections),
        **inputs,
    )
    return new_head, change
