"""Serve a saved site version, so the UI can show it in a sandboxed iframe.

The pages are generated, so they are treated as untrusted: the response carries a policy that
allows nothing but the site's own files and its Google Fonts, and the UI embeds the frame with
`sandbox` and no `allow-same-origin`. Links between the pages are relative and stay inside the
same version.
"""

import mimetypes
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Request, Response

from webkiln_api import store
from webkiln_api.config import get_api_settings
from webkiln_api.openapi import ERROR_RESPONSES

router = APIRouter(prefix="/preview", tags=["preview"], responses=ERROR_RESPONSES)

# Text types the browser must know exactly. Anything else is guessed from the extension.
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".txt": "text/plain; charset=utf-8",
    ".xml": "application/xml",
}


def content_type_of(path: str) -> str:
    suffix = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return CONTENT_TYPES.get(suffix) or mimetypes.guess_type(path)[0] or "text/plain; charset=utf-8"


def policy() -> str:
    """No script, no network, no form, no outside frames. Fonts come from Google Fonts, which the
    pages' head links, and only the UI (or the same origin, when both sit behind one domain) may
    put a page in a frame."""
    return "; ".join(
        [
            "default-src 'none'",
            "style-src 'self' https://fonts.googleapis.com",
            "font-src https://fonts.gstatic.com",
            "img-src 'self' data:",
            "base-uri 'none'",
            "form-action 'none'",
            f"frame-ancestors 'self' {get_api_settings().UI_ORIGIN}",
        ]
    )


@router.get("/{thread_id}/{version}/{path:path}")
async def preview_file(
    thread_id: str,
    version: Annotated[int, Path(ge=1)],
    path: str,
    request: Request,
) -> Response:
    """One file of one saved version. An empty path is the home page."""
    files = await store.site_files(request.app.state.sessions, thread_id, version)
    if not files:
        raise HTTPException(status_code=404, detail="Site version not found")
    name = path or "index.html"
    content = files.get(name)
    if content is None:
        raise HTTPException(status_code=404, detail="File not found")
    return Response(
        content,
        media_type=content_type_of(name),
        headers={
            "Content-Security-Policy": policy(),
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            # Version 1 can be replaced until the first edit, so the browser always asks again.
            "Cache-Control": "no-cache",
        },
    )
