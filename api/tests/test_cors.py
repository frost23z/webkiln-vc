"""The browser asks before it sends a cross-origin request that carries a custom header."""

from fastapi.testclient import TestClient
from webkiln_api.app import create_app
from webkiln_api.config import get_api_settings


def preflight(headers: str, method: str = "POST") -> tuple[int, str | None]:
    origin = get_api_settings().UI_ORIGIN
    client = TestClient(create_app())
    response = client.options(
        "/runs/some-thread/resume",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": headers,
        },
    )
    return response.status_code, response.headers.get("access-control-allow-origin")


def test_the_resume_request_of_the_ui_passes_the_preflight() -> None:
    # The UI sends this header on every Approve and Request changes click (ui/lib/api/client.ts).
    status, allowed_origin = preflight("content-type,idempotency-key")

    assert status == 200
    assert allowed_origin == get_api_settings().UI_ORIGIN


def test_a_header_the_api_does_not_know_is_still_refused() -> None:
    status, _ = preflight("content-type,x-unknown")

    assert status == 400
