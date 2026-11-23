from collections.abc import Generator
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from webkiln_api.app import create_app

BRIEF = {
    "brand_name": "Cedar Stay",
    "prompt": "A boutique hotel website with booking",
    "niche": "Hospitality",
}


@pytest.fixture
def client() -> Generator[TestClient]:
    with TestClient(create_app()) as client:
        yield client


def test_valid_brief_returns_a_thread_id(client: TestClient) -> None:
    response = client.post("/runs", json=BRIEF)

    assert response.status_code == 201
    body = response.json()
    UUID(body["thread_id"])  # raises if it is not a uuid
    assert body["status"] == "created"


def test_each_run_gets_its_own_thread_id(client: TestClient) -> None:
    first = client.post("/runs", json=BRIEF).json()["thread_id"]
    second = client.post("/runs", json=BRIEF).json()["thread_id"]

    assert first != second


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("brand_name", "x"),
        ("brand_name", "   "),
        ("prompt", "too short"),
        ("niche", "n" * 81),
    ],
)
def test_invalid_brief_returns_422(client: TestClient, field: str, value: str) -> None:
    response = client.post("/runs", json={**BRIEF, field: value})

    assert response.status_code == 422


def test_missing_field_returns_422(client: TestClient) -> None:
    response = client.post("/runs", json={"brand_name": "Cedar Stay"})

    assert response.status_code == 422
