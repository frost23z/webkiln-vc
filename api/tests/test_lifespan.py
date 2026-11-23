from fastapi.testclient import TestClient
from webkiln_api.app import create_app


def test_graph_is_compiled_once_at_startup() -> None:
    app = create_app()

    with TestClient(app) as client:
        graph = app.state.graph
        client.get("/health")
        client.get("/health")

        assert graph is not None
        assert app.state.graph is graph  # not rebuilt per request
