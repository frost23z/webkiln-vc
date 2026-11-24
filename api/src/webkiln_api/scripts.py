import json
from pathlib import Path

from webkiln_api.main import app


def export_openapi_schema() -> None:
    schema = app.openapi()

    output = Path(__file__).resolve().parents[2] / "openapi.json"
    output.write_text(
        json.dumps(schema, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"OpenAPI schema exported to {output}")
