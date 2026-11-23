agents:
  uv run --directory agents --package webkiln-agents langgraph dev

api:
  uv run --directory api --package webkiln-api fastapi dev
