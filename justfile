agents:
  uv run --directory agents --package webkiln-agents langgraph dev

api:
  uv run --directory api --package webkiln-api fastapi dev

ui-types:
  uv run export-openapi-schema
  pnpm --dir ui exec openapi-typescript ../api/openapi.json --o lib/generated/schema.d.ts

db-up:
  docker compose up -d --wait db

db-down:
  docker compose down

db-migrate:
  uv run --directory api --package webkiln-api alembic upgrade head

db-revision message:
  uv run --directory api --package webkiln-api alembic revision --autogenerate -m "{{message}}"

check-structured-output:
  uv run --directory agents --package webkiln-agents check-structured-output

check-memory:
  uv run --directory agents --package webkiln-agents check-memory
