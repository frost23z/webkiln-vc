# Webkiln API

FastAPI service that runs the agents graph. See the README in the repository root for setup.

- `POST /runs` starts a run, `GET /runs/{id}` returns it, `GET /runs/{id}/stream` streams its
  events and `POST /runs/{id}/retry` runs its failed or waiting tasks again.
- Runs and their task outputs are stored in Postgres. A run that was interrupted by a restart
  continues at the next start.
- The tables are created and updated on startup (Alembic). Settings come from `.env` here or one
  folder up: `DB_URL`, `UI_ORIGIN` and the agents' keys.
