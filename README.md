# Webkiln

An AI website builder. You describe a site, and a Manager agent plans the work and runs a team of
agents on it. The repository holds three parts:

- `agents/`: the LangGraph graph (Manager, scheduler, research workers, failure handling).
- `api/`: the FastAPI service that runs the graph, stores runs in Postgres and streams progress.
- `ui/`: the Next.js app where you write the brief and follow the run.

What works today: the Manager analyses the brief, then the niche, audience, competitor and SEO
research agents run in parallel, and the UI shows their progress and reports. When a worker fails,
the Manager decides whether to retry it, skip it or ask you. The brand agent then writes a brand
guideline and the run pauses for your decision: approve it, edit colours, fonts, tagline or
positioning by hand, or ask for changes in words (up to four rounds). The paused run survives an
API restart. The wireframe and generation agents come in later milestones, so a run ends after the
approval for now.

## Requirements

Python 3.14 with [uv](https://docs.astral.sh/uv/), Node.js with [pnpm](https://pnpm.io/),
[just](https://just.systems/) and Docker (for Postgres).

## Setup

1. Copy `.env.example` to `.env` and add your keys. Every key is optional, but a research agent
   needs at least one provider from its model chain (Google, Groq or OpenRouter). Without
   `TAVILY_API_KEY` the competitor report is flagged as unsourced.
2. Copy `ui/.env.example` to `ui/.env.local` if the API is not at `http://localhost:8000`.
3. Install the dependencies:

   ```bash
   uv sync --all-packages
   pnpm --dir ui install
   ```

4. Check that your keys and model IDs work (free-tier model IDs change often):

   ```bash
   just check-structured-output
   ```

## Run

```bash
just db-up                  # Postgres in Docker
just api                    # the API on :8000, creates its tables on startup
pnpm --dir ui dev           # the UI on :3000
```

`just agents` starts LangGraph Studio for working on the graph alone.

If you change the API's models, regenerate the UI types with `just ui-types`. After a change to
the tables, `just db-revision "message"` writes a migration. The API applies migrations itself, so
`just db-migrate` is only needed to run them without starting the API.

## Check

```bash
uv run pytest agents api   # needs the database from `just db-up`
uvx ruff check . && uvx ruff format --check .
uv run pyrefly check
pnpm --dir ui exec tsc --noEmit && pnpm --dir ui lint
```

## Limits

- Nothing here has been run against the live providers by the build process. Run
  `just check-structured-output` with your keys before the first real run.
- Only the research and brand agents have workers. The wireframe and the site arrive in the next
  milestones, so an approved run ends with those tasks still pending.
- The approval card and the resume endpoint have only been tested with fake models, not in a
  browser or against the live providers.
