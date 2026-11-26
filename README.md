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
API restart. After the approval the wireframe agent plans the pages and sections, and the generate
step writes the copy of every page (one model call per page) and the notes for the pictures. Code
renders the pages from templates and checks the site before it is saved as version 1.

The finished site opens in a preview beside a chat. You can ask for a change in words: new words
for one section, new colours or fonts, a section added or removed, or a new page. One Manager call
reads the message with the last six chat turns and a line for each earlier edit, and code checks
the result against the site before anything is written. Every change is saved as a new version
that holds only the files that differ, you can open any version and restore it (a restore is a new
version too), and an edit made against an older version than the current one is refused. The
preview is a sandboxed iframe. A question, an unclear message and a refused edit are answered in
the chat and make no version. An edit that is cut off by an API restart is finished when the API
starts again.

The system also has a long-term memory, kept in Postgres with pgvector. When you approve the brand,
or ask for changes, it keeps the approved guideline, the research summaries, your hand edits, your
words and any lasting preferences it finds in them. A new run, even for another project, starts the
brand agent with the preferences and the project's own history, and the progress lines say what was
remembered. The memory also keeps the approved brand and the site's architecture (pages, sections
and menu) in one slot each. A colour or font edit, a structure edit and a restore replace the slot,
so a later session never recalls a colour or a page that the site no longer has, and the old
versions are kept. The chat router and the section writer read it when they work on an edit, and an
edit's decision and any lasting preference in the message are saved. A changed fact becomes a new version and the old one is kept. Memory belongs to one
browser: the UI makes an id once, keeps it in `localStorage` and sends it as `X-User-Id`. It is not
a login, and a run with no id has no memory. Memory is on when `GOOGLE_API_KEYS` is set, because the
embeddings come from Google, and off otherwise.

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

4. Check that your keys and model IDs work (free-tier model IDs change often), and that the
   embeddings behave as the memory expects:

   ```bash
   just check-structured-output
   just check-memory
   ```

   `check-memory` embeds a few pairs of sentences with your Google key and prints how similar each
   pair is. Use it to tune `SUPERSEDE_ABOVE` in `agents/src/webkiln_agents/memory/service.py`, the
   similarity at which a new preference replaces an old one.

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
- The memory has only been tested with fake embeddings and fake models. The 0.8 similarity at which
  a preference replaces an older one was tuned with `just check-memory` on six sentence pairs
  only, so it may need another look with more real preferences. The golden-query tests use fake embeddings, so they check the
  pipeline and say nothing about how well the real model ranks. The brand agent, the Manager's
  feedback node and the edit router and writer read memory. The chat history is not summarised by
  a model: the router reads the last six turns and a one-line digest of each earlier edit (the
  newest 30), so a long session can still lose the wording of a very old message.
- Chat edits cannot yet change the SEO of a page, the layout of a section or an image. The router
  says so and changes nothing.
- The site is plain HTML and CSS. Pictures are placeholders that keep a prompt and alt text for a
  real image later. The forms the wireframe asks for are drawn but post nowhere, because the
  site has no backend. Prices and testimonials in the copy are examples for the owner to replace,
  and the copy writer is told to leave out contact details and policies that the brief does not
  give.
- A generate task that fails (a page that cannot be written, or a site that fails its checks) and
  is started again writes every page again, so it spends one model call per page.
- The approval card and the resume endpoint have only been tested with fake models, not in a
  browser or against the live providers.
