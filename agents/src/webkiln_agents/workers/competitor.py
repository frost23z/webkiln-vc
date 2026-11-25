"""The competitor worker: web search first, then one model call that reads the results.

The queries are built in code, so the worker costs one model call. If search is not configured
or fails, the worker goes on with the model's own knowledge and flags the report `unsourced`.
"""

import asyncio
import ipaddress
import logging
from urllib.parse import urlparse

from webkiln_agents.manager.context import Context
from webkiln_agents.models import ConfigError
from webkiln_agents.search import SearchProvider, SearchResult
from webkiln_agents.workers.base import WorkerInput, WorkerResult
from webkiln_agents.workers.outputs import CompetitorAnalysis, CompetitorReport
from webkiln_agents.workers.research import ResearchSpec, block, research_messages
from webkiln_agents.workers.structured import generate

logger = logging.getLogger(__name__)

SPEC = ResearchSpec(
    "competitor",
    "competitor analyst",
    "Identify the real competitors of this business, with the address of each one's own site. "
    "Describe how they position themselves, the site structures and features they share, their "
    "content and messaging, and where this site can stand out. Use the search results when "
    "there are any, and name only businesses you can stand behind.",
    CompetitorAnalysis,
)

MAX_SEARCHES = 3  # a hard cap per task: each search costs one Tavily credit
MIN_COMPETITORS = 2
PROMPT_EXCERPT = 100  # characters of the user's prompt used in a query


def search_queries(brief: dict[str, str]) -> list[str]:
    """Up to `MAX_SEARCHES` queries from the brief, with no model call."""
    niche = " ".join(brief["niche"].split())
    wanted = " ".join(brief["prompt"].split())[:PROMPT_EXCERPT]
    return [
        f"top {niche} companies websites",
        f"{niche} {wanted} competitors",
        f"{niche} website features customers expect",
    ][:MAX_SEARCHES]


async def gather_results(search: SearchProvider, queries: list[str]) -> list[SearchResult]:
    """Run the queries together. A failed query is logged by type and skipped. The results are
    joined in query order, without repeated URLs."""
    answers = await asyncio.gather(*(search.search(q) for q in queries), return_exceptions=True)
    seen: set[str] = set()
    results: list[SearchResult] = []
    for answer in answers:
        if isinstance(answer, BaseException):
            logger.warning("competitor search failed: %s", type(answer).__name__)
            continue
        for item in answer:
            if item.url not in seen:
                seen.add(item.url)
                results.append(item)
    return results


def results_block(results: list[SearchResult]) -> str:
    entries = [f"[{i}] {r.title}\n{r.url}\n{r.content}" for i, r in enumerate(results, start=1)]
    return block("search_results", "\n\n".join(entries))


def is_site_url(url: str | None) -> bool:
    """Whether `url` is a public web address: http or https, with a real host name."""
    if not url:
        return False
    try:
        parsed = urlparse(url.strip())
        host = (parsed.hostname or "").lower()
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"} or "." not in host or host == "localhost":
        return False
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return not host.endswith((".local", ".internal", ".example", ".invalid", ".test"))
    return False  # an address of a machine is not a competitor's site


def check_competitors(report: CompetitorAnalysis, *, sourced: bool) -> list[str]:
    """With search, a competitor needs an address that parses. Too few left is a problem."""
    if not sourced:
        return []
    good = [c for c in report.competitors if is_site_url(c.url)]
    if len(good) >= MIN_COMPETITORS:
        return []
    return [
        f"only {len(good)} competitors have a valid https:// address, and {MIN_COMPETITORS} are "
        "needed: give each competitor the address of its own site"
    ]


async def competitor_worker(packet: WorkerInput, context: Context) -> WorkerResult:
    if context.llm is None:
        msg = "No model provider is configured for the competitor agent"
        raise ConfigError(msg)
    results: list[SearchResult] = []
    if context.search is not None:
        results = await gather_results(context.search, search_queries(packet["brief"]))
    sourced = bool(results)
    extra = (
        results_block(results)
        if sourced
        else block("search_results", "No search results are available for this task.")
    )
    call = context.llm.structured(SPEC.agent, CompetitorAnalysis)
    analysis = await generate(
        call,
        research_messages(SPEC, packet, extra),
        extra_check=lambda found: check_competitors(found, sourced=sourced),
    )
    fields = analysis.model_dump()
    if sourced:  # an entry without a valid address is dropped, as the plan says
        fields["competitors"] = [c.model_dump() for c in analysis.competitors if is_site_url(c.url)]
    report = CompetitorReport(**fields, unsourced=not sourced, sources=[r.url for r in results])
    summary = report.summary if sourced else f"(unsourced) {report.summary}"
    return WorkerResult(summary=summary[:1500], data=report.model_dump(mode="json"))
