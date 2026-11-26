"""Get one valid, checked output from a model. The retry loop of every worker is here."""

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from webkiln_agents.llm import StructuredCall
from webkiln_agents.workers.errors import WorkerOutputError

logger = logging.getLogger(__name__)

# A rejected answer may be sent back this many times, with the problems written out.
MAX_FIXES = 2

_PLACEHOLDER = re.compile(
    r"lorem ipsum|\bTBD\b|\bTODO\b|\bN/A\b|placeholder|\[(?:insert|your|add)[^\]]*\]|^\.{3}$",
    re.IGNORECASE,
)


def _strings(value: Any, path: str = "") -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, dict):
        return [
            item for key, v in value.items() for item in _strings(v, f"{path}.{key}".strip("."))
        ]
    if isinstance(value, list):
        return [item for i, v in enumerate(value) for item in _strings(v, f"{path}[{i}]")]
    return []


def check_output(output: BaseModel) -> list[str]:
    """Plain-code checks that a schema cannot express: blank text and placeholder text.

    Sizes and required fields are already enforced by the output's own model. Returns every
    problem found, ready to be pasted into the retry prompt.
    """
    problems: list[str] = []
    for path, text in _strings(output.model_dump(mode="json")):
        if not text.strip():
            problems.append(f"{path} is empty")
        elif _PLACEHOLDER.search(text.strip()):
            problems.append(f"{path} contains placeholder text: {text[:60]!r}")
    return problems


def _describe(error: Exception) -> list[str]:
    if isinstance(error, ValidationError):
        return [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()[:8]]
    return [f"the answer could not be parsed: {str(error)[:300]}"]


async def generate[T: BaseModel](
    call: StructuredCall[T],
    messages: list[BaseMessage],
    *,
    extra_check: Callable[[T], list[str]] | None = None,
    soft_check: Callable[[T], list[str]] | None = None,
) -> T:
    """Ask until the answer is valid and passes the checks, or raise `WorkerOutputError`.

    A parse error, a model that did not comply (`None`) and a failed check each cost one try.
    Other errors (a rate limit, an outage) are not caught: the graph's retry policy handles them.
    `extra_check` is a function from the output to a list of problems, for rules of one worker.
    `soft_check` is the same for style rules: its problems are sent back like the others, but if
    they are all that is left on the last try, the answer is accepted rather than failing the run.
    """
    problems: list[str] = []
    for attempt in range(MAX_FIXES + 1):
        prompt = messages
        if problems:
            fixes = "\n".join(f"- {p}" for p in problems)
            prompt = [
                *messages,
                HumanMessage(
                    "Your last answer was rejected. Fix every problem and answer again:\n" + fixes
                ),
            ]
        try:
            result = await call.ainvoke(prompt)
        except (ValidationError, OutputParserException, json.JSONDecodeError) as error:
            problems = _describe(error)
            continue
        if result is None:
            problems = ["you did not return the structured answer"]
            continue
        hard = check_output(result) + (extra_check(result) if extra_check else [])
        soft = soft_check(result) if soft_check else []
        problems = hard + soft
        if not problems:
            return result
        if not hard and attempt == MAX_FIXES:
            logger.warning("Accepting an answer with style problems: %s", "; ".join(soft))
            return result
    msg = "No valid output after retries: " + "; ".join(problems)
    raise WorkerOutputError(msg[:600])
