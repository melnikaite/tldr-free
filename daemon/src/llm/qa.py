"""Single-job Q&A — layered answering: material → AI knowledge → web.

Public surface:
    async def stream_answer(*, job, question: str, output_language: str)
        -> AsyncIterator[str | dict[str, Any]]

        Builds context = job.raw_text if it fits, else job.summary_md.

        Flow (the model never gets to veto a search with a single tool pick —
        that is the failure mode small models like Gemma 4 abuse, refusing to
        search at the slightest excuse):

          1. PLAN — one forced `plan` tool call returns
             {material_sufficient: bool, search_query: str}. The bar for
             `material_sufficient=true` is high: the material must explicitly
             and completely answer. Anything else — uncertainty, a missing
             specific detail, a parse error, a backend without tool support —
             defaults to SEARCHING. An extra web trip is cheap; a missed one
             leaves the user short.
          2. SEARCH — when not sufficient, fetch + trafilatura-clean the DDG
             results (real page content, not just snippets).
          3. SYNTHESIS — stream the answer grounded in, by priority, the
             material, then the model's own knowledge, then the web results
             (cited). The prompt actively encourages going beyond the material.

        Yields:
          - str — token delta for the final answer
          - dict — stage event, e.g. {"type":"stage","stage":"searching","detail":"<query>"}

Called from api/ai.py POST /ai/qa.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import get_config
from src.llm import client as llm_client
from src.llm.tokens import count_tokens
from src.workers import search as _search
from src.workers import timecodes as _timecodes

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Reserve room for the prompt scaffolding, tool results, and the answer.
_PROMPT_OVERHEAD_TOKENS = 4000

# Single forced tool. The model must fill in the plan — it cannot refuse to
# decide. material_sufficient is deliberately strict; we treat anything other
# than a clean True as "search".
_PLAN_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "plan",
        "description": (
            "Decide whether the provided material alone fully and specifically "
            "answers the user's question, and propose a web search query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "material_sufficient": {
                    "type": "boolean",
                    "description": (
                        "true ONLY if the material explicitly contains the "
                        "complete, specific answer to every part of the "
                        "question. false if the material only mentions the "
                        "topic but lacks the specific fact, number, name, or "
                        "detail asked for, or if current/external/additional "
                        "information would help. When in doubt, false."
                    ),
                },
                "search_query": {
                    "type": "string",
                    "description": (
                        "A concise web search query in the user's language to "
                        "find or enrich the answer online. Always provide one."
                    ),
                },
            },
            "required": ["material_sufficient", "search_query"],
        },
    },
}

# A line that is nothing but HTML tags + whitespace — a run of <br>, a stray
# </blockquote>, etc. A small local model degenerating at the end of an answer
# emits these as filler. A line with real text between tags (e.g. a link,
# "<a ...>Источник</a>") does NOT match, so genuine content survives; the
# frequency penalty on generation is what stops the loop producing them.
_MARKUP_ONLY_LINE = re.compile(r"^\s*(?:<[^>]+>\s*)+$")


def clean_answer(text: str) -> str:
    """Strip degenerate filler a small model appends to a Q&A answer.

    Observed tail-noise that carries no information:
      - a dump of bare ``[MM:SS]`` markers, one per line (the whole transcript);
      - lines made only of HTML tags (runs of ``<br>``, a stray ``</blockquote>``).
    We drop both, collapse the blank runs left behind, and trim. Lines with real
    text — an inline timecode next to a sentence, a link with a label — are kept.

    Pure: same input → same output.
    """
    if not text:
        return text
    cleaned = _timecodes.strip_bare_timecode_lines(text)
    if "<" in cleaned:
        kept = [ln for ln in cleaned.split("\n") if not _MARKUP_ONLY_LINE.match(ln)]
        cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return cleaned


@lru_cache(maxsize=4)
def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _select_context(job: Any) -> str:
    """Pick raw_text if it fits in the model's context; otherwise summary_md."""
    raw = getattr(job, "raw_text", None) or ""
    summary = getattr(job, "summary_md", None) or ""
    budget = get_config().llm.context_length - _PROMPT_OVERHEAD_TOKENS
    if raw and count_tokens(raw) <= budget:
        return raw
    return summary


def _plan_context(job: Any, *, fallback: str) -> str:
    """Compact context for the cheap PLAN call — prefer the summary over the
    full transcript. The plan only judges sufficiency + drafts a query, so the
    summary is enough; it also halves the plan-call prefill and biases toward
    searching (the summary omits detail), which is the behaviour we want."""
    return getattr(job, "summary_md", None) or fallback


def _plan_messages(*, title: str, context: str, question: str) -> list[dict[str, Any]]:
    prompt = _load_prompt("qa_plan.txt").format(
        title=title, context=context, question=question
    )
    return [{"role": "user", "content": prompt}]


def _answer_messages(
    *,
    output_language: str,
    title: str,
    context: str,
    question: str,
    web_results: str,
) -> list[dict[str, Any]]:
    prompt = _load_prompt("qa.txt").format(
        output_language=output_language,
        title=title,
        context=context,
        question=question,
        web_results=web_results or "(no web search was run)",
    )
    return [{"role": "user", "content": prompt}]


def _parse_plan(response: Any) -> tuple[bool, str]:
    """Extract (material_sufficient, search_query) from the plan tool call.

    Defaults to (False, "") — i.e. SEARCH — on any malformed / missing call,
    so a flaky tool response biases toward searching rather than dodging it.
    """
    try:
        tool_calls = response.choices[0].message.tool_calls or []
    except (AttributeError, IndexError):
        return False, ""
    for tc in tool_calls:
        if tc.function.name != "plan":
            continue
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, TypeError):
            return False, ""
        sufficient = args.get("material_sufficient")
        query = args.get("search_query") or ""
        # Strict: only an explicit boolean True counts as sufficient.
        return (sufficient is True), str(query).strip()
    return False, ""


async def stream_answer(
    *,
    job: Any,
    question: str,
    output_language: str,
) -> AsyncIterator[str | dict[str, Any]]:
    """Yield token deltas (str) or stage dicts (dict) for a QA turn.

    See module docstring for the plan → search → synthesis flow.
    """
    context = _select_context(job)
    title = getattr(job, "title", None) or ""

    # Step 1: PLAN. Forced single tool — the model must decide; we bias to
    # search on any failure. Judged against the compact summary, not the full
    # transcript, to keep this call cheap.
    sufficient = False
    query = question
    try:
        plan = await llm_client.complete_with_messages(
            _plan_messages(
                title=title,
                context=_plan_context(job, fallback=context),
                question=question,
            ),
            tools=[_PLAN_TOOL],
            tool_choice={"type": "function", "function": {"name": "plan"}},
            max_tokens=300,
            temperature=0.0,
        )
        sufficient, parsed_query = _parse_plan(plan)
        if parsed_query:
            query = parsed_query
    except Exception:
        # Backend lacks tool support / errored — default to searching.
        log.warning("QA plan call failed; defaulting to web search", exc_info=True)
        sufficient = False

    # Step 2: SEARCH unless the material clearly suffices on its own.
    web_results = ""
    if not sufficient:
        yield {"type": "stage", "stage": "searching", "detail": query}
        try:
            results = await _search.ddg_search_with_content(query)
            web_results = _search.format_results(results, full_content=True)
        except Exception:
            log.warning("QA web search failed; answering without it", exc_info=True)
            web_results = ""

    # Step 3: SYNTHESIS. Stream the grounded answer (material → knowledge → web).
    messages = _answer_messages(
        output_language=output_language,
        title=title,
        context=context,
        question=question,
        web_results=web_results,
    )
    async for delta in llm_client.stream_with_messages(
        messages,
        max_tokens=2000,
        temperature=0.3,
    ):
        yield delta


__all__ = ["clean_answer", "stream_answer"]
