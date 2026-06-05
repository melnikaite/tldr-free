"""Single-job Q&A — streaming with DuckDuckGo web search + page fetching.

Public surface:
    async def stream_answer(*, job, question: str, output_language: str)
        -> AsyncIterator[str | dict[str, Any]]

        Builds context = job.raw_text if it fits, else job.summary_md.

        Routing is language-agnostic and decided by the model itself, not
        by keyword matching. We offer TWO tools and require the model to
        pick exactly one (tool_choice="required"):

          - web_search(query): the answer is NOT in the material, OR the
            user wants external / current / online information, OR the user
            asked to search / look up / find something (in ANY language).
          - answer_from_material(): the answer IS in the provided material.

        Because the model must choose a tool, it can't "forget" to search
        or refuse with "the material doesn't contain that" — the decision
        is structural, not dependent on the model volunteering a tool call
        on tool_choice="auto" (which small models like Gemma 4 do unreliably).

        web_search path fetches real page content (trafilatura-cleaned) in
        parallel, not just DDG snippets.

        Yields:
          - str — token delta for the final answer
          - dict — stage event, e.g. {"type":"stage","stage":"searching","detail":"<query>"}

        Graceful degradation: if the backend doesn't support tool calling
        (or tool_choice="required"), falls back to a plain stream_complete.

Called from api/ai.py POST /ai/qa.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import get_config
from src.llm import client as llm_client
from src.llm.tokens import count_tokens
from src.workers import search as _search

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Reserve room for the prompt scaffolding, tool results, and the answer.
_PROMPT_OVERHEAD_TOKENS = 4000

# Two-tool router. The model MUST pick exactly one (tool_choice="required"),
# so the search-vs-material decision is structural rather than relying on the
# model to volunteer a web_search call on its own. The descriptions are
# phrased around USER INTENT so the model maps requests in any language to the
# right tool — no keyword lists, no per-language maintenance.
_WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web (DuckDuckGo) and read the resulting pages. Choose "
            "this when ANY of the following is true, regardless of the language "
            "the user writes in:\n"
            "- the answer is NOT clearly present in the provided material;\n"
            "- the user asks for current, recent, or external information "
            "(news, prices, reviews, what other people say, status, releases);\n"
            "- the user asks you to search, look up, find, google, or check "
            "something online.\n"
            "When in doubt between this and answer_from_material, prefer "
            "web_search — it is the safe fallback."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Concise search query in the user's language. Include "
                        "the topic from the material if it adds useful context."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

_ANSWER_FROM_MATERIAL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "answer_from_material",
        "description": (
            "Answer directly from the material that was provided to you, with no "
            "web search. Choose this ONLY when the material clearly contains the "
            "answer to the user's question and no external or current information "
            "is needed."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

_TOOLS = [_WEB_SEARCH_TOOL, _ANSWER_FROM_MATERIAL_TOOL]


@lru_cache(maxsize=1)
def _load_prompt() -> str:
    return (_PROMPTS_DIR / "qa.txt").read_text(encoding="utf-8")


def _select_context(job: Any) -> str:
    """Pick raw_text if it fits in the model's context; otherwise summary_md."""
    raw = getattr(job, "raw_text", None) or ""
    summary = getattr(job, "summary_md", None) or ""
    budget = get_config().llm.context_length - _PROMPT_OVERHEAD_TOKENS
    if raw and count_tokens(raw) <= budget:
        return raw
    return summary


def _build_messages(
    *,
    output_language: str,
    title: str,
    context: str,
    question: str,
) -> list[dict[str, Any]]:
    prompt = _load_prompt().format(
        output_language=output_language,
        title=title,
        context=context,
        question=question,
    )
    return [{"role": "user", "content": prompt}]


def _assistant_tool_call_msg(tc: Any) -> dict[str, Any]:
    """Build the assistant message echoing a single tool call for history."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
        ],
    }


def _pick_tool_call(tool_calls: list[Any]) -> Any | None:
    """Return the chosen tool call. web_search wins if the model emits both."""
    if not tool_calls:
        return None
    for tc in tool_calls:
        if tc.function.name == "web_search":
            return tc
    return tool_calls[0]


async def stream_answer(
    *,
    job: Any,
    question: str,
    output_language: str,
) -> AsyncIterator[str | dict[str, Any]]:
    """Yield token deltas (str) or stage dicts (dict) for a QA turn.

    Flow:
    1. Non-streaming call with both tools + tool_choice="required" — the model
       must pick web_search or answer_from_material (language-agnostic routing).
    2a. web_search → emit ``searching`` stage, fetch + clean pages, stream answer.
    2b. answer_from_material → stream the answer grounded in the material.
    3. Fallback: if the tool call raises (backend lacks tool support /
       "required"), stream a plain completion from the base prompt.
    """
    context = _select_context(job)
    title = getattr(job, "title", None) or ""
    messages = _build_messages(
        output_language=output_language,
        title=title,
        context=context,
        question=question,
    )

    # Step 1: forced tool choice — the model routes the request itself.
    try:
        response = await llm_client.complete_with_messages(
            messages,
            tools=_TOOLS,
            tool_choice="required",
            max_tokens=500,
            temperature=0.3,
        )
    except Exception:
        log.warning(
            "tool-capable request failed; falling back to plain stream", exc_info=True
        )
        async for delta in llm_client.stream_complete(
            messages[0]["content"],
            max_tokens=2000,
            temperature=0.3,
            respect_pause=False,
        ):
            yield delta
        return

    choice = response.choices[0]
    chosen = _pick_tool_call(choice.message.tool_calls or [])

    # Defensive: some backends honour "required" loosely and may return content
    # with no tool call. If so, just yield whatever the model said.
    if chosen is None:
        content = choice.message.content or ""
        if content:
            yield content
        return

    messages.append(_assistant_tool_call_msg(chosen))

    if chosen.function.name == "web_search":
        try:
            args = json.loads(chosen.function.arguments)
            query = args.get("query") or question
        except (json.JSONDecodeError, KeyError):
            query = question

        yield {"type": "stage", "stage": "searching", "detail": query}

        # Enriched search: fetch + trafilatura-clean the actual pages.
        results = await _search.ddg_search_with_content(query)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": chosen.id,
                "content": _search.format_results(results, full_content=True),
            }
        )
    else:
        # answer_from_material — acknowledge the tool and ask for the answer.
        messages.append(
            {
                "role": "tool",
                "tool_call_id": chosen.id,
                "content": "Now answer the user's question using the material above.",
            }
        )

    # Step 2: stream the grounded final answer (no tools this turn).
    async for delta in llm_client.stream_with_messages(
        messages, max_tokens=2000, temperature=0.3
    ):
        yield delta


__all__ = ["stream_answer"]
