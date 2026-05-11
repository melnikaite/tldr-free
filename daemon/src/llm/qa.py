"""Single-job Q&A — streaming with optional DuckDuckGo web search tool.

Public surface:
    async def stream_answer(*, job, question: str, output_language: str)
        -> AsyncIterator[str | dict[str, Any]]

        Builds context = job.raw_text if it fits else job.summary_md.
        Offers a ``web_search`` tool to the LLM; if invoked, runs DuckDuckGo,
        injects results into the conversation, then streams the final answer.

        Yields:
          - ``str`` — token delta for the final answer
          - ``dict`` — stage event, e.g. {"type": "stage", "stage": "searching",
            "detail": "<query>"}  (consumed by api/ai.py and forwarded as SSE)

        Graceful degradation: if the backend does not support tool calling
        (returns an error on the first call), falls back to plain
        stream_complete with the original prompt.

Called from ``api/ai.py`` POST /ai/qa.
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

# Reserve room for the prompt scaffolding + the answer.
_PROMPT_OVERHEAD_TOKENS = 4000

# Tool definition sent to the LLM on every QA call.
#
# Wording matters for small models (Gemma 4 E4B in particular). Be explicit
# about WHEN to call: the model will not invoke this unless triggers are
# spelled out concretely. "Search the web" alone is too generic — list the
# user intents (latest/news/today/find/check/source) that should fire it.
_WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web via DuckDuckGo. Call this tool whenever the answer "
            "to the user's question is not plainly stated in the material that "
            "was provided to you. This is the default fallback — do NOT respond "
            "with 'the material does not contain that' without first trying a "
            "web_search. Also call this tool for any current/recent information "
            "(news, prices, weather, releases, status) and whenever the user "
            "asks to search, look up, find, google, or check something. The "
            "ONLY time to skip the tool is when the answer is already clearly "
            "in the provided material."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Concise search query in the user's language. "
                        "Include the topic from the material if it adds context."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


@lru_cache(maxsize=1)
def _load_prompt() -> str:
    return (_PROMPTS_DIR / "qa.txt").read_text(encoding="utf-8")


def _select_context(job: Any) -> str:
    """Pick raw_text if it fits in the model's context; otherwise summary_md.

    Falls back to "" when neither is set.
    """
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


async def stream_answer(
    *,
    job: Any,
    question: str,
    output_language: str,
) -> AsyncIterator[str | dict[str, Any]]:
    """Yield token deltas (str) or stage dicts (dict) for a QA turn.

    Flow:
    1. Non-streaming call with the web_search tool offered → detect tool use.
    2a. Tool called → emit ``searching`` stage, run DDG, append results,
        stream the final answer.
    2b. No tool call → yield the direct answer content.
    3. Fallback: if step 1 raises (backend unsupported), stream without tools.
    """
    context = _select_context(job)
    title = getattr(job, "title", None) or ""
    messages = _build_messages(
        output_language=output_language,
        title=title,
        context=context,
        question=question,
    )

    # Step 1: non-streaming call with tools so we can detect tool invocations.
    try:
        response = await llm_client.complete_with_messages(
            messages,
            tools=[_WEB_SEARCH_TOOL],
            max_tokens=2000,
            temperature=0.3,
        )
    except Exception:
        log.warning("tool-capable request failed; falling back to plain stream", exc_info=True)
        async for delta in llm_client.stream_complete(
            messages[0]["content"],
            max_tokens=2000,
            temperature=0.3,
            respect_pause=False,
        ):
            yield delta
        return

    choice = response.choices[0]
    tool_calls = choice.message.tool_calls

    if tool_calls:
        # Append the assistant message (with tool_calls) to the history.
        messages.append(
            {
                "role": "assistant",
                "content": choice.message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            if tc.function.name != "web_search":
                continue
            try:
                args = json.loads(tc.function.arguments)
                query = args.get("query") or question
            except (json.JSONDecodeError, KeyError):
                query = question

            log.info("web_search tool called: %r", query)
            yield {"type": "stage", "stage": "searching", "detail": query}

            results = await _search.ddg_search(query)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _search.format_results(results),
                }
            )

        # Step 2a: stream the grounded final answer.
        async for delta in llm_client.stream_with_messages(
            messages,
            max_tokens=2000,
            temperature=0.3,
        ):
            yield delta

    else:
        # Step 2b: model answered directly — yield content as a single delta.
        content = choice.message.content or ""
        if content:
            yield content


__all__ = ["stream_answer"]
