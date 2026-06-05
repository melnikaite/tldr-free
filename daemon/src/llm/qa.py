"""Single-job Q&A — streaming with DuckDuckGo web search + page fetching.

Public surface:
    async def stream_answer(*, job, question: str, output_language: str)
        -> AsyncIterator[str | dict[str, Any]]

        Builds context = job.raw_text if it fits, else job.summary_md.
        Offers a ``web_search`` tool to the LLM. If invoked, fetches real
        page content (trafilatura-cleaned), injects into conversation, then
        streams the final answer.

        Two extra reliability layers on top of the basic tool-use flow:

        1. Forced search for explicit user intent: when the question
           contains clear search-request words ("поищи", "найди", "ищи",
           "search", "find", "look up", …) we set tool_choice="required"
           so the LLM *must* call the tool rather than answering from
           general knowledge or ignoring the request.

        2. Auto-retry on refusal: after a no-tool-call response, we scan
           the content for refusal patterns ("материал не содержит",
           "не нашёл", "not found", "cannot find", …). When detected we
           immediately retry with tool_choice="required" and the original
           question as the query. This catches cases where Gemma answers
           "I couldn't find it" without actually calling the tool.

        Yields:
          - str — token delta for the final answer
          - dict — stage event, e.g. {"type":"stage","stage":"searching","detail":"<query>"}

        Graceful degradation: if the backend doesn't support tool calling
        (raises on the first call), falls back to plain stream_complete.

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

# Tool definition sent to the LLM on every QA call.
#
# Wording matters for small models (Gemma 4 E4B in particular). Be explicit
# about WHEN to call: the model will not invoke this unless triggers are
# spelled out concretely.
_WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web via DuckDuckGo and retrieve full page content. "
            "Call this tool whenever the answer to the user's question is not "
            "plainly stated in the material provided to you. This is the default "
            "fallback — do NOT respond with 'the material does not contain that' "
            "without first trying a web_search. Also call this tool for any "
            "current/recent information (news, prices, weather, releases, status) "
            "and whenever the user asks to search, look up, find, google, or check "
            "something. The ONLY time to skip the tool is when the answer is "
            "already clearly in the provided material."
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

# Words / phrases in the user's question that signal an explicit intent to
# search the web. When present, we use tool_choice="required" so the model
# must call the tool instead of answering from memory or the material.
# Covers common patterns in Russian and English.
_EXPLICIT_SEARCH_PATTERNS: tuple[str, ...] = (
    # Russian
    "поищи", "поиск", "найди", "найти", "ищи", "искать",
    "загугли", "погугли", "проверь", "узнай в интернете",
    "ищи в интернете", "найди в интернете", "поищи в интернете",
    # English
    "search", "look up", "look it up", "find online", "google",
    "check online", "search the web", "search online",
)

# Patterns in the model's response that indicate it refused to search even
# though it probably should have. Used by the auto-retry logic.
_REFUSAL_PATTERNS: tuple[str, ...] = (
    # Russian
    "не содержит", "нет информации", "не нашёл", "не смог найти",
    "не могу найти", "отсутствует", "не упоминается",
    "в материале нет", "нет данных",
    # English
    "does not contain", "not found in", "cannot find", "couldn't find",
    "no information", "not mentioned", "not available in",
    "the material does not",
)


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


def _wants_search(question: str) -> bool:
    """Return True when the question contains an explicit search-request word."""
    lower = question.lower()
    return any(pat in lower for pat in _EXPLICIT_SEARCH_PATTERNS)


def _looks_like_refusal(content: str) -> bool:
    """Return True when the model's answer looks like a search refusal."""
    lower = content.lower()
    return any(pat in lower for pat in _REFUSAL_PATTERNS)


async def _run_search_and_append(
    query: str,
    tool_call_id: str,
    messages: list[dict[str, Any]],
) -> None:
    """Execute DDG search with full page fetching and append results to messages."""
    log.info("web_search tool called: %r", query)

    # Use enriched search (fetches real page content, not just snippets).
    results = await _search.ddg_search_with_content(query)
    content = _search.format_results(results, full_content=True)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
    )


async def stream_answer(
    *,
    job: Any,
    question: str,
    output_language: str,
) -> AsyncIterator[str | dict[str, Any]]:
    """Yield token deltas (str) or stage dicts (dict) for a QA turn.

    Flow:
    1. Non-streaming call with web_search tool offered.
       - If question contains explicit search intent → tool_choice="required"
         (forces the model to call the tool, no refusal possible).
       - Otherwise → tool_choice="auto" (model decides).
    2a. Tool called → emit ``searching`` stage, run DDG + fetch pages,
        inject cleaned content, stream final answer.
    2b. No tool call, but answer looks like a refusal → auto-retry with
        tool_choice="required" using the original question as the query.
    2c. No tool call, answer looks fine → yield content directly.
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

    # Determine tool_choice before the first call.
    force_search = _wants_search(question)
    tool_choice: str | dict[str, Any] = (
        {"type": "function", "function": {"name": "web_search"}}
        if force_search
        else "auto"
    )
    if force_search:
        log.info("qa: explicit search intent detected — tool_choice=required")

    # Step 1: non-streaming call to detect/trigger tool use.
    try:
        response = await llm_client.complete_with_messages(
            messages,
            tools=[_WEB_SEARCH_TOOL],
            tool_choice=tool_choice,
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
    direct_content = choice.message.content or ""

    # Step 2b: no tool call but response looks like a refusal → auto-retry.
    if not tool_calls and _looks_like_refusal(direct_content):
        log.info(
            "qa: model refused without searching; auto-retrying with forced search. "
            "Refusal snippet: %r",
            direct_content[:120],
        )
        yield {"type": "stage", "stage": "searching", "detail": question}

        # Build a synthetic tool call so _run_search_and_append can append results.
        fake_tool_id = "auto_retry_0"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": fake_tool_id,
                        "type": "function",
                        "function": {"name": "web_search", "arguments": json.dumps({"query": question})},
                    }
                ],
            }
        )
        await _run_search_and_append(question, fake_tool_id, messages)

        async for delta in llm_client.stream_with_messages(
            messages, max_tokens=2000, temperature=0.3
        ):
            yield delta
        return

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

            yield {"type": "stage", "stage": "searching", "detail": query}
            await _run_search_and_append(query, tc.id, messages)

        # Step 2a: stream the grounded final answer.
        async for delta in llm_client.stream_with_messages(
            messages, max_tokens=2000, temperature=0.3
        ):
            yield delta

    else:
        # Step 2c: model answered directly from material — yield as-is.
        if direct_content:
            yield direct_content


__all__ = ["stream_answer"]
