"""DuckDuckGo web search — async wrapper around the synchronous DDGS API.

Used as a tool in the Q&A pipeline when the LLM decides external
information would improve the answer. No API key required.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)

_MAX_RESULTS = 5


async def ddg_search(query: str, max_results: int = _MAX_RESULTS) -> list[dict[str, Any]]:
    """Run a DuckDuckGo text search; return up to ``max_results`` results.

    Each result dict has keys: ``title``, ``href``, ``body``.

    Runs the synchronous DDGS API in a thread-pool executor so the event loop
    is never blocked. Returns an empty list on any error.
    """

    def _sync() -> list[dict[str, Any]]:
        from ddgs import DDGS  # imported here to keep startup fast

        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _sync)
    except Exception:
        log.exception("DuckDuckGo search failed for query %r", query)
        return []


def format_results(results: list[dict[str, Any]]) -> str:
    """Format search results as a plain-text block suitable for LLM context."""
    if not results:
        return "No results found."
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        href = r.get("href", "")
        body = r.get("body", "")
        lines.append(f"{i}. {title}\n   {href}\n   {body}")
    return "\n\n".join(lines)
