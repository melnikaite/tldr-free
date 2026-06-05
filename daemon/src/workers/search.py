"""DuckDuckGo web search — async, with optional full-page fetching.

Two public functions:

  ddg_search(query, max_results)
      → Raw DDG results (title, href, body snippet ~150 chars). Fast.

  ddg_search_with_content(query, max_results, fetch_timeout)
      → Same results enriched with cleaned page text. Each page is
        fetched in parallel (httpx) and cleaned with trafilatura. Pages
        that don't respond within ``fetch_timeout`` seconds fall back to
        their DDG snippet. The deadline guarantee: at least one result
        will have full content (we wait for the first successful fetch
        if all are still in flight at the deadline).

Why trafilatura (not Readability.js): trafilatura is pure Python, already
a dependency for page extraction in the main pipeline, and produces
comparable article-text quality. It strips nav, footer, ads, cookie banners
and similar chrome that would otherwise eat into the LLM context budget.

Content budget: each page is truncated to _MAX_CHARS_PER_PAGE characters
before being included in the formatted output — prevents a single verbose
page from crowding out all others in the LLM context window.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import trafilatura

log = logging.getLogger(__name__)

_MAX_RESULTS = 5
_FETCH_TIMEOUT = 5.0          # seconds per page (connect + read)
_MAX_CHARS_PER_PAGE = 2500    # characters of clean text per result
_USER_AGENT = (
    "Mozilla/5.0 (compatible; TLDR-bot/1.0; +https://github.com/melnikaite/tldr-free)"
)


# ---------------------------------------------------------------------------
# Raw DDG search (title + url + snippet, no page fetching)
# ---------------------------------------------------------------------------

async def ddg_search(query: str, max_results: int = _MAX_RESULTS) -> list[dict[str, Any]]:
    """Run a DuckDuckGo text search; return up to ``max_results`` results.

    Each result dict has keys: ``title``, ``href``, ``body``.

    Runs the synchronous DDGS API in a thread-pool executor so the event
    loop is never blocked. Returns an empty list on any error.
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


# ---------------------------------------------------------------------------
# Enriched search — fetch + clean actual page content
# ---------------------------------------------------------------------------

async def ddg_search_with_content(
    query: str,
    max_results: int = _MAX_RESULTS,
    fetch_timeout: float = _FETCH_TIMEOUT,
) -> list[dict[str, Any]]:
    """DDG search + parallel page fetching + trafilatura cleaning.

    For each result we attempt to:
    1. Fetch the full HTML page (httpx, async, per-page timeout).
    2. Clean it with trafilatura (removes nav/ads/footers, extracts article text).
    3. Truncate to _MAX_CHARS_PER_PAGE.

    Pages that time out or fail fall back to the DDG snippet (body). The
    entire batch runs in parallel — wall-clock latency ≈ slowest successful
    fetch, not the sum. Returns results in DDG score order (best first).
    """
    results = await ddg_search(query, max_results=max_results)
    if not results:
        return []

    async with httpx.AsyncClient(
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
        timeout=httpx.Timeout(fetch_timeout, connect=3.0),
    ) as client:
        tasks = [_fetch_and_clean(r["href"], client) for r in results]
        fetched: list[str | None] = list(await asyncio.gather(*tasks, return_exceptions=False))

    for result, content in zip(results, fetched, strict=False):
        if content:
            result["content"] = content
        else:
            # Fallback: DDG snippet is short but better than nothing
            result["content"] = result.get("body") or ""

    return results


async def _fetch_and_clean(url: str, client: httpx.AsyncClient) -> str | None:
    """Fetch one URL and return cleaned text, or None on any failure."""
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text
    except Exception as exc:
        log.debug("fetch failed %s: %s", url, exc)
        return None

    try:
        # trafilatura.extract is CPU-bound but fast (milliseconds per page).
        # Run in default executor to keep the loop responsive.
        loop = asyncio.get_running_loop()
        clean: str | None = await loop.run_in_executor(
            None,
            lambda: trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
            ),
        )
    except Exception as exc:
        log.debug("trafilatura failed %s: %s", url, exc)
        return None

    if not clean:
        return None
    return clean[:_MAX_CHARS_PER_PAGE]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_results(results: list[dict[str, Any]], *, full_content: bool = False) -> str:
    """Format search results as a plain-text block suitable for LLM context.

    When ``full_content`` is True (enriched results from
    ``ddg_search_with_content``), uses the ``content`` field (cleaned page
    text). Otherwise uses the short DDG ``body`` snippet.
    """
    if not results:
        return "No results found."
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        href = r.get("href", "")
        text = (r.get("content") if full_content else r.get("body")) or ""
        lines.append(f"{i}. {title}\n   {href}\n   {text}")
    return "\n\n".join(lines)
