"""Page text extraction fallback.

Public surface:
    async def extract_with_trafilatura(url: str) -> tuple[str | None, str]
        Used only when the extension didn't supply ``page_text``. Fetches the
        URL via trafilatura's downloader, extracts main text and (optionally)
        the page title from metadata. Returns ``(title, text)``; either may be
        empty / None.

Trafilatura is synchronous; the public surface here is async — we wrap the
blocking calls with ``asyncio.to_thread`` so the event loop doesn't stall.
"""

from __future__ import annotations

import asyncio
import logging

import trafilatura

log = logging.getLogger(__name__)


def _extract_sync(url: str) -> tuple[str | None, str]:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return (None, "")

    # ``with_metadata=True`` instructs trafilatura to inject metadata at the
    # top of the output. We instead read structured metadata for the title and
    # extract the body separately so the title isn't entangled with the text.
    try:
        meta = trafilatura.extract_metadata(downloaded)
        title = (meta.title if meta is not None else None) or None
    except Exception:
        # ``extract_metadata`` is best-effort; never let it block extraction.
        title = None

    text = trafilatura.extract(
        downloaded,
        output_format="txt",
        include_comments=False,
        include_tables=True,
    ) or ""

    return (title, text.strip())


async def extract_with_trafilatura(url: str) -> tuple[str | None, str]:
    """Fetch + extract a page; returns ``(title, text)``."""
    return await asyncio.to_thread(_extract_sync, url)


__all__ = ["extract_with_trafilatura"]
