"""Tests for workers.search — DDG search + parallel page fetch + cleaning.

The DDG call and httpx fetching are mocked; we verify the orchestration:
- ddg_search_with_content enriches each result with cleaned page text,
- pages that fail to fetch fall back to the DDG snippet,
- format_results(full_content=True) uses the cleaned content.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.workers import search as search_mod


@pytest.mark.asyncio
async def test_search_with_content_enriches_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each DDG result gets a ``content`` field from the fetched+cleaned page."""

    async def fake_ddg(query: str, max_results: int = 5) -> list[dict[str, Any]]:
        return [
            {"title": "A", "href": "https://a.test", "body": "snippet A"},
            {"title": "B", "href": "https://b.test", "body": "snippet B"},
        ]

    async def fake_fetch(url: str, client: Any) -> str | None:
        return f"cleaned content from {url}"

    monkeypatch.setattr(search_mod, "ddg_search", fake_ddg)
    monkeypatch.setattr(search_mod, "_fetch_and_clean", fake_fetch)

    results = await search_mod.ddg_search_with_content("anything")

    assert len(results) == 2
    assert results[0]["content"] == "cleaned content from https://a.test"
    assert results[1]["content"] == "cleaned content from https://b.test"


@pytest.mark.asyncio
async def test_search_with_content_falls_back_to_snippet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a page fetch returns None, content falls back to the DDG snippet."""

    async def fake_ddg(query: str, max_results: int = 5) -> list[dict[str, Any]]:
        return [
            {"title": "OK", "href": "https://ok.test", "body": "snippet OK"},
            {"title": "Fail", "href": "https://fail.test", "body": "snippet FAIL"},
        ]

    async def fake_fetch(url: str, client: Any) -> str | None:
        return "real content" if "ok" in url else None

    monkeypatch.setattr(search_mod, "ddg_search", fake_ddg)
    monkeypatch.setattr(search_mod, "_fetch_and_clean", fake_fetch)

    results = await search_mod.ddg_search_with_content("anything")

    assert results[0]["content"] == "real content"
    # Failed fetch → fall back to the snippet body.
    assert results[1]["content"] == "snippet FAIL"


@pytest.mark.asyncio
async def test_search_with_content_empty_when_no_ddg_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_ddg(query: str, max_results: int = 5) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(search_mod, "ddg_search", fake_ddg)
    results = await search_mod.ddg_search_with_content("anything")
    assert results == []


def test_format_results_full_content_uses_content_field() -> None:
    results = [
        {"title": "T1", "href": "https://1.test", "body": "snip", "content": "FULL ONE"},
        {"title": "T2", "href": "https://2.test", "body": "snip2", "content": "FULL TWO"},
    ]
    out = search_mod.format_results(results, full_content=True)
    assert "FULL ONE" in out
    assert "FULL TWO" in out
    assert "snip" not in out  # snippet not used in full-content mode


def test_format_results_snippet_mode_uses_body() -> None:
    results = [{"title": "T", "href": "https://x.test", "body": "the snippet", "content": "ignored"}]
    out = search_mod.format_results(results, full_content=False)
    assert "the snippet" in out
    assert "ignored" not in out


def test_format_results_empty() -> None:
    assert search_mod.format_results([]) == "No results found."
