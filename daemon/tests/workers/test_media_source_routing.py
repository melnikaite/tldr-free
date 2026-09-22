"""Tests for workers.pipeline._run_media's URL-routing rule.

For kind=media jobs the daemon receives both the page ``url`` and the
``media_url`` the extension scraped from the DOM. ``_run_media`` decides
which one to hand to yt-dlp for captions/title/Whisper: the page URL wins
only when it differs from ``media_url`` AND yt-dlp has a dedicated
extractor for it (checked locally via ``youtube.has_dedicated_extractor``)
AND a metadata probe of the page URL confirms a real, single-video result
(non-empty, not a playlist, extractor not ``generic``). Otherwise
``media_url`` is used, unchanged from pre-existing behaviour.

These tests never touch real yt-dlp/network — ``youtube.has_dedicated_extractor``,
``youtube.fetch_video_metadata`` and ``youtube.download_subtitles`` are all
monkeypatched, and ``_finish_caption_fast_path`` is monkeypatched too so we
can assert on the URL it receives without needing a real LLM summarizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.storage import repo
from src.storage.db import dispose_engine, init_engine
from src.storage.migrations import run_migrations
from src.workers import broker as broker_mod
from src.workers import control as control_mod
from src.workers import pipeline as pipeline_mod
from src.workers import queue as queue_mod
from src.workers.queue import get_queue

PAGE_URL = "https://www.zdf.de/play/serien/spaeti-102/grumpy-elster-100"
MEDIA_URL = (
    "https://nrodlzdf-a.akamaihd.net/dach/zdf/25/04/250415_2145_sendung_sae/"
    "1/250415_2145_sendung_sae_a1a2_4328k_p19v17.webm"
)


@pytest.fixture(autouse=True)
def _reset_singletons() -> Any:
    control_mod.reset_control()
    queue_mod.reset_queue()
    broker_mod.reset_broker()
    yield
    control_mod.reset_control()
    queue_mod.reset_queue()
    broker_mod.reset_broker()


@pytest.fixture
def isolated_db(tmp_path: Path) -> Any:
    db_path = tmp_path / "media_routing.db"
    engine = init_engine(db_path)
    run_migrations(engine)
    try:
        yield engine
    finally:
        dispose_engine()


def _make_job(isolated_db: Any) -> Any:
    return repo.create_job(
        url=PAGE_URL, kind="media", title="Grumpy Elster (scraped)", media_url=MEDIA_URL,
    )


async def _run(
    monkeypatch: pytest.MonkeyPatch,
    isolated_db: Any,
    *,
    has_dedicated: bool,
    metadata: dict[str, Any] | None,
    caption_segments: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Run ``_run_media`` for a fresh job with the given fakes wired in.

    Returns a dict capturing every URL each downstream consumer saw:
    ``download_subtitles_url`` and either ``finish_caption_fast_path_url``
    (captions found) or ``whisper_task_url`` (captions absent).
    """
    job = _make_job(isolated_db)

    calls: dict[str, Any] = {}

    def fake_has_dedicated_extractor(url: str) -> bool:
        calls["has_dedicated_extractor_url"] = url
        return has_dedicated

    async def fake_fetch_video_metadata(*, url: str, cookies: Any, scratch_dir: Any) -> dict[str, Any]:
        calls.setdefault("fetch_video_metadata_urls", []).append(url)
        return dict(metadata) if metadata is not None else {}

    async def fake_download_subtitles(*, url: str, **_kwargs: Any) -> list[dict[str, Any]] | None:
        calls["download_subtitles_url"] = url
        return caption_segments

    async def fake_finish_caption_fast_path(job_id: str, *, url: str, **_kwargs: Any) -> None:
        calls["finish_caption_fast_path_url"] = url

    monkeypatch.setattr(pipeline_mod.youtube, "has_dedicated_extractor", fake_has_dedicated_extractor)
    monkeypatch.setattr(pipeline_mod.youtube, "fetch_video_metadata", fake_fetch_video_metadata)
    monkeypatch.setattr(pipeline_mod.youtube, "download_subtitles", fake_download_subtitles)
    monkeypatch.setattr(pipeline_mod, "_finish_caption_fast_path", fake_finish_caption_fast_path)

    await pipeline_mod._run_media(
        job.id,
        url=PAGE_URL,
        media_url=MEDIA_URL,
        page_title="Grumpy Elster (scraped)",
        page_text=None,
        cookies=[],
    )

    if caption_segments is None:
        queue = get_queue()
        task = await queue.get()
        calls["whisper_task_url"] = task.url

    return calls


@pytest.mark.asyncio
async def test_dedicated_extractor_single_video_page_url_wins_for_captions(
    monkeypatch: pytest.MonkeyPatch, isolated_db: Any,
) -> None:
    calls = await _run(
        monkeypatch, isolated_db,
        has_dedicated=True,
        metadata={"title": "Grumpy Elster", "extractor": "zdf", "is_playlist": False},
        caption_segments=[{"start": 0.0, "duration": 2.0, "text": "hallo"}],
    )
    assert calls["has_dedicated_extractor_url"] == PAGE_URL
    assert calls["fetch_video_metadata_urls"] == [PAGE_URL]
    assert calls["download_subtitles_url"] == PAGE_URL
    assert calls["finish_caption_fast_path_url"] == PAGE_URL


@pytest.mark.asyncio
async def test_dedicated_extractor_single_video_page_url_wins_for_whisper(
    monkeypatch: pytest.MonkeyPatch, isolated_db: Any,
) -> None:
    """Same resolved source, but no captions found -> Whisper still gets the page URL."""
    calls = await _run(
        monkeypatch, isolated_db,
        has_dedicated=True,
        metadata={"title": "Grumpy Elster", "extractor": "zdf", "is_playlist": False},
        caption_segments=None,
    )
    assert calls["download_subtitles_url"] == PAGE_URL
    assert calls["whisper_task_url"] == PAGE_URL


@pytest.mark.asyncio
async def test_metadata_probe_empty_dict_keeps_media_url(
    monkeypatch: pytest.MonkeyPatch, isolated_db: Any,
) -> None:
    calls = await _run(
        monkeypatch, isolated_db,
        has_dedicated=True,
        metadata={},
        caption_segments=None,
    )
    assert calls["fetch_video_metadata_urls"] == [PAGE_URL]
    assert calls["download_subtitles_url"] == MEDIA_URL
    assert calls["whisper_task_url"] == MEDIA_URL


@pytest.mark.asyncio
async def test_metadata_probe_playlist_keeps_media_url(
    monkeypatch: pytest.MonkeyPatch, isolated_db: Any,
) -> None:
    calls = await _run(
        monkeypatch, isolated_db,
        has_dedicated=True,
        metadata={"title": "Some Playlist", "extractor": "youtube:tab", "is_playlist": True},
        caption_segments=None,
    )
    assert calls["download_subtitles_url"] == MEDIA_URL
    assert calls["whisper_task_url"] == MEDIA_URL


@pytest.mark.asyncio
async def test_metadata_probe_generic_extractor_keeps_media_url(
    monkeypatch: pytest.MonkeyPatch, isolated_db: Any,
) -> None:
    calls = await _run(
        monkeypatch, isolated_db,
        has_dedicated=True,
        metadata={"title": "250415_2145_sendung_sae_a1a2_4328k_p19v17", "extractor": "generic", "is_playlist": False},
        caption_segments=None,
    )
    assert calls["download_subtitles_url"] == MEDIA_URL
    assert calls["whisper_task_url"] == MEDIA_URL


@pytest.mark.asyncio
async def test_no_dedicated_extractor_keeps_media_url_and_skips_probe(
    monkeypatch: pytest.MonkeyPatch, isolated_db: Any,
) -> None:
    calls = await _run(
        monkeypatch, isolated_db,
        has_dedicated=False,
        metadata={"title": "irrelevant", "extractor": "vimeo", "is_playlist": False},
        caption_segments=None,
    )
    assert calls["has_dedicated_extractor_url"] == PAGE_URL
    # The metadata probe must never run when there's no dedicated extractor.
    assert "fetch_video_metadata_urls" not in calls
    assert calls["download_subtitles_url"] == MEDIA_URL
    assert calls["whisper_task_url"] == MEDIA_URL
