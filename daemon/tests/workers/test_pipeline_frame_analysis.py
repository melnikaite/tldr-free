"""Tests for workers.pipeline._run_frame_analysis — the summary-time
frame-analysis step (stage 2 of the video-frame-understanding feature).

Mocks ``workers.deixis.candidates_for_job`` (never touching the real
regex/segment-parsing implementation) and ``llm.vision.fetch_moment_frames``
/ ``llm.vision.analyze_summary_frames`` (never touching real yt-dlp/ffmpeg
downloads or a real LLM) so these tests exercise only the pipeline step's
own orchestration: candidate filtering/selection, the download/vision
conveyor, and degrade-on-failure.

No sleep-races: the conveyor-overlap test asserts on an explicit ordering
list built from fakes' own instrumentation, not wall-clock timing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from src.llm.vision import VisionResult
from src.storage import repo
from src.storage.db import dispose_engine, init_engine
from src.storage.migrations import run_migrations
from src.workers import broker as broker_mod
from src.workers import control as control_mod
from src.workers import frames as frames_mod
from src.workers import pipeline as pipeline_mod
from src.workers.broker import get_broker
from src.workers.deixis import DeixisCandidate, DeixisCategory


@pytest.fixture(autouse=True)
def _reset_singletons() -> Any:
    control_mod.reset_control()
    broker_mod.reset_broker()
    yield
    control_mod.reset_control()
    broker_mod.reset_broker()


@pytest.fixture
def isolated_db(tmp_path: Path) -> Any:
    db_path = tmp_path / "frame-analysis.db"
    engine = init_engine(db_path)
    run_migrations(engine)
    try:
        yield engine
    finally:
        dispose_engine()


def _cand(seconds: float, category: DeixisCategory, phrase: str = "this cream") -> DeixisCandidate:
    return DeixisCandidate(timestamp=seconds, phrase=phrase, category=category, confidence=0.8)


def _make_job(isolated_db: Any) -> Any:
    return repo.create_job(url="https://example.com/video", kind="youtube")


# ---------------------------------------------------------------------------
# No-candidates path is a true no-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_candidates_is_a_true_noop(
    isolated_db: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _make_job(isolated_db)
    monkeypatch.setattr(pipeline_mod.deixis, "candidates_for_job", lambda j: [])

    async def boom_fetch(*a: Any, **k: Any) -> list[Path]:
        raise AssertionError("fetch_moment_frames must not be called for a no-op job")

    async def boom_vision(*a: Any, **k: Any) -> VisionResult:
        raise AssertionError("analyze_summary_frames must not be called for a no-op job")

    monkeypatch.setattr(pipeline_mod.llm_vision, "fetch_moment_frames", boom_fetch)
    monkeypatch.setattr(pipeline_mod.llm_vision, "analyze_summary_frames", boom_vision)

    result = await pipeline_mod._run_frame_analysis(job.id, get_broker(), [])
    assert result == []


# ---------------------------------------------------------------------------
# A media job with no stored media_url must degrade to a true no-op too —
# end-to-end through the real llm.vision.fetch_moment_frames (only the
# bottom-most workers.frames.fetch_frames is mocked), so this exercises the
# actual resolve_frame_source_url call, not a stand-in for it.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_media_job_without_stored_media_url_degrades_to_noop(
    isolated_db: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before migration v12 every kind=media job has media_url=None. Frame
    analysis for it must degrade to finding nothing — never fetch job.url
    (the page, which has no video on it) and never raise out of
    run_pipeline."""
    job = repo.create_job(url="https://example.com/article", kind="media")
    assert job.media_url is None

    obj = _cand(10.0, DeixisCategory.OBJECT, "this cream")
    monkeypatch.setattr(pipeline_mod.deixis, "candidates_for_job", lambda j: [obj])

    async def boom_fetch_frames(**kwargs: Any) -> list[Path]:
        raise AssertionError(
            f"workers.frames.fetch_frames must not be called (got url={kwargs.get('url')!r})"
        )

    # Deliberately NOT mocking llm_vision.fetch_moment_frames itself — the
    # point is to exercise the real resolve_frame_source_url call inside it.
    monkeypatch.setattr(frames_mod, "fetch_frames", boom_fetch_frames)

    result = await pipeline_mod._run_frame_analysis(job.id, get_broker(), [])
    assert result == []


# ---------------------------------------------------------------------------
# EXTERNAL candidates are skipped entirely — no frame fetch for them
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_external_candidates_never_fetched(
    isolated_db: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _make_job(isolated_db)
    external = _cand(5.0, DeixisCategory.EXTERNAL, "link in the description")
    obj = _cand(10.0, DeixisCategory.OBJECT, "this cream")
    monkeypatch.setattr(
        pipeline_mod.deixis, "candidates_for_job", lambda j: [external, obj]
    )

    fetched_timestamps: list[float] = []

    async def fake_fetch(*, job: Any, candidate: DeixisCandidate, cookies: Any = None) -> list[Path]:
        fetched_timestamps.append(candidate.timestamp)
        return [Path("/tmp/frame_01.jpg")]

    async def fake_vision(frame_paths: Any, *, candidate: DeixisCandidate, **k: Any) -> VisionResult:
        return VisionResult(finding="A red tub.", relevant=True, best_frame_index=1)

    monkeypatch.setattr(pipeline_mod.llm_vision, "fetch_moment_frames", fake_fetch)
    monkeypatch.setattr(pipeline_mod.llm_vision, "analyze_summary_frames", fake_vision)

    result = await pipeline_mod._run_frame_analysis(job.id, get_broker(), [])

    assert fetched_timestamps == [10.0]  # EXTERNAL (5.0) never reached fetch
    assert [f["seconds"] for f in result] == [10.0]


# ---------------------------------------------------------------------------
# Budget exhaustion: chronological, first-fit selection stops once the next
# moment would push the per-job frame budget over MAX_FRAMES_PER_JOB.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_exhaustion_stops_selection(
    isolated_db: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _make_job(isolated_db)
    # DEFAULT_NUM_FRAMES=5, MAX_FRAMES_PER_JOB=24 -> exactly 4 candidates fit
    # (20 <= 24; a 5th would need 25 > 24). Provide 6 to prove the rest never
    # get fetched at all.
    candidates = [_cand(float(i * 10), DeixisCategory.OBJECT) for i in range(6)]
    monkeypatch.setattr(pipeline_mod.deixis, "candidates_for_job", lambda j: candidates)

    fetched: list[float] = []

    async def fake_fetch(*, job: Any, candidate: DeixisCandidate, cookies: Any = None) -> list[Path]:
        fetched.append(candidate.timestamp)
        return [Path("/tmp/frame_01.jpg")]

    async def fake_vision(frame_paths: Any, *, candidate: DeixisCandidate, **k: Any) -> VisionResult:
        return VisionResult(finding="ok", relevant=True, best_frame_index=1)

    monkeypatch.setattr(pipeline_mod.llm_vision, "fetch_moment_frames", fake_fetch)
    monkeypatch.setattr(pipeline_mod.llm_vision, "analyze_summary_frames", fake_vision)

    assert 4 * frames_mod.DEFAULT_NUM_FRAMES <= frames_mod.MAX_FRAMES_PER_JOB
    assert 5 * frames_mod.DEFAULT_NUM_FRAMES > frames_mod.MAX_FRAMES_PER_JOB

    result = await pipeline_mod._run_frame_analysis(job.id, get_broker(), [])

    assert fetched == [0.0, 10.0, 20.0, 30.0]
    assert len(result) == 4


# ---------------------------------------------------------------------------
# Per-moment failure degrades without failing the whole step
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_failure_degrades_one_moment(
    isolated_db: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _make_job(isolated_db)
    bad = _cand(10.0, DeixisCategory.OBJECT, "this cream")
    good = _cand(20.0, DeixisCategory.ACTION, "watch this")
    monkeypatch.setattr(pipeline_mod.deixis, "candidates_for_job", lambda j: [bad, good])

    async def flaky_fetch(*, job: Any, candidate: DeixisCandidate, cookies: Any = None) -> list[Path]:
        if candidate.timestamp == 10.0:
            raise RuntimeError("section download failed")
        return [Path("/tmp/frame_01.jpg")]

    vision_calls: list[float] = []

    async def fake_vision(frame_paths: Any, *, candidate: DeixisCandidate, **k: Any) -> VisionResult:
        vision_calls.append(candidate.timestamp)
        return VisionResult(finding="A hand folding paper.", relevant=True, best_frame_index=1)

    monkeypatch.setattr(pipeline_mod.llm_vision, "fetch_moment_frames", flaky_fetch)
    monkeypatch.setattr(pipeline_mod.llm_vision, "analyze_summary_frames", fake_vision)

    result = await pipeline_mod._run_frame_analysis(job.id, get_broker(), [])

    # The failed moment never reached the vision call at all, and the whole
    # step still returned the surviving moment's finding rather than raising.
    assert vision_calls == [20.0]
    assert [f["seconds"] for f in result] == [20.0]


@pytest.mark.asyncio
async def test_vision_call_failure_degrades_one_moment(
    isolated_db: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _make_job(isolated_db)
    bad = _cand(10.0, DeixisCategory.OBJECT, "this cream")
    good = _cand(20.0, DeixisCategory.ACTION, "watch this")
    monkeypatch.setattr(pipeline_mod.deixis, "candidates_for_job", lambda j: [bad, good])

    async def fake_fetch(*, job: Any, candidate: DeixisCandidate, cookies: Any = None) -> list[Path]:
        return [Path("/tmp/frame_01.jpg")]

    async def flaky_vision(frame_paths: Any, *, candidate: DeixisCandidate, **k: Any) -> VisionResult:
        if candidate.timestamp == 10.0:
            raise RuntimeError("backend errored")
        return VisionResult(finding="A hand folding paper.", relevant=True, best_frame_index=1)

    monkeypatch.setattr(pipeline_mod.llm_vision, "fetch_moment_frames", fake_fetch)
    monkeypatch.setattr(pipeline_mod.llm_vision, "analyze_summary_frames", flaky_vision)

    result = await pipeline_mod._run_frame_analysis(job.id, get_broker(), [])

    assert [f["seconds"] for f in result] == [20.0]


# ---------------------------------------------------------------------------
# relevant=False produces no finding (and no thumbnail)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_irrelevant_moment_produces_no_finding(
    isolated_db: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _make_job(isolated_db)
    talking_head = _cand(10.0, DeixisCategory.OBJECT, "this")
    monkeypatch.setattr(
        pipeline_mod.deixis, "candidates_for_job", lambda j: [talking_head]
    )

    async def fake_fetch(*, job: Any, candidate: DeixisCandidate, cookies: Any = None) -> list[Path]:
        return [Path("/tmp/frame_01.jpg")]

    async def fake_vision(frame_paths: Any, *, candidate: DeixisCandidate, **k: Any) -> VisionResult:
        return VisionResult(
            finding="Just the speaker talking to camera; nothing shown.",
            relevant=False,
            best_frame_index=1,
        )

    monkeypatch.setattr(pipeline_mod.llm_vision, "fetch_moment_frames", fake_fetch)
    monkeypatch.setattr(pipeline_mod.llm_vision, "analyze_summary_frames", fake_vision)

    result = await pipeline_mod._run_frame_analysis(job.id, get_broker(), [])
    assert result == []


# ---------------------------------------------------------------------------
# Conveyor: moment i+1's download starts while moment i's vision call is
# still in flight. Asserted via an explicit event ordering, not timing.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conveyor_overlaps_download_with_vision_call(
    isolated_db: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _make_job(isolated_db)
    first = _cand(10.0, DeixisCategory.OBJECT, "this cream")
    second = _cand(20.0, DeixisCategory.ACTION, "watch this")
    monkeypatch.setattr(
        pipeline_mod.deixis, "candidates_for_job", lambda j: [first, second]
    )

    order: list[str] = []

    async def recording_fetch(*, job: Any, candidate: DeixisCandidate, cookies: Any = None) -> list[Path]:
        order.append(f"fetch_start:{candidate.timestamp:.0f}")
        order.append(f"fetch_end:{candidate.timestamp:.0f}")
        return [Path("/tmp/frame_01.jpg")]

    async def recording_vision(
        frame_paths: Any, *, candidate: DeixisCandidate, **k: Any
    ) -> VisionResult:
        order.append(f"vision_start:{candidate.timestamp:.0f}")
        # Yield control here — this is the window in which moment i+1's
        # prefetch (already scheduled via asyncio.create_task before this
        # await) actually gets to run.
        await asyncio.sleep(0.05)
        order.append(f"vision_end:{candidate.timestamp:.0f}")
        return VisionResult(finding="ok", relevant=True, best_frame_index=1)

    monkeypatch.setattr(pipeline_mod.llm_vision, "fetch_moment_frames", recording_fetch)
    monkeypatch.setattr(pipeline_mod.llm_vision, "analyze_summary_frames", recording_vision)

    result = await pipeline_mod._run_frame_analysis(job.id, get_broker(), [])

    assert len(result) == 2
    # The second moment's download must start (and, since it's fast here,
    # finish) WHILE the first moment's vision call is still awaiting —
    # i.e. before "vision_end:10", not after.
    assert order.index("fetch_start:20") < order.index("vision_end:10")
    assert order.index("fetch_end:20") < order.index("vision_end:10")
    # And it must not have started before the first moment's vision call did
    # — the conveyor only ever runs 1 moment ahead, not eagerly upfront.
    assert order.index("vision_start:10") < order.index("fetch_start:20")


# ---------------------------------------------------------------------------
# Ingestion-time cookies reach every frame download in this step — the gap
# fixed here: `fetch_moment_frames` used to always be called with no cookies
# at all, so a cookie-gated video's frame fetch 403'd on every moment. Both
# the initial fetch and the conveyor's prefetch call must forward them.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_frame_analysis_forwards_cookies_to_every_fetch(
    isolated_db: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _make_job(isolated_db)
    first = _cand(10.0, DeixisCategory.OBJECT, "this cream")
    second = _cand(20.0, DeixisCategory.ACTION, "watch this")
    monkeypatch.setattr(
        pipeline_mod.deixis, "candidates_for_job", lambda j: [first, second]
    )

    received_cookies: list[Any] = []

    async def fake_fetch(
        *, job: Any, candidate: DeixisCandidate, cookies: Any = None
    ) -> list[Path]:
        received_cookies.append(cookies)
        return [Path("/tmp/frame_01.jpg")]

    async def fake_vision(frame_paths: Any, *, candidate: DeixisCandidate, **k: Any) -> VisionResult:
        return VisionResult(finding="ok", relevant=True, best_frame_index=1)

    monkeypatch.setattr(pipeline_mod.llm_vision, "fetch_moment_frames", fake_fetch)
    monkeypatch.setattr(pipeline_mod.llm_vision, "analyze_summary_frames", fake_vision)

    sentinel_cookies = [{"name": "session", "value": "abc123"}]
    result = await pipeline_mod._run_frame_analysis(job.id, get_broker(), sentinel_cookies)

    assert len(result) == 2
    # Both the initial fetch AND the prefetched next-moment fetch received
    # the SAME cookies list handed to _run_frame_analysis.
    assert received_cookies == [sentinel_cookies, sentinel_cookies]
