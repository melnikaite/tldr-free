"""``_checkpoint_pause`` is the contract every pipeline step honours.

Soft-pause semantics, exactly what the user asked for:

- Whatever step is *in flight* finishes normally.
- The very next step — download / transcribe / summarize / whatever —
  waits at the checkpoint until the user clicks Resume, then carries on
  from where it stopped.
- While parked we surface ``progress_stage="paused"`` so the Library row
  shows "Paused"; on resume we restore the stage we were heading into so
  the row goes back to e.g. "Transcribing".

These tests assert that contract on the pipeline + runner helpers without
any LLM/yt-dlp moving parts in the picture.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from src.api.schemas import JobStatus
from src.storage import repo
from src.storage.db import dispose_engine, init_engine
from src.storage.migrations import run_migrations
from src.workers import broker as broker_mod
from src.workers import control as control_mod
from src.workers import pipeline as pipeline_mod
from src.workers import queue as queue_mod
from src.workers import runner as runner_mod
from src.workers.broker import get_broker, get_event_broker


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
    db_path = tmp_path / "checkpoint.db"
    engine = init_engine(db_path)
    run_migrations(engine)
    try:
        yield engine
    finally:
        dispose_engine()


# ---------------------------------------------------------------------------
# pipeline._checkpoint_pause (uses src.storage.repo directly)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_checkpoint_pause_noop_when_not_paused(isolated_db: Any) -> None:
    """No paused → checkpoint is instant, no DB writes, no events."""
    job = repo.create_job(url="https://x", kind="page", title="t")
    # Subscribe AFTER create_job so the "created" event isn't sitting in the
    # queue waiting for us — we want this queue to capture only side effects
    # of the checkpoint call.
    queue = get_event_broker().subscribe()

    await pipeline_mod._checkpoint_pause(job.id, get_broker(), "extracting")

    await asyncio.sleep(0)
    assert queue.empty()
    fresh = repo.get_job(job.id)
    assert fresh is not None
    assert fresh.progress_stage is None  # nothing changed


@pytest.mark.asyncio
async def test_pipeline_checkpoint_pause_blocks_then_restores_stage(
    isolated_db: Any,
) -> None:
    """Paused at checkpoint → row flips to 'paused', awaits resume, then
    restores the on_resume_stage so Library knows where the job is heading."""
    job = repo.create_job(url="https://x", kind="page", title="t")
    control_mod.get_control().pause()

    waiter = asyncio.create_task(
        pipeline_mod._checkpoint_pause(job.id, get_broker(), "summarizing"),
    )

    # Give the checkpoint a chance to run the first half (mark paused).
    await asyncio.sleep(0.05)
    assert not waiter.done()

    fresh = repo.get_job(job.id)
    assert fresh is not None
    assert fresh.progress_stage == "paused"

    # Resume → checkpoint exits and restores the next stage.
    control_mod.get_control().resume()
    await asyncio.wait_for(waiter, timeout=2.0)

    fresh = repo.get_job(job.id)
    assert fresh is not None
    assert fresh.progress_stage == "summarizing"


# ---------------------------------------------------------------------------
# runner._checkpoint_pause (uses repo_module DI for tests)
# ---------------------------------------------------------------------------


class _FakeRepoMin:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def update_status(self, job_id: str, **kwargs: Any) -> None:
        self.calls.append((job_id, kwargs))


@pytest.mark.asyncio
async def test_runner_checkpoint_pause_noop_when_not_paused() -> None:
    repo_mod = _FakeRepoMin()
    await runner_mod._checkpoint_pause("j1", repo_mod, "transcribing")
    assert repo_mod.calls == []


@pytest.mark.asyncio
async def test_runner_checkpoint_pause_marks_paused_then_restores() -> None:
    """Same shape as the pipeline test but threading the repo_module DI."""
    repo_mod = _FakeRepoMin()
    control_mod.get_control().pause()

    waiter = asyncio.create_task(
        runner_mod._checkpoint_pause("j2", repo_mod, "transcribing"),
    )
    await asyncio.sleep(0.05)
    assert not waiter.done()

    # First call: marked paused.
    assert len(repo_mod.calls) == 1
    assert repo_mod.calls[0][1]["progress_stage"] == "paused"

    control_mod.get_control().resume()
    await asyncio.wait_for(waiter, timeout=2.0)

    # Second call: restored to the on_resume stage.
    assert len(repo_mod.calls) == 2
    assert repo_mod.calls[1][1]["progress_stage"] == "transcribing"
    assert repo_mod.calls[1][1]["status"] == JobStatus.RUNNING.value
