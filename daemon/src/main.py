"""FastAPI entry point. Run via:

    uvicorn src.main:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import ai, events, health, jobs, workers
from src.config import DAEMON_VERSION, get_config
from src.storage import repo
from src.storage.db import dispose_engine, init_engine
from src.storage.migrations import run_migrations
from src.workers.queue import get_queue, re_enqueue_pending
from src.workers.retention import retention_worker
from src.workers.runner import whisper_worker

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = get_config()
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    log.info("TLDR daemon v%s starting", DAEMON_VERSION)

    engine = init_engine()
    applied = run_migrations(engine)
    if applied:
        log.info("storage: applied migrations: %s", applied)
    else:
        log.info("storage: schema up to date")

    # Whisper queue + worker.
    queue = get_queue()
    try:
        re_enqueued = await re_enqueue_pending(queue, repo)
        if re_enqueued:
            log.info("workers: re-enqueued %d pending youtube job(s)", re_enqueued)
    except Exception:
        log.exception("workers: re-enqueue on startup failed")

    worker_task: asyncio.Task[None] = asyncio.create_task(
        whisper_worker(queue, repo),
        name="whisper-worker",
    )

    # Periodic retention sweep (deletes jobs older than config.storage.retention_days).
    retention_task: asyncio.Task[None] = asyncio.create_task(
        retention_worker(),
        name="retention-worker",
    )

    try:
        yield
    finally:
        log.info("TLDR daemon shutting down")
        worker_task.cancel()
        retention_task.cancel()
        await asyncio.gather(worker_task, retention_task, return_exceptions=True)
        dispose_engine()


app = FastAPI(
    title="TLDR",
    version=DAEMON_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://*"],
    allow_origin_regex=r"chrome-extension://.*",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router)
app.include_router(ai.router)
app.include_router(events.router)
app.include_router(workers.router)
app.include_router(health.router)
