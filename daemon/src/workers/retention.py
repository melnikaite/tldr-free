"""Periodic retention sweep — delete jobs older than ``config.storage.retention_days``.

Started from ``main.lifespan`` as a long-running coroutine. One sweep on
startup, then every ``_INTERVAL_HOURS`` hours forever.

If ``retention_days`` is 0 the worker logs once and exits — retention is
disabled.

Shutdown: cancellation from the lifespan. The current sweep is cheap
(single DELETE per old row) so we don't try to interrupt mid-sweep.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

from src.config import get_config
from src.storage import repo

log = logging.getLogger(__name__)

_INTERVAL_SECONDS = 6 * 60 * 60     # 6 hours


async def retention_worker() -> None:
    """Loop forever, sweeping old jobs every ``_INTERVAL_SECONDS``."""
    cfg = get_config()
    days = cfg.storage.retention_days
    if days <= 0:
        log.info("retention worker disabled (retention_days=%d)", days)
        return

    log.info("retention worker started (retention_days=%d)", days)
    while True:
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            n = repo.delete_jobs_older_than(cutoff)
            if n:
                log.info("retention sweep deleted %d job(s) older than %s", n, cutoff)
            else:
                log.debug("retention sweep deleted 0 jobs (cutoff=%s)", cutoff)
        except asyncio.CancelledError:
            log.info("retention worker cancelled")
            raise
        except Exception:
            log.exception("retention sweep failed; will retry on next interval")
        await asyncio.sleep(_INTERVAL_SECONDS)


__all__ = ["retention_worker"]
