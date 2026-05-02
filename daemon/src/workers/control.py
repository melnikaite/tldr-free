"""Global throttle for background ML work.

Two knobs that the user controls:

- ``pause()`` / ``resume()`` flips a flag checked before every heavy step
  in pipelines and the Whisper worker. In-flight work runs to completion;
  newly-started work waits at ``wait_if_paused()`` until ``resume()``. State
  is in-memory, resets on daemon restart.
- ``cooldown_seconds`` from ``config.workers.cooldown_seconds`` is applied
  by long-running workers between consecutive jobs (default 0 = no wait).

QA stays unblocked — those calls don't go through this gate, since the user
is actively waiting for the answer.

``pause()`` / ``resume()`` publish ``workers_event`` to the global event
broker as a side effect so the Side panel / Library reflect the new state
without polling. Late imports keep this module free of compile-time
dependencies on ``workers.broker`` / ``workers.queue``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

log = logging.getLogger(__name__)


def _publish_state() -> None:
    try:
        from src.workers.broker import get_event_broker, workers_event
        from src.workers.queue import get_queue
    except Exception:
        return
    size, running = get_queue().snapshot()
    state: dict[str, Any] = {
        "paused": _control._paused,
        "queue_size": size,
        "running": running,
    }
    with contextlib.suppress(Exception):
        get_event_broker().publish(workers_event(state))


class WorkerControl:
    _PAUSE_POLL_SECONDS: float = 1.0

    def __init__(self) -> None:
        self._paused = False

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        if not self._paused:
            log.info("workers: paused (in-flight tasks finish; new ones wait until resume)")
        self._paused = True
        _publish_state()

    def resume(self) -> None:
        if self._paused:
            log.info("workers: resumed")
        self._paused = False
        _publish_state()

    async def wait_if_paused(self) -> None:
        """Block while paused. Returns immediately when not paused."""
        while self._paused:
            await asyncio.sleep(self._PAUSE_POLL_SECONDS)


_control = WorkerControl()


def get_control() -> WorkerControl:
    return _control


def reset_control() -> None:
    """Test helper — clear pause state between tests."""
    _control._paused = False


__all__ = ["WorkerControl", "get_control", "reset_control"]
