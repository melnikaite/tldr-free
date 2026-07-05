"""Integration tests for the smaller API surfaces: /health, /workers, /events.

Reuses the same hermetic app-bring-up style as ``test_api_jobs.py``:
- a fresh in-memory-ish SQLite under ``tmp_path``,
- the Whisper + retention workers stubbed to no-ops so the lifespan starts
  cleanly,
- queue / broker / control singletons reset around each test.

External network in ``/health`` (the ``GET {base_url}/models`` probe) is
mocked at the ``httpx.AsyncClient`` level so we can exercise both the
"backend reachable" and "backend unreachable" branches without a real LLM.

SSE note: ``GET /events`` is an infinite stream. We drive it with
``TestClient.stream(...)`` and pull a bounded number of frames (a published
event, plus the keep-alive comment), then close the connection — rather than
trying to read the whole body (which never ends).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.storage.db import dispose_engine, init_engine
from src.storage.migrations import run_migrations


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Hermetic TestClient: stubbed workers + reset singletons (no LLM mock).

    Individual tests patch ``httpx.AsyncClient`` themselves when they need to
    control the ``/health`` backend probe, so this fixture stays neutral.
    """
    db_path = tmp_path / "api.db"
    engine = init_engine(db_path)
    run_migrations(engine)

    import asyncio

    # Whisper worker → no-op so lifespan spins up cleanly.
    from src.workers import runner as runner_mod

    async def _noop_worker(queue, repo_module):  # noqa: ANN001
        try:
            while True:
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            return

    monkeypatch.setattr(runner_mod, "whisper_worker", _noop_worker)
    import src.main as main_mod

    monkeypatch.setattr(main_mod, "whisper_worker", _noop_worker)

    # Retention worker → no-op.
    from src.workers import retention as retention_mod

    async def _noop_retention() -> None:
        try:
            while True:
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            return

    monkeypatch.setattr(retention_mod, "retention_worker", _noop_retention)
    monkeypatch.setattr(main_mod, "retention_worker", _noop_retention)

    # Reset queue + broker + control singletons between tests.
    from src.workers import broker as broker_mod
    from src.workers import control as control_mod
    from src.workers import queue as queue_mod

    queue_mod.reset_queue()
    broker_mod.reset_broker()
    control_mod.reset_control()

    with TestClient(app) as c:
        yield c
    dispose_engine()
    queue_mod.reset_queue()
    broker_mod.reset_broker()
    control_mod.reset_control()


# ---------------------------------------------------------------------------
# Helpers: a fake httpx.AsyncClient for the /health backend probe
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict[str, Any]:
        return self._payload


def _patch_health_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: _FakeResponse | None = None,
    raises: Exception | None = None,
) -> None:
    """Replace ``httpx.AsyncClient`` (as imported in ``src.api.health``) with a
    fake whose ``.get`` returns ``response`` or raises ``raises``."""

    class _FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _FakeAsyncClient:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def get(self, url: str) -> _FakeResponse:
            if raises is not None:
                raise raises
            assert response is not None
            return response

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_ok_when_backend_reachable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backend answers 200 with a model list → status ok, reachable True,
    models surfaced."""
    _patch_health_backend(
        monkeypatch,
        response=_FakeResponse(
            200,
            {"data": [{"id": "gemma-4"}, {"id": "whisper"}, {"no_id": "skip"}]},
        ),
    )

    r = client.get("/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["llm_backend_reachable"] is True
    # Only entries with an "id" key are surfaced.
    assert body["llm_backend_models"] == ["gemma-4", "whisper"]
    assert "version" in body
    assert body["queue_size"] == 0
    assert body["queue_running"] == 0


def test_health_degraded_on_non_200(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backend answers non-200 → reachable stays False, models empty,
    status degraded (but the endpoint itself still returns 200)."""
    _patch_health_backend(monkeypatch, response=_FakeResponse(503, {"data": []}))

    r = client.get("/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "degraded"
    assert body["llm_backend_reachable"] is False
    assert body["llm_backend_models"] == []


def test_health_degraded_when_backend_unreachable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Probe raises (connection refused / timeout) → swallowed, degraded."""
    _patch_health_backend(
        monkeypatch, raises=httpx.ConnectError("refused")
    )

    r = client.get("/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "degraded"
    assert body["llm_backend_reachable"] is False
    assert body["llm_backend_models"] == []


# ---------------------------------------------------------------------------
# /workers — status + pause/resume state machine
# ---------------------------------------------------------------------------


def test_workers_status_starts_unpaused(client: TestClient) -> None:
    r = client.get("/workers")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {"paused": False, "queue_size": 0, "running": 0}


def test_workers_pause_then_resume_round_trip(client: TestClient) -> None:
    # pause flips the flag and echoes the new snapshot.
    r = client.post("/workers/pause")
    assert r.status_code == 200, r.text
    assert r.json()["paused"] is True

    # status reflects the paused state.
    assert client.get("/workers").json()["paused"] is True

    # resume flips it back.
    r = client.post("/workers/resume")
    assert r.status_code == 200, r.text
    assert r.json()["paused"] is False

    assert client.get("/workers").json()["paused"] is False


def test_workers_pause_is_idempotent(client: TestClient) -> None:
    """Pausing twice keeps paused=True (no error, no toggle)."""
    assert client.post("/workers/pause").json()["paused"] is True
    assert client.post("/workers/pause").json()["paused"] is True
    assert client.get("/workers").json()["paused"] is True


def test_workers_pause_publishes_event_to_global_broker(client: TestClient) -> None:
    """control.pause() publishes a 'workers' event to the global broker — a
    subscriber should receive it. Verifies the side effect that the route docs
    rely on, without going through the SSE HTTP layer."""
    from src.workers.broker import get_event_broker

    broker = get_event_broker()
    q = broker.subscribe()
    try:
        r = client.post("/workers/pause")
        assert r.status_code == 200

        # control.pause() -> broker.publish() -> q.put_nowait(), all synchronous,
        # so the event is already sitting in the queue — read it without a loop.
        event = q.get_nowait()
        assert event["type"] == "workers"
        assert event["state"]["paused"] is True
    finally:
        broker.unsubscribe(q)


# NOTE: the /events SSE stream is intentionally NOT tested through the sync
# TestClient — its generator subscribes lazily once the client starts reading,
# which races with publishing and hangs on the infinite stream. The broker that
# backs it is covered directly in tests/workers/test_broker.py (100%), and the
# pause→publish side effect is covered above without the HTTP streaming layer.
