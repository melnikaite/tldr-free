"""GET /health — daemon liveness + LLM backend reachability.

Pings ``config.llm.base_url + /models`` to check the configured backend
(mlx-server, Ollama, LM Studio, …) is reachable, and reports the live
Whisper queue size from ``workers.queue``.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter

from src.api.schemas import HealthResponse
from src.config import DAEMON_VERSION, get_config
from src.workers.queue import get_queue

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health() -> HealthResponse:
    config = get_config()

    backend_reachable = False
    backend_models: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{config.llm.base_url}/models")
            if r.status_code == 200:
                backend_reachable = True
                payload = r.json()
                backend_models = [m["id"] for m in payload.get("data", []) if "id" in m]
    except Exception:
        pass

    queue_size, queue_running = get_queue().snapshot()

    return HealthResponse(
        status="ok" if backend_reachable else "degraded",
        queue_size=queue_size,
        queue_running=queue_running,
        llm_backend_reachable=backend_reachable,
        llm_backend_models=backend_models,
        version=DAEMON_VERSION,
    )
