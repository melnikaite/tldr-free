"""GET /health — daemon liveness + LLM backend reachability.

Pings ``config.llm.base_url + /models`` to check the configured backend
(mlx-server, Ollama, LM Studio, cloud providers, …) is reachable, and reports
the live Whisper queue size from ``workers.queue``.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter

from src.api.schemas import HealthResponse
from src.config import DAEMON_VERSION, get_config
from src.workers.queue import get_queue

router = APIRouter(prefix="/health", tags=["health"])

# Cloud backends (OpenAI, Anthropic-via-proxy, OpenRouter, …) can take longer
# than a local server to complete the TLS handshake + first byte, especially
# on a cold connection. 2s was fine for localhost-only mlx-server/LocalAI but
# flagged real remote backends as "unreachable" under normal latency.
_HEALTH_PROBE_TIMEOUT_SECONDS = 5.0


@router.get("", response_model=HealthResponse)
async def health() -> HealthResponse:
    config = get_config()

    backend_reachable = False
    backend_models: list[str] = []
    backend_error: str | None = None
    try:
        headers: dict[str, str] = {}
        api_key = config.llm.effective_api_key
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        async with httpx.AsyncClient(timeout=_HEALTH_PROBE_TIMEOUT_SECONDS) as client:
            r = await client.get(f"{config.llm.base_url}/models", headers=headers)
            if r.status_code == 200:
                backend_reachable = True
                payload = r.json()
                backend_models = [m["id"] for m in payload.get("data", []) if "id" in m]
            elif r.status_code in (401, 403):
                backend_error = f"{r.status_code} unauthorized — check llm.api_key"
    except Exception:
        pass

    queue_size, queue_running = get_queue().snapshot()

    return HealthResponse(
        status="ok" if backend_reachable else "degraded",
        queue_size=queue_size,
        queue_running=queue_running,
        llm_backend_reachable=backend_reachable,
        llm_backend_models=backend_models,
        llm_backend_error=backend_error,
        version=DAEMON_VERSION,
    )
