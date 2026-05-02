"""Single-job Q&A — streaming.

Public surface:
    async def stream_answer(*, job, question: str, output_language: str) -> AsyncIterator[str]
        Builds context = job.raw_text if it fits else job.summary_md.
        Loads prompts/qa.txt, formats with output_language, calls the LLM with
        stream=True, yields token deltas as they arrive.

Called from ``api/ai.py``'s POST /ai/stream when the request body has a
``question`` field; the route wraps each yielded delta as an
``AIDeltaEvent`` SSE frame and emits an ``AIDoneEvent`` at the end.

The prompt instructs the LLM to include [MM:SS] markers in the answer when
they help locate the relevant moment in the source video.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import get_config
from src.llm import client as llm_client
from src.llm.tokens import count_tokens

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Reserve room for the prompt scaffolding + the answer — we don't want to
# squeeze context to the bone.
_PROMPT_OVERHEAD_TOKENS = 4000

# We accept any object with `.title`, `.raw_text`, `.summary_md` attributes —
# storage.db.Job is the runtime type, but tests pass dataclasses or simple
# namespaces. Typed as `Any` (not a Protocol) so callers don't need to
# subclass anything.


@lru_cache(maxsize=1)
def _load_prompt() -> str:
    return (_PROMPTS_DIR / "qa.txt").read_text(encoding="utf-8")


def _select_context(job: Any) -> str:
    """Pick raw_text if it fits in the model's context; otherwise summary_md.

    Falls back to "" when neither is set.
    """
    raw = (getattr(job, "raw_text", None) or "")
    summary = (getattr(job, "summary_md", None) or "")
    budget = get_config().llm.context_length - _PROMPT_OVERHEAD_TOKENS
    if raw and count_tokens(raw) <= budget:
        return raw
    return summary


async def stream_answer(
    *,
    job: Any,
    question: str,
    output_language: str,
) -> AsyncIterator[str]:
    """Yield assistant token deltas for a Q&A turn over `job`.

    Loads `prompts/qa.txt`, formats it with `{title}`, `{context}`, `{question}`,
    `{output_language}`, then streams the model's reply.
    """
    context = _select_context(job)
    title = (getattr(job, "title", None) or "")
    prompt = _load_prompt().format(
        output_language=output_language,
        title=title,
        context=context,
        question=question,
    )
    # respect_pause=False — Q&A bypasses the global pause gate because the
    # user is actively waiting on the answer.
    async for delta in llm_client.stream_complete(
        prompt, max_tokens=2000, temperature=0.3, respect_pause=False
    ):
        yield delta


__all__ = ["stream_answer"]
