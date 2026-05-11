"""Streaming summarization (single-pass / map-reduce).

    async def stream_summarize(text, *, title, output_language)
            -> AsyncIterator[str]
        Single-pass when the input fits under config.llm.single_pass_token_limit:
        yields tokens directly from the LLM. Otherwise falls back to map-reduce:
        runs the map phase silently (chunks summarised one at a time, since
        ``llm.client`` serialises every LLM call to spare the local mlx-server)
        and then streams the final reduce phase. Preserves [MM:SS] markers
        in the input.

Prompts: prompts/summary_single.txt, summary_chunk.txt, summary_reduce.txt.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache
from pathlib import Path

from src.config import get_config
from src.llm import client as llm_client
from src.llm.chunking import split_for_summary
from src.llm.tokens import count_tokens

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=8)
def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _safe_title(title: str | None) -> str:
    """Plug into prompts where {title} is expected. Empty when missing —
    no language-specific placeholder text."""
    return title.strip() if title and title.strip() else ""


def _build_single_pass_prompt(
    text: str, *, title: str | None, output_language: str
) -> str:
    template = _load_prompt("summary_single.txt")
    return template.format(
        output_language=output_language,
        title=_safe_title(title),
        text=text,
    )


async def _stream_single_pass(
    text: str,
    *,
    title: str | None,
    output_language: str,
) -> AsyncIterator[str]:
    prompt = _build_single_pass_prompt(
        text, title=title, output_language=output_language
    )
    async for delta in llm_client.stream_complete(prompt, max_tokens=2000, temperature=0.3):
        yield delta


async def _summarize_chunk(
    chunk: str,
    *,
    title: str | None,
    output_language: str,
    n: int,
    total: int,
) -> str:
    template = _load_prompt("summary_chunk.txt")
    prompt = template.format(
        output_language=output_language,
        title=_safe_title(title),
        chunk=chunk,
        n=n,
        total=total,
    )
    return (await llm_client.complete(prompt, max_tokens=1500, temperature=0.3)).strip()


async def _stream_reduce(
    partials: list[str],
    *,
    title: str | None,
    output_language: str,
) -> AsyncIterator[str]:
    combined = "\n\n---\n\n".join(partials)
    template = _load_prompt("summary_reduce.txt")
    prompt = template.format(
        output_language=output_language,
        title=_safe_title(title),
        combined=combined,
    )
    async for delta in llm_client.stream_complete(prompt, max_tokens=2000, temperature=0.3):
        yield delta


async def stream_summarize(
    text: str,
    *,
    title: str | None,
    output_language: str,
) -> AsyncIterator[str]:
    """Stream a summary of ``text`` token by token.

    For inputs below ``config.llm.single_pass_token_limit`` we ask the LLM
    once with streaming. For longer inputs we fall back to map-reduce: the
    map phase runs silently (chunks are summarised one at a time — streaming
    each would interleave nonsense, and ``llm.client._llm_lock()`` serialises
    every call anyway), then the reduce phase streams its output to the caller.
    """
    if not text or not text.strip():
        return

    threshold = get_config().llm.single_pass_token_limit
    if count_tokens(text) < threshold:
        async for delta in _stream_single_pass(
            text, title=title, output_language=output_language
        ):
            yield delta
        return

    # Map-reduce path. Run map phase to completion, then stream the reduce.
    chunks = split_for_summary(text, target_tokens=4000, overlap_tokens=200)
    if not chunks:
        return
    if len(chunks) == 1:
        async for delta in _stream_single_pass(
            chunks[0], title=title, output_language=output_language
        ):
            yield delta
        return

    total = len(chunks)
    # Sequential map phase — llm.client._LLM_LOCK serialises every call anyway,
    # so spawning N tasks just queues them on the lock without any speedup.
    partials: list[str] = []
    for i, c in enumerate(chunks):
        partials.append(
            await _summarize_chunk(
                c,
                title=title,
                output_language=output_language,
                n=i + 1,
                total=total,
            )
        )
    async for delta in _stream_reduce(
        list(partials), title=title, output_language=output_language
    ):
        yield delta


__all__ = ["stream_summarize"]
