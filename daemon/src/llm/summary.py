"""Streaming summarization (single-pass / map-reduce).

    async def stream_summarize(text, *, title, output_language)
            -> AsyncIterator[str]
        Single-pass when the input fits under config.llm.single_pass_token_limit:
        yields tokens directly from the LLM. Otherwise falls back to map-reduce:
        runs the map phase silently (chunks summarised one at a time, since
        ``llm.client`` serialises every LLM call to spare the local mlx-server)
        and then streams the final reduce phase. Preserves [MM:SS] markers
        in the input.

Trust but verify, not trust: single-pass is only taken when
``count_tokens(chunk) < threshold``, never merely "len(chunks) == 1" — a
lone chunk isn't proof it fits. Same idea on the reduce side:
``_fold_partials`` checks the partials' joined size against that same
threshold before the final reduce and folds hierarchically if it doesn't fit.

Prompts: prompts/summary_single.txt, summary_chunk.txt, summary_reduce.txt.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache
from pathlib import Path

from src.config import get_config
from src.llm import client as llm_client
from src.llm.chunking import pack_lines, split_for_summary
from src.llm.tokens import count_tokens

# Ceiling on the map-phase chunk budget; stream_summarize takes the actual
# budget as min(this, single_pass_token_limit - 1), so a small configured
# threshold shrinks chunks accordingly instead of being ignored.
_CHUNK_TARGET_TOKENS = 4000
_CHUNK_OVERLAP_TOKENS = 200

# Hard cap on reduce-phase folding rounds (see _fold_partials) — pure safety
# net against an infinite loop, never expected to bind in practice: each
# round groups at least a pair of partials together (or bails out early when
# it can't), so a handful of rounds collapses any realistic partial count.
_MAX_REDUCE_FOLD_ITERS = 6

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Injected into the summary prompts when the source is a speech-to-text
# transcript (Whisper / auto-captions). Gives the model licence to fix obvious
# recognition errors instead of faithfully echoing garbled terms (e.g. a
# German "GSM-R" heard as "ГСМР … геометрия").
_TRANSCRIPT_SOURCE_NOTE = (
    "The text below is an automatic speech-to-text transcription of audio. It "
    "may contain recognition errors — misheard proper names, foreign words, "
    "and acronyms (an acronym may be spelled phonetically or split apart). Use "
    "context and general knowledge to silently correct obvious such errors in "
    "your summary; do not invent facts the text does not support."
)

# Injected when the source is a written document (web page / PDF), which has no
# timestamps. Counters the prompt's "include timestamps" rule so a small local
# model doesn't fabricate "[00:42]" markers next to key points (the pipeline
# also strips any that slip through — see workers.timecodes.strip_all_timecodes).
_DOCUMENT_SOURCE_NOTE = (
    "The text below is a written document (web page or PDF). It has NO "
    "timestamps. Do NOT add any [MM:SS] or [HH:MM:SS] markers to your summary."
)


@lru_cache(maxsize=8)
def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _safe_title(title: str | None) -> str:
    """Plug into prompts where {title} is expected. Empty when missing —
    no language-specific placeholder text."""
    return title.strip() if title and title.strip() else ""


def _source_note(*, from_audio_transcript: bool) -> str:
    return _TRANSCRIPT_SOURCE_NOTE if from_audio_transcript else _DOCUMENT_SOURCE_NOTE


def _build_single_pass_prompt(
    text: str, *, title: str | None, output_language: str, source_note: str
) -> str:
    template = _load_prompt("summary_single.txt")
    return template.format(
        output_language=output_language,
        title=_safe_title(title),
        text=text,
        source_note=source_note,
    )


async def _stream_single_pass(
    text: str,
    *,
    title: str | None,
    output_language: str,
    source_note: str,
) -> AsyncIterator[str]:
    prompt = _build_single_pass_prompt(
        text, title=title, output_language=output_language, source_note=source_note
    )
    async for delta in llm_client.stream_complete(prompt, max_tokens=2000, temperature=0.3):
        yield delta


async def _summarize_chunk(
    chunk: str,
    *,
    title: str | None,
    output_language: str,
    source_note: str,
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
        source_note=source_note,
    )
    return (await llm_client.complete(prompt, max_tokens=1500, temperature=0.3)).strip()


def _build_reduce_prompt(
    partials: list[str], *, title: str | None, output_language: str, source_note: str
) -> str:
    combined = "\n\n---\n\n".join(partials)
    template = _load_prompt("summary_reduce.txt")
    return template.format(
        output_language=output_language,
        title=_safe_title(title),
        combined=combined,
        source_note=source_note,
    )


async def _stream_reduce(
    partials: list[str],
    *,
    title: str | None,
    output_language: str,
    source_note: str,
) -> AsyncIterator[str]:
    prompt = _build_reduce_prompt(
        partials, title=title, output_language=output_language, source_note=source_note
    )
    async for delta in llm_client.stream_complete(prompt, max_tokens=2000, temperature=0.3):
        yield delta


async def _intermediate_reduce(
    group: list[str],
    *,
    title: str | None,
    output_language: str,
    source_note: str,
) -> str:
    """One non-streaming reduce pass over a GROUP of partial summaries.

    Only used when the final reduce's input would itself blow the context
    budget (see ``_fold_partials``). Reuses ``summary_reduce.txt`` rather
    than adding a second prompt file: its instructions ("combine these chunk
    summaries into Overview + Key points, preserve markers") describe
    exactly the operation an intermediate fold needs too — the caller just
    treats the output as one more partial instead of the final answer.
    """
    prompt = _build_reduce_prompt(
        group, title=title, output_language=output_language, source_note=source_note
    )
    return (await llm_client.complete(prompt, max_tokens=1500, temperature=0.3)).strip()


async def _fold_partials(
    partials: list[str],
    *,
    title: str | None,
    output_language: str,
    source_note: str,
    budget: int,
) -> list[str]:
    """Collapse `partials` until their joined size fits `budget`: even with
    every map-phase chunk correctly bounded, enough of them can still
    produce partials whose concatenation alone exceeds the reduce budget.

    Groups with ``pack_lines`` (a partial is atomic — never split mid-
    sentence, only combined with neighbours) and reduces each multi-item
    group via ``_intermediate_reduce``. Bounded by ``_MAX_REDUCE_FOLD_ITERS``
    and bails out early if a round can't shrink the count — both are just
    infinite-loop guards; the caller streams whatever comes back regardless.
    """
    current = partials
    for _ in range(_MAX_REDUCE_FOLD_ITERS):
        if len(current) <= 1:
            return current
        combined_tokens = count_tokens("\n\n---\n\n".join(current))
        if combined_tokens <= budget:
            return current
        groups = pack_lines(current, target_tokens=budget)
        if len(groups) >= len(current):
            # Packing couldn't shrink the count (every partial is already
            # its own oversized group) — further rounds can't help either.
            return current
        folded: list[str] = []
        for group in groups:
            if len(group) == 1:
                folded.append(group[0])
            else:
                folded.append(
                    await _intermediate_reduce(
                        group,
                        title=title,
                        output_language=output_language,
                        source_note=source_note,
                    )
                )
        current = folded
    return current


async def stream_summarize(
    text: str,
    *,
    title: str | None,
    output_language: str,
    from_audio_transcript: bool = False,
) -> AsyncIterator[str]:
    """Stream a summary of ``text`` token by token.

    For inputs below ``config.llm.single_pass_token_limit`` we ask the LLM
    once with streaming. For longer inputs we fall back to map-reduce: the
    map phase runs silently (chunks are summarised one at a time — streaming
    each would interleave nonsense, and ``llm.client._llm_lock()`` serialises
    every call anyway), then the reduce phase streams its output to the caller.

    ``from_audio_transcript`` adds a note telling the model the source is a
    speech-to-text transcript that may contain recognition errors to correct.
    """
    if not text or not text.strip():
        return

    note = _source_note(from_audio_transcript=from_audio_transcript)
    threshold = get_config().llm.single_pass_token_limit
    if count_tokens(text) < threshold:
        async for delta in _stream_single_pass(
            text, title=title, output_language=output_language, source_note=note
        ):
            yield delta
        return

    # Map-reduce path. Run map phase to completion, then stream the reduce.
    # Derive the chunk budget from `threshold` itself (capped at the usual
    # 4000) so a smaller-than-usual single_pass_token_limit can't hand back
    # a chunk too big to ever single-pass — with a normal config this is
    # just 4000, unchanged.
    chunk_target = min(_CHUNK_TARGET_TOKENS, max(1, threshold - 1))
    chunks = split_for_summary(text, target_tokens=chunk_target, overlap_tokens=_CHUNK_OVERLAP_TOKENS)
    if not chunks:
        return
    # Do NOT read "one chunk" as "safe to single-pass" — check the real size.
    # Should be unreachable now that chunk_target already respects threshold,
    # except for an unsplittable single "word" (chunking.py's own last-resort
    # case); if so, fall through to the ordinary map phase below instead of
    # single-passing an oversized chunk.
    if len(chunks) == 1 and count_tokens(chunks[0]) < threshold:
        async for delta in _stream_single_pass(
            chunks[0], title=title, output_language=output_language, source_note=note
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
                source_note=note,
                n=i + 1,
                total=total,
            )
        )
    # Reduce phase's own budget guard — see _fold_partials docstring for why
    # this is a separate failure mode from the map-phase chunk-size fix
    # above (correctly-bounded chunks can still produce enough partials that
    # THEIR join alone exceeds the model's context).
    partials = await _fold_partials(
        list(partials),
        title=title,
        output_language=output_language,
        source_note=note,
        budget=threshold,
    )
    async for delta in _stream_reduce(
        partials, title=title, output_language=output_language, source_note=note
    ):
        yield delta


__all__ = ["stream_summarize"]
