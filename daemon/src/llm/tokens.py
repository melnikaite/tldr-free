"""Token counting via tiktoken cl100k_base (proxy for Qwen tokenizer).

Public surface:
    count_tokens(text: str) -> int
    make_filler_text(token_count: int) -> str
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the number of tokens in `text` per cl100k_base.

    Used as a budget proxy for Qwen — close enough for chunking decisions.
    """
    if not text:
        return 0
    return len(_encoding().encode(text))


def make_filler_text(token_count: int) -> str:
    """Build a text blob whose cl100k_base token count is exactly
    ``token_count`` — used by ``POST /config/test`` to probe a backend's
    real context ceiling with a deliberately oversized request.

    Repeats a single filler token id, so the count is exact against THIS
    tokenizer regardless of ``token_count``'s size. A real backend almost
    certainly uses a different tokenizer and will report a different count
    for the same text — that's fine, callers parse the backend's OWN
    reported numbers out of its error message rather than trusting this
    count to travel unchanged over the wire; this function only needs to
    produce "big enough to trip a real context ceiling", not an exact
    number anyone downstream relies on.
    """
    if token_count <= 0:
        return ""
    enc = _encoding()
    filler_id = enc.encode(" the")[0]
    return enc.decode([filler_id] * token_count)
