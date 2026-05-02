"""Token counting via tiktoken cl100k_base (proxy for Qwen tokenizer).

Public surface:
    count_tokens(text: str) -> int
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
