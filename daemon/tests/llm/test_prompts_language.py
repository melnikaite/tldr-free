"""Verify every prompt template:
- Accepts the placeholders the production code passes
- Has no leftover `{...}` placeholders after a representative format() call
- Can be formatted with output_language="English" without error
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "src" / "prompts"

_PROMPT_KWARGS: dict[str, dict[str, object]] = {
    "summary_single.txt": {
        "output_language": "English",
        "title": "Sample title",
        "text": "Some text body.",
    },
    "summary_chunk.txt": {
        "output_language": "English",
        "title": "Sample title",
        "chunk": "Chunk body.",
        "n": 1,
        "total": 3,
    },
    "summary_reduce.txt": {
        "output_language": "English",
        "title": "Sample title",
        "combined": "Partial 1\n\n---\n\nPartial 2",
    },
    "qa.txt": {
        "output_language": "English",
        "title": "Sample title",
        "context": "Material body.",
        "question": "What's the main point?",
    },
}


@pytest.mark.parametrize("name,kwargs", list(_PROMPT_KWARGS.items()))
def test_prompt_formats_cleanly(name: str, kwargs: dict[str, object]) -> None:
    template = (_PROMPTS_DIR / name).read_text(encoding="utf-8")
    formatted = template.format(**kwargs)

    # No leftover placeholders. Allow literal '{' if escaped as '{{' in
    # source — but the prompts as drafted have no such escapes, so any
    # remaining `{name}` is a bug.
    leftovers = re.findall(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", formatted)
    assert not leftovers, f"{name}: unfilled placeholders {leftovers}"

    # output_language must be substituted into the body somewhere.
    assert "English" in formatted, f"{name}: output_language not visible in body"


def test_all_prompt_files_present() -> None:
    """All four prompt files exist."""
    for name in _PROMPT_KWARGS:
        path = _PROMPTS_DIR / name
        assert path.is_file(), f"Missing prompt file: {path}"


def test_prompts_use_output_language_placeholder() -> None:
    """Each prompt template must reference {output_language} at least once
    (so config-driven language switching actually works)."""
    for name in _PROMPT_KWARGS:
        body = (_PROMPTS_DIR / name).read_text(encoding="utf-8")
        assert "{output_language}" in body, (
            f"{name} does not reference {{output_language}} — language is hardcoded"
        )
