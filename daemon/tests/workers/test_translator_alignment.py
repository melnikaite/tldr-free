"""Tests for the translator's deterministic verification/repair machinery:

- ``_align_translation`` (marker-based line alignment, degeneration check,
  markerless fallback)
- ``_stream_group`` (the streaming repetition-loop guard)
- ``_translate_group`` (bisection + leaf fallback + call-budget exhaustion)
- end-to-end through ``_run``: a run with some fallback lines settles on
  ``status="partial"`` with a populated ``error``.

These complement ``tests/workers/test_translator.py`` (dedup/retry/restart
plumbing) — this file is about the "never trust the model's line
alignment" contract itself.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from src.llm.languages import normalize_lang
from src.storage import repo
from src.storage.db import (
    TranscriptTranslation,
    dispose_engine,
    init_engine,
    session_scope,
)
from src.storage.migrations import run_migrations
from src.workers import translator

_RU = normalize_lang("ru")
_PROMPT = "{transcript}"  # trivial template — these tests don't care about phrasing


@pytest.fixture
def isolated_db(tmp_path: Path):
    db_path = tmp_path / "translator_alignment.db"
    engine = init_engine(db_path)
    run_migrations(engine)
    try:
        yield engine
    finally:
        dispose_engine()


# ---------------------------------------------------------------------------
# _align_translation
# ---------------------------------------------------------------------------


def test_align_exact_one_to_one() -> None:
    input_lines = ["[00:00] hello", "[00:01] world"]
    output_lines = ["[00:00] привет", "[00:01] мир"]
    assert translator._align_translation(input_lines, output_lines) == output_lines


def test_align_absorbs_extra_unmarked_output_lines() -> None:
    """A model that splits one input line into two output lines (the
    second with no marker) gets merged back into one aligned line."""
    input_lines = ["[00:00] hello world"]
    output_lines = ["[00:00] hello", "world"]
    result = translator._align_translation(input_lines, output_lines)
    assert result == ["[00:00] hello world"]


def test_align_missing_marker_returns_none() -> None:
    input_lines = ["[00:00] a", "[00:01] b"]
    output_lines = ["[00:00] A"]  # [00:01] never shows up
    assert translator._align_translation(input_lines, output_lines) is None


def test_align_duplicate_markers_align_in_order() -> None:
    input_lines = ["[00:00] a", "[00:00] b"]
    output_lines = ["[00:00] A", "[00:00] B"]
    result = translator._align_translation(input_lines, output_lines)
    assert result == ["[00:00] A", "[00:00] B"]


def test_align_degenerate_run_returns_none() -> None:
    input_lines = ["[00:00] a"]
    output_lines = ["[00:00] loop"] + ["loop"] * 10
    assert translator._align_translation(input_lines, output_lines) is None


def test_align_degenerate_threshold_is_relative_to_the_input() -> None:
    """A source that itself legitimately repeats one line 28 times
    (measured on a real job: 28 consecutive Whisper ``[05:28] Ja.``
    segments) must not be rejected just because a FAITHFUL translation
    repeats just as much. The translator must not lean on the input
    already being clean (``timecodes.collapse_repeated_segments`` is a
    separate, upstream concern)."""
    input_lines = ["[00:00] Ja."] * 28
    output_lines = ["[00:00] Да."] * 28
    result = translator._align_translation(input_lines, output_lines)
    assert result == output_lines


def test_align_degenerate_run_beyond_input_justification_still_rejected() -> None:
    """The relative threshold is not a blank check — a source with only a
    short (3x) legitimate repeat does not excuse an output that loops far
    beyond it (30x)."""
    input_lines = ["[00:00] a"] * 3 + ["[00:01] b"]
    output_lines = ["[00:00] loop"] * 30
    assert translator._align_translation(input_lines, output_lines) is None


def test_align_rebuilds_marker_from_input_not_model() -> None:
    """The model drops the leading zero ([1:02] instead of [01:02]) — same
    time, mangled formatting. Alignment must still succeed AND the
    reconstructed line must carry the INPUT's own marker text verbatim,
    never the model's mangled copy."""
    input_lines = ["[01:02] hello"]
    output_lines = ["[1:02] привет"]
    result = translator._align_translation(input_lines, output_lines)
    assert result is not None
    assert result[0].startswith("[01:02]")
    assert "[1:02]" not in result[0]


def test_align_markerless_source_accepts_nonempty_output() -> None:
    """raw_text fallback (PDF/HTML) has no [MM:SS] markers at all —
    verification degrades to an emptiness/degeneration check."""
    input_lines = ["some paragraph text", "more text, no markers here"]
    output_lines = ["translated paragraph", "translated more text"]
    result = translator._align_translation(input_lines, output_lines)
    assert result == output_lines


def test_align_markerless_source_rejects_empty_output() -> None:
    input_lines = ["some paragraph text", "more text, no markers here"]
    assert translator._align_translation(input_lines, ["", ""]) is None


# ---------------------------------------------------------------------------
# Echo detection (group-level) — the model copied the input instead of
# translating it. Measured live (job cWiAdufn-6j8, EN transcript -> RU):
# qwen3-1.7b returned 139/139 lines byte-identical to the source and the
# old structural-only check accepted it as "done". See _group_is_echo.
# ---------------------------------------------------------------------------


def test_align_full_group_echo_is_rejected() -> None:
    """A group of N>1 lines that comes back byte-identical to the input
    (marker-bearing case) is NOT accepted — it must go down the same path
    as any other alignment failure (bisection / fallback), never straight
    to "done"."""
    input_lines = ["[00:00] hello", "[00:01] world", "[00:02] again"]
    output_lines = list(input_lines)  # byte-identical echo
    assert translator._align_translation(
        input_lines, output_lines, source_lang="en", target_lang="ru",
    ) is None


def test_align_single_line_echo_is_accepted() -> None:
    """Regression guard against an overly aggressive rule: once bisection
    has narrowed a mismatch down to ONE line, that line matching the
    source is normal (a number, a proper noun, "OK", "2024"), not evidence
    of an echo — a group of size 1 must never be flagged."""
    input_lines = ["[00:00] 2024"]
    output_lines = ["[00:00] 2024"]
    result = translator._align_translation(
        input_lines, output_lines, source_lang="en", target_lang="ru",
    )
    assert result == output_lines


def test_align_partial_echo_within_group_is_not_flagged() -> None:
    """A group where SOME lines happen to match the source (numbers, a
    proper noun) and others were genuinely translated is accepted whole —
    the echo signature is the WHOLE group matching, never a subset."""
    input_lines = ["[00:00] 2024", "[00:01] OK", "[00:02] hello world"]
    output_lines = ["[00:00] 2024", "[00:01] OK", "[00:02] привет мир"]
    result = translator._align_translation(
        input_lines, output_lines, source_lang="en", target_lang="ru",
    )
    assert result == output_lines


def test_align_full_echo_with_matching_source_and_target_lang_is_accepted() -> None:
    """Behavior unchanged from before the echo check existed when the
    target language equals the source language — a same-language
    "translation" is SUPPOSED to come back identical (though in practice
    ``enqueue_translation`` already short-circuits this case before any
    LLM call is made; this test pins the defensive guard inside
    ``_align_translation`` itself)."""
    input_lines = ["[00:00] hello", "[00:01] world", "[00:02] again"]
    output_lines = list(input_lines)
    result = translator._align_translation(
        input_lines, output_lines, source_lang="en", target_lang="en",
    )
    assert result == output_lines


def test_align_markerless_full_group_echo_is_rejected() -> None:
    """Markerless input (PDF/HTML raw_text) can still be an echo — the
    group-level check applies on top of the emptiness/degeneration-only
    check that markerless input degrades to. Predictable behavior: a
    markerless group of N>1 lines coming back byte-identical is rejected
    exactly like the marker-bearing case."""
    input_lines = ["some paragraph text", "more prose, no markers here"]
    output_lines = list(input_lines)
    assert translator._align_translation(
        input_lines, output_lines, source_lang="en", target_lang="ru",
    ) is None


def test_align_markerless_single_line_echo_is_accepted() -> None:
    """Predictable behavior, other half: a markerless group of exactly one
    line matching the source is accepted, same size-1 exemption as the
    marker-bearing case."""
    input_lines = ["Acme Corp"]
    output_lines = ["Acme Corp"]
    result = translator._align_translation(
        input_lines, output_lines, source_lang="en", target_lang="ru",
    )
    assert result == output_lines


def test_align_echo_check_does_not_guess_when_source_lang_unknown() -> None:
    """When ``job.transcript_language`` was never detected (``None``), the
    echo check must not guess via an alphabet heuristic — it runs the
    check rather than assuming same-language. A full-group echo with
    ``source_lang=None`` is still rejected."""
    input_lines = ["[00:00] hello", "[00:01] world", "[00:02] again"]
    output_lines = list(input_lines)
    assert translator._align_translation(
        input_lines, output_lines, source_lang=None, target_lang="ru",
    ) is None


@pytest.mark.asyncio
async def test_translate_group_full_echo_goes_to_fallback_not_silently_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end through ``_translate_group``: a model that copies its
    entire input back (measured live: qwen3-1.7b, 139/139 lines) must not
    reach ``done`` — the echo check flags the group at every level it's
    retried at, going down the ordinary bisection/fallback path exactly
    like any other verification failure.

    Note the interaction with the size-1 exemption (rule 2): if bisection
    were allowed to run all the way down to single-line granularity, a
    stub that echoes UNCONDITIONALLY at every call size would eventually
    have every line accepted at the leaf (matching a lone line is not
    itself evidence of an echo) — so this test caps ``_MAX_BISECT_DEPTH``
    at 1 to pin the case that actually matters: an echo group that can't
    be fully resolved down to exempted single lines within the
    depth/budget it's given must come back as a real, counted fallback,
    not silently accepted.
    """
    monkeypatch.setattr(translator, "_MAX_BISECT_DEPTH", 1)

    async def echo_stream(
        prompt: str, *, max_tokens: int, temperature: float, respect_pause: bool,
    ) -> AsyncIterator[str]:
        # Copies the input verbatim, like a model that ignored the
        # translate instruction entirely.
        yield prompt + "\n"

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", echo_stream)

    lines = _lines(8)
    budget = translator._CallBudget(50)
    output, fallback_count = await translator._translate_group(
        lines, _RU, _PROMPT, depth=0, budget=budget, source_lang="en",
    )
    # Every line is still present (never lost) but explicitly flagged as
    # a fallback rather than accepted as "translated".
    assert output == lines
    assert fallback_count == len(lines)


@pytest.mark.asyncio
async def test_run_partial_error_counts_echoed_lines_honestly(
    isolated_db, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``error`` message on a ``partial`` row must honestly count the
    lines that fell back — including ones that fell back because they
    were detected as an echo, not just structural mismatches. Same
    shallow-depth setup as the ``_translate_group`` test above, for the
    same reason: an echo that CAN be bisected all the way down to
    exempted single lines is expected (by design, rule 2) to end up
    "done" — this test pins the case where it can't."""
    monkeypatch.setattr(translator, "_MAX_BISECT_DEPTH", 1)

    async def echo_stream(
        prompt: str, *, max_tokens: int, temperature: float, respect_pause: bool,
    ) -> AsyncIterator[str]:
        yield _transcript_section(prompt)

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", echo_stream)

    raw_text = "\n".join(_lines(8)) + "\n"
    job = _seed_job(raw_text=raw_text)

    await translator.enqueue_translation(job.id, "ru")
    await _drain_tasks()

    with session_scope() as session:
        row = session.get(TranscriptTranslation, (job.id, "ru"))
        assert row is not None
        assert row.status == "partial"
        assert row.error is not None and "8 of 8" in row.error
        assert row.text is not None
        for line in _lines(8):
            assert line in row.text


# ---------------------------------------------------------------------------
# Task 2 diagnostic: does the observed fallback boundary line up with
# bisection-in-half, or does it just track wherever the model's own
# behavior changes?
#
# Live observation (two different models, same job): lines 0-33 came back
# as an echo and were accepted (pre-fix bug), 34-126 were genuinely
# translated, 127-138 honestly fell back. 34 ~= 139/4 raised the
# possibility that this is just deterministic bisection-in-half applied
# twice (139 -> 69/70 -> 34/35/35/35), not a separate bug.
#
# This test builds a ~139-line group against a stub LLM that echoes one
# sub-range, translates another, and always garbles a third (forcing a
# real fallback), with the echo/garble boundaries placed AWAY from any
# bisection cut point of 139 (69, then 34/35, then 17/18, ...). If the
# final fallback boundary in the OUTPUT were an artifact of bisection
# halving, it would snap toward one of those cut points; if it's
# content-driven, it lands exactly where the stub's own behavior changes.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bisection_boundary_tracks_content_not_halving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n = 139
    lines = _lines_n(n)
    # Deliberately NOT aligned with 139's bisection cut points (69, 34/35,
    # 17/18, ...): the echo/translate boundary sits at 50, the
    # translate/garble boundary at 110.
    echo_end = 50
    garble_start = 110

    async def stub(
        prompt: str, *, max_tokens: int, temperature: float, respect_pause: bool,
    ) -> AsyncIterator[str]:
        prompt_lines = prompt.rstrip("\n").split("\n")
        out = []
        for line in prompt_lines:
            m = re.match(r"^(\[(\d{2}):(\d{2})\])(.*)$", line)
            assert m is not None
            marker, mm, ss, rest = m.group(1), m.group(2), m.group(3), m.group(4)
            idx = int(mm) * 60 + int(ss)
            if idx < echo_end:
                out.append(f"{marker}{rest}")  # echo
            elif idx < garble_start:
                out.append(f"{marker} translated{rest}")
            else:
                out.append("garbage, no marker at all")  # forces a real failure
        yield "\n".join(out) + "\n"

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", stub)

    budget = translator._CallBudget(1000)
    output, _fallback_count = await translator._translate_group(
        lines, _RU, _PROMPT, depth=0, budget=budget, source_lang="en",
    )

    untranslated = [
        i for i, (src, out) in enumerate(zip(lines, output, strict=True)) if src == out
    ]

    # The observed boundary must match the CONTENT boundary (50, 110) —
    # not any of 139's bisection cut points (69, 34, 35, 17, 18, ...).
    # This is what actually happens: bisection only decides WHEN to
    # retry (on a verification failure), never WHERE the model's own
    # echo/translate/garble behavior changes within a call that verifies
    # fine as a whole. So the hypothesis in the live-data comment above
    # does NOT hold as a general mechanism — a boundary that happens to
    # sit near a bisection cut point (like the observed ~34) is
    # coincidental, not caused by "deterministic bisection-in-half
    # applied twice".
    assert untranslated[0] == 0
    assert untranslated[-1] == n - 1
    # The echoed prefix (content-driven) survives untouched...
    assert set(range(0, echo_end)) <= set(untranslated)
    # ...but genuinely translated lines in between must NOT appear as
    # "untranslated" just because they sit near a bisection cut point.
    assert 60 not in untranslated
    assert 100 not in untranslated


# ---------------------------------------------------------------------------
# _stream_group — streaming loop guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_group_aborts_runaway_repetition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model that loops on the same line forever must not be read to
    completion — the guard trips and the stream is abandoned early."""

    async def looping_stream(
        prompt: str, *, max_tokens: int, temperature: float, respect_pause: bool,
    ) -> AsyncIterator[str]:
        # Yield far more repeats than the guard threshold; if the guard
        # didn't work this generator would run to 500 lines.
        for _ in range(500):
            yield "[00:00] стоп стоп стоп\n"

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", looping_stream)

    output = await translator._stream_group(["[00:00] hello"], _RU, _PROMPT)
    non_empty = [line for line in output if line.strip()]
    # Guard trips shortly after _MAX_REPEATED_LINES — nowhere near 500.
    assert len(non_empty) <= translator._MAX_REPEATED_LINES + 2


@pytest.mark.asyncio
async def test_stream_group_does_not_abort_a_faithful_long_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The streaming guard's threshold floats up to match the INPUT's own
    repetition — a source with a 28-line legitimate repeat must be read
    to completion, not cut off at the fixed floor of 6."""
    input_lines = ["[00:00] Ja."] * 28

    async def faithful_repeat_stream(
        prompt: str, *, max_tokens: int, temperature: float, respect_pause: bool,
    ) -> AsyncIterator[str]:
        for _ in range(28):
            yield "[00:00] Да.\n"

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", faithful_repeat_stream)

    output = await translator._stream_group(input_lines, _RU, _PROMPT)
    non_empty = [line for line in output if line.strip()]
    assert len(non_empty) == 28


# ---------------------------------------------------------------------------
# _translate_group — bisection + leaf fallback + call budget
# ---------------------------------------------------------------------------


def _lines(n: int) -> list[str]:
    return [f"[00:{i:02d}] hello {i}" for i in range(n)]


def _transcript_section(prompt: str) -> str:
    """The real prompt template (loaded by ``_run`` via ``_load_prompt``)
    wraps the transcript in instructional preamble — pull just the
    transcript back out, mirroring what a real backend receives."""
    marker = "Input transcript:\n"
    if marker in prompt:
        return prompt.split(marker, 1)[1]
    return prompt


@pytest.mark.asyncio
async def test_bisection_isolates_a_single_failing_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fake LLM that only ever mistranslates ONE specific line still
    produces a full, aligned output for every other line — bisection
    narrows the mismatch down to just that line, which falls back to the
    source text (counted)."""
    failing_marker = "[00:03]"

    async def fake_stream(
        prompt: str, *, max_tokens: int, temperature: float, respect_pause: bool,
    ) -> AsyncIterator[str]:
        lines = prompt.rstrip("\n").split("\n")
        out = []
        for line in lines:
            m = re.match(r"^(\[\d{2}:\d{2}\])(.*)$", line)
            assert m is not None
            marker, rest = m.group(1), m.group(2)
            if marker == failing_marker:
                out.append("garbage output with no marker at all")
            else:
                out.append(f"{marker} translated{rest}")
        yield "\n".join(out) + "\n"

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", fake_stream)

    lines = _lines(8)
    budget = translator._CallBudget(50)
    output, fallback_count = await translator._translate_group(
        lines, _RU, _PROMPT, depth=0, budget=budget,
    )

    assert fallback_count == 1
    assert len(output) == 8
    # The failing line survived verbatim from the source.
    assert lines[3] in output
    # Every other line was actually translated.
    for i in range(len(lines)):
        if i == 3:
            continue
        assert any(out_line.startswith(f"[00:{i:02d}] translated") for out_line in output)


def _lines_n(n: int) -> list[str]:
    """Like ``_lines`` but safe for n > 60 (rolls minutes over)."""
    return [f"[{i // 60:02d}:{i % 60:02d}] hello {i}" for i in range(n)]


@pytest.mark.asyncio
async def test_bisection_reaches_single_line_granularity_on_real_sized_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression pin for the depth/granularity relationship:
    ``pack_lines`` routinely produces 100+-line groups (measured live: a
    656-line transcript packed into 4 groups of 145-177 lines each), and
    ONE bad line among them must isolate to a fallback of exactly 1 —
    not an ~11-line block, which is what happened when
    ``_MAX_BISECT_DEPTH`` bottomed out above single-line granularity on a
    group this size. If a future change to the depth constant regresses
    this, this test catches it.

    The failing line is the LAST one — bisection's ``lines[mid:]`` half
    always contains the last index, so this pins the worst-case (deepest)
    path: 200 → 100 → 50 → 25 → 13 → 7 → 4 → 2 → 1, exactly
    ``_MAX_BISECT_DEPTH`` (8) splits.
    """
    n = 200
    lines = _lines_n(n)
    failing_marker = lines[-1].split("]", 1)[0] + "]"

    async def fake_stream(
        prompt: str, *, max_tokens: int, temperature: float, respect_pause: bool,
    ) -> AsyncIterator[str]:
        prompt_lines = prompt.rstrip("\n").split("\n")
        out = []
        for line in prompt_lines:
            m = re.match(r"^(\[\d{2}:\d{2}\])(.*)$", line)
            assert m is not None
            marker, rest = m.group(1), m.group(2)
            if marker == failing_marker:
                out.append("garbage output with no marker at all")
            else:
                out.append(f"{marker} translated{rest}")
        yield "\n".join(out) + "\n"

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", fake_stream)

    budget = translator._CallBudget(200)
    output, fallback_count = await translator._translate_group(
        lines, _RU, _PROMPT, depth=0, budget=budget,
    )

    assert fallback_count == 1
    assert len(output) == n
    assert lines[-1] in output
    for i in range(n - 1):
        marker = lines[i].split("]", 1)[0] + "]"
        assert any(
            out_line.startswith(f"{marker} translated") for out_line in output
        )


@pytest.mark.asyncio
async def test_translate_group_faithful_long_repeat_does_not_bisect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A group whose source legitimately repeats one line 28 times must
    align on the FIRST call — no bisection, zero fallback. Regression for
    the fixed-threshold bug where a genuinely repetitive source forced
    every sub-group into fallback."""
    lines = ["[00:00] Ja."] * 28
    call_count = 0

    async def faithful_repeat_stream(
        prompt: str, *, max_tokens: int, temperature: float, respect_pause: bool,
    ) -> AsyncIterator[str]:
        nonlocal call_count
        call_count += 1
        yield "\n".join(["[00:00] Да."] * 28) + "\n"

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", faithful_repeat_stream)

    budget = translator._CallBudget(50)
    output, fallback_count = await translator._translate_group(
        lines, _RU, _PROMPT, depth=0, budget=budget,
    )

    assert fallback_count == 0
    assert output == ["[00:00] Да."] * 28
    assert call_count == 1  # aligned immediately — no bisection needed


@pytest.mark.asyncio
async def test_on_delta_propagates_through_bisection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Progress must not freeze for the whole duration of a bisecting
    group — ``on_delta`` has to reach every recursive/retry
    ``_stream_group`` call, not just the initial top-level attempt."""
    delta_calls = 0

    async def always_mismatch(
        prompt: str, *, max_tokens: int, temperature: float, respect_pause: bool,
    ) -> AsyncIterator[str]:
        yield "always wrong, no markers whatsoever"

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", always_mismatch)

    def on_delta(_received_text: str) -> None:
        nonlocal delta_calls
        delta_calls += 1

    lines = _lines(4)
    budget = translator._CallBudget(50)
    await translator._translate_group(
        lines, _RU, _PROMPT, depth=0, budget=budget, on_delta=on_delta,
    )

    # Every mismatching call — root + every bisection/retry leaf — streams
    # at least one delta, so on_delta must fire more than once (were it
    # only wired to the top-level call, this would be exactly 1).
    assert delta_calls > 1


@pytest.mark.asyncio
async def test_call_budget_exhaustion_falls_back_without_fanning_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model that mismatches on every single call must not be allowed to
    bisect indefinitely — once the budget is spent, remaining lines fall
    back to source immediately."""
    call_count = 0

    async def always_mismatch(
        prompt: str, *, max_tokens: int, temperature: float, respect_pause: bool,
    ) -> AsyncIterator[str]:
        nonlocal call_count
        call_count += 1
        yield "always wrong, no markers whatsoever\n"

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", always_mismatch)

    lines = _lines(16)
    budget = translator._CallBudget(5)
    output, fallback_count = await translator._translate_group(
        lines, _RU, _PROMPT, depth=0, budget=budget,
    )

    # Nothing is lost — every source line is present somewhere in the output.
    assert fallback_count == 16
    assert output == lines
    # The budget was never exceeded.
    assert call_count <= 5


@pytest.mark.asyncio
async def test_bisection_depth_limit_falls_back_whole_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def always_mismatch(
        prompt: str, *, max_tokens: int, temperature: float, respect_pause: bool,
    ) -> AsyncIterator[str]:
        yield "always wrong\n"

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", always_mismatch)

    lines = _lines(4)
    budget = translator._CallBudget(1000)
    output, fallback_count = await translator._translate_group(
        lines, _RU, _PROMPT, depth=translator._MAX_BISECT_DEPTH, budget=budget,
    )
    assert output == lines
    assert fallback_count == 4


# ---------------------------------------------------------------------------
# End-to-end through _run: partial status
# ---------------------------------------------------------------------------


def _seed_job(*, raw_text: str, source_lang: str = "en") -> Any:
    job = repo.create_job(url="https://x/align", kind="media", title="Test")
    repo.mark_done(
        job.id,
        raw_text=raw_text,
        summary_md="ok",
        transcript_source="whisper",
        transcript_language=source_lang,
    )
    fresh = repo.get_job(job.id)
    assert fresh is not None
    return fresh


async def _drain_tasks() -> None:
    for _ in range(200):
        if not translator._BACKGROUND_TASKS:
            return
        await asyncio.sleep(0.02)
    raise AssertionError("translator background task did not finish in time")


@pytest.mark.asyncio
async def test_run_yields_partial_status_with_error_on_fallback(
    isolated_db, monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing_marker = "[00:03]"

    async def fake_stream(
        prompt: str, *, max_tokens: int, temperature: float, respect_pause: bool,
    ) -> AsyncIterator[str]:
        lines = _transcript_section(prompt).rstrip("\n").split("\n")
        out = []
        for line in lines:
            m = re.match(r"^(\[\d{2}:\d{2}\])(.*)$", line)
            if m is None:
                out.append(line)
                continue
            marker, rest = m.group(1), m.group(2)
            if marker == failing_marker:
                out.append("garbage with no marker")
            else:
                out.append(f"{marker} translated{rest}")
        yield "\n".join(out) + "\n"

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", fake_stream)

    raw_text = "\n".join(_lines(6)) + "\n"
    job = _seed_job(raw_text=raw_text)

    await translator.enqueue_translation(job.id, "ru")
    await _drain_tasks()

    with session_scope() as session:
        row = session.get(TranscriptTranslation, (job.id, "ru"))
        assert row is not None
        assert row.status == "partial"
        assert row.progress_percent == 100
        assert row.error is not None and "1 of 6" in row.error
        assert row.text is not None
        # The failing line is present verbatim (untranslated) in the text.
        assert "[00:03] hello 3" in row.text
        # Every other line got translated.
        assert "[00:00] translated" in row.text


@pytest.mark.asyncio
async def test_run_happy_path_still_yields_done_and_full_output(
    isolated_db, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def clean_stream(
        prompt: str, *, max_tokens: int, temperature: float, respect_pause: bool,
    ) -> AsyncIterator[str]:
        lines = _transcript_section(prompt).rstrip("\n").split("\n")
        out = []
        for line in lines:
            m = re.match(r"^(\[\d{2}:\d{2}\])(.*)$", line)
            assert m is not None
            marker, rest = m.group(1), m.group(2)
            out.append(f"{marker} translated{rest}")
        yield "\n".join(out) + "\n"

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", clean_stream)

    raw_text = "\n".join(_lines(10)) + "\n"
    job = _seed_job(raw_text=raw_text)

    await translator.enqueue_translation(job.id, "ru")
    await _drain_tasks()

    with session_scope() as session:
        row = session.get(TranscriptTranslation, (job.id, "ru"))
        assert row is not None
        assert row.status == "done"
        assert row.progress_percent == 100
        assert row.error is None
        assert row.text is not None
        assert row.text.count("\n") == 9  # 10 lines, 9 internal newlines
        for i in range(10):
            assert f"[00:{i:02d}] translated hello {i}" in row.text


@pytest.mark.asyncio
async def test_run_markerless_source_still_works(
    isolated_db, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PDF/HTML-style raw_text with no [MM:SS] markers at all still
    produces a done translation via the emptiness/degeneration-only
    check."""

    async def clean_stream(
        prompt: str, *, max_tokens: int, temperature: float, respect_pause: bool,
    ) -> AsyncIterator[str]:
        lines = _transcript_section(prompt).rstrip("\n").split("\n")
        yield "\n".join(f"translated: {line}" for line in lines) + "\n"

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", clean_stream)

    raw_text = "\n".join(f"paragraph number {i} with plain prose." for i in range(5))
    job = _seed_job(raw_text=raw_text)

    await translator.enqueue_translation(job.id, "ru")
    await _drain_tasks()

    with session_scope() as session:
        row = session.get(TranscriptTranslation, (job.id, "ru"))
        assert row is not None
        assert row.status == "done"
        assert row.text is not None
        assert "translated: paragraph number 0" in row.text


@pytest.mark.asyncio
async def test_retry_after_partial_clears_stale_error_on_full_success(
    isolated_db, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A row sitting in ``partial`` (with a populated ``error``) that gets
    retried and this time aligns perfectly must come back ``done`` with
    NO leftover error text — ``_update_status`` only ever WRITES
    ``error`` when given a non-None value, so the reset path
    (``retry_all_failed`` → ``_reset_failed_rows``) is what's responsible
    for clearing the stale value before the retry runs. This pins that
    invariant end-to-end rather than assuming it holds."""
    job = _seed_job(raw_text="\n".join(_lines(4)) + "\n")
    with session_scope() as session:
        session.add(TranscriptTranslation(
            job_id=job.id, language_code="ru", status="partial",
            progress_percent=100,
            text="stale partial text",
            error="2 of 4 lines could not be translated and are shown in the original language",
        ))

    async def clean_stream(
        prompt: str, *, max_tokens: int, temperature: float, respect_pause: bool,
    ) -> AsyncIterator[str]:
        lines = _transcript_section(prompt).rstrip("\n").split("\n")
        out = []
        for line in lines:
            m = re.match(r"^(\[\d{2}:\d{2}\])(.*)$", line)
            assert m is not None
            marker, rest = m.group(1), m.group(2)
            out.append(f"{marker} translated{rest}")
        yield "\n".join(out) + "\n"

    from src.llm import client as llm_client
    monkeypatch.setattr(llm_client, "stream_complete", clean_stream)

    retried = await translator.retry_all_failed(job.id)
    assert [r["language_code"] for r in retried] == ["ru"]

    await _drain_tasks()

    with session_scope() as session:
        row = session.get(TranscriptTranslation, (job.id, "ru"))
        assert row is not None
        assert row.status == "done"
        assert row.error is None
        assert row.text is not None and "stale partial text" not in row.text
