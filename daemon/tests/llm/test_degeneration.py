"""Tests for llm.degeneration — the single repetition-loop guard.

Covers the real Q&A incident (message 267: a closing paragraph repeated 4x,
two copies reworded), period-1/2/4 cycles, filler runs (<br>, bare
timecodes), legit-content negatives, the streaming guard (closes the
upstream generator), and the wiring into summary + Q&A.
"""

from __future__ import annotations

import json
import logging
import types
from pathlib import Path
from typing import Any

import pytest

from src.llm import client as llm_client
from src.llm import degeneration
from src.llm import summary as summary_mod
from src.llm.degeneration import find_loop, guard_stream, trim

_FIXTURES = Path(__file__).parent / "fixtures"
INCIDENT_267 = (_FIXTURES / "qa_loop_msg267.txt").read_text(encoding="utf-8")
# The closing paragraph that the model then looped on.
_LOOP_HEAD = "Таким образом, утверждения в вопросе не соответствуют"

PARA_A = (
    "The rate limiter keeps a sliding window per API key and rejects requests "
    "once the window is full, which the author says prevents bursts from "
    "starving other tenants on the shared cluster during peak hours."
)
PARA_B = (
    "Caching is layered: an in-process LRU in front of Redis, and the author "
    "measured that the LRU alone absorbs most reads for hot keys, so Redis "
    "traffic dropped sharply after it was introduced last spring."
)
PARA_C = (
    "Deployments go through a canary stage that receives a small share of "
    "traffic for an hour before the rollout continues automatically if error "
    "rates stay flat compared with the previous release."
)
PARA_D = (
    "On-call engineers get paged only for symptoms visible to users, never for "
    "internal causes, which the team credits with halving the number of night "
    "pages within a single quarter of adopting the policy."
)
INTRO = "Here is what the material says about the system design.\n\n"


async def _aiter(items: list[str]):
    for it in items:
        yield it


async def _collect(stream) -> str:
    return "".join([d async for d in stream])


def _chunks(text: str, size: int = 17) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


# ---------------------------------------------------------------------------
# find_loop / trim: positives
# ---------------------------------------------------------------------------


def test_incident_267_is_cut_before_second_copy() -> None:
    assert INCIDENT_267.count(_LOOP_HEAD) >= 2
    out = trim(INCIDENT_267)
    # Everything up to the first copy of the closing paragraph survives...
    assert out.startswith("Инсулинорезистентность")
    assert len(out) > 2500
    # ...and no copy of the looped paragraph is left behind.
    assert _LOOP_HEAD not in out
    assert out == out.rstrip()


def test_incident_267_detected_while_streaming() -> None:
    # Every 64-char prefix check, as guard_stream does: must fire before the end.
    hits = [e for e in range(64, len(INCIDENT_267), 64) if find_loop(INCIDENT_267[:e])]
    assert hits and hits[0] < len(INCIDENT_267) * 0.8


def test_period_1_paragraph_loop() -> None:
    text = INTRO + "\n\n".join([PARA_A, PARA_B] + [PARA_C] * 4)
    out = trim(text)
    assert out.count(PARA_C) == 1
    assert out.endswith(PARA_C)


def test_period_2_cycle() -> None:
    text = INTRO + "\n\n".join([PARA_A] + [PARA_B, PARA_C] * 3)
    out = trim(text)
    assert out.count(PARA_B) == 1 and out.count(PARA_C) == 1


def test_period_4_cycle_of_numbered_bullets() -> None:
    # Numbers differ per copy ("5." vs "1.") — must still match.
    items = [PARA_A, PARA_B, PARA_C, PARA_D]
    lines = [f"{n + 1}. {items[n % 4]}" for n in range(12)]
    out = trim(INTRO + "\n".join(lines))
    for p in items:
        assert out.count(p) == 1
    assert out.rstrip().endswith(PARA_D)


def test_reworded_copies_still_caught() -> None:
    variant = PARA_C.replace("for an hour", "for roughly an hour").replace(
        "automatically", "on its own"
    )
    text = INTRO + "\n\n".join([PARA_A, PARA_C, PARA_C, variant, PARA_C])
    out = trim(text)
    assert out.count("canary stage") == 1


def test_timecoded_bullet_loop_with_varying_timecodes() -> None:
    line = "He prefers setups where volume on green candles is higher than on red ones."
    bullets = [f"- [{20 + i:02d}:{i * 7 % 60:02d}] {line}" for i in range(6)]
    text = INTRO + "- [10:00] An unrelated opening point about risk sizing.\n" + "\n".join(bullets)
    out = trim(text)
    assert out.count(line) == 1


def test_br_filler_run() -> None:
    assert trim("Real answer.\n" + "<br>\n" * 6) == "Real answer."
    assert find_loop("Real answer.\n" + "<br>\n" * 5) is None


def test_bare_timecode_dump() -> None:
    tail = "".join(f"[{m:02d}:00]\n" for m in range(8))
    assert trim("Answer about the video.\n" + tail) == "Answer about the video."


def test_mixed_tag_and_timecode_filler() -> None:
    text = "Answer.\n</blockquote>\n<br>\n[00:00]\n<br>\n[01:00]\n<br>"
    assert trim(text) == "Answer."


# ---------------------------------------------------------------------------
# negatives: legit content
# ---------------------------------------------------------------------------


def _assert_clean_everywhere(text: str) -> None:
    for e in [*range(64, len(text), 64), len(text)]:
        assert find_loop(text[:e]) is None, text[:e][-200:]


def test_clean_prose() -> None:
    _assert_clean_everywhere(INTRO + "\n\n".join([PARA_A, PARA_B, PARA_C, PARA_D]))


def test_inline_timecodes() -> None:
    _assert_clean_everywhere("First point [00:30]. Second point [01:00]. Third [02:00].")


def test_timecoded_list_with_shared_phrasing() -> None:
    topics = [
        "pricing of the base model",
        "battery life in daily use",
        "accuracy of sleep tracking",
        "comfort of the strap at night",
        "the subscription model",
        "heart rate during intervals",
        "the companion app's coaching",
        "water resistance while swimming",
        "how stress scores are computed",
        "export of raw data",
    ]
    text = "## Key points\n" + "\n".join(
        f"- [{i:02d}:{i * 3 % 60:02d}] The author says that {t} matters more than expected."
        for i, t in enumerate(topics)
    )
    _assert_clean_everywhere(text)


def test_short_identical_list_items() -> None:
    _assert_clean_everywhere("Answers:\n" + "- Yes.\n- No.\n" * 6)


def test_markdown_table() -> None:
    rows = "\n".join(
        f"| Device {i} | {10 + i} days | {i * 3}% error | Good choice for those who want a tracker |"
        for i in range(8)
    )
    _assert_clean_everywhere("| Device | Battery | Error | Verdict |\n|---|---|---|---|\n" + rows)


def test_quote_or_takeaway_repeated_once() -> None:
    # A takeaway restated verbatim in a bullet (2 copies) is not a loop.
    text = "## Key takeaways\n" + PARA_A + "\n\n## Key points\n- " + PARA_B + "\n- " + PARA_A
    _assert_clean_everywhere(text)
    assert trim(text) == text


def test_empty() -> None:
    assert find_loop("") is None
    assert trim("") == ""


# ---------------------------------------------------------------------------
# streaming guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guard_stream_passes_clean_stream_through() -> None:
    text = INTRO + "\n\n".join([PARA_A, PARA_B, PARA_C, PARA_D])
    assert await _collect(guard_stream(_aiter(_chunks(text)))) == text


@pytest.mark.asyncio
async def test_guard_stream_stops_and_closes_infinite_loop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    closed = False

    async def gen():
        nonlocal closed
        try:
            yield INTRO
            while True:
                for p in (PARA_A, PARA_B):
                    for c in _chunks(p + "\n\n"):
                        yield c
        finally:
            closed = True

    with caplog.at_level(logging.WARNING, logger="src.llm.degeneration"):
        out = await _collect(guard_stream(gen(), label="t"))
    assert closed
    assert len(out) < 2000
    assert trim(out).count(PARA_A) == 1
    assert any("degeneration loop detected (t)" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_guard_stream_incident_267_stops_early() -> None:
    out = await _collect(guard_stream(_aiter(_chunks(INCIDENT_267, 11))))
    assert len(out) < len(INCIDENT_267)
    assert _LOOP_HEAD not in trim(out)


# ---------------------------------------------------------------------------
# wiring: summary + Q&A
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_summarize_stops_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_stream(prompt: str, **kwargs: object):
        async def gen():
            yield "## Key takeaways\n\n"
            while True:
                yield PARA_A + "\n"
                yield PARA_B + "\n"

        return gen()

    monkeypatch.setattr(llm_client, "stream_complete", fake_stream)
    out = await _collect(
        summary_mod.stream_summarize("short source text" * 3, title="T", output_language="English")
    )
    assert trim(out).count(PARA_A) == 1


@pytest.mark.asyncio
async def test_qa_done_event_and_stored_answer_are_trimmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.api import ai as ai_mod

    closed = False

    async def fake_answer(**kwargs: Any):
        nonlocal closed
        try:
            yield {"type": "stage", "stage": "thinking"}
            for c in _chunks(INCIDENT_267, 23):
                yield c
            while True:  # keep looping like a model stuck until max_tokens
                yield "\n\n" + INCIDENT_267[INCIDENT_267.index(_LOOP_HEAD) :][:1100]
        finally:
            closed = True

    stored: list[str] = []

    def fake_add_message(job_id: str, *, role: str, content: str, **kw: Any):
        if role == "assistant":
            stored.append(content)
        return types.SimpleNamespace(id=len(stored))

    monkeypatch.setattr(ai_mod.llm_qa, "stream_answer", fake_answer)
    monkeypatch.setattr(ai_mod.repo, "add_message", fake_add_message)
    job = types.SimpleNamespace(id="j1", raw_text="x", summary_md="y", transcript_source="whisper")
    frames = [f async for f in ai_mod._qa_stream(job, "q")]
    done = json.loads(frames[-1].removeprefix("data: "))
    assert done["type"] == "done"
    assert closed
    assert _LOOP_HEAD not in done["content"]
    assert stored == [done["content"]]
    assert done["content"].startswith("Инсулинорезистентность")


def test_module_constants_sane() -> None:
    assert degeneration._MIN_PRIOR >= 2  # 2nd copy alone must never cut
