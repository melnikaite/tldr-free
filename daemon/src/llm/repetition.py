"""Guard against a local model locking into a line-repetition loop.

Public surface:
    async def abort_on_repeated_lines(stream, *, min_line_chars=.., occurrence_threshold=..)
            -> AsyncIterator[str]
        Wraps a raw LLM delta stream. Once a countable line reaches its
        ``occurrence_threshold``-th occurrence ANYWHERE in the text so far
        (not necessarily adjacent to its previous occurrence), that
        occurrence is dropped (never yielded) and the underlying stream is
        closed early — the caller never sees the repeated tail, and the LLM
        is stopped instead of burning its whole ``max_tokens`` budget on
        the loop.

    def trim_repeated_tail(text, *, min_line_chars=.., occurrence_threshold=..) -> str
        Same detection, applied after the fact to an already-complete
        string (used for the non-streaming map/reduce calls, which have no
        stream to abort — but a degenerate chunk summary must not be fed
        into the next reduce round either).

Why this exists: measured on real jobs, a small local model finishing a
summary sometimes locks into a loop over a small set of lines and repeats
them until ``max_tokens`` is exhausted. A single line repeated back-to-back
is just the simplest (period-1) case of this — measured incidents also
include a period-2 alternation between two bullets and a period-4 cycle
over four, none of which ever places two occurrences of the same line
NEXT to each other. So the rule can't be "N consecutive repeats"; it has
to be "N occurrences of the same countable line, anywhere in the text so
far". The Q&A path already has a comparable guard (``api/ai.py``'s
``_DEGEN_TAIL_RE`` + ``llm/qa.py``'s ``clean_answer``), but that one
matches boilerplate FILLER via regex (a run of ``<br>`` tags, bare
timecodes) — wrong shape for this failure, which is an arbitrary,
otherwise-legitimate-looking line (or small set of lines) repeated
verbatim. Distantly related to
``workers/timecodes.collapse_repeated_segments`` (also a repeat-collapse),
but that one only ever collapses ADJACENT duplicates in already-split
Whisper segments; this operates on raw streamed/complete text and must
catch repeats that are not adjacent at all.

Threshold calibration — measured against every stored summary in the
production DB (99 jobs), varying the occurrence threshold used by
``trim_repeated_tail``:

    threshold | jobs flagged | known-bad caught | false positives
    2         | 7            | 6/6               | 1
    3         | 6            | 6/6               | 0
    4         | 6            | 6/6               | 0
    5         | 6            | 6/6               | 0

  - ``_MIN_REPEAT_LINE_CHARS`` (20): short lines are EXCLUDED from
    occurrence counting entirely — they never contribute to, or get
    flagged by, the threshold. A markdown bullet marker plus a terse
    legitimate answer ("- Да.", "- Yes.", "- N/A") sits comfortably under
    this, so genuine short-item repetition (a list of one-word verdicts,
    repeated arbitrarily many times) never trips the guard. A real prose
    bullet ("- The API rate limit is..." etc.) is well over it. This
    minimum is what makes a low occurrence threshold safe at all — without
    it, threshold 2 also flags legitimate short repeated items.
  - ``_REPEAT_OCCURRENCE_THRESHOLD`` (3): three occurrences of the same
    countable line anywhere in the text. Threshold 2 catches every known
    incident but also flags one legitimate summary (a benign single
    duplicate long line) — one false positive across 99 real jobs.
    Threshold 3 is the first value with ZERO false positives while still
    catching all six known-bad jobs (which repeat their loop line 8 to 25
    times); raising it further (4, 5) catches the same six with no
    additional benefit, so 3 is the tightest value that stays clean.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

log = logging.getLogger(__name__)

# See module docstring "Threshold calibration" for the reasoning and the
# measured table behind both numbers.
_MIN_REPEAT_LINE_CHARS = 20
_REPEAT_OCCURRENCE_THRESHOLD = 3


def _is_countable(line: str, min_line_chars: int) -> bool:
    return len(line.strip()) >= min_line_chars


def trim_repeated_tail(
    text: str,
    *,
    min_line_chars: int = _MIN_REPEAT_LINE_CHARS,
    occurrence_threshold: int = _REPEAT_OCCURRENCE_THRESHOLD,
) -> str:
    """Cut ``text`` at the point some line's repeat count reaches the loop
    threshold — anywhere in the text, not just back-to-back.

    Walks ``text`` line by line, counting occurrences of each "countable"
    (``>= min_line_chars``) line seen so far, keyed by its exact text. As
    soon as any line's count would reach ``occurrence_threshold``, that
    line and everything after it (in the whole text) is dropped — trailing
    blank lines left behind are trimmed too — i.e. every countable line
    keeps at most ``occurrence_threshold - 1`` surviving occurrences, and
    the text ends at the first line that would have broken that limit.
    This catches a period-1 back-to-back repeat as a special case, but also
    an ABAB alternation or any other small-set cycle, since it never
    requires two occurrences to be adjacent. Short lines are never counted
    and never trigger a cut — see module docstring for why that keeps
    legitimate short repeated list items safe.

    No-op (returns ``text`` unchanged) when no line ever reaches the
    threshold. Pure: same input -> same output.
    """
    if not text:
        return text
    lines = text.split("\n")
    counts: dict[str, int] = {}
    cut_at: int | None = None
    for i, line in enumerate(lines):
        if not _is_countable(line, min_line_chars):
            continue
        seen = counts.get(line, 0)
        if seen >= occurrence_threshold - 1:
            cut_at = i
            break
        counts[line] = seen + 1

    if cut_at is None:
        return text
    kept = lines[:cut_at]
    while kept and not kept[-1].strip():
        kept.pop()
    log.warning(
        "trim_repeated_tail: dropped a degenerate repeated-line loop "
        "starting at line %d (of %d)",
        cut_at,
        len(lines),
    )
    return "\n".join(kept)


class _RepeatGuardState:
    """Incremental line-repeat detector backing ``abort_on_repeated_lines``.

    Generalisation of a simple "hold while matching one reference line"
    state machine to "hold while matching any line already seen
    ``occurrence_threshold - 1`` times" — those are the only lines whose
    NEXT occurrence would trigger a cut, so they are the only ones worth
    buffering for. Mirrors ``workers.timecodes._MarkerCapState``'s design:
    holds text back only while it MIGHT still turn into a triggering
    repeat, and flushes immediately the moment that stops being possible —
    so ordinary (non-repeating, or not-yet-suspicious) generation streams
    through with no added latency. A brand new line, or one seen only once
    before, is never held at all (it cannot possibly trigger a cut by
    completing); only once a line has already recurred once BEFORE does its
    next occurrence get buffered while it's still a live candidate — and
    even then, real generation diverges from every candidate within the
    first few characters in the overwhelming common case.

    ``candidates`` is recomputed only right after a line completes (the
    only point at which ``counts`` can change), not per character — a
    summary is on the order of tens of lines, so a plain set + linear scan
    is simpler than anything trie-like and the cost is irrelevant.
    """

    __slots__ = (
        "min_line_chars",
        "occurrence_threshold",
        "counts",
        "candidates",
        "cur_line",
        "held",
        "aborted",
    )

    def __init__(self, *, min_line_chars: int, occurrence_threshold: int) -> None:
        self.min_line_chars = min_line_chars
        self.occurrence_threshold = occurrence_threshold
        self.counts: dict[str, int] = {}
        self.candidates: set[str] = set()
        self.cur_line = ""
        self.held = ""
        self.aborted = False

    def _recompute_candidates(self) -> None:
        threshold = self.occurrence_threshold
        self.candidates = {line for line, n in self.counts.items() if n >= threshold - 1}

    def feed(self, char: str) -> str:
        """Feed one character; return the text (possibly empty) now safe to
        emit. Once ``self.aborted`` is set, the caller must stop feeding
        and close the underlying stream — no more text is ever released."""
        if self.aborted:
            return ""

        if char == "\n":
            line = self.cur_line
            self.cur_line = ""
            if _is_countable(line, self.min_line_chars):
                seen = self.counts.get(line, 0)
                if seen >= self.occurrence_threshold - 1:
                    # This occurrence completes the loop threshold — drop it
                    # (never yield the held buffer) and signal the caller to
                    # stop pulling from the stream.
                    self.held = ""
                    self.aborted = True
                    log.warning(
                        "abort_on_repeated_lines: aborting stream — a line "
                        "reached its %d-th occurrence (loop detected)",
                        seen + 1,
                    )
                    return ""
                self.counts[line] = seen + 1
                self._recompute_candidates()
            out = self.held + "\n"
            self.held = ""
            return out

        self.cur_line += char
        if self.candidates and any(self.cur_line == c[: len(self.cur_line)] for c in self.candidates):
            # Still a viable candidate for completing some already-twice-seen
            # line — hold rather than yield, since releasing it now and then
            # aborting a moment later would leak part of the repeated line.
            self.held += char
            return ""
        # Not (or no longer) a candidate — release whatever was held plus
        # this character, all at once.
        out = self.held + char
        self.held = ""
        return out

    def flush(self) -> str:
        """Resolve whatever's left buffered when the upstream stream ends
        (with no trailing newline). Never seen a terminating "\\n", so this
        can't have been confirmed as a repeat — release it as-is."""
        out = self.held
        self.held = ""
        return out


async def abort_on_repeated_lines(
    stream: AsyncIterator[str],
    *,
    min_line_chars: int = _MIN_REPEAT_LINE_CHARS,
    occurrence_threshold: int = _REPEAT_OCCURRENCE_THRESHOLD,
) -> AsyncIterator[str]:
    """Wrap an LLM delta stream, aborting once some line's occurrence count
    (anywhere in the text so far, not just back-to-back) reaches the loop
    threshold. See module docstring for the failure mode and threshold
    rationale.

    Yields the same text the wrapped stream would have, MINUS the
    occurrence that completes the loop and anything the model would have
    generated after it: as soon as that line is detected, the underlying
    stream is closed (``aclose()``, best-effort — a plain async iterator
    without one is left alone, since dropping the reference is enough for
    it to be garbage-collected) and this generator stops.
    """
    state = _RepeatGuardState(
        min_line_chars=min_line_chars, occurrence_threshold=occurrence_threshold
    )
    async for delta in stream:
        out_parts: list[str] = []
        for char in delta:
            piece = state.feed(char)
            if piece:
                out_parts.append(piece)
            if state.aborted:
                break
        if out_parts:
            yield "".join(out_parts)
        if state.aborted:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    log.debug("abort_on_repeated_lines: aclose() on stream failed", exc_info=True)
            return
    tail = state.flush()
    if tail:
        yield tail


__all__ = ["abort_on_repeated_lines", "trim_repeated_tail"]
