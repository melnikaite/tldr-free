"""Guard against a local model locking into a line-repetition loop.

Public surface:
    async def abort_on_repeated_lines(stream, *, min_line_chars=.., occurrence_threshold=..)
            -> AsyncIterator[str]
        Wraps a raw LLM delta stream and enforces a per-line occurrence
        limit on every "countable" line, counted ANYWHERE in the text so
        far (not necessarily adjacent to its previous occurrence): the 1st
        occurrence is yielded normally; the 2nd occurrence is SUPPRESSED —
        dropped from the output (its text and its terminating newline)
        without stopping the stream, so generation keeps flowing; the
        ``occurrence_threshold``-th occurrence (3rd, by default) is also
        suppressed and additionally aborts the stream early — the
        underlying stream is closed and nothing the model would have
        generated after that point is ever yielded. Net effect: the caller
        never sees ANY duplicate of a countable line, and the LLM is
        stopped instead of burning its whole ``max_tokens`` budget on a
        genuine loop.

    def trim_repeated_tail(text, *, min_line_chars=.., occurrence_threshold=..) -> str
        Same per-line occurrence rule (1st kept, 2nd suppressed in place,
        ``occurrence_threshold``-th suppressed AND cuts everything after
        it), applied after the fact to an already-complete string. A 2nd
        occurrence is dropped even when the text never reaches a 3rd — no
        duplicate of a countable line ever survives, cut or no cut. Used
        for the non-streaming map/reduce calls, which have no stream to
        abort — but a degenerate chunk summary must not be fed into the
        next reduce round either.

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

A cut alone is not enough to keep duplicates out of what the user sees:
even once a loop is stopped at its 3rd occurrence, the two occurrences
before it (which is what "at most ``occurrence_threshold - 1`` surviving
occurrences" used to mean) still reached the saved summary — visibly
duplicated. So the 2nd occurrence of any countable line is now suppressed
unconditionally, whether or not the text ever goes on to a 3rd: the cut is
still what stops a genuine loop from consuming the whole generation
budget, but no duplicate line reaches the output either way, looping or
not.

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

    Re-read that "1 false positive" in light of occurrence-2 suppression,
    added alongside the cut: what the measurement calls a false positive at
    threshold 2 is a summary where some long line legitimately appears
    exactly twice — no loop, no 3rd occurrence. Threshold 2 would have CUT
    that summary right there, discarding everything after a wholly
    legitimate duplicate; that is what made it a false positive worth
    avoiding. Under the CURRENT rule (threshold 3, plus suppression at
    occurrence 2), that same benign duplicate is silently DROPPED — only
    the repeated line's 2nd occurrence disappears, nothing after it is
    touched — instead of triggering a cut. That is the desired outcome, not
    a defect: the reader loses one duplicated line, not the rest of the
    document. This does not change the calibration decision (the threshold
    stays 3, since a genuine loop must still not be allowed to run to a 3rd
    occurrence uncut); it means threshold 2's "1 false positive" no longer
    describes a failure mode worth avoiding at all — it describes exactly
    what occurrence-2 suppression now does on purpose, at threshold 3, to
    every countable line that happens to repeat exactly once.
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
    """Enforce the per-line occurrence rule over the whole of ``text`` —
    anywhere in the text, not just back-to-back.

    Walks ``text`` line by line, counting occurrences of each "countable"
    (``>= min_line_chars``) line seen so far, keyed by its exact text. A
    countable line's 1st occurrence is kept; its 2nd occurrence is dropped
    in place (its text and the newline that followed it — no blank line is
    left behind, the surrounding lines simply become adjacent) but walking
    continues; its ``occurrence_threshold``-th occurrence (3rd, by
    default) is ALSO dropped and additionally cuts the text there — that
    line and everything after it (in the whole text) is discarded, and
    trailing blank lines left behind by the cut are trimmed too. So no
    countable line ever survives with a duplicate in the output, whether or
    not a cut ever happens. This catches a period-1 back-to-back repeat as
    a special case, but also an ABAB alternation or any other small-set
    cycle, since it never requires two occurrences to be adjacent. Short
    lines are never counted, never dropped, and never trigger a cut — see
    module docstring for why that keeps legitimate short repeated list
    items safe.

    No-op (returns ``text`` unchanged) when no countable line ever repeats
    at all. Pure: same input -> same output.
    """
    if not text:
        return text
    lines = text.split("\n")
    counts: dict[str, int] = {}
    kept: list[str] = []
    cut_at: int | None = None
    suppressed_any = False
    for i, line in enumerate(lines):
        if not _is_countable(line, min_line_chars):
            kept.append(line)
            continue
        seen = counts.get(line, 0)
        if seen >= occurrence_threshold - 1:
            # This occurrence would reach the loop threshold — drop it and
            # everything after it (handled below), without appending.
            cut_at = i
            break
        counts[line] = seen + 1
        if seen >= 1:
            # Not the first occurrence — suppress it in place (don't append),
            # but keep counting, since a later occurrence of this same line
            # is what triggers the cut. Deliberately keyed off "have we seen
            # this line before", NOT off ``occurrence_threshold``: only the
            # CUT depends on the threshold; the no-duplicates rule holds at
            # every threshold.
            suppressed_any = True
            continue
        kept.append(line)

    if cut_at is None:
        if not suppressed_any:
            return text
        return "\n".join(kept)

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
    state machine to "hold while matching any line already seen at least
    once" — every such line's NEXT occurrence needs to be intercepted, either
    to be suppressed (2nd occurrence) or to be suppressed AND to abort the
    stream (``occurrence_threshold``-th occurrence), so those are the lines
    worth buffering for. A brand new line (never seen before) is never held
    at all — its own first occurrence cannot possibly need suppressing.
    Mirrors ``workers.timecodes._MarkerCapState``'s design: holds text back
    only while it MIGHT still turn into an intercepted repeat, and flushes
    immediately the moment that stops being possible — so ordinary
    (non-repeating, or not-yet-suspicious) generation streams through with
    no added latency; real generation diverges from every candidate within
    the first few characters in the overwhelming common case, and the
    divergence check re-runs on every character, so held text is released
    the instant a match fails rather than waiting for the line to end.

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
        # Any line seen at least once is a live candidate: its very next
        # occurrence must be intercepted, whether that means merely
        # suppressing it (every repeat) or suppressing it AND aborting (the
        # ``occurrence_threshold``-th). Lines are only ever inserted into
        # ``counts`` with n >= 1, so every previously-seen line stays a
        # candidate for the rest of the stream — hence the plain truthiness
        # of ``counts`` rather than a threshold-derived comparison.
        self.candidates = set(self.counts)

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
                if seen >= 1:
                    # Not the first occurrence — drop it (text and its
                    # newline) without aborting: generation keeps flowing,
                    # but no duplicate ever reaches the caller. Independent
                    # of ``occurrence_threshold``, same as in
                    # ``trim_repeated_tail`` — only the abort above is
                    # threshold-driven.
                    self.held = ""
                    return ""
            out = self.held + "\n"
            self.held = ""
            return out

        self.cur_line += char
        if self.candidates and any(self.cur_line == c[: len(self.cur_line)] for c in self.candidates):
            # Still a viable candidate for completing some already-seen
            # line — hold rather than yield, since releasing it now and then
            # suppressing or aborting a moment later would leak part of the
            # repeated line.
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
        can't have been confirmed as a repeat — release it as-is. This is
        deliberately left alone by the occurrence-2 suppression rule too:
        an unterminated line was never counted (counting only happens on
        "\\n", see ``feed`` above), so there is no completed occurrence
        number to suppress or emit — releasing it unconditionally is the
        only option that doesn't risk silently eating a chunk of genuine,
        still-arriving output just because it happened to prefix-match a
        candidate."""
        out = self.held
        self.held = ""
        return out


async def abort_on_repeated_lines(
    stream: AsyncIterator[str],
    *,
    min_line_chars: int = _MIN_REPEAT_LINE_CHARS,
    occurrence_threshold: int = _REPEAT_OCCURRENCE_THRESHOLD,
) -> AsyncIterator[str]:
    """Wrap an LLM delta stream, suppressing every countable line's 2nd
    occurrence and aborting once some line's occurrence count (anywhere in
    the text so far, not just back-to-back) reaches the loop threshold. See
    module docstring for the failure mode and threshold rationale.

    Yields the same text the wrapped stream would have, MINUS every
    countable line's 2nd occurrence (dropped in place, generation
    continues) and MINUS the occurrence that completes the loop plus
    anything the model would have generated after it: as soon as that line
    is detected, the underlying stream is closed (``aclose()``,
    best-effort — a plain async iterator without one is left alone, since
    dropping the reference is enough for it to be garbage-collected) and
    this generator stops.
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
