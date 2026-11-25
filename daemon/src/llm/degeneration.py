"""One shape-agnostic guard against a local LLM locking into a repetition loop.

Public surface:
    find_loop(text) -> LoopCut | None
        Pure detector. Returns where to cut ``text`` (and how much that
        discards) when its TAIL is degenerate, else ``None``.
    trim(text, *, label="") -> str
        ``text`` cut back to where the loop began (right-stripped), or
        ``text`` unchanged. Logs a WARNING when it cuts. Used for
        non-streaming calls and on the accumulated text of every stream.
    LoopWatch(label).feed(delta) -> bool
        Incremental check for callers that consume a stream themselves
        (``api/ai.py``: the Q&A stream interleaves text with stage events).
    guard_stream(stream, *, label="") -> AsyncIterator[str]
        Passes deltas through unchanged; on detection closes the underlying
        stream (``aclose()`` -> the HTTP request to the backend is
        cancelled) and stops.

Streaming never retracts deltas already sent: the caller ``trim``s its
accumulated text, and that trimmed text is what gets persisted and sent in
the ``done`` event. Both the Q&A chat (``sidepanel/chat.js``) and the
summary view (``sidepanel/app.js``) re-render from the ``done`` payload,
replacing the raw streamed text — so the user ends up seeing the trimmed
version.

Why one detector: small local models (gemma-4, qwen3-vl-8b) sometimes finish
the real content and then loop until ``max_tokens``. Measured shapes:
  - a run of filler — ``<br>`` tags, stray closing tags, bare ``[MM:SS]``;
  - one line repeated verbatim, or a period-2 / period-4 cycle of bullets
    (repeats are NOT adjacent);
  - a whole closing paragraph (~1.1k chars) repeated 4 times, two copies
    verbatim and two slightly reworded (Q&A message 267, qwen3-vl-8b).
The previous guards were one per shape (a tail regex in ``api/ai.py``,
exact-line occurrence counting in ``llm/repetition.py``) and the last shape
slipped past both. This module catches all of them by content, not shape.

Algorithm:
  1. Tokenize: lowercase ``\\w+`` words; every HTML tag becomes ``<tag>``,
     every ``[MM:SS]`` / ``[HH:MM:SS]`` becomes ``<tc>`` (so "[24:18] X" and
     "[26:42] X" match). Markdown — including ordered-list markers, so
     "11. Item" and "15. Item" match — punctuation and whitespace are
     dropped, so a re-wrapped or re-bulleted copy still matches. Other
     numbers stay literal: a table whose rows differ only in their figures
     is legit content, not a loop. Each token keeps its char offset.
  2. Filler rule: the text ends with ``_FILLER_RUN`` (6)+ consecutive
     ``<tag>``/``<tc>`` tokens and no word after them -> cut at the run.
     (The shape the old ``api/ai.py`` ``_DEGEN_TAIL_RE`` caught.)
  3. Shingle rule: word ``_SHINGLE``-grams (6); for each, count how many
     times the same shingle occurred EARLIER anywhere in the text. If at
     least ``_MIN_CONTAINMENT`` (0.8) of the last ``_WINDOW`` (32) shingles
     have ``>= _MIN_PRIOR`` (2) earlier occurrences, the tail is a loop.
     Containment over a window (not exact lines) tolerates reworded
     copies; earlier-anywhere (not adjacent) catches period-N cycles.
     Requiring the THIRD occurrence is what keeps it safe: models
     legitimately restate a takeaway once (summary "Key takeaways" vs the
     matching bullet, a Q&A conclusion rephrasing the paragraph above); a
     real loop always goes on to a third copy.
  4. Cut point: walk back from the tail over shingles seen at least ONCE
     before, tolerating gaps shorter than ``_GAP`` (16) unseen shingles (a
     reworded clause), to the start of the 2nd copy; then snap to the start
     of that line (or, inside a >``_MAX_LINE_SNAP``-char paragraph, to the
     previous sentence end). The first copy is kept, every later one dropped.

Cost: one regex tokenization + one dict pass over the whole text, re-run
every ``_CHECK_EVERY_CHARS`` (64) streamed chars; texts are <= ~10k chars
(max_tokens 2000), so a few ms per check at worst.

Calibration (2026-11-25), live DB read-only: 37 assistant Q&A messages,
208 ``job.summary_md``, 56 ``transcript_translation.text`` (301 texts).
Simulated streaming: ``find_loop`` on every 64-char prefix plus the full
text, first hit wins. The old exact-line ``trim_repeated_tail`` flagged NONE
of the stored summaries (its 6 known-bad jobs were since re-processed), so
the known-bad set is: Q&A 231 (paragraph loop, verbatim), 253 (period-4
cycle of numbered items), 267 (the incident: closing paragraph x4, two
copies reworded) and summary rb1qogRprkdD (timecoded bullet x4, verbs
alternating, truncated by max_tokens).

    window | containment | min prior | caught (of 4)     | false positives
    24     | 0.8         | 1         | 4                 | 7 (Q&A 257, 263; 5 summaries)
    32     | 0.8         | 1         | 4                 | 4 (Q&A 263; 3 summaries)
    32     | 0.6         | 2         | 4                 | 1 (IrwzJiAKton7)
    24-60  | 0.8         | 2         | 3 (no rb1q)       | 0

  "min prior 1" (cut on the 2nd copy) false-flags every summary whose key
  takeaway is restated in a bullet and Q&A answers ending with a rephrased
  conclusion. With min prior 2 the window size (24..60) doesn't change the
  result; 32 (~200-250 chars of prose) is the middle. The miss is
  rb1qogRprkdD: its copies alternate "объясняет"/"отмечает", so third
  occurrences only cover ~66% of the window — indistinguishable from the
  legit IrwzJiAKton7 (0.62, a takeaway restated in two bullets) without a
  false positive; a longer loop of that shape is still caught once each
  variant reaches its 3rd copy. One translation (oEsWDk1tI9-fru) is
  flagged too — a faithful translation of a Whisper loop in the source;
  irrelevant because the translator does not use this module (it keeps
  its own line-aligned guard whose threshold floats with the input's
  repetition, see ``workers/translator.py``).
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

log = logging.getLogger(__name__)

# See module docstring for the calibration behind these numbers.
_SHINGLE = 6
_WINDOW = 32
_MIN_CONTAINMENT = 0.8
_MIN_PRIOR = 2
_GAP = 16
_FILLER_RUN = 6
_CHECK_EVERY_CHARS = 64
_MAX_LINE_SNAP = 300

_TOKEN_RE = re.compile(
    r"(?m)(<[^<>\n]{1,80}>)|(\[\d{1,2}:\d{2}(?::\d{2})?\])|(^[ \t]*\d{1,3}[.)](?=\s))|(\w+)"
)
_SENTENCE_END_RE = re.compile(r"[.!?…:;]\s")

_TAG = "<tag>"
_TC = "<tc>"


@dataclass(frozen=True)
class LoopCut:
    offset: int  # keep text[:offset]
    discarded: int  # chars dropped


def _tokenize(text: str) -> tuple[list[str], list[int]]:
    toks: list[str] = []
    offs: list[int] = []
    for m in _TOKEN_RE.finditer(text):
        if m.group(1):
            toks.append(_TAG)
        elif m.group(2):
            toks.append(_TC)
        elif m.group(3):
            continue  # ordered-list marker: markup, like "-" or "**"
        else:
            toks.append(m.group(4).lower())
        offs.append(m.start())
    return toks, offs


def _snap_back(text: str, offset: int) -> int:
    """Move ``offset`` back to the start of its line, so no half-duplicate
    paragraph/list item survives. If the line is long prose (the repeat
    began deep inside a paragraph), fall back to the previous sentence end
    instead of throwing away the whole paragraph."""
    line_start = text.rfind("\n", 0, offset) + 1
    if offset - line_start <= _MAX_LINE_SNAP:
        return line_start
    best = line_start
    for m in _SENTENCE_END_RE.finditer(text, line_start, offset):
        best = m.end()
    return best


def _filler_cut(toks: list[str], offs: list[int]) -> int | None:
    n = 0
    for t in reversed(toks):
        if t not in (_TAG, _TC):
            break
        n += 1
    if n >= _FILLER_RUN:
        return offs[len(toks) - n]
    return None


def _shingle_cut(toks: list[str], offs: list[int]) -> int | None:
    n_sh = len(toks) - _SHINGLE + 1
    if n_sh < _WINDOW:
        return None
    counts: dict[tuple[str, ...], int] = {}
    prior = [0] * n_sh  # occurrences of this shingle BEFORE position i
    for i in range(n_sh):
        sh = tuple(toks[i : i + _SHINGLE])
        prior[i] = counts.get(sh, 0)
        counts[sh] = prior[i] + 1
    hot = sum(1 for c in prior[n_sh - _WINDOW :] if c >= _MIN_PRIOR)
    if hot < _MIN_CONTAINMENT * _WINDOW:
        return None
    # Walk back over the repeated span (2nd copy onwards) to where it began.
    start = n_sh - 1
    gap = 0
    for i in range(n_sh - 1, -1, -1):
        if prior[i] >= 1:
            start = i
            gap = 0
        else:
            gap += 1
            if gap >= _GAP:
                break
    return offs[start]


def find_loop(text: str) -> LoopCut | None:
    """Detect a degenerate tail in ``text``; see module docstring."""
    if not text:
        return None
    toks, offs = _tokenize(text)
    if not toks:
        return None
    raw = _filler_cut(toks, offs)
    if raw is None:
        raw = _shingle_cut(toks, offs)
    if raw is None:
        return None
    offset = len(text[: _snap_back(text, raw)].rstrip())
    return LoopCut(offset=offset, discarded=len(text) - offset)


def _suffix(label: str) -> str:
    return f" ({label})" if label else ""


def trim(text: str, *, label: str = "") -> str:
    """Return ``text`` with a degenerate tail cut off (or unchanged)."""
    cut = find_loop(text)
    if cut is None:
        return text
    log.warning(
        "degeneration loop cut%s: kept %d chars, discarded %d",
        _suffix(label),
        cut.offset,
        cut.discarded,
    )
    return text[: cut.offset]


class LoopWatch:
    """Incremental detector over a stream of text deltas.

    ``feed(delta)`` returns True once the accumulated text loops; the caller
    must then stop/close its stream and ``trim`` the accumulated text.
    """

    __slots__ = ("_acc", "_since", "label")

    def __init__(self, label: str = "") -> None:
        self._acc: list[str] = []
        self._since = 0
        self.label = label

    def feed(self, delta: str) -> bool:
        self._acc.append(delta)
        self._since += len(delta)
        if self._since < _CHECK_EVERY_CHARS:
            return False
        self._since = 0
        text = "".join(self._acc)
        self._acc = [text]
        if find_loop(text) is None:
            return False
        log.warning(
            "degeneration loop detected%s at %d chars — stopping the LLM stream",
            _suffix(self.label),
            len(text),
        )
        return True


async def aclose_quietly(stream: object) -> None:
    """Close an async generator (cancelling the LLM request behind it);
    a plain iterator without ``aclose`` is left to garbage collection."""
    aclose = getattr(stream, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception:
        log.debug("aclose() on a looped stream failed", exc_info=True)


async def guard_stream(stream: AsyncIterator[str], *, label: str = "") -> AsyncIterator[str]:
    """Pass ``stream`` through; stop (closing it) once its tail loops."""
    watch = LoopWatch(label)
    async for delta in stream:
        yield delta
        if watch.feed(delta):
            await aclose_quietly(stream)
            return


__all__ = ["LoopCut", "LoopWatch", "aclose_quietly", "find_loop", "guard_stream", "trim"]
