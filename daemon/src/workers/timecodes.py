"""Build raw_text with inline ``[MM:SS]`` / ``[HH:MM:SS]`` markers.

This is the SINGLE place where timecode markers are formatted. Both the
YouTube fast path (``youtube-transcript-api``) and the Whisper path
(``mlx-server``) go through ``build_marked_text`` so summaries and Q&A see
one uniform marker format.

Public surface:
    build_marked_text(segments: list[Segment], window_seconds: int) -> str

A ``Segment`` is a plain dict with ``start`` and ``text`` (we don't use
``duration``/``end`` except to decide HH:MM:SS vs MM:SS):
    {"start": float, "duration": float, "text": str}   # youtube-transcript-api
    {"start": float, "end":      float, "text": str}   # whisper (synthesised
                                                       #  as one whole-audio
                                                       #  segment — mlx-server
                                                       #  v1.8 stopped returning
                                                       #  per-segment timestamps)

We only need ``start`` and ``text`` to bucket. ``duration``/``end`` are
used to determine whether the total span is over an hour (so we switch
to ``HH:MM:SS``).

Algorithm:
1. Determine the maximum start time in the input — picks ``HH:MM:SS``
   when it's >= 3600 s, else ``MM:SS``.
2. Bucket each segment by ``floor(start / window_seconds)``.
3. Concatenate the text of each bucket with a single space, trimmed.
4. Emit one line per non-empty bucket: ``"[MM:SS] text\n"``.

The output is deterministic and pure: same input → same output.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

# Format strings for ``str.format(...)`` so callers / tests can refer to the
# exact formatting without parsing f-strings out of the source.
_MM_SS = "{m:02d}:{s:02d}"
_HH_MM_SS = "{h:d}:{m:02d}:{s:02d}"


def _format_marker(seconds: int, *, use_hours: bool) -> str:
    if use_hours:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return _HH_MM_SS.format(h=h, m=m, s=s)
    m, s = divmod(seconds, 60)
    return _MM_SS.format(m=m, s=s)


def _segment_start(seg: Mapping[str, Any]) -> float:
    start = seg.get("start")
    if start is None:
        return 0.0
    return float(start)


def _segment_text(seg: Mapping[str, Any]) -> str:
    text = seg.get("text") or ""
    return str(text).strip()


def build_marked_text(segments: list[dict[str, Any]], window_seconds: int) -> str:
    """Bucket ``segments`` and produce a flat text with ``[MM:SS]`` markers.

    ``segments`` is a list of dicts (or any Mapping) with ``start`` and ``text``
    keys. ``window_seconds`` controls the bucket size — ``30`` means each line
    summarises ~30 seconds of speech.
    """
    if not segments or window_seconds <= 0:
        return ""

    # Decide the marker format once, based on the latest start time.
    max_start = max((_segment_start(s) for s in segments), default=0.0)
    use_hours = max_start >= 3600.0

    # Bucket segments by floor(start / window_seconds). Ordered dicts preserve
    # insertion order — but for safety we explicitly sort buckets by index at
    # the end, since segments may not arrive in order.
    buckets: dict[int, list[str]] = {}
    for seg in segments:
        text = _segment_text(seg)
        if not text:
            continue
        idx = int(math.floor(_segment_start(seg) / window_seconds))
        if idx < 0:
            idx = 0
        buckets.setdefault(idx, []).append(text)

    if not buckets:
        return ""

    lines: list[str] = []
    for idx in sorted(buckets):
        line_text = " ".join(buckets[idx]).strip()
        if not line_text:
            continue
        seconds = idx * window_seconds
        marker = _format_marker(seconds, use_hours=use_hours)
        lines.append(f"[{marker}] {line_text}")

    if not lines:
        return ""
    return "\n".join(lines) + "\n"


__all__ = ["build_marked_text"]
