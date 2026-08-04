"""Find moments where a video's SPEECH points at its PICTURE.

Feeds the (not-yet-wired) video-frame-understanding feature: when a speaker
says "do it like this" / "this cream here" / "the article number is in the
description", the transcript alone can't answer a follow-up question about
what's on screen at that moment. Fetching frames costs a download (and,
downstream, an LLM vision call), so we want to look ONLY where the speech
actually gestures at the picture — not at every sentence containing "this".

This module is PURE TEXT ANALYSIS: no network, no LLM, no frame fetching.
Given the segment list already persisted on the job
(``Job.raw_segments_json``) and the transcript language, it returns an
ordered list of :class:`DeixisCandidate` — timestamp + triggering phrase +
category — for a later step to act on.

Public surface:
    find_deixis_candidates(segments, language, *, max_candidates=DEFAULT_MAX_CANDIDATES)
        -> list[DeixisCandidate]

Categories (``DeixisCategory``) — downstream treats each differently:
    ACTION   a demonstrated action ("do it like this", "вот так",
             "so macht man das") — worth several CONSECUTIVE frames.
    OBJECT   a shown object ("this cream", "вот такой", "dieses Produkt")
             — worth ONE good frame, candidate for label OCR.
    EXTERNAL an explicit defer-to-elsewhere reference ("link in the
             description", "артикул в описании", "unten verlinkt") — no
             frame helps here; the point is to detect that the material
             defers to something OUTSIDE itself, so a later step can
             search instead of fetching a frame.

MARKER-SELECTION REASONING
---------------------------
A bare demonstrative ("this" / "это" / "das") is far too common on its own
to be a signal — in ordinary speech it mostly refers back to something
already SAID, not to something SHOWN. Measured directly against live jobs
in this DB: the Russian demonstrative "вот это" alone fires dozens of times
in a film-analysis video ("вот это переживание", "вот это работа детектора
ошибок") where the speaker is elaborating an abstract idea, not pointing at
a picture. Same story for bare German "diese"/"dieser"/"dieses" ("Ich habe
ihr sogar diese Lampe gekauft" — narrating a past purchase, nothing on
screen to look at) and for English "like that" used as the VERB "to like"
("the Monkey doesn't like that plan") rather than the comparison adverbial.

So every marker here is a PHRASE that pairs a deictic word with something
that only makes sense next to a visual/imperative cue:
- a demonstrative + a manner/location word ("вот ТАК", "this WAY", "das
  hier", "hier IST") — points at how/where something looks, not just that
  it exists;
- an imperative of seeing combined with the deictic ("take a LOOK", "вот
  СМОТРИТЕ", "schaut MAL", "hier SEHT ihr") — an instruction to look,
  which bare "смотрите" alone doesn't guarantee (it's also used as a
  rhetorical discourse filler, "смотрите, так как вы..." ~= "look, given
  that you...", not a look-here cue);
- a fixed, unambiguous location/object phrase for OBJECT ("вот он/она/оно
  /они" = "here it is", "this side", "this button") — validated against
  real data: "такие препараты, как кипферон. Вот они" is the speaker
  literally holding up medication boxes.

EXTERNAL markers are the least ambiguous of all three categories — "in the
description", "в описании", "in der Beschreibung" essentially never occur
without meaning "look outside this video" — so they carry the highest
weights.

Adding a language: append a new list to ``_MARKERS_BY_LANGUAGE`` — nothing
else in this module needs to change. Keep entries as
``(pattern, category, weight)`` triples; ``weight`` is this module's own
notion of how unambiguous the phrase is (0..1, see ``DeixisCandidate.
confidence``), used only to pick survivors when ``max_candidates`` caps the
result — it never changes the OUTPUT ORDER, which is always transcript
order.

MATCHING QUALITY, CAPS AND ORDERING
------------------------------------
``find_deixis_candidates`` is pure: same input -> same output.

1. Every marker is tried against every segment's text (case-insensitive).
2. Raw hits landing within ``COLLAPSE_WINDOW_SECONDS`` of the previous hit
   are chained into one cluster (the same gesture/product-show usually
   spans several consecutive segments a second or two apart) — the
   cluster keeps its EARLIEST timestamp (where the gesture starts) and the
   phrase/category of its HIGHEST-weight member (its most informative
   label).
3. If more than ``max_candidates`` clusters survive, only the
   ``max_candidates`` most CONFIDENT are kept (each later candidate costs
   a download) — but the returned list is always re-sorted back into
   transcript order afterward, so ordering never reveals which ones were
   dropped for confidence vs. simply not existing.

Language handling: ``language`` picks which marker table to use (matched
on the first two characters, case-insensitively, so "en-US"/"EN" both hit
"en"). When it's ``None`` or not one of the known codes, we fall back to
the UNION of every known language's markers rather than returning nothing
— a missing/unrecognised language code must not silently disable the
feature.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.workers.timecodes import _segment_start, _segment_text


class DeixisCategory(StrEnum):
    """How downstream should react to a candidate — see module docstring."""

    ACTION = "action"
    OBJECT = "object"
    EXTERNAL = "external"


@dataclass(frozen=True)
class DeixisCandidate:
    """One moment where the speech plausibly points at the video's picture.

    ``timestamp`` is the segment start (seconds) the triggering phrase was
    first seen at — a couple of seconds of slack either side is expected
    when the caller fetches frames.
    ``phrase`` is the exact matched substring (original casing), useful for
    logging/debugging and for a human reviewing candidate quality.
    ``confidence`` is this module's own weight for the phrase (0..1) — used
    only to decide which candidates survive ``max_candidates``; it is not a
    calibrated probability.
    """

    timestamp: float
    phrase: str
    category: DeixisCategory
    confidence: float


@dataclass(frozen=True)
class _Marker:
    pattern: re.Pattern[str]
    category: DeixisCategory
    weight: float


def _marker(phrase_regex: str, category: DeixisCategory, weight: float) -> _Marker:
    return _Marker(re.compile(phrase_regex, re.IGNORECASE), category, weight)


# ---------------------------------------------------------------------------
# Marker tables — one per language, easy to extend (see module docstring).
# ---------------------------------------------------------------------------

_EN_MARKERS: tuple[_Marker, ...] = (
    # ACTION — demonstrated action ("do it like this").
    _marker(r"\bdo it like this\b", DeixisCategory.ACTION, 0.9),
    _marker(r"\bwatch this\b", DeixisCategory.ACTION, 0.85),
    _marker(r"\bcheck this out\b", DeixisCategory.ACTION, 0.8),
    _marker(r"\blook at this\b", DeixisCategory.ACTION, 0.8),
    # NOTE: deliberately no bare "\blike this\b" marker — "I like this
    # idea/song" (verb "to like" + demonstrative object) is at least as
    # common as the deictic adverbial ("fold it like this"), and plain
    # substring matching can't tell the two apart. "like so" doesn't have
    # that verb reading ("*I like so" is ungrammatical), so it's kept.
    _marker(r"\blike so\b", DeixisCategory.ACTION, 0.75),
    _marker(r"\btake a look\b", DeixisCategory.ACTION, 0.7),
    _marker(r"\bthis way\b", DeixisCategory.ACTION, 0.6),
    # OBJECT — a shown object/body part/UI element ("this cream"). "thing"
    # is deliberately excluded from the noun list below — "this thing" is
    # extremely common filler ("this whole thing is annoying") with no
    # visual referent, unlike "this one"/"this side" which is how people
    # actually talk about a shown comparison (validated: "this one leg",
    # "this side" in a real exercise-demo transcript).
    _marker(
        r"\bthis (?:one|side|button|product|cream|item|part|area|spot|"
        r"label|box|package|bottle|page|chart|graph|image|photo|"
        r"slide|diagram)\b",
        DeixisCategory.OBJECT,
        0.65,
    ),
    _marker(r"\byou can see\b", DeixisCategory.OBJECT, 0.55),
    # EXTERNAL — defers to something outside the material.
    _marker(r"\blink(?:ed)? (?:in|below) the description\b", DeixisCategory.EXTERNAL, 0.9),
    _marker(r"\bin the description\b", DeixisCategory.EXTERNAL, 0.85),
    _marker(r"\bdescription below\b", DeixisCategory.EXTERNAL, 0.85),
    _marker(r"\blink(?:ed)? below\b", DeixisCategory.EXTERNAL, 0.85),
    _marker(r"\bshow ?notes\b", DeixisCategory.EXTERNAL, 0.7),
)

_RU_MARKERS: tuple[_Marker, ...] = (
    # ACTION — "вот так" (like this) — validated heavily against a real
    # drawing-tutorial job (24 hits, all genuine "watch me draw it like
    # this" moments).
    _marker(r"\bвот\s+так\b", DeixisCategory.ACTION, 0.75),
    _marker(r"\bвот\s+таким\s+образом\b", DeixisCategory.ACTION, 0.8),
    _marker(r"\bвот\s+смотрите\b", DeixisCategory.ACTION, 0.7),
    # "смотрите/смотри, КАК ..." ("look HOW ...") is a process cue, unlike
    # "смотрите, ЧТО ..." ("look WHAT's here") below, which points at a
    # thing rather than a demonstrated process — validated on a real job:
    # "И смотрите, что у нас на обложке тетрадки" is the speaker showing a
    # notebook cover, not demonstrating an action.
    _marker(r"\bсмотри(?:те)?,?\s+как\b", DeixisCategory.ACTION, 0.75),
    # OBJECT — "вот такой/такая/такое/такие" (an object of this kind) and
    # "вот он/она/оно/они" (here it is) — validated: "такие препараты, как
    # кипферон. Вот они" (speaker holding up medication boxes). The leading
    # ``\b`` matters: without it "живот такой" (stomach ...) would match
    # "вот такой" mid-word.
    _marker(r"\bвот\s+тако[а-яё]*\b", DeixisCategory.OBJECT, 0.65),
    _marker(r"\bвот\s+(?:он|она|оно|они)\b", DeixisCategory.OBJECT, 0.75),
    _marker(r"\bсмотри(?:те)?,?\s+что\b", DeixisCategory.OBJECT, 0.7),
    # EXTERNAL — defers to something outside the material.
    _marker(r"\bв\s+описании\b", DeixisCategory.EXTERNAL, 0.85),
    _marker(r"\bссылк\w*\s+в\s+описании\b", DeixisCategory.EXTERNAL, 0.9),
    _marker(r"\bартикул\w*\b", DeixisCategory.EXTERNAL, 0.8),
)

_DE_MARKERS: tuple[_Marker, ...] = (
    # ACTION — "so macht man das" / "so geht das" (that's how it's done).
    _marker(r"\bso\s+macht\s+man\s+das\b", DeixisCategory.ACTION, 0.85),
    _marker(r"\bso\s+geht\s+das\b", DeixisCategory.ACTION, 0.8),
    _marker(r"\bhier\s+seht\s+ihr\b", DeixisCategory.ACTION, 0.85),
    _marker(r"\bschaut\s+mal\b", DeixisCategory.ACTION, 0.7),
    _marker(r"\bguckt\s+mal\b", DeixisCategory.ACTION, 0.7),
    _marker(r"\bmach\s+es\s+so\b", DeixisCategory.ACTION, 0.75),
    # OBJECT — a demonstrative reinforced with "hier" (here), since bare
    # "diese(r|s)" alone is too weak (see module docstring — measured on a
    # real job: "diese Lampe" narrates a past purchase, nothing shown).
    # Allows up to two words between the demonstrative and "hier" so
    # "dieses Produkt hier" / "diese Creme hier" match, not just a bare
    # "dieses hier".
    _marker(r"\bdies(?:e|er|es)\b(?:\s+\w+){0,2}\s+hier\b", DeixisCategory.OBJECT, 0.7),
    _marker(r"\bdas\s+hier\b", DeixisCategory.OBJECT, 0.65),
    _marker(r"\bhier\s+ist\b", DeixisCategory.OBJECT, 0.6),
    # EXTERNAL — defers to something outside the material.
    _marker(r"\blink\s+in\s+der\s+beschreibung\b", DeixisCategory.EXTERNAL, 0.9),
    _marker(r"\bin\s+der\s+beschreibung\b", DeixisCategory.EXTERNAL, 0.85),
    _marker(r"\bunten\s+verlinkt\b", DeixisCategory.EXTERNAL, 0.85),
    _marker(r"\bbeschreibung\s+unten\b", DeixisCategory.EXTERNAL, 0.8),
)

_MARKERS_BY_LANGUAGE: dict[str, tuple[_Marker, ...]] = {
    "en": _EN_MARKERS,
    "ru": _RU_MARKERS,
    "de": _DE_MARKERS,
}

# "The same gesture usually spans several segments" — chain raw hits within
# this many seconds of the previous hit into one candidate. A "couple of
# seconds" per the spec; segments in this DB run 1-5s apart, so this is
# generous enough to bridge consecutive segments of one gesture without
# bridging across unrelated later mentions.
COLLAPSE_WINDOW_SECONDS = 3.0

# Each surviving candidate costs a frame download later, so the default cap
# is deliberately small. Callable-overridable via ``max_candidates``.
DEFAULT_MAX_CANDIDATES = 8


@dataclass
class _RawHit:
    timestamp: float
    phrase: str
    category: DeixisCategory
    weight: float


def _markers_for_language(language: str | None) -> tuple[_Marker, ...]:
    """Pick the marker table for ``language``, or the union of all of them.

    Matched on the first two characters (case-insensitive) so "en-US" / "EN"
    both resolve to "en". Falls back to every known language's markers when
    ``language`` is ``None`` or not recognised — see module docstring for
    why a missing language must not silently disable the feature.
    """
    if language:
        code = language.strip().lower()[:2]
        markers = _MARKERS_BY_LANGUAGE.get(code)
        if markers is not None:
            return markers
    combined: list[_Marker] = []
    for markers in _MARKERS_BY_LANGUAGE.values():
        combined.extend(markers)
    return tuple(combined)


def _find_raw_hits(
    segments: Sequence[Mapping[str, Any]], markers: tuple[_Marker, ...]
) -> list[_RawHit]:
    hits: list[_RawHit] = []
    for seg in segments:
        text = _segment_text(seg)
        if not text:
            continue
        start = _segment_start(seg)
        for marker in markers:
            for match in marker.pattern.finditer(text):
                hits.append(_RawHit(start, match.group(0), marker.category, marker.weight))
    hits.sort(key=lambda h: h.timestamp)
    return hits


def _collapse_hits(hits: list[_RawHit]) -> list[DeixisCandidate]:
    """Chain-merge hits within ``COLLAPSE_WINDOW_SECONDS`` of the previous one.

    Each cluster keeps the EARLIEST timestamp (gesture start) and the
    phrase/category of its highest-weight member.
    """
    if not hits:
        return []

    clusters: list[list[_RawHit]] = []
    for hit in hits:
        if clusters and hit.timestamp - clusters[-1][-1].timestamp <= COLLAPSE_WINDOW_SECONDS:
            clusters[-1].append(hit)
        else:
            clusters.append([hit])

    candidates: list[DeixisCandidate] = []
    for cluster in clusters:
        best = max(cluster, key=lambda h: h.weight)
        candidates.append(
            DeixisCandidate(
                timestamp=cluster[0].timestamp,
                phrase=best.phrase,
                category=best.category,
                confidence=best.weight,
            )
        )
    return candidates


def find_deixis_candidates(
    segments: Sequence[Mapping[str, Any]] | None,
    language: str | None,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[DeixisCandidate]:
    """Find moments where the transcript's speech points at the video's picture.

    ``segments`` is the same shape ``Job.raw_segments_json`` persists — a
    list of dicts (or any Mapping) with ``start``/``text`` (see
    ``workers.timecodes._segment_start`` / ``_segment_text``, reused here
    rather than reimplemented). ``language`` is the short transcript
    language code (``Job.transcript_language``, e.g. ``"en"``/``"ru"``/
    ``"de"``); ``None`` or an unrecognised code falls back to matching every
    known language's markers (see ``_markers_for_language``).

    Returns candidates in TRANSCRIPT ORDER (ascending timestamp), each
    representing one distinct moment — nearby matches within
    ``COLLAPSE_WINDOW_SECONDS`` are collapsed into one (see
    ``_collapse_hits``). When more than ``max_candidates`` distinct moments
    are found, only the most CONFIDENT ``max_candidates`` survive — each one
    costs a frame download downstream — but the survivors are still returned
    in transcript order, not confidence order.

    Pure: same input -> same output. Empty/``None`` segments return ``[]``.
    """
    if not segments:
        return []

    markers = _markers_for_language(language)
    hits = _find_raw_hits(segments, markers)
    candidates = _collapse_hits(hits)

    if max_candidates >= 0 and len(candidates) > max_candidates:
        kept = sorted(candidates, key=lambda c: c.confidence, reverse=True)[:max_candidates]
        kept_ids = {id(c) for c in kept}
        candidates = [c for c in candidates if id(c) in kept_ids]

    return candidates


__all__ = [
    "COLLAPSE_WINDOW_SECONDS",
    "DEFAULT_MAX_CANDIDATES",
    "DeixisCandidate",
    "DeixisCategory",
    "find_deixis_candidates",
]
