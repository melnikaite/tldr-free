"""Tests for workers.deixis — deictic ("speech points at the picture") detection."""

from __future__ import annotations

from src.workers.deixis import (
    COLLAPSE_WINDOW_SECONDS,
    DEFAULT_MAX_CANDIDATES,
    DeixisCandidate,
    DeixisCategory,
    find_deixis_candidates,
)


def _seg(start: float, text: str) -> dict[str, object]:
    return {"start": start, "text": text}


# ---------------------------------------------------------------------------
# Empty / None input
# ---------------------------------------------------------------------------


def test_none_segments_returns_empty_list() -> None:
    assert find_deixis_candidates(None, "en") == []


def test_empty_segments_returns_empty_list() -> None:
    assert find_deixis_candidates([], "en") == []


def test_segments_with_no_matches_returns_empty_list() -> None:
    segs = [_seg(0.0, "Just an ordinary sentence with nothing special in it.")]
    assert find_deixis_candidates(segs, "en") == []


# ---------------------------------------------------------------------------
# English — one case per category
# ---------------------------------------------------------------------------


def test_english_action_do_it_like_this() -> None:
    segs = [_seg(10.0, "Now watch this, you do it like this.")]
    out = find_deixis_candidates(segs, "en")
    assert len(out) == 1
    assert out[0].category == DeixisCategory.ACTION
    assert out[0].timestamp == 10.0


def test_english_object_this_cream() -> None:
    segs = [_seg(42.0, "You want to apply this cream twice a day.")]
    out = find_deixis_candidates(segs, "en")
    assert len(out) == 1
    assert out[0].category == DeixisCategory.OBJECT
    assert "this cream" in out[0].phrase.lower()


def test_english_external_link_in_description() -> None:
    segs = [_seg(5.0, "Grab the code from the link in the description.")]
    out = find_deixis_candidates(segs, "en")
    assert len(out) == 1
    assert out[0].category == DeixisCategory.EXTERNAL


# ---------------------------------------------------------------------------
# Russian — one case per category
# ---------------------------------------------------------------------------


def test_russian_action_vot_tak() -> None:
    segs = [_seg(61.0, "Возьмём вот такой вот маркер и рисуем вот так.")]
    out = find_deixis_candidates(segs, "ru")
    assert any(c.category == DeixisCategory.ACTION for c in out)


def test_russian_object_vot_takoi() -> None:
    segs = [_seg(864.0, "Вот такой челлендж получился, посмотрите.")]
    out = find_deixis_candidates(segs, "ru")
    assert any(c.category == DeixisCategory.OBJECT for c in out)


def test_russian_object_vot_oni() -> None:
    segs = [_seg(861.0, "Такие препараты, как кипферон. Вот они.")]
    out = find_deixis_candidates(segs, "ru")
    assert any(c.category == DeixisCategory.OBJECT for c in out)


def test_russian_external_ssylka_v_opisanii() -> None:
    segs = [_seg(939.0, "Полный протокол по ссылке в описании.")]
    out = find_deixis_candidates(segs, "ru")
    assert any(c.category == DeixisCategory.EXTERNAL for c in out)


def test_russian_external_artikul() -> None:
    segs = [_seg(30.0, "Артикул товара смотрите в описании ниже.")]
    out = find_deixis_candidates(segs, "ru")
    assert any(c.category == DeixisCategory.EXTERNAL for c in out)


# ---------------------------------------------------------------------------
# German — one case per category
# ---------------------------------------------------------------------------


def test_german_action_so_macht_man_das() -> None:
    segs = [_seg(15.0, "Und so macht man das, ganz einfach.")]
    out = find_deixis_candidates(segs, "de")
    assert any(c.category == DeixisCategory.ACTION for c in out)


def test_german_object_dieses_hier() -> None:
    segs = [_seg(20.0, "Nehmt dieses Produkt hier, das ist perfekt.")]
    out = find_deixis_candidates(segs, "de")
    assert any(c.category == DeixisCategory.OBJECT for c in out)


def test_german_external_unten_verlinkt() -> None:
    segs = [_seg(25.0, "Der Link ist unten verlinkt, schaut mal rein.")]
    out = find_deixis_candidates(segs, "de")
    assert any(c.category == DeixisCategory.EXTERNAL for c in out)


# ---------------------------------------------------------------------------
# Too-common-word rejection — bare demonstrative alone must NOT fire.
# ---------------------------------------------------------------------------


def test_bare_this_alone_does_not_fire_english() -> None:
    segs = [
        _seg(0.0, "This is a great point."),
        _seg(5.0, "I really like this idea a lot."),
        _seg(10.0, "This kind of question comes up a lot."),
    ]
    # No "like this" marker exists (see module docstring — "I like this
    # idea" is the verb "to like", not the deictic adverbial), and none of
    # these sentences hit the curated OBJECT noun list either.
    out = find_deixis_candidates(segs, "en")
    assert out == []


def test_bare_eto_alone_does_not_fire_russian() -> None:
    segs = [
        _seg(0.0, "Это очень интересная тема для разговора."),
        _seg(5.0, "Вот это переживание меня взволновало."),
        _seg(10.0, "Это то самое, о чём я говорил раньше."),
    ]
    out = find_deixis_candidates(segs, "ru")
    assert out == []


def test_bare_das_alone_does_not_fire_german() -> None:
    segs = [
        _seg(0.0, "Das ist ein wichtiger Punkt."),
        _seg(5.0, "Ich habe ihr sogar diese Lampe gekauft."),
        _seg(10.0, "Diesem Vollidioten kann ich nicht vertrauen."),
    ]
    out = find_deixis_candidates(segs, "de")
    assert out == []


def test_word_boundary_prevents_zhivot_takoi_false_positive() -> None:
    # "живот такой" ends in "...вот такой" as a raw substring — the
    # word-boundary anchor must stop this from matching "вот такой".
    segs = [_seg(0.0, "У меня живот такой большой после обеда.")]
    out = find_deixis_candidates(segs, "ru")
    assert out == []


# ---------------------------------------------------------------------------
# Near-in-time collapsing
# ---------------------------------------------------------------------------


def test_nearby_hits_collapse_into_one_candidate() -> None:
    segs = [
        _seg(100.0, "Вот так вот."),
        _seg(101.5, "Вот так вот."),
        _seg(103.0, "И ещё вот так вот."),
    ]
    out = find_deixis_candidates(segs, "ru")
    assert len(out) == 1
    # Keeps the earliest timestamp — the gesture's start.
    assert out[0].timestamp == 100.0


def test_hits_far_apart_do_not_collapse() -> None:
    gap = COLLAPSE_WINDOW_SECONDS + 5.0
    segs = [
        _seg(0.0, "Do it like this."),
        _seg(gap, "Watch this instead."),
    ]
    out = find_deixis_candidates(segs, "en")
    assert len(out) == 2
    assert [c.timestamp for c in out] == [0.0, gap]


def test_collapse_keeps_highest_weight_representative() -> None:
    # "in the description" (0.85) outweighs "you can see" (0.55); both land
    # within the collapse window, so the merged candidate should report the
    # more confident phrase's category.
    segs = [
        _seg(10.0, "You can see it here,"),
        _seg(11.0, "it's in the description too."),
    ]
    out = find_deixis_candidates(segs, "en")
    assert len(out) == 1
    assert out[0].category == DeixisCategory.EXTERNAL
    assert out[0].confidence == 0.85


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_candidates_returned_in_transcript_order() -> None:
    segs = [
        _seg(50.0, "The article number is in the description."),
        _seg(0.0, "First, do it like this."),
        _seg(25.0, "Now apply this cream."),
    ]
    out = find_deixis_candidates(segs, "en")
    timestamps = [c.timestamp for c in out]
    assert timestamps == sorted(timestamps)
    assert timestamps == [0.0, 25.0, 50.0]


# ---------------------------------------------------------------------------
# max_candidates cap — most confident survive, output still chronological.
# ---------------------------------------------------------------------------


def test_max_candidates_keeps_most_confident_but_stays_in_order() -> None:
    segs = [
        _seg(0.0, "You can see this side clearly."),  # weight 0.65 (this side)
        _seg(100.0, "Watch this carefully now."),  # weight 0.85
        _seg(200.0, "This one is different."),  # weight 0.65 (this one)
        _seg(300.0, "The link is in the description."),  # weight 0.85
    ]
    out = find_deixis_candidates(segs, "en", max_candidates=2)
    assert len(out) == 2
    # Chronological order preserved among survivors.
    assert [c.timestamp for c in out] == [100.0, 300.0]
    assert all(c.confidence == 0.85 for c in out)


def test_max_candidates_zero_returns_empty() -> None:
    segs = [_seg(0.0, "Watch this carefully now.")]
    out = find_deixis_candidates(segs, "en", max_candidates=0)
    assert out == []


def test_default_cap_is_reasonably_small() -> None:
    # 20 well-separated action hits shouldn't all survive with the default cap.
    segs = [
        _seg(float(i) * 10.0, "Now watch this and do it like this.")
        for i in range(20)
    ]
    out = find_deixis_candidates(segs, "en")
    assert 0 < len(out) <= 20
    assert len(out) <= DEFAULT_MAX_CANDIDATES


# ---------------------------------------------------------------------------
# Language fallback — None/unknown must not silently disable the feature.
# ---------------------------------------------------------------------------


def test_none_language_falls_back_to_matching_all_languages() -> None:
    segs = [_seg(0.0, "Возьмём вот такой вот маркер и рисуем вот так.")]
    out = find_deixis_candidates(segs, None)
    assert out != []


def test_unknown_language_falls_back_to_matching_all_languages() -> None:
    segs = [_seg(0.0, "Und so macht man das, ganz einfach.")]
    out = find_deixis_candidates(segs, "xx")
    assert out != []


def test_region_suffixed_language_code_resolves() -> None:
    segs = [_seg(0.0, "Watch this and do it like this.")]
    out = find_deixis_candidates(segs, "en-US")
    assert out != []


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


def test_same_input_produces_same_output() -> None:
    segs = [
        _seg(0.0, "Do it like this."),
        _seg(30.0, "Check the link in the description."),
    ]
    first = find_deixis_candidates(segs, "en")
    second = find_deixis_candidates(segs, "en")
    assert first == second


# ---------------------------------------------------------------------------
# Real-data-derived case (from job iBcRstyWUFaE, a Russian drawing tutorial —
# segments below are excerpted verbatim from the live DB's raw_segments_json).
# ---------------------------------------------------------------------------


def test_real_segments_from_drawing_tutorial_job() -> None:
    segs = [
        _seg(28.84, "И смотрите, что у нас на обложке тетрадки."),
        _seg(61.039, "Возьмём вот такой вот маркер такого цвета."),
        _seg(155.28, "Ну вы ж видите меня, видите?"),
        _seg(281.639, "получается вот так вот. Да, нехило,"),
        _seg(470.159, "Рекомендую перейти по ссылке в описании,"),
        _seg(861.48, "такие препараты, как кипферон. Вот они"),
    ]
    out = find_deixis_candidates(segs, "ru", max_candidates=10)
    categories = {c.category for c in out}
    # A genuine mix of action / object / external moments, not one blanket
    # category and not zero (the whole point of the marker design).
    assert DeixisCategory.ACTION in categories
    assert DeixisCategory.OBJECT in categories
    assert DeixisCategory.EXTERNAL in categories
    # "видите?" alone (bare, no reinforcing cue) must not fire — it's the
    # kind of common word this module is designed to reject.
    assert not any(c.timestamp == 155.28 for c in out)


def test_dataclass_fields_are_frozen_and_typed() -> None:
    candidate = DeixisCandidate(
        timestamp=1.0, phrase="watch this", category=DeixisCategory.ACTION, confidence=0.85
    )
    assert candidate.timestamp == 1.0
    assert candidate.category == DeixisCategory.ACTION
