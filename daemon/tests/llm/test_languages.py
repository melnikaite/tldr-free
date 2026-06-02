"""Language code normalization (used by the translate endpoint)."""

from __future__ import annotations

import pytest

from src.llm.languages import (
    Language,
    UnknownLanguageError,
    normalize_lang,
    supported_codes,
)


def test_normalize_iso_639_1_code() -> None:
    lang = normalize_lang("ru")
    assert lang.code == "ru"
    assert lang.english_name == "Russian"


def test_normalize_is_case_insensitive() -> None:
    assert normalize_lang("RU").code == "ru"
    assert normalize_lang("German").code == "de"
    assert normalize_lang("DEUTSCH").code == "de"


def test_normalize_strips_region_tag() -> None:
    """`en-US` → `en`. Some browsers / yt-dlp emit region-tagged codes."""
    assert normalize_lang("en-US").code == "en"
    assert normalize_lang("zh-CN").code == "zh"


def test_normalize_accepts_iso_639_2_codes() -> None:
    assert normalize_lang("rus").code == "ru"
    assert normalize_lang("eng").code == "en"
    assert normalize_lang("deu").code == "de"


def test_normalize_accepts_autonyms() -> None:
    assert normalize_lang("Русский").code == "ru"
    assert normalize_lang("Deutsch").code == "de"
    assert normalize_lang("日本語").code == "ja"


def test_unknown_language_raises_with_supported_list() -> None:
    with pytest.raises(UnknownLanguageError) as ei:
        normalize_lang("klingon")
    msg = str(ei.value)
    assert "klingon" in msg
    # Useful UX: tell the caller what IS supported.
    for code in ("en", "ru", "ja"):
        assert code in msg


def test_empty_input_rejected() -> None:
    with pytest.raises(UnknownLanguageError):
        normalize_lang("")
    with pytest.raises(UnknownLanguageError):
        normalize_lang("   ")


def test_supported_codes_contains_basics() -> None:
    codes = set(supported_codes())
    # Should at least include the obvious top languages.
    for c in ("en", "ru", "de", "fr", "es", "ja", "zh"):
        assert c in codes


def test_returned_language_is_immutable() -> None:
    a = normalize_lang("ru")
    b = normalize_lang("ru")
    # Same Language record returned (we cache in _LOOKUP).
    assert a is b
    assert isinstance(a, Language)
