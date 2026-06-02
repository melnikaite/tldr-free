"""Language code normalization for transcript translations.

The user types a code (``ru``, ``zh``, ``ja``) or English name
(``russian``, ``chinese``) in the sidepanel's language input. The
daemon's translate endpoint passes that string to ``normalize_lang``
which canonicalises it to an ISO-639-1 two-letter code and a
human-readable English name. Unknown input → ``UnknownLanguageError``,
the API turns that into a 400 the UI displays.

Scope: only the ~20 languages Gemma 4 was trained on with at least
decent translation quality. Keeps the chip list short, avoids users
asking for translations the model can't actually do well.

Adding a language: append to ``_LANGUAGES``. Aliases (autonyms, common
misspellings) go in ``_ALIASES``. No need to touch any other file.
"""

from __future__ import annotations

from dataclasses import dataclass


class UnknownLanguageError(ValueError):
    """Raised when ``normalize_lang`` can't resolve the input to a known code."""


@dataclass(frozen=True)
class Language:
    code: str          # ISO-639-1 lowercase
    english_name: str  # used in the translate prompt to instruct Gemma
    autonym: str       # how the language refers to itself


# Curated list of languages Gemma 4 handles well. Order is roughly by
# usefulness for the typical TLDR user. Adding to this is cheap; removing
# means users with cached translations in that code see a 400 on retry —
# don't churn this without thought.
_LANGUAGES: tuple[Language, ...] = (
    Language("en", "English",            "English"),
    Language("ru", "Russian",            "Русский"),
    Language("de", "German",             "Deutsch"),
    Language("fr", "French",             "Français"),
    Language("es", "Spanish",            "Español"),
    Language("it", "Italian",            "Italiano"),
    Language("pt", "Portuguese",         "Português"),
    Language("nl", "Dutch",              "Nederlands"),
    Language("pl", "Polish",             "Polski"),
    Language("tr", "Turkish",            "Türkçe"),
    Language("uk", "Ukrainian",          "Українська"),
    Language("cs", "Czech",              "Čeština"),
    Language("sv", "Swedish",            "Svenska"),
    Language("ja", "Japanese",           "日本語"),
    Language("ko", "Korean",             "한국어"),
    Language("zh", "Chinese (Simplified)", "中文"),
    Language("ar", "Arabic",             "العربية"),
    Language("hi", "Hindi",              "हिन्दी"),
    Language("he", "Hebrew",             "עברית"),
    Language("vi", "Vietnamese",         "Tiếng Việt"),
)

# Extra spellings that should resolve to a known code. Lowercased here;
# input is lowercased before lookup, so the table doesn't need case
# variants.
_ALIASES: dict[str, str] = {
    # ISO-639-2 / common alternates
    "eng": "en",
    "rus": "ru",
    "ger": "de", "deu": "de",
    "fre": "fr", "fra": "fr",
    "spa": "es",
    "ita": "it",
    "por": "pt",
    "nld": "nl",
    "pol": "pl",
    "tur": "tr",
    "ukr": "uk",
    "ces": "cs", "cze": "cs",
    "swe": "sv",
    "jpn": "ja",
    "kor": "ko",
    "chi": "zh", "zho": "zh",
    "ara": "ar",
    "hin": "hi",
    "heb": "he",
    "vie": "vi",
}


def _build_lookup() -> dict[str, Language]:
    """Index for fast ``normalize_lang`` — code, English name, autonym, alias."""
    lookup: dict[str, Language] = {}
    for lang in _LANGUAGES:
        lookup[lang.code] = lang
        lookup[lang.english_name.lower()] = lang
        lookup[lang.autonym.lower()] = lang
    for alias, code in _ALIASES.items():
        if code in lookup:
            lookup[alias] = lookup[code]
    return lookup


_LOOKUP = _build_lookup()


def normalize_lang(raw: str) -> Language:
    """Resolve ``raw`` (case-insensitive) to a :class:`Language`.

    Accepts ISO-639-1 codes (``"ru"``), ISO-639-2 codes (``"rus"``),
    English names (``"russian"``, ``"Russian"``), autonyms (``"русский"``),
    or any alias from ``_ALIASES``. Region tags are stripped (``"en-US"``
    → ``"en"``).

    Raises ``UnknownLanguageError`` with a list of supported codes if the
    input doesn't match anything we know — the caller turns that into a
    400 with a helpful message.
    """
    if not isinstance(raw, str):
        raise UnknownLanguageError("language must be a string")
    s = raw.strip().lower()
    if not s:
        raise UnknownLanguageError("language is empty")
    if "-" in s and len(s.split("-", 1)[0]) == 2:
        s = s.split("-", 1)[0]
    lang = _LOOKUP.get(s)
    if lang is None:
        supported = ", ".join(language.code for language in _LANGUAGES)
        raise UnknownLanguageError(
            f"unsupported language {raw!r}. Supported: {supported}"
        )
    return lang


def supported_codes() -> list[str]:
    """List of ISO-639-1 codes the daemon will translate to. Exposed via
    a future ``/transcript/languages`` endpoint if we want an explicit
    listing for the UI — for now the front-end relies on chip + free input."""
    return [lang.code for lang in _LANGUAGES]


__all__ = [
    "Language",
    "UnknownLanguageError",
    "normalize_lang",
    "supported_codes",
]
