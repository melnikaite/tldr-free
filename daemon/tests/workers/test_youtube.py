"""Tests for workers.youtube.

Covers extract_video_id URL parsing and exception classification through
fetch_transcript_with_retry with a monkeypatched YouTubeTranscriptApi.
"""

from __future__ import annotations

import itertools
import json
import string
from pathlib import Path
from typing import Any

import pytest
import requests
from youtube_transcript_api import Transcript, TranscriptList
from youtube_transcript_api._errors import (
    AgeRestricted,
    CouldNotRetrieveTranscript,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)

from src.api.schemas import Cookie
from src.workers import youtube
from src.workers.errors import (
    ExhaustedRetriesError,
    NetworkTranscriptError,
    PermanentTranscriptError,
    TransientTranscriptError,
)

# ---------------------------------------------------------------------------
# extract_video_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://music.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?t=15", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/v/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ],
)
def test_extract_video_id_handles_common_forms(url: str, expected: str) -> None:
    assert youtube.extract_video_id(url) == expected


@pytest.mark.parametrize(
    "url, expected",
    [
        # Extra path/query segments after the id are ignored.
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ?feature=share", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ?start=10", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ/", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ/", "dQw4w9WgXcQ"),
        # Multiple v= params — first one wins.
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&v=ZZZZZZZZZZZ", "dQw4w9WgXcQ"),
        # Subdomains beyond the common ones still resolve.
        ("https://gaming.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ],
)
def test_extract_video_id_handles_edge_forms(url: str, expected: str) -> None:
    assert youtube.extract_video_id(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "",
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/",
        "https://www.youtube.com/watch",
        "https://www.youtube.com/watch?x=foo",
        "https://www.youtube.com/playlist?list=PLxxxxx",
        "not a url",
        # youtu.be host but no/invalid id in path.
        "https://youtu.be/",
        "https://youtu.be/short",  # too short for the 11-char pattern
        "https://youtu.be/waytoolongvideoid",  # too long
        # youtube.com host but id fails the strict pattern.
        "https://www.youtube.com/watch?v=tooShort",
        "https://www.youtube.com/shorts/bad!chars99",
        "https://www.youtube.com/embed/",
        # Recognised host, unrecognised path shape.
        "https://www.youtube.com/feed/subscriptions",
    ],
)
def test_extract_video_id_rejects_invalid(url: str) -> None:
    with pytest.raises(ValueError):
        youtube.extract_video_id(url)


# ---------------------------------------------------------------------------
# _classify_transcript_exception — direct (pure) branch coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc_cls",
    [TranscriptsDisabled, VideoUnavailable, AgeRestricted],
)
def test_classify_permanent(exc_cls) -> None:  # noqa: ANN001
    exc = _build_yt_api_exception(exc_cls)
    out = youtube._classify_transcript_exception(exc)
    assert isinstance(out, PermanentTranscriptError)


@pytest.mark.parametrize(
    "exc_cls",
    [IpBlocked, RequestBlocked],
)
def test_classify_transient(exc_cls) -> None:  # noqa: ANN001
    exc = _build_yt_api_exception(exc_cls)
    out = youtube._classify_transcript_exception(exc)
    assert isinstance(out, TransientTranscriptError)


def test_classify_request_exception_is_network() -> None:
    out = youtube._classify_transcript_exception(
        requests.exceptions.ConnectTimeout("boom")
    )
    assert isinstance(out, NetworkTranscriptError)


def test_classify_could_not_retrieve_is_transient() -> None:
    # CouldNotRetrieveTranscript (base of the library hierarchy, not in either
    # explicit tuple) → treated as transient so the caller still defers.
    exc = CouldNotRetrieveTranscript(video_id="abc")
    out = youtube._classify_transcript_exception(exc)
    assert isinstance(out, TransientTranscriptError)


def test_classify_unknown_exception_is_transient() -> None:
    out = youtube._classify_transcript_exception(RuntimeError("surprise"))
    assert isinstance(out, TransientTranscriptError)


# ---------------------------------------------------------------------------
# _pick_subtitle_lang — pure language-selection logic
# ---------------------------------------------------------------------------


def test_pick_subtitle_lang_empty_returns_none() -> None:
    assert youtube._pick_subtitle_lang({}, {}, "en", ["en", "fr"]) is None


def test_pick_subtitle_lang_prefers_original() -> None:
    available = {"en": [], "fr": [], "de": []}
    # Original language wins even when it sits later in preferences.
    assert youtube._pick_subtitle_lang(available, {}, "de", ["en", "fr"]) == "de"


def test_pick_subtitle_lang_falls_back_to_preferences_in_order() -> None:
    available = {"fr": [], "de": []}
    # Original not available → first matching preference wins.
    assert youtube._pick_subtitle_lang(available, {}, "en", ["es", "fr", "de"]) == "fr"


def test_pick_subtitle_lang_skips_none_original() -> None:
    available = {"fr": [], "de": []}
    assert youtube._pick_subtitle_lang(available, {}, None, ["de"]) == "de"


def test_pick_subtitle_lang_falls_back_to_output_language() -> None:
    available = {"fr": [], "ru": []}
    # Neither original nor any preference matches, but output.language does.
    assert (
        youtube._pick_subtitle_lang(available, {}, "en", ["es", "de"], output_language="ru")
        == "ru"
    )


def test_pick_subtitle_lang_falls_back_to_manual_track() -> None:
    available = {"zh": [], "ar": [], "de": [], "fr": []}
    manual = {"fr": []}
    # Nothing else matches, but "fr" is a real, manually-created track.
    assert (
        youtube._pick_subtitle_lang(available, manual, "en", ["es"], output_language="ru")
        == "fr"
    )


def test_pick_subtitle_lang_never_picks_arbitrary_machine_translation() -> None:
    """157 machine-translated auto-caption languages, no manual track, and
    no match on original/preferences/output_language: the old code picked
    ``sorted(available)[0]`` here (alphabetically "aa", Afar) — silently
    summarising a random translation is worse than giving up. The fix must
    return None so the caller retries or defers to Whisper instead."""
    # 157 synthetic two-letter codes, matching the real-world count from the
    # probe (see task evidence: ESjRLuqF-do had 157 automatic_captions
    # languages). "aa" (Afar) sorts first alphabetically — exactly what the
    # old code picked.
    codes: list[str] = []
    for a, b in itertools.product(string.ascii_lowercase, repeat=2):
        code = a + b
        if code in ("en", "ru"):  # must not accidentally satisfy the preferences below
            continue
        codes.append(code)
        if len(codes) == 157:
            break
    available = {code: [] for code in codes}
    assert "aa" in available
    assert "en" not in available and "ru" not in available
    assert len(available) == 157
    result = youtube._pick_subtitle_lang(
        available, {}, original_lang=None, preferences=["en", "ru"], output_language="ru",
    )
    assert result is None
    assert result != sorted(available.keys())[0]


def test_pick_subtitle_lang_generalises_to_generic_site_single_manual_track() -> None:
    """The generic-media shape (Phase 1): a non-YouTube site typically has
    exactly one manually-created track and NO automatic_captions at all —
    unlike YouTube, which usually offers dozens of machine-translated auto
    tracks alongside at most one real one. ``_pick_subtitle_lang`` is shared
    unchanged between both callers of ``_download_subtitles_sync``; this
    documents that the existing priority chain already covers the generic
    case (falls to the "any manually-created track" step) without any
    YouTube-specific reasoning."""
    available = {"deu": []}  # merged auto ({}) + manual ({"deu": []})
    manual = {"deu": []}
    # Original language/preferences/output_language all miss (foreign site,
    # default config) — must still land on the one real track that exists.
    result = youtube._pick_subtitle_lang(
        available, manual, original_lang=None, preferences=["en", "ru"], output_language="en",
    )
    assert result == "deu"


# ---------------------------------------------------------------------------
# _parse_subtitle_json3 — pure parsing of YouTube's json3 caption format
# ---------------------------------------------------------------------------


def _write_json3(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "subs.json3"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_parse_subtitle_json3_basic(tmp_path: Path) -> None:
    payload = {
        "events": [
            {
                "tStartMs": 1500,
                "dDurationMs": 2000,
                "segs": [{"utf8": "hello "}, {"utf8": "world"}],
            }
        ]
    }
    out = youtube._parse_subtitle_json3(_write_json3(tmp_path, payload))
    assert out == [{"start": 1.5, "duration": 2.0, "text": "hello world"}]


def test_parse_subtitle_json3_skips_blank_and_cue_only(tmp_path: Path) -> None:
    payload = {
        "events": [
            # Missing tStartMs → skipped.
            {"dDurationMs": 1000, "segs": [{"utf8": "no start"}]},
            # Empty text after join/strip → skipped.
            {"tStartMs": 0, "dDurationMs": 500, "segs": [{"utf8": "  "}]},
            # No segs key at all → empty text → skipped.
            {"tStartMs": 1000, "dDurationMs": 500},
            # Valid one with newline normalised to space.
            {"tStartMs": 2000, "dDurationMs": 1000, "segs": [{"utf8": "a\nb"}]},
        ]
    }
    out = youtube._parse_subtitle_json3(_write_json3(tmp_path, payload))
    assert out == [{"start": 2.0, "duration": 1.0, "text": "a b"}]


def test_parse_subtitle_json3_defaults_missing_duration(tmp_path: Path) -> None:
    payload = {"events": [{"tStartMs": 3000, "segs": [{"utf8": "x"}]}]}
    out = youtube._parse_subtitle_json3(_write_json3(tmp_path, payload))
    assert out == [{"start": 3.0, "duration": 0.0, "text": "x"}]


def test_parse_subtitle_json3_no_events(tmp_path: Path) -> None:
    assert youtube._parse_subtitle_json3(_write_json3(tmp_path, {})) == []


# ---------------------------------------------------------------------------
# _parse_subtitle_vtt — pure parsing of WebVTT (the format non-YouTube sites
# actually serve: ZDF, ARD, Vimeo, TED, Coursera, …)
# ---------------------------------------------------------------------------


def _write_vtt(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "subs.vtt"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def test_parse_subtitle_vtt_basic(tmp_path: Path) -> None:
    path = _write_vtt(
        tmp_path,
        ["WEBVTT", "", "00:00:01.500 --> 00:00:03.500", "hello world", ""],
    )
    out = youtube._parse_subtitle_vtt(path)
    assert out == [{"start": 1.5, "duration": 2.0, "text": "hello world"}]


def test_parse_subtitle_vtt_multiline_cue_joined_with_space(tmp_path: Path) -> None:
    path = _write_vtt(
        tmp_path,
        ["WEBVTT", "", "00:00:00.000 --> 00:00:02.000", "line one", "line two", ""],
    )
    out = youtube._parse_subtitle_vtt(path)
    assert out == [{"start": 0.0, "duration": 2.0, "text": "line one line two"}]


def test_parse_subtitle_vtt_strips_cue_id_settings_and_inline_tags(tmp_path: Path) -> None:
    """Cue identifier lines, trailing cue settings (``align:``/``position:``)
    and inline markup (``<c>``, karaoke timestamps like ``<00:00:06.000>``)
    must all be stripped down to plain text."""
    path = _write_vtt(
        tmp_path,
        [
            "WEBVTT",
            "",
            "1",
            "00:00:05.000 --> 00:00:07.000 align:start position:0%",
            "<c.colorE5E5E5>Hey <00:00:06.000>Fred</c>",
            "",
        ],
    )
    out = youtube._parse_subtitle_vtt(path)
    assert out == [{"start": 5.0, "duration": 2.0, "text": "Hey Fred"}]


def test_parse_subtitle_vtt_supports_hours_and_comma_decimal(tmp_path: Path) -> None:
    """SRT-flavoured WebVTT (comma decimal separator) and hour-including
    timestamps must both parse — some non-YouTube extractors emit either."""
    path = _write_vtt(
        tmp_path,
        ["WEBVTT", "", "01:00:00,250 --> 01:00:02,750", "an hour in", ""],
    )
    out = youtube._parse_subtitle_vtt(path)
    assert out == [{"start": 3600.25, "duration": 2.5, "text": "an hour in"}]


def test_parse_subtitle_vtt_skips_header_note_and_empty_cues(tmp_path: Path) -> None:
    path = _write_vtt(
        tmp_path,
        [
            "WEBVTT",
            "Kind: captions",
            "Language: de",
            "",
            "NOTE this is a comment block",
            "",
            "00:00:01.000 --> 00:00:02.000",
            "",
            "00:00:03.000 --> 00:00:04.000",
            "real cue",
            "",
        ],
    )
    out = youtube._parse_subtitle_vtt(path)
    assert out == [{"start": 3.0, "duration": 1.0, "text": "real cue"}]


def test_parse_subtitle_vtt_no_cues_returns_empty_list(tmp_path: Path) -> None:
    assert youtube._parse_subtitle_vtt(_write_vtt(tmp_path, ["WEBVTT", ""])) == []


# ---------------------------------------------------------------------------
# fetch_transcript_with_retry — error classification
# ---------------------------------------------------------------------------


class _FakeFetched:
    """Stand-in for FetchedTranscript. Iteration yields snippet objects."""

    def __init__(self, snippets: list[dict]) -> None:
        self._snippets = [_FakeSnippet(**s) for s in snippets]

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._snippets)


class _FakeSnippet:
    def __init__(self, text: str, start: float, duration: float) -> None:
        self.text = text
        self.start = start
        self.duration = duration


def _make_transcript(
    *, language_code: str, is_generated: bool, result: list[dict] | BaseException,
) -> Transcript:
    """Build a real ``Transcript`` whose ``.fetch()`` is stubbed.

    Using the real class (rather than a hand-rolled fake) means
    ``TranscriptList.find_transcript`` / ``__iter__`` — which
    ``_select_youtube_api_transcript`` relies on — run for real in tests,
    not just against a re-implementation of their behaviour.
    """
    transcript = Transcript(
        None,  # type: ignore[arg-type]  # http_client — unused; fetch is stubbed below
        "video",
        "https://example.invalid/captions",
        language_code,
        language_code,
        is_generated,
        [],
    )

    def _fetch(preserve_formatting: bool = False) -> Any:  # noqa: ARG001
        if isinstance(result, BaseException):
            raise result
        return _FakeFetched(result)

    transcript.fetch = _fetch  # type: ignore[method-assign]
    return transcript


def _make_transcript_list(
    *,
    video_id: str = "video",
    manual: dict[str, Transcript] | None = None,
    generated: dict[str, Transcript] | None = None,
) -> TranscriptList:
    return TranscriptList(video_id, manual or {}, generated or {}, [])


def _make_fake_api(side_effect):  # type: ignore[no-untyped-def]
    """Return a class that mimics YouTubeTranscriptApi but uses side_effect on list()."""

    class FakeAPI:
        def __init__(self, *args, **kwargs):  # noqa: ANN001
            pass

        def list(self, video_id):  # noqa: ANN001
            if callable(side_effect):
                return side_effect(video_id)
            raise side_effect

    return FakeAPI


def _build_yt_api_exception(cls):  # type: ignore[no-untyped-def]
    """Build an instance of a youtube_transcript_api error.

    These constructors take a video_id parameter; the message text isn't important
    for classification.
    """
    return cls(video_id="testvideoid")


@pytest.mark.asyncio
async def test_permanent_transcript_disabled_raises_permanent(monkeypatch) -> None:  # noqa: ANN001
    fake_api = _make_fake_api(_build_yt_api_exception(TranscriptsDisabled))
    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", fake_api)

    with pytest.raises(PermanentTranscriptError):
        await youtube.fetch_transcript_with_retry(
            video_id="abc",
            cookies=[],
            max_attempts=3,
            backoff_seconds=[0, 0, 0],
            preferences=["en"],
        )


@pytest.mark.asyncio
async def test_permanent_no_transcript_found_raises_permanent(monkeypatch) -> None:  # noqa: ANN001
    err = NoTranscriptFound(
        video_id="abc",
        requested_language_codes=["en"],
        transcript_data=None,
    )
    fake_api = _make_fake_api(err)
    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", fake_api)

    with pytest.raises(PermanentTranscriptError):
        await youtube.fetch_transcript_with_retry(
            video_id="abc",
            cookies=[],
            max_attempts=3,
            backoff_seconds=[0, 0, 0],
            preferences=["en"],
        )


@pytest.mark.asyncio
async def test_permanent_video_unavailable_raises_permanent(monkeypatch) -> None:  # noqa: ANN001
    fake_api = _make_fake_api(_build_yt_api_exception(VideoUnavailable))
    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", fake_api)

    with pytest.raises(PermanentTranscriptError):
        await youtube.fetch_transcript_with_retry(
            video_id="abc",
            cookies=[],
            max_attempts=3,
            backoff_seconds=[0, 0, 0],
            preferences=["en"],
        )


@pytest.mark.asyncio
async def test_permanent_age_restricted_raises_permanent(monkeypatch) -> None:  # noqa: ANN001
    fake_api = _make_fake_api(_build_yt_api_exception(AgeRestricted))
    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", fake_api)

    with pytest.raises(PermanentTranscriptError):
        await youtube.fetch_transcript_with_retry(
            video_id="abc",
            cookies=[],
            max_attempts=3,
            backoff_seconds=[0, 0, 0],
            preferences=["en"],
        )


@pytest.mark.asyncio
async def test_transient_ip_blocked_raises_exhausted_after_retries(monkeypatch) -> None:  # noqa: ANN001
    fake_api = _make_fake_api(_build_yt_api_exception(IpBlocked))
    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", fake_api)

    with pytest.raises(ExhaustedRetriesError) as exc_info:
        await youtube.fetch_transcript_with_retry(
            video_id="abc",
            cookies=[],
            max_attempts=2,
            backoff_seconds=[0, 0],
            preferences=["en"],
        )
    # Code should propagate from the wrapped TransientTranscriptError.
    assert exc_info.value.code == "transcript_blocked"


@pytest.mark.asyncio
async def test_transient_request_blocked_raises_exhausted(monkeypatch) -> None:  # noqa: ANN001
    fake_api = _make_fake_api(_build_yt_api_exception(RequestBlocked))
    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", fake_api)

    with pytest.raises(ExhaustedRetriesError):
        await youtube.fetch_transcript_with_retry(
            video_id="abc",
            cookies=[],
            max_attempts=2,
            backoff_seconds=[0, 0],
            preferences=["en"],
        )


@pytest.mark.asyncio
async def test_successful_fetch_returns_segments(monkeypatch) -> None:  # noqa: ANN001
    snippets = [
        {"text": "hello", "start": 0.0, "duration": 5.0},
        {"text": "world", "start": 5.0, "duration": 5.0},
    ]

    def _ok(video_id: str):  # type: ignore[no-untyped-def]
        return _make_transcript_list(
            generated={"en": _make_transcript(language_code="en", is_generated=True, result=snippets)},
        )

    fake_api = _make_fake_api(_ok)
    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", fake_api)

    out = await youtube.fetch_transcript_with_retry(
        video_id="abc",
        cookies=[],
        max_attempts=3,
        backoff_seconds=[0, 0, 0],
        preferences=["en"],
    )
    assert out == [
        {"text": "hello", "start": 0.0, "duration": 5.0},
        {"text": "world", "start": 5.0, "duration": 5.0},
    ]


@pytest.mark.asyncio
async def test_retry_then_success(monkeypatch) -> None:  # noqa: ANN001
    """One transient fail then a successful fetch — should not raise."""
    state = {"calls": 0}

    def _flaky(video_id: str):  # type: ignore[no-untyped-def]
        state["calls"] += 1
        if state["calls"] == 1:
            raise _build_yt_api_exception(IpBlocked)
        return _make_transcript_list(
            generated={
                "en": _make_transcript(
                    language_code="en", is_generated=True,
                    result=[{"text": "ok", "start": 0.0, "duration": 1.0}],
                ),
            },
        )

    fake_api = _make_fake_api(_flaky)
    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", fake_api)

    out = await youtube.fetch_transcript_with_retry(
        video_id="abc",
        cookies=[],
        max_attempts=3,
        backoff_seconds=[0, 0, 0],
        preferences=["en"],
    )
    assert state["calls"] == 2
    assert out == [{"text": "ok", "start": 0.0, "duration": 1.0}]


# ---------------------------------------------------------------------------
# Bug 1 — the fast path must not hard-code English (youtube-transcript-api)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fast_path_fetches_russian_only_video(monkeypatch) -> None:  # noqa: ANN001
    """A video with only a Russian auto-caption track (no English at all)
    must still be fetched by the fast path — this was the actual failure
    mode: youtube-transcript-api's ``fetch()`` defaults to ``languages=("en",)``
    when no languages are passed, so a Russian-only video raised
    "No transcripts were found for any of the requested language codes:
    ('en',)" every time, even though .list() shows the video has Russian
    captions available."""
    ru_snippets = [{"text": "привет", "start": 0.0, "duration": 2.0}]

    def _ru_only(video_id: str):  # type: ignore[no-untyped-def]
        return _make_transcript_list(
            generated={"ru": _make_transcript(language_code="ru", is_generated=True, result=ru_snippets)},
        )

    fake_api = _make_fake_api(_ru_only)
    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", fake_api)

    out = await youtube.fetch_transcript_with_retry(
        video_id="ru-video",
        cookies=[],
        max_attempts=1,
        backoff_seconds=[0],
        preferences=["en", "ru"],  # default youtube.subtitle_lang_preferences
        output_language="en",  # output.language does not match either — must not matter
    )
    assert out == [{"text": "привет", "start": 0.0, "duration": 2.0}]


def test_select_youtube_api_transcript_no_candidates_falls_back_to_only_track() -> None:
    """No original-language signal (multiple/zero generated tracks), no
    preference/output_language match — but there IS exactly one manually
    created track, so it must be used rather than raising."""
    transcript_list = _make_transcript_list(
        manual={"de": _make_transcript(language_code="de", is_generated=False, result=[])},
    )
    picked = youtube._select_youtube_api_transcript(
        transcript_list, preferences=["en", "ru"], output_language="fr",
    )
    assert picked.language_code == "de"


def test_select_youtube_api_transcript_raises_when_truly_empty() -> None:
    transcript_list = _make_transcript_list()
    with pytest.raises(NoTranscriptFound):
        youtube._select_youtube_api_transcript(
            transcript_list, preferences=["en", "ru"], output_language=None,
        )


# ---------------------------------------------------------------------------
# Bug 2 — live_chat must never be treated as a subtitle track
# ---------------------------------------------------------------------------


def _write_json3_file(path: Path, events: list[dict]) -> None:
    path.write_text(json.dumps({"events": events}), encoding="utf-8")


class _FakeYoutubeDL:
    """Stand-in for yt_dlp.YoutubeDL covering the probe + download passes
    ``_download_subtitles_sync`` performs.

    ``probe_info`` is returned verbatim for the probe pass (no
    ``subtitleslangs`` in opts). For the download pass, writes a file for
    the requested language at the conventional ``<id>.<lang>.<ext>`` path
    (``ext`` defaults to ``"json3"``, matching real yt-dlp/YouTube; set it to
    ``"vtt"`` or anything else to exercise format-negotiation dispatch) and
    returns an info dict pointing at it, so the parse step exercises a real
    file on disk exactly like the real yt-dlp flow does.
    """

    def __init__(self, opts: dict[str, Any]) -> None:
        self.opts = opts

    def __enter__(self) -> _FakeYoutubeDL:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def extract_info(self, url: str, download: bool):  # noqa: ANN001, ARG002
        if "subtitleslangs" not in self.opts:
            return dict(_FakeYoutubeDL.probe_info)
        chosen = self.opts["subtitleslangs"][0]
        out_dir = Path(self.opts["outtmpl"]).parent
        video_id = "video1"
        ext = _FakeYoutubeDL.ext
        sub_path = out_dir / f"{video_id}.{chosen}.{ext}"
        if ext == "json3":
            _write_json3_file(sub_path, _FakeYoutubeDL.events_by_lang[chosen])
        else:
            sub_path.write_text(_FakeYoutubeDL.raw_text_by_lang[chosen], encoding="utf-8")
        return {
            "id": video_id,
            "requested_subtitles": {chosen: {"filepath": str(sub_path)}},
        }

    # Set per-test via monkeypatch before instantiation.
    probe_info: dict[str, Any] = {}
    events_by_lang: dict[str, list[dict]] = {}
    ext: str = "json3"
    raw_text_by_lang: dict[str, str] = {}


def _patch_fake_ydl(  # noqa: ANN001
    monkeypatch,
    probe_info: dict[str, Any],
    events_by_lang: dict[str, list[dict]],
    *,
    ext: str = "json3",
    raw_text_by_lang: dict[str, str] | None = None,
) -> None:
    _FakeYoutubeDL.probe_info = probe_info
    _FakeYoutubeDL.events_by_lang = events_by_lang
    _FakeYoutubeDL.ext = ext
    _FakeYoutubeDL.raw_text_by_lang = raw_text_by_lang or {}
    monkeypatch.setattr("yt_dlp.YoutubeDL", _FakeYoutubeDL)
    # ffmpeg/deno opt-builders shell out to resolve host binaries; keep them
    # inert so tests don't depend on what's installed on the host.
    monkeypatch.setattr(youtube, "_ffmpeg_opt", lambda: {})
    monkeypatch.setattr(youtube, "_jsruntime_opt", lambda: {})


def test_download_subtitles_sync_skips_live_chat_and_picks_real_auto(
    monkeypatch, tmp_path: Path,  # noqa: ANN001
) -> None:
    """live_chat is manual-only and would outrank 157 real auto languages
    once manual is merged on top of auto — it must be filtered before that
    merge, and parsing it (line-delimited JSON, not json3) must never be
    attempted."""
    probe_info = {
        "language": "ru",
        "subtitles": {"live_chat": [{"ext": "json", "url": "https://example.invalid/chat"}]},
        "automatic_captions": {
            "ru": [{"ext": "json3", "url": "https://example.invalid/ru"}],
            "en": [{"ext": "json3", "url": "https://example.invalid/en"}],
        },
    }
    events_by_lang = {
        "ru": [{"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "привет"}]}],
    }
    _patch_fake_ydl(monkeypatch, probe_info, events_by_lang)

    out = youtube._download_subtitles_sync(
        url="https://www.youtube.com/watch?v=video1",
        cookies=[],
        dir=tmp_path,
        lang_preferences=["en", "ru"],
    )
    assert out == [{"start": 0.0, "duration": 1.0, "text": "привет"}]


def test_download_subtitles_sync_no_real_tracks_returns_none(
    monkeypatch, tmp_path: Path,  # noqa: ANN001
) -> None:
    """live_chat as the ONLY entry (no real captions at all) must yield
    "no caption track", not an attempt to parse the chat replay."""
    probe_info = {
        "language": None,
        "subtitles": {"live_chat": [{"ext": "json"}]},
        "automatic_captions": {},
    }
    _patch_fake_ydl(monkeypatch, probe_info, {})

    out = youtube._download_subtitles_sync(
        url="https://www.youtube.com/watch?v=video1",
        cookies=[],
        dir=tmp_path,
        lang_preferences=["en", "ru"],
    )
    assert out is None


# ---------------------------------------------------------------------------
# Phase 1 — format negotiation: non-YouTube sites serve WebVTT, not json3
# ---------------------------------------------------------------------------


def test_download_subtitles_sync_dispatches_to_vtt_parser(
    monkeypatch, tmp_path: Path,  # noqa: ANN001
) -> None:
    """A generic site (ZDF/ARD/Vimeo/TED/Coursera/…) offers only a manual
    track in WebVTT, never json3. ``_download_subtitles_sync`` must still
    request+download it (``_SUBTITLE_FORMAT_PREFERENCE`` falls back past
    json3 to vtt) and dispatch to ``_parse_subtitle_vtt`` based on the
    actual downloaded file's extension, not assume json3."""
    probe_info = {
        "language": "deu",
        "subtitles": {"deu": [{"ext": "vtt", "url": "https://example.invalid/deu.vtt"}]},
        "automatic_captions": {},
    }
    vtt_text = "\n".join(
        ["WEBVTT", "", "00:00:01.000 --> 00:00:04.000", "Hallo Welt", ""]
    )
    _patch_fake_ydl(
        monkeypatch, probe_info, {}, ext="vtt", raw_text_by_lang={"deu": vtt_text},
    )

    out = youtube._download_subtitles_sync(
        url="https://www.zdf.de/play/example",
        cookies=[],
        dir=tmp_path,
        lang_preferences=["en", "ru"],
    )
    assert out == [{"start": 1.0, "duration": 3.0, "text": "Hallo Welt"}]


def test_download_subtitles_sync_unsupported_format_returns_none(
    monkeypatch, tmp_path: Path,  # noqa: ANN001
) -> None:
    """If yt-dlp lands on a format we have no parser for (e.g. a site
    offering only ttml/dfxp), the probe must fail soft — None, so the caller
    falls through to Whisper — rather than crash trying to parse it."""
    probe_info = {
        "language": "deu",
        "subtitles": {"deu": [{"ext": "ttml", "url": "https://example.invalid/deu.ttml"}]},
        "automatic_captions": {},
    }
    _patch_fake_ydl(
        monkeypatch, probe_info, {}, ext="ttml", raw_text_by_lang={"deu": "<tt></tt>"},
    )

    out = youtube._download_subtitles_sync(
        url="https://www.zdf.de/play/example",
        cookies=[],
        dir=tmp_path,
        lang_preferences=["en", "ru"],
    )
    assert out is None


# ---------------------------------------------------------------------------
# Bug 3 — retry with backoff (and a cookie-less attempt) before Whisper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_subtitles_retries_then_succeeds(monkeypatch) -> None:  # noqa: ANN001
    """First attempt (with cookies) fails; second attempt (cookie-less)
    succeeds — the overall call must return the successful result rather
    than conceding to Whisper after the first miss."""
    calls: list[dict[str, Any]] = []

    def _fake_sync(*, url, cookies, dir, lang_preferences, output_language=None):  # noqa: ANN001
        calls.append({"cookies": cookies})
        if len(calls) == 1:
            return None  # first attempt: no usable track
        return [{"start": 0.0, "duration": 1.0, "text": "ok"}]

    monkeypatch.setattr(youtube, "_download_subtitles_sync", _fake_sync)

    cookie = Cookie(name="SID", value="x", domain=".youtube.com")
    out = await youtube.download_subtitles(
        url="https://www.youtube.com/watch?v=video1",
        cookies=[cookie],
        dir=Path("/unused"),
        lang_preferences=["en", "ru"],
        max_attempts=3,
        backoff_seconds=[0, 0],
    )

    assert out == [{"start": 0.0, "duration": 1.0, "text": "ok"}]
    assert len(calls) == 2
    # First attempt used the caller's cookies...
    assert calls[0]["cookies"] == [cookie]
    # ...but the retry dropped them, since cookie-less requests have been
    # observed to succeed on exactly the videos a cookied one failed on.
    assert calls[1]["cookies"] == []


@pytest.mark.asyncio
async def test_download_subtitles_retries_on_exception(monkeypatch) -> None:  # noqa: ANN001
    """A parse failure (e.g. a JSONDecodeError from a corrupt/unexpected
    download) must be retried too, not treated as an immediate concession
    to Whisper."""
    calls = {"n": 0}

    def _fake_sync(*, url, cookies, dir, lang_preferences, output_language=None):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            raise json.JSONDecodeError("boom", "doc", 0)
        return [{"start": 0.0, "duration": 1.0, "text": "recovered"}]

    monkeypatch.setattr(youtube, "_download_subtitles_sync", _fake_sync)

    out = await youtube.download_subtitles(
        url="https://www.youtube.com/watch?v=video1",
        cookies=[],
        dir=Path("/unused"),
        lang_preferences=["en"],
        max_attempts=2,
        backoff_seconds=[0],
    )
    assert out == [{"start": 0.0, "duration": 1.0, "text": "recovered"}]
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_download_subtitles_gives_up_after_max_attempts(monkeypatch) -> None:  # noqa: ANN001
    calls = {"n": 0}

    def _always_fails(*, url, cookies, dir, lang_preferences, output_language=None):  # noqa: ANN001
        calls["n"] += 1
        return None

    monkeypatch.setattr(youtube, "_download_subtitles_sync", _always_fails)

    out = await youtube.download_subtitles(
        url="https://www.youtube.com/watch?v=video1",
        cookies=[],
        dir=Path("/unused"),
        lang_preferences=["en"],
        max_attempts=3,
        backoff_seconds=[0],
    )
    assert out is None
    assert calls["n"] == 3


# ---------------------------------------------------------------------------
# Phase 1 — retry_on_no_track=False: the generic media path's retry policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_subtitles_no_track_not_retried_when_disabled(monkeypatch) -> None:  # noqa: ANN001
    """Generic media path (pipeline._run_media passes retry_on_no_track=False):
    a CLEAN "no usable captions" result must be accepted after the first
    attempt, not retried — unlike YouTube, "this site has no subtitle track"
    reproduces identically on every attempt, so retrying just burns extra
    probes + backoff sleeps on the common captionless case."""
    calls = {"n": 0}

    def _fake_sync(*, url, cookies, dir, lang_preferences, output_language=None):  # noqa: ANN001
        calls["n"] += 1
        return None  # clean "no track" outcome, every time

    monkeypatch.setattr(youtube, "_download_subtitles_sync", _fake_sync)

    out = await youtube.download_subtitles(
        url="https://example.invalid/media",
        cookies=[],
        dir=Path("/unused"),
        lang_preferences=["en"],
        max_attempts=3,
        backoff_seconds=[0, 0],
        retry_on_no_track=False,
    )
    assert out is None
    assert calls["n"] == 1  # no retry burned on a clean "no track" result


@pytest.mark.asyncio
async def test_download_subtitles_still_retries_exceptions_when_no_track_disabled(  # noqa: ANN001
    monkeypatch,
) -> None:
    """retry_on_no_track=False only short-circuits a CLEAN no-track result.
    A genuine transient failure (here: a parse exception on the first
    attempt) must still be retried — that's the "genuine transient failures
    ... still are [retried]" half of the policy."""
    calls = {"n": 0}

    def _fake_sync(*, url, cookies, dir, lang_preferences, output_language=None):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            raise json.JSONDecodeError("boom", "doc", 0)
        return [{"start": 0.0, "duration": 1.0, "text": "recovered"}]

    monkeypatch.setattr(youtube, "_download_subtitles_sync", _fake_sync)

    out = await youtube.download_subtitles(
        url="https://example.invalid/media",
        cookies=[],
        dir=Path("/unused"),
        lang_preferences=["en"],
        max_attempts=3,
        backoff_seconds=[0],
        retry_on_no_track=False,
    )
    assert out == [{"start": 0.0, "duration": 1.0, "text": "recovered"}]
    assert calls["n"] == 2
