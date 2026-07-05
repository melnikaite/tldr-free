"""Tests for workers.youtube.

Covers extract_video_id URL parsing and exception classification through
fetch_transcript_with_retry with a monkeypatched YouTubeTranscriptApi.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests
from youtube_transcript_api._errors import (
    AgeRestricted,
    CouldNotRetrieveTranscript,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)

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
    assert youtube._pick_subtitle_lang({}, "en", ["en", "fr"]) is None


def test_pick_subtitle_lang_prefers_original() -> None:
    available = {"en": [], "fr": [], "de": []}
    # Original language wins even when it sits later in preferences.
    assert youtube._pick_subtitle_lang(available, "de", ["en", "fr"]) == "de"


def test_pick_subtitle_lang_falls_back_to_preferences_in_order() -> None:
    available = {"fr": [], "de": []}
    # Original not available → first matching preference wins.
    assert youtube._pick_subtitle_lang(available, "en", ["es", "fr", "de"]) == "fr"


def test_pick_subtitle_lang_skips_none_original() -> None:
    available = {"fr": [], "de": []}
    assert youtube._pick_subtitle_lang(available, None, ["de"]) == "de"


def test_pick_subtitle_lang_alphabetical_last_resort() -> None:
    available = {"zh": [], "ar": [], "de": []}
    # Neither original nor any preference matches → alphabetically first key.
    assert youtube._pick_subtitle_lang(available, "en", ["es", "fr"]) == "ar"


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


def _make_fake_api(side_effect):  # type: ignore[no-untyped-def]
    """Return a class that mimics YouTubeTranscriptApi but uses side_effect on fetch."""

    class FakeAPI:
        def __init__(self, *args, **kwargs):  # noqa: ANN001
            pass

        def fetch(self, video_id, languages=("en",), preserve_formatting=False):  # noqa: ANN001
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
        )


@pytest.mark.asyncio
async def test_successful_fetch_returns_segments(monkeypatch) -> None:  # noqa: ANN001
    snippets = [
        {"text": "hello", "start": 0.0, "duration": 5.0},
        {"text": "world", "start": 5.0, "duration": 5.0},
    ]

    def _ok(video_id: str):  # type: ignore[no-untyped-def]
        return _FakeFetched(snippets)

    fake_api = _make_fake_api(_ok)
    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", fake_api)

    out = await youtube.fetch_transcript_with_retry(
        video_id="abc",
        cookies=[],
        max_attempts=3,
        backoff_seconds=[0, 0, 0],
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
        return _FakeFetched([{"text": "ok", "start": 0.0, "duration": 1.0}])

    fake_api = _make_fake_api(_flaky)
    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", fake_api)

    out = await youtube.fetch_transcript_with_retry(
        video_id="abc",
        cookies=[],
        max_attempts=3,
        backoff_seconds=[0, 0, 0],
    )
    assert state["calls"] == 2
    assert out == [{"text": "ok", "start": 0.0, "duration": 1.0}]
