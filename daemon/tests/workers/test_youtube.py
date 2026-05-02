"""Tests for workers.youtube.

Covers extract_video_id URL parsing and exception classification through
fetch_transcript_with_retry with a monkeypatched YouTubeTranscriptApi.
"""

from __future__ import annotations

import pytest
from youtube_transcript_api._errors import (
    AgeRestricted,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)

from src.workers import youtube
from src.workers.errors import (
    ExhaustedRetriesError,
    PermanentTranscriptError,
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
    "url",
    [
        "",
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/",
        "https://www.youtube.com/watch",
        "https://www.youtube.com/watch?x=foo",
        "https://www.youtube.com/playlist?list=PLxxxxx",
        "not a url",
    ],
)
def test_extract_video_id_rejects_invalid(url: str) -> None:
    with pytest.raises(ValueError):
        youtube.extract_video_id(url)


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
