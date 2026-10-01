"""``JobCreateRequest.media_headers`` allow-list."""

from __future__ import annotations

from src.api.schemas import MEDIA_HEADER_MAX_LEN, JobCreateRequest, sanitize_media_headers


def test_keeps_only_allowlisted_headers_with_canonical_casing() -> None:
    req = JobCreateRequest(
        url="https://example.com/movie",
        kind="media",
        media_url="https://cdn.example.net/master.m3u8",
        media_headers={
            "referer": "https://player.example.net/embed/1",
            "ORIGIN": "https://player.example.net",
            "User-Agent": "Mozilla/5.0",
            "Cookie": "session=secret",
            "Authorization": "Bearer x",
            "X-Forwarded-For": "1.2.3.4",
        },
    )
    assert req.media_headers == {
        "Referer": "https://player.example.net/embed/1",
        "Origin": "https://player.example.net",
        "User-Agent": "Mozilla/5.0",
    }


def test_drops_empty_and_crlf_values_and_caps_length() -> None:
    out = sanitize_media_headers({
        "Referer": "   ",
        "Origin": "https://a.example\r\nX-Evil: 1",
        "User-Agent": "u" * (MEDIA_HEADER_MAX_LEN + 50),
    })
    assert out == {"User-Agent": "u" * MEDIA_HEADER_MAX_LEN}


def test_nothing_left_becomes_none() -> None:
    assert sanitize_media_headers({"Cookie": "a=b"}) is None
    assert sanitize_media_headers(None) is None
    assert JobCreateRequest(url="https://x", media_headers={}).media_headers is None
