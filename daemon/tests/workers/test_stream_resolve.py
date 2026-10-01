"""workers/stream_resolve.py — playlist classification, selection, VTT concat."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.workers import pipeline, stream_resolve
from src.workers.stream_resolve import (
    concat_vtt_segments,
    parse_playlist,
    select,
    subtitles_cover,
)

MASTER_URL = "https://cdn.example/v/123/master.m3u8?token=abc"
MASTER = """#EXTM3U
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="aud",NAME="Rus. Prof",LANGUAGE="rus",DEFAULT=YES,URI="audio/ru/index.m3u8"
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="aud",NAME="German Original",DEFAULT=NO,URI="https://cdn.example/v/123/audio/de/index.m3u8"
#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",NAME="Eng. full",LANGUAGE="en",URI="subs/en/index.m3u8"
#EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=640x360,AUDIO="aud",SUBTITLES="subs"
video/360/index.m3u8
"""
AUDIO_RU = "https://cdn.example/v/123/audio/ru/index.m3u8"
AUDIO_DE = "https://cdn.example/v/123/audio/de/index.m3u8"
SUBS_EN = "https://cdn.example/v/123/subs/en/index.m3u8"
VIDEO_360 = "https://cdn.example/v/123/video/360/index.m3u8"


def media_pl(url: str, n: int, ext: str = "ts", dur: float = 10.0) -> str:
    body = "#EXTM3U\n#EXT-X-TARGETDURATION:10\n"
    for i in range(n):
        body += f"#EXTINF:{dur},\nseg{i}.{ext}\n"
    return body + "#EXT-X-ENDLIST\n"


@dataclass
class S:
    url: str
    kind: str
    first_seen: float
    last_seen: float


def parsed(**bodies: str) -> dict[str, Any]:
    out = {}
    for url, text in bodies.items():
        p = parse_playlist(url, text)
        assert p is not None
        out[url] = p
    return out


def test_parse_master_resolves_relative_and_absolute_uris() -> None:
    pl = parse_playlist(MASTER_URL, MASTER)
    assert pl is not None and pl.is_master
    audio = [r for r in pl.renditions if r.type == "AUDIO"]
    assert [r.uri for r in audio] == [AUDIO_RU, AUDIO_DE]
    assert audio[0].default and not audio[1].default
    assert pl.variant_uris == [VIDEO_360]
    assert parse_playlist("https://x/a", "<html>") is None


def test_parse_media_playlist_duration_and_subtitle_detection() -> None:
    pl = parse_playlist(SUBS_EN, media_pl(SUBS_EN, 3, "vtt"))
    assert pl is not None and not pl.is_master
    assert pl.duration == 30.0
    assert pl.is_subtitle_media
    assert pl.segment_uris[0] == "https://cdn.example/v/123/subs/en/seg0.vtt"


def test_duration_is_the_extinf_sum_of_a_finished_playlist() -> None:
    url = "https://c/a/index.m3u8"
    body = "#EXTM3U\n#EXTINF:9.009,\ns0.ts\n#EXTINF:10.01,\ns1.ts\n#EXTINF:4.5,\ns2.ts\n"
    live = parse_playlist(url, body)
    assert live is not None and live.duration is None  # no ENDLIST: live window, no guess
    vod = parse_playlist(url, body + "#EXT-X-ENDLIST\n")
    assert vod is not None and vod.duration == pytest.approx(23.519)
    # The chosen audio rendition's sum is the resolution's duration…
    streams = [S(MASTER_URL, "hls", 1, 1), S(AUDIO_DE, "hls", 2, 2)]
    pls = parsed(**{MASTER_URL: MASTER, AUDIO_DE: media_pl(AUDIO_DE, 5, dur=6.0)})
    assert select(streams, pls, default_url=MASTER_URL).duration == 30.0
    # …and with no audio rendition, a fetched video variant's.
    plain = "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\nlow.m3u8\n"
    m, low = "https://c/m.m3u8", "https://c/low.m3u8"
    res = select([S(m, "hls", 1, 1)], parsed(**{m: plain, low: media_pl(low, 7)}), default_url=m)
    assert res.download_url == m and res.duration == 70.0


def test_audio_rendition_fetched_last_wins() -> None:
    streams = [
        S(MASTER_URL, "hls", 1, 1),
        S(AUDIO_RU, "hls", 2, 2),
        S(AUDIO_DE, "hls", 5, 6),   # user switched to German
        S(VIDEO_360, "hls", 2, 7),
    ]
    pls = parsed(**{
        MASTER_URL: MASTER, AUDIO_RU: media_pl(AUDIO_RU, 300),
        AUDIO_DE: media_pl(AUDIO_DE, 300), VIDEO_360: media_pl(VIDEO_360, 300),
    })
    res = select(streams, pls, default_url=VIDEO_360)
    assert res.master_url == MASTER_URL
    assert res.audio is not None and res.audio.name == "German Original"
    assert res.download_url == AUDIO_DE
    assert res.duration == 3000.0
    assert res.subtitle is None  # subtitle rendition never fetched = not enabled
    sel = res.selection(subtitles_used=False)
    assert sel["audio_name"] == "German Original"


def test_player_query_tokens_still_match_renditions() -> None:
    # The player appends its own token to the URIs the master lists; the
    # selection must still see the fetch, and download the tokenised URL.
    de_tok = AUDIO_DE + "?t=xyz"
    subs_tok = SUBS_EN + "?t=xyz"
    streams = [
        S(MASTER_URL, "hls", 1, 1),
        S(AUDIO_RU + "?t=xyz", "hls", 2, 2),
        S(de_tok, "hls", 5, 6),
        S(subs_tok, "hls", 3, 3),
    ]
    pls = parsed(**{
        MASTER_URL: MASTER,
        de_tok: media_pl(de_tok, 300),
        subs_tok: media_pl(subs_tok, 300, ext="vtt"),
    })
    res = select(streams, pls, default_url=MASTER_URL)
    assert res.audio is not None and res.audio.name == "German Original"
    assert res.download_url == de_tok
    assert res.duration == 3000.0
    assert res.subtitle is not None and res.subtitle.url == subs_tok
    assert res.subtitle.name == "Eng. full"


def test_audio_falls_back_to_default_then_master() -> None:
    res = select([S(MASTER_URL, "hls", 1, 1)], parsed(**{MASTER_URL: MASTER}),
                 default_url="https://other/x.m3u8")
    assert res.download_url == AUDIO_RU  # DEFAULT=YES
    plain = "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\nlow.m3u8\n"
    res = select([S("https://c/m.m3u8", "hls", 1, 1)], parsed(**{"https://c/m.m3u8": plain}),
                 default_url="https://c/m.m3u8")
    assert res.audio is None and res.download_url == "https://c/m.m3u8"


def test_no_master_keeps_default_url() -> None:
    url = "https://c/only/index.m3u8"
    res = select([S(url, "hls", 1, 1)], parsed(**{url: media_pl(url, 3)}), default_url=url)
    assert res.master_url is None and res.download_url == url and res.duration == 30.0


def test_dub_switch_with_separate_masters_picks_latest_master() -> None:
    a = "https://p.example/hls/ru/master.m3u8"
    b = "https://p.example/hls/de/master.m3u8"
    body = "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\n360.m3u8\n"
    streams = [S(a, "hls", 1, 2), S(a.replace("master", "360"), "hls", 2, 3),
               S(b, "hls", 10, 10), S(b.replace("master", "360"), "hls", 11, 12)]
    res = select(streams, parsed(**{a: body, b: body}), default_url=a)
    assert res.master_url == b and res.download_url == b


def test_subtitle_rendition_selected_and_segments_not_double_counted() -> None:
    seg0 = "https://cdn.example/v/123/subs/en/seg0.vtt"
    streams = [
        S(MASTER_URL, "hls", 1, 1),
        S(SUBS_EN, "hls", 3, 3),
        S(seg0, "subtitle", 4, 4),  # a segment of the playlist, not its own track
    ]
    pls = parsed(**{MASTER_URL: MASTER, SUBS_EN: media_pl(SUBS_EN, 2, "vtt")})
    res = select(streams, pls, default_url=MASTER_URL)
    assert res.subtitle is not None
    assert res.subtitle.url == SUBS_EN and res.subtitle.via == "hls"
    assert res.subtitle.language == "en" and res.subtitle.name == "Eng. full"


def test_most_recent_subtitle_wins_direct_file() -> None:
    vtt = "https://subs.example/eng_full.vtt"
    streams = [S(MASTER_URL, "hls", 1, 1), S(SUBS_EN, "hls", 3, 3), S(vtt, "subtitle", 8, 9)]
    pls = parsed(**{MASTER_URL: MASTER, SUBS_EN: media_pl(SUBS_EN, 2, "vtt")})
    res = select(streams, pls, default_url=MASTER_URL)
    assert res.subtitle is not None and res.subtitle.via == "file" and res.subtitle.url == vtt


def test_language_from_rendition_name_when_no_language_attr() -> None:
    body = ('#EXTM3U\n#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="s",NAME="German Original",'
            'URI="de.m3u8"\n#EXT-X-STREAM-INF:BANDWIDTH=1\nv.m3u8\n')
    m, sub = "https://c/master.m3u8", "https://c/de.m3u8"
    res = select([S(m, "hls", 1, 1), S(sub, "hls", 2, 2)], parsed(**{m: body}), default_url=m)
    assert res.subtitle is not None and res.subtitle.language == "de"


def _cues(n: int, step: float = 10.0) -> list[dict[str, Any]]:
    return [{"start": i * step, "duration": 2.0, "text": f"line {i}"} for i in range(n)]


def test_subtitles_cover_requires_cues_and_coverage() -> None:
    assert not subtitles_cover(None, 100)
    assert not subtitles_cover(_cues(3), None)
    assert subtitles_cover(_cues(10), None)
    assert subtitles_cover(_cues(100), 1000.0)          # last cue ends at 992 s
    assert not subtitles_cover(_cues(10), 1000.0)       # stops at ~92 s of 1000


VTT_SEG = """WEBVTT
X-TIMESTAMP-MAP=MPEGTS:{mpegts},LOCAL:00:00:00.000

{a} --> {b}
<i>Hello</i> &amp; {{\\an8}}welcome

"""


def test_concat_same_map_keeps_absolute_times_and_dedupes() -> None:
    s1 = VTT_SEG.format(mpegts=900000, a="00:00:01.000", b="00:00:03.000")
    s2 = VTT_SEG.format(mpegts=900000, a="00:00:01.000", b="00:00:03.000")  # repeated cue
    s3 = VTT_SEG.format(mpegts=900000, a="00:00:12.000", b="00:00:14.000")
    out = concat_vtt_segments([s1, s2, s3], [0.0, 10.0, 20.0])
    assert [c["start"] for c in out] == [1.0, 12.0]
    assert out[0]["text"] == "Hello & welcome"


def test_concat_applies_increasing_timestamp_maps() -> None:
    s1 = VTT_SEG.format(mpegts=900000, a="00:00:01.000", b="00:00:02.000")
    s2 = VTT_SEG.format(mpegts=900000 + 90000 * 10, a="00:00:01.000", b="00:00:02.000")
    out = concat_vtt_segments([s1, s2], [0.0, 10.0])
    assert [c["start"] for c in out] == [1.0, 11.0]


def test_concat_segment_local_times_without_map_are_shifted() -> None:
    body = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhi\n"
    body2 = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nthere\n"
    out = concat_vtt_segments([body, body2], [0.0, 10.0])
    assert [c["start"] for c in out] == [1.0, 11.0]


@pytest.mark.asyncio
async def test_pipeline_short_subtitles_fall_back_to_asr(monkeypatch) -> None:  # noqa: ANN001
    res = stream_resolve.Resolution(
        download_url=AUDIO_DE, master_url=MASTER_URL, audio=None,
        subtitle=stream_resolve.SubtitleChoice(SUBS_EN, "hls", "Eng", "en", 3),
        duration=3000.0,
    )

    async def fake_resolve(*_a: Any, **_k: Any) -> Any:
        return res

    async def fake_subs(*_a: Any, **_k: Any) -> Any:
        return _cues(10)  # covers ~92 s of 3000 s

    finished: list[Any] = []

    async def fake_finish(*_a: Any, **k: Any) -> None:
        finished.append(k)

    persisted: list[Any] = []
    monkeypatch.setattr(stream_resolve, "resolve", fake_resolve)
    monkeypatch.setattr(stream_resolve, "fetch_subtitle_segments", fake_subs)
    monkeypatch.setattr(pipeline, "_finish_caption_fast_path", fake_finish)
    monkeypatch.setattr(pipeline.repo, "set_media_resolution",
                        lambda job_id, **k: persisted.append(k))

    out = await pipeline._resolve_sniffed(
        "j1", sniffed_streams=[S(MASTER_URL, "hls", 1, 1)], media_url=MASTER_URL,
        media_headers=None, cookies=[], page_title=None, cfg=None,
    )
    assert out == AUDIO_DE and not finished
    assert persisted[0]["media_frame_url"] == MASTER_URL
    assert persisted[0]["duration_seconds"] == 3000
    assert '"subtitles_used":false' in persisted[0]["media_selection_json"]

    async def full_subs(*_a: Any, **_k: Any) -> Any:
        return _cues(300)

    monkeypatch.setattr(stream_resolve, "fetch_subtitle_segments", full_subs)
    out = await pipeline._resolve_sniffed(
        "j1", sniffed_streams=[S(MASTER_URL, "hls", 1, 1)], media_url=MASTER_URL,
        media_headers={"Referer": "https://p/"}, cookies=[], page_title=None, cfg=None,
    )
    assert out is None
    assert finished[0]["transcript_language"] == "en"
    assert finished[0]["media_duration"] == 3000.0
    assert finished[0]["http_headers"] == {"Referer": "https://p/"}
