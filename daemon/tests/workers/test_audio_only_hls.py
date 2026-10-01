"""Audio-only download for muxed HLS streams (no audio-only format).

Unit tests for the decision, plus an integration test against a synthetic,
AES-128-encrypted, Referer-protected HLS stream served from localhost.
"""

from __future__ import annotations

import http.server
import json
import shutil
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.workers import youtube
from src.workers.ffmpeg import resolve_ffmpeg_dir

# ---------------------------------------------------------------------------
# Decision unit tests
# ---------------------------------------------------------------------------

_MUXED = {"format_id": "0", "vcodec": "avc1.64001f", "acodec": "mp4a.40.2", "protocol": "m3u8_native"}


def _info(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {**_MUXED, "formats": [dict(_MUXED)]}
    base.update(over)
    return base


def test_muxed_hls_without_audio_only_format_needs_branch() -> None:
    assert youtube._needs_audio_only_download(_info())


def test_unknown_vcodec_counts_as_maybe_video() -> None:
    fmt = {"format_id": "0", "protocol": "m3u8_native"}
    assert youtube._needs_audio_only_download({**fmt, "formats": [fmt]})


def test_audio_only_format_available_skips_branch() -> None:
    audio = {"format_id": "a", "vcodec": "none", "acodec": "mp4a.40.2", "protocol": "m3u8_native"}
    assert not youtube._needs_audio_only_download(_info(formats=[dict(_MUXED), audio]))


def test_selected_audio_only_format_skips_branch() -> None:
    audio = {"format_id": "a", "vcodec": "none", "acodec": "opus", "protocol": "https"}
    assert not youtube._needs_audio_only_download({**audio, "formats": [audio]})


def test_non_hls_protocol_skips_branch() -> None:
    assert not youtube._needs_audio_only_download(_info(protocol="https"))


def test_video_only_stream_skips_branch() -> None:
    assert not youtube._needs_audio_only_download(_info(acodec="none"))


def test_merge_of_requested_formats_skips_branch() -> None:
    assert not youtube._needs_audio_only_download(_info(requested_formats=[{}, {}]))


# ---------------------------------------------------------------------------
# Integration: synthetic encrypted muxed HLS behind a Referer check
# ---------------------------------------------------------------------------

_REFERER = "https://player.example.test/embed/42"
_DURATION = 6.0


def _ffmpeg_bins() -> tuple[str, str] | None:
    d = resolve_ffmpeg_dir() if not shutil.which("ffmpeg") else None
    ff = shutil.which("ffmpeg") or (d and str(Path(d) / "ffmpeg"))
    fp = shutil.which("ffprobe") or (d and str(Path(d) / "ffprobe"))
    if not ff or not fp:
        return None
    return ff, fp


@pytest.fixture
def hls_server(tmp_path: Path) -> Iterator[tuple[str, Path, list[tuple[str, str | None]]]]:
    bins = _ffmpeg_bins()
    if bins is None:
        pytest.skip("ffmpeg/ffprobe not available")
    ffmpeg_bin, _ = bins
    root = tmp_path / "www"
    root.mkdir()
    (root / "enc.key").write_bytes(bytes(range(16)))
    keyinfo = tmp_path / "keyinfo"
    keyinfo.write_text(f"enc.key\n{root / 'enc.key'}\n")
    subprocess.run(
        [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=25:duration={_DURATION}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={_DURATION}",
            "-c:v", "libx264", "-preset", "ultrafast", "-b:v", "4M",
            "-c:a", "aac", "-b:a", "64k", "-shortest",
            "-hls_time", "2", "-hls_playlist_type", "vod",
            "-hls_key_info_file", str(keyinfo),
            "-hls_segment_filename", str(root / "seg%03d.ts"),
            str(root / "index.m3u8"),
        ],
        check=True,
    )
    log: list[tuple[str, str | None]] = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a: Any, **kw: Any) -> None:
            super().__init__(*a, directory=str(root), **kw)

        def do_GET(self) -> None:
            ref = self.headers.get("Referer")
            log.append((self.path.split("?")[0], ref))
            if ref != _REFERER:
                self.send_error(403)
                return
            super().do_GET()

        def guess_type(self, path: Any) -> str:
            if str(path).endswith(".m3u8"):
                return "application/vnd.apple.mpegurl"
            return super().guess_type(path)

        def log_message(self, *a: Any) -> None:
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/index.m3u8", root, log
    finally:
        srv.shutdown()
        srv.server_close()


def _probe(ffprobe: str, path: Path) -> dict[str, Any]:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    data: dict[str, Any] = json.loads(out)
    return data


def test_muxed_hls_downloads_audio_only(
    hls_server: tuple[str, Path, list[tuple[str, str | None]]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, root, log = hls_server
    bins = _ffmpeg_bins()
    assert bins is not None
    _, ffprobe_bin = bins
    monkeypatch.setattr(
        youtube, "get_config",
        lambda: SimpleNamespace(youtube=SimpleNamespace(
            ytdlp_sleep_interval=[0, 0], audio_format="opus", audio_bitrate_max=64,
        )),
    )

    # Capture the file yt-dlp downloaded (before FFmpegExtractAudio re-encodes)
    # so we can prove video never reached disk.
    import yt_dlp

    downloaded: list[tuple[Path, int, dict[str, Any]]] = []
    real_init = yt_dlp.YoutubeDL.__init__

    def init(self: Any, params: dict[str, Any] | None = None, *a: Any, **kw: Any) -> None:
        params = dict(params or {})

        def hook(d: dict[str, Any]) -> None:
            if d.get("status") == "started" and d.get("postprocessor") == "ExtractAudio":
                f = Path(d["info_dict"]["filepath"])
                downloaded.append((f, f.stat().st_size, _probe(ffprobe_bin, f)))

        params["postprocessor_hooks"] = [hook]
        real_init(self, params, *a, **kw)

    monkeypatch.setattr(yt_dlp.YoutubeDL, "__init__", init)

    out_dir = tmp_path / "out"
    path, duration = youtube._download_audio_sync(
        url=url, cookies=[], dir=out_dir, http_headers={"Referer": _REFERER},
    )

    # Intermediate (on-disk download) is audio-only and tiny.
    assert downloaded, "ExtractAudio postprocessor never ran"
    _, raw_size, raw = downloaded[0]
    kinds = {s["codec_type"] for s in raw["streams"]}
    assert kinds == {"audio"}
    muxed_bytes = sum(p.stat().st_size for p in root.glob("seg*.ts"))
    assert raw_size < muxed_bytes / 10

    # Final file: audio, no video, correct duration.
    final = _probe(ffprobe_bin, path)
    assert {s["codec_type"] for s in final["streams"]} == {"audio"}
    assert float(final["format"]["duration"]) == pytest.approx(_DURATION, abs=0.5)
    if duration is not None:
        assert duration == pytest.approx(_DURATION, abs=0.5)

    # Referer on playlist, every segment and the key; no 403s happened.
    paths = {p for p, _ in log}
    assert "/index.m3u8" in paths and "/enc.key" in paths
    assert {f"/{p.name}" for p in root.glob("seg*.ts")} <= paths
    assert all(ref == _REFERER for _, ref in log)
