"""Tests for workers.frames — video-frame extraction near a timestamp.

No network access anywhere here: yt-dlp is driven through its Python
``YoutubeDL`` API (as every other yt-dlp caller in this daemon does), so the
seam we mock is ``frames._new_ytdl`` — the one-line constructor wrapper —
rather than subprocess. ffmpeg genuinely is a subprocess call, so
``subprocess.run`` is mocked the same way ``workers.transcribe`` mocks its
own ffmpeg calls. Assertions focus on the exact options/argv built: the
``*START-END`` section syntax, the resolved ffmpeg/deno paths actually being
spread into yt-dlp's opts, the output template, the window arithmetic
(including the near-zero clamp), the frame caps, and the non-zero-exit
failure paths for both yt-dlp and ffmpeg.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from yt_dlp.utils import download_range_func

from src.api.schemas import Cookie
from src.workers import frames
from src.workers.errors import FrameExtractionError

# ---------------------------------------------------------------------------
# Window arithmetic
# ---------------------------------------------------------------------------


def test_compute_window_typical_timestamp() -> None:
    start, end = frames._compute_window(100.0)
    assert start == 100.0 - frames.WINDOW_BEFORE_SECONDS
    assert end == 100.0 + frames.WINDOW_AFTER_SECONDS


def test_compute_window_near_zero_does_not_go_negative() -> None:
    # 1s minus the 2s "before" slack would be negative — must clamp to 0.
    start, end = frames._compute_window(1.0)
    assert start == 0.0
    assert end == 1.0 + frames.WINDOW_AFTER_SECONDS


def test_compute_window_at_exactly_zero() -> None:
    start, end = frames._compute_window(0.0)
    assert start == 0.0
    assert end == frames.WINDOW_AFTER_SECONDS
    assert end > start


def test_compute_window_negative_timestamp_treated_as_zero() -> None:
    start, end = frames._compute_window(-5.0)
    assert start == 0.0
    assert end == frames.WINDOW_AFTER_SECONDS


# ---------------------------------------------------------------------------
# Section-syntax formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "00:00"),
        (5, "00:05"),
        (65, "01:05"),
        (3661, "1:01:01"),
        (12.6, "00:13"),  # rounds to nearest second
    ],
)
def test_fmt_hms(seconds: float, expected: str) -> None:
    assert frames._fmt_hms(seconds) == expected


def test_format_section_arg_matches_star_range_syntax() -> None:
    assert frames._format_section_arg(5, 12) == "*00:05-00:12"


def test_format_section_arg_reflects_clamped_window_near_zero() -> None:
    start, end = frames._compute_window(1.0)
    assert frames._format_section_arg(start, end) == "*00:00-00:06"


# ---------------------------------------------------------------------------
# Frame-count clamping (per-call cap)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "requested, expected",
    [
        (1, 1),
        (frames.DEFAULT_NUM_FRAMES, frames.DEFAULT_NUM_FRAMES),
        (frames.MAX_FRAMES_PER_CALL, frames.MAX_FRAMES_PER_CALL),
        (frames.MAX_FRAMES_PER_CALL + 50, frames.MAX_FRAMES_PER_CALL),
        (0, 1),
        (-3, 1),
    ],
)
def test_effective_num_frames_is_clamped(requested: int, expected: int) -> None:
    assert frames._effective_num_frames(requested) == expected


# ---------------------------------------------------------------------------
# _ffmpeg_opt / _jsruntime_opt
# ---------------------------------------------------------------------------


def test_ffmpeg_opt_present_when_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(frames, "resolve_ffmpeg_dir", lambda: "/opt/homebrew/bin")
    assert frames._ffmpeg_opt() == {"ffmpeg_location": "/opt/homebrew/bin"}


def test_ffmpeg_opt_empty_when_unresolved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(frames, "resolve_ffmpeg_dir", lambda: None)
    assert frames._ffmpeg_opt() == {}


def test_jsruntime_opt_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        frames, "deno_runtime_opt", lambda: {"js_runtimes": {"deno": {"path": "/x/deno"}}}
    )
    assert frames._jsruntime_opt() == {"js_runtimes": {"deno": {"path": "/x/deno"}}}


# ---------------------------------------------------------------------------
# Fake YoutubeDL for exercising _download_section_sync / _download_full_sync
# ---------------------------------------------------------------------------


class _FakeYoutubeDL:
    """Captures the opts it was constructed with; extract_info is scripted
    per-test via ``info_or_exc`` and, on success, writes a stub file at the
    output template so path resolution has something real to find."""

    last_opts: dict[str, Any] | None = None

    def __init__(self, opts: dict[str, Any]) -> None:
        self.opts = opts
        _FakeYoutubeDL.last_opts = opts

    def __enter__(self) -> _FakeYoutubeDL:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def extract_info(self, url: str, download: bool = True) -> dict[str, Any]:
        raise NotImplementedError

    def prepare_filename(self, info: dict[str, Any]) -> str:
        return info["_test_path"]


def _install_fake_ytdl(
    monkeypatch: pytest.MonkeyPatch,
    *,
    write_path: Path | None = None,
    raise_exc: Exception | None = None,
    via_requested_downloads: bool = True,
) -> None:
    def factory(opts: dict[str, Any]) -> _FakeYoutubeDL:
        ydl = _FakeYoutubeDL(opts)

        def extract_info(url: str, download: bool = True) -> dict[str, Any]:
            if raise_exc is not None:
                raise raise_exc
            assert write_path is not None
            write_path.parent.mkdir(parents=True, exist_ok=True)
            write_path.write_bytes(b"fake video bytes")
            if via_requested_downloads:
                return {
                    "requested_downloads": [{"filepath": str(write_path)}],
                    "id": "abc12345678",
                }
            return {"_test_path": str(write_path), "id": "abc12345678"}

        ydl.extract_info = extract_info  # type: ignore[method-assign]
        return ydl

    monkeypatch.setattr(frames, "_new_ytdl", factory)


# ---------------------------------------------------------------------------
# _download_section_sync — opts built, path resolution, non-zero-exit path
# ---------------------------------------------------------------------------


def test_download_section_sync_builds_expected_opts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(frames, "resolve_ffmpeg_dir", lambda: "/opt/homebrew/bin")
    monkeypatch.setattr(
        frames, "deno_runtime_opt", lambda: {"js_runtimes": {"deno": {"path": "/data/deno/deno"}}}
    )
    out_file = tmp_path / "abc12345678.section.mp4"
    _install_fake_ytdl(monkeypatch, write_path=out_file)

    result = frames._download_section_sync(
        url="https://www.youtube.com/watch?v=abc12345678",
        start=5.0,
        end=12.0,
        cookies=[],
        dir=tmp_path,
        max_height=480,
    )

    assert result == out_file
    opts = _FakeYoutubeDL.last_opts
    assert opts is not None
    assert opts["format"] == "bestvideo[height<=480]/worst[height<=480]/worst"
    assert opts["download_ranges"] == download_range_func([], [(5.0, 12.0)])
    assert opts["force_keyframes_at_cuts"] is False
    assert opts["ffmpeg_location"] == "/opt/homebrew/bin"
    assert opts["js_runtimes"] == {"deno": {"path": "/data/deno/deno"}}
    assert opts["outtmpl"] == str(tmp_path / "%(id)s.section.%(ext)s")
    assert "cookiefile" not in opts


def test_download_section_sync_omits_ffmpeg_and_deno_when_unresolved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(frames, "resolve_ffmpeg_dir", lambda: None)
    monkeypatch.setattr(frames, "deno_runtime_opt", lambda: {})
    out_file = tmp_path / "vid.section.mp4"
    _install_fake_ytdl(monkeypatch, write_path=out_file)

    frames._download_section_sync(
        url="https://example.com/video", start=0.0, end=5.0,
        cookies=[], dir=tmp_path, max_height=480,
    )

    opts = _FakeYoutubeDL.last_opts
    assert opts is not None
    assert "ffmpeg_location" not in opts
    assert "js_runtimes" not in opts


def test_download_section_sync_writes_cookiefile_when_cookies_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(frames, "resolve_ffmpeg_dir", lambda: None)
    monkeypatch.setattr(frames, "deno_runtime_opt", lambda: {})
    out_file = tmp_path / "vid.section.mp4"
    _install_fake_ytdl(monkeypatch, write_path=out_file)

    cookie = Cookie(name="sid", value="abc", domain=".youtube.com")
    frames._download_section_sync(
        url="https://example.com/video", start=0.0, end=5.0,
        cookies=[cookie], dir=tmp_path, max_height=480,
    )

    opts = _FakeYoutubeDL.last_opts
    assert opts is not None
    assert "cookiefile" in opts
    # The cookie file is unlinked in the finally block — it must not linger.
    assert not Path(opts["cookiefile"]).exists()


def test_download_section_sync_falls_back_to_prepare_filename(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(frames, "resolve_ffmpeg_dir", lambda: None)
    monkeypatch.setattr(frames, "deno_runtime_opt", lambda: {})
    out_file = tmp_path / "vid.section.mp4"
    _install_fake_ytdl(monkeypatch, write_path=out_file, via_requested_downloads=False)

    result = frames._download_section_sync(
        url="https://example.com/video", start=0.0, end=5.0,
        cookies=[], dir=tmp_path, max_height=480,
    )
    assert result == out_file


def test_download_section_sync_propagates_on_ytdlp_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The path where yt-dlp exits non-zero — DownloadError (or any exception
    yt-dlp raises) must propagate untranslated; the async orchestrator
    (fetch_frames) is what decides whether to fall back or wrap it."""
    monkeypatch.setattr(frames, "resolve_ffmpeg_dir", lambda: None)
    monkeypatch.setattr(frames, "deno_runtime_opt", lambda: {})
    _install_fake_ytdl(monkeypatch, raise_exc=RuntimeError("yt-dlp exited with an error"))

    with pytest.raises(RuntimeError, match="yt-dlp exited with an error"):
        frames._download_section_sync(
            url="https://example.com/video", start=0.0, end=5.0,
            cookies=[], dir=tmp_path, max_height=480,
        )


# ---------------------------------------------------------------------------
# _download_full_sync — no download_ranges, max_filesize present
# ---------------------------------------------------------------------------


def test_download_full_sync_has_no_download_ranges_and_sets_max_filesize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(frames, "resolve_ffmpeg_dir", lambda: "/opt/homebrew/bin")
    monkeypatch.setattr(frames, "deno_runtime_opt", lambda: {})
    out_file = tmp_path / "vid.full.mp4"
    _install_fake_ytdl(monkeypatch, write_path=out_file)

    frames._download_full_sync(
        url="https://example.com/video", cookies=[], dir=tmp_path,
        max_height=480, max_filesize_bytes=123456,
    )

    opts = _FakeYoutubeDL.last_opts
    assert opts is not None
    assert "download_ranges" not in opts
    assert opts["max_filesize"] == 123456
    assert opts["format"] == "worst[height<=480]/worst"


# ---------------------------------------------------------------------------
# ffmpeg argv construction
# ---------------------------------------------------------------------------


def test_build_ffmpeg_frame_cmd_without_seek() -> None:
    cmd = frames._build_ffmpeg_frame_cmd(
        ffmpeg_bin="/opt/homebrew/bin/ffmpeg",
        video_path=Path("/tmp/section.mp4"),
        out_pattern=Path("/tmp/out/frame_%02d.jpg"),
        num_frames=5,
        seek_start=None,
        seek_duration=None,
    )
    assert cmd[0] == "/opt/homebrew/bin/ffmpeg"
    assert "-ss" not in cmd
    assert "-t" not in cmd
    assert cmd[cmd.index("-i") + 1] == "/tmp/section.mp4"
    assert cmd[cmd.index("-vf") + 1] == f"fps={frames.FRAME_FPS},scale={frames.FRAME_WIDTH_PX}:-2"
    assert cmd[cmd.index("-vframes") + 1] == "5"
    assert cmd[cmd.index("-q:v") + 1] == str(frames.FRAME_JPEG_QUALITY)
    assert cmd[-1] == "/tmp/out/frame_%02d.jpg"


def test_build_ffmpeg_frame_cmd_with_seek() -> None:
    cmd = frames._build_ffmpeg_frame_cmd(
        ffmpeg_bin="/opt/homebrew/bin/ffmpeg",
        video_path=Path("/tmp/full.mp4"),
        out_pattern=Path("/tmp/out/frame_%02d.jpg"),
        num_frames=3,
        seek_start=10.0,
        seek_duration=7.0,
    )
    assert cmd[cmd.index("-ss") + 1] == "10.000"
    # -ss must come before -i (fast demuxer-level seek).
    assert cmd.index("-ss") < cmd.index("-i")
    assert cmd[cmd.index("-t") + 1] == "7.000"
    # -t comes after -i (limits post-seek duration).
    assert cmd.index("-i") < cmd.index("-t")


# ---------------------------------------------------------------------------
# _extract_frames_sync — success, ffmpeg failure, no ffmpeg resolved
# ---------------------------------------------------------------------------


def test_extract_frames_sync_happy_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(frames, "_ffmpeg_bin", lambda: "/usr/bin/ffmpeg")

    def fake_run(cmd: list[str], **kwargs: object) -> None:
        out_dir = Path(cmd[-1]).parent
        for i in range(1, 4):
            (out_dir / f"frame_{i:02d}.jpg").write_bytes(b"jpeg")

    monkeypatch.setattr(frames.subprocess, "run", fake_run)

    out_dir = tmp_path / "out"
    result = frames._extract_frames_sync(
        video_path=tmp_path / "section.mp4", out_dir=out_dir,
        num_frames=3, seek_start=None, seek_duration=None,
    )
    assert len(result) == 3
    assert all(p.exists() for p in result)


def test_extract_frames_sync_raises_when_no_ffmpeg_resolved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(frames, "_ffmpeg_bin", lambda: None)
    with pytest.raises(FrameExtractionError, match="ffmpeg is not available"):
        frames._extract_frames_sync(
            video_path=tmp_path / "section.mp4", out_dir=tmp_path / "out",
            num_frames=3, seek_start=None, seek_duration=None,
        )


def test_extract_frames_sync_raises_on_ffmpeg_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(frames, "_ffmpeg_bin", lambda: "/usr/bin/ffmpeg")

    def fake_run(cmd: list[str], **kwargs: object) -> None:
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr=b"boom")

    monkeypatch.setattr(frames.subprocess, "run", fake_run)

    with pytest.raises(FrameExtractionError, match="ffmpeg frame extraction failed"):
        frames._extract_frames_sync(
            video_path=tmp_path / "section.mp4", out_dir=tmp_path / "out",
            num_frames=3, seek_start=None, seek_duration=None,
        )


def test_extract_frames_sync_raises_when_no_frames_produced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(frames, "_ffmpeg_bin", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(frames.subprocess, "run", lambda cmd, **k: None)

    with pytest.raises(FrameExtractionError, match="produced no frames"):
        frames._extract_frames_sync(
            video_path=tmp_path / "section.mp4", out_dir=tmp_path / "out",
            num_frames=3, seek_start=None, seek_duration=None,
        )


# ---------------------------------------------------------------------------
# Public API: fetch_frames — end to end (mocked), caps, fallback
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _config_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point storage.data_dir at a throwaway tmp_path for every test in this
    module, mirroring test_retention.py's pattern of patching the live
    config singleton rather than constructing a new one."""
    cfg = frames.get_config()
    monkeypatch.setattr(cfg.storage, "data_dir", str(tmp_path))
    return tmp_path


def _fake_ffmpeg_writes_frames(count: int):  # type: ignore[no-untyped-def]
    def fake_run(cmd: list[str], **kwargs: object) -> None:
        out_dir = Path(cmd[-1]).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        n = int(cmd[cmd.index("-vframes") + 1])
        for i in range(1, min(n, count) + 1):
            (out_dir / f"frame_{i:02d}.jpg").write_bytes(b"jpeg")

    return fake_run


async def test_fetch_frames_happy_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_file = tmp_path / "frames_scratch_out.mp4"

    def factory(opts: dict[str, Any]) -> _FakeYoutubeDL:
        ydl = _FakeYoutubeDL(opts)

        def extract_info(url: str, download: bool = True) -> dict[str, Any]:
            Path(opts["outtmpl"].replace("%(id)s", "vid").replace("%(ext)s", "mp4")).write_bytes(
                b"fake video"
            )
            path = opts["outtmpl"].replace("%(id)s", "vid").replace("%(ext)s", "mp4")
            return {"requested_downloads": [{"filepath": path}]}

        ydl.extract_info = extract_info  # type: ignore[method-assign]
        return ydl

    monkeypatch.setattr(frames, "_new_ytdl", factory)
    monkeypatch.setattr(frames, "resolve_ffmpeg_dir", lambda: str(tmp_path))
    monkeypatch.setattr(frames, "_ffmpeg_bin", lambda: str(tmp_path / "ffmpeg"))
    monkeypatch.setattr(frames.subprocess, "run", _fake_ffmpeg_writes_frames(5))

    result = await frames.fetch_frames(
        job_id="job123", url="https://www.youtube.com/watch?v=abc12345678",
        timestamp_seconds=30.0, num_frames=5,
    )

    assert len(result) == 5
    for p in result:
        assert p.exists()
        assert p.suffix == ".jpg"
    assert result[0].parent == tmp_path / "frames" / "job123" / "t30"
    # The scratch download is fully cleaned up (TemporaryDirectory removed).
    assert not (tmp_path / "frames_scratch").exists() or not any(
        (tmp_path / "frames_scratch").iterdir()
    )
    del out_file  # unused placeholder kept for clarity of intent above


async def test_fetch_frames_returns_empty_when_job_cap_already_spent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Pre-existing frames live under a DIFFERENT timestamp bucket (t999) than
    # the one this call targets (t10) — fetch_frames clears its own target
    # bucket up front (see the stale-directory comment in frames.py), so the
    # cap-filling frames must sit elsewhere to actually exercise the cap.
    job_dir = tmp_path / "frames" / "jobfull" / "t999"
    job_dir.mkdir(parents=True)
    for i in range(frames.MAX_FRAMES_PER_JOB):
        (job_dir / f"frame_{i:02d}.jpg").write_bytes(b"jpeg")

    def boom(*a: object, **k: object) -> None:
        pytest.fail("should not attempt a download once the job cap is spent")

    monkeypatch.setattr(frames, "_download_video_section", boom)

    result = await frames.fetch_frames(
        job_id="jobfull", url="https://example.com/video", timestamp_seconds=10.0,
    )
    assert result == []


async def test_fetch_frames_clamps_to_remaining_job_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Pre-populate MAX_FRAMES_PER_JOB - 2 frames under a DIFFERENT timestamp
    # than the one this call targets (t100 vs t0 below) — this call's own
    # output directory must stay untouched by the pre-existing-frames setup,
    # otherwise it's testing the same-timestamp overwrite behaviour instead
    # of cross-call cumulative budget clamping.
    existing_dir = tmp_path / "frames" / "jobpartial" / "t100"
    existing_dir.mkdir(parents=True)
    for i in range(frames.MAX_FRAMES_PER_JOB - 2):
        (existing_dir / f"frame_{i:02d}.jpg").write_bytes(b"jpeg")

    out_file = tmp_path / "vid.mp4"

    def factory(opts: dict[str, Any]) -> _FakeYoutubeDL:
        ydl = _FakeYoutubeDL(opts)

        def extract_info(url: str, download: bool = True) -> dict[str, Any]:
            out_file.write_bytes(b"fake video")
            return {"requested_downloads": [{"filepath": str(out_file)}]}

        ydl.extract_info = extract_info  # type: ignore[method-assign]
        return ydl

    monkeypatch.setattr(frames, "_new_ytdl", factory)
    monkeypatch.setattr(frames, "resolve_ffmpeg_dir", lambda: None)
    monkeypatch.setattr(frames, "_ffmpeg_bin", lambda: str(tmp_path / "ffmpeg"))

    captured: dict[str, int] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> None:
        n = int(cmd[cmd.index("-vframes") + 1])
        captured["n"] = n
        out_dir = Path(cmd[-1]).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (out_dir / f"frame_{i:02d}.jpg").write_bytes(b"jpeg")

    monkeypatch.setattr(frames.subprocess, "run", fake_run)

    result = await frames.fetch_frames(
        job_id="jobpartial", url="https://example.com/video",
        timestamp_seconds=0.0, num_frames=frames.DEFAULT_NUM_FRAMES,
    )
    assert captured["n"] == 2
    assert len(result) == 2


async def test_fetch_frames_raises_without_fallback_when_section_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def factory(opts: dict[str, Any]) -> _FakeYoutubeDL:
        ydl = _FakeYoutubeDL(opts)

        def extract_info(url: str, download: bool = True) -> dict[str, Any]:
            raise RuntimeError("section download refused")

        ydl.extract_info = extract_info  # type: ignore[method-assign]
        return ydl

    monkeypatch.setattr(frames, "_new_ytdl", factory)
    monkeypatch.setattr(frames, "resolve_ffmpeg_dir", lambda: None)

    with pytest.raises(FrameExtractionError, match="section download failed"):
        await frames.fetch_frames(
            job_id="jobX", url="https://example.com/video",
            timestamp_seconds=10.0, allow_full_download=False,
        )


async def test_fetch_frames_falls_back_to_full_download_when_opted_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    full_file = tmp_path / "full.mp4"
    calls = {"section": 0, "full": 0}

    def factory(opts: dict[str, Any]) -> _FakeYoutubeDL:
        ydl = _FakeYoutubeDL(opts)

        def extract_info(url: str, download: bool = True) -> dict[str, Any]:
            if "download_ranges" in opts:
                calls["section"] += 1
                raise RuntimeError("section download refused")
            calls["full"] += 1
            full_file.write_bytes(b"x" * 1000)
            return {"requested_downloads": [{"filepath": str(full_file)}]}

        ydl.extract_info = extract_info  # type: ignore[method-assign]
        return ydl

    monkeypatch.setattr(frames, "_new_ytdl", factory)
    monkeypatch.setattr(frames, "resolve_ffmpeg_dir", lambda: None)
    monkeypatch.setattr(frames, "_ffmpeg_bin", lambda: str(tmp_path / "ffmpeg"))
    monkeypatch.setattr(frames.subprocess, "run", _fake_ffmpeg_writes_frames(2))

    result = await frames.fetch_frames(
        job_id="jobFallback", url="https://example.com/video",
        timestamp_seconds=10.0, num_frames=2, allow_full_download=True,
    )

    assert calls == {"section": 1, "full": 1}
    assert len(result) == 2


async def test_fetch_frames_full_download_over_size_guard_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    full_file = tmp_path / "full.mp4"

    def factory(opts: dict[str, Any]) -> _FakeYoutubeDL:
        ydl = _FakeYoutubeDL(opts)

        def extract_info(url: str, download: bool = True) -> dict[str, Any]:
            if "download_ranges" in opts:
                raise RuntimeError("section refused")
            full_file.write_bytes(b"x" * 2048)
            return {"requested_downloads": [{"filepath": str(full_file)}]}

        ydl.extract_info = extract_info  # type: ignore[method-assign]
        return ydl

    monkeypatch.setattr(frames, "_new_ytdl", factory)
    monkeypatch.setattr(frames, "resolve_ffmpeg_dir", lambda: None)

    with pytest.raises(FrameExtractionError, match="over the .*-byte guard"):
        await frames.fetch_frames(
            job_id="jobBig", url="https://example.com/video",
            timestamp_seconds=10.0, allow_full_download=True,
            full_download_max_bytes=1024,
        )


# ---------------------------------------------------------------------------
# delete_job_frames — retention hook
# ---------------------------------------------------------------------------


def test_delete_job_frames_removes_directory(tmp_path: Path) -> None:
    job_dir = tmp_path / "frames" / "jobY" / "t5"
    job_dir.mkdir(parents=True)
    (job_dir / "frame_00.jpg").write_bytes(b"jpeg")

    assert frames.delete_job_frames("jobY") is True
    assert not (tmp_path / "frames" / "jobY").exists()


def test_delete_job_frames_returns_false_when_nothing_to_delete(tmp_path: Path) -> None:
    assert frames.delete_job_frames("no-such-job") is False
