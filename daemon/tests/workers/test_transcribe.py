"""Tests for workers.transcribe — single-shot vs chunked transcription."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from src.workers import transcribe


def _payload(segments: list[dict], language: str = "en") -> dict:
    return {"segments": segments, "language": language, "duration": 100.0}


@pytest.mark.asyncio
async def test_small_file_takes_single_post(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x" * 1024)  # 1 KB, well under the cap

    calls: list[Path] = []

    async def fake_post(path: Path, **_kwargs: object) -> dict:
        calls.append(path)
        return _payload([{"start": 1.0, "end": 2.0, "text": "hi"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)
    monkeypatch.setattr(
        transcribe, "_transcribe_chunked", lambda *a, **k: pytest.fail("should not chunk")
    )
    # total_duration (100s) is unrelated to the fake single-segment payload
    # (ends at 2s) — not testing coverage here, so make the coverage-retry
    # cut a guaranteed no-op rather than depend on ffmpeg failing on fake bytes.
    monkeypatch.setattr(transcribe, "_cut_audio_segment", lambda *a, **k: None)

    result = await transcribe.transcribe_audio(audio, total_duration=100.0)
    assert calls == [audio]
    assert result.segments == [{"start": 1.0, "end": 2.0, "text": "hi"}]
    assert result.language == "en"


@pytest.mark.asyncio
async def test_chunked_merges_with_offsets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = tmp_path / "big.opus"
    audio.write_bytes(b"x" * 2048)

    # Two chunks at offsets 0 and 600s.
    c0, c1 = tmp_path / "c0.opus", tmp_path / "c1.opus"
    c0.write_bytes(b"0")
    c1.write_bytes(b"1")
    monkeypatch.setattr(
        transcribe, "_split_audio", lambda *a, **k: [(c0, 0.0), (c1, 600.0)]
    )
    # Segment coverage here (5-10 s / 3-8 s local) is far short of the
    # nominal ~400s/chunk this setup implies — not testing coverage in this
    # test, so make the retry cut a guaranteed no-op.
    monkeypatch.setattr(transcribe, "_cut_audio_segment", lambda *a, **k: None)

    async def fake_post(path: Path, **_kwargs: object) -> dict:
        if path == c0:
            return _payload([{"start": 5.0, "end": 10.0, "text": "first"}], "ru")
        return _payload([{"start": 3.0, "end": 8.0, "text": "second"}], "en")

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result = await transcribe._transcribe_chunked(
        audio, total_duration=1200.0, max_bytes=1024, per_job_lock=asyncio.Semaphore(1)
    )
    # Second chunk's timestamps are shifted by its 600 s offset.
    assert result.segments == [
        {"start": 5.0, "end": 10.0, "text": "first"},
        {"start": 603.0, "end": 608.0, "text": "second"},
    ]
    # Language comes from the first chunk that reports one.
    assert result.language == "ru"
    assert result.duration_seconds == 1200.0


@pytest.mark.asyncio
async def test_chunked_falls_back_when_no_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = tmp_path / "big.opus"
    audio.write_bytes(b"x" * 2048)
    monkeypatch.setattr(transcribe, "_probe_duration", lambda _p: None)

    async def fake_post(path: Path, **_kwargs: object) -> dict:
        return _payload([{"start": 0.0, "end": 1.0, "text": "whole"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)
    monkeypatch.setattr(
        transcribe, "_split_audio", lambda *a, **k: pytest.fail("should not split")
    )

    result = await transcribe._transcribe_chunked(
        audio, total_duration=None, max_bytes=1024, per_job_lock=asyncio.Semaphore(1)
    )
    assert result.segments == [{"start": 0.0, "end": 1.0, "text": "whole"}]


@pytest.mark.asyncio
async def test_large_file_routes_to_chunked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = tmp_path / "big.opus"
    audio.write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MB

    # Force a tiny cap so the 2 MB file is "oversized".
    cfg = transcribe.get_config()
    monkeypatch.setattr(cfg.whisper, "max_upload_mb", 1)
    monkeypatch.setattr(transcribe, "get_config", lambda: cfg)

    async def fake_chunked(*_a: object, **_k: object) -> transcribe.TranscribeResult:
        return transcribe.TranscribeResult(segments=[], language=None, duration_seconds=1.0)

    monkeypatch.setattr(transcribe, "_transcribe_chunked", fake_chunked)
    monkeypatch.setattr(
        transcribe, "_post_audio", lambda *a, **k: pytest.fail("should chunk")
    )

    result = await transcribe.transcribe_audio(audio, total_duration=10.0)
    assert result.duration_seconds == 1.0


@pytest.mark.asyncio
async def test_transcribe_audio_collapses_repeated_segments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The single-request path collapses a Whisper repetition-loop run to
    ONE instance (Phase 6 didn't change that — see the ruling's second
    constraint). Real ffmpeg can't cut the fake audio bytes here, so the
    coverage recheck fails outright (recut_unavailable) and — per the
    Phase 6 owner ruling ("a wrong transcript beats a hole") — the kept
    instance is restored/extended across the WHOLE span and marked
    low_confidence instead of leaving 9s of silence after it."""
    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x" * 1024)  # under the cap → single-request path

    loop_text = "I'm not sure if I'm doing that right."
    segments = [
        {"start": float(i), "end": float(i + 1), "text": loop_text} for i in range(10)
    ]

    async def fake_post(_path: Path, **_kwargs: object) -> dict:
        return _payload(segments)

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result = await transcribe.transcribe_audio(audio, total_duration=10.0)
    # Still exactly ONE instance of the repeated text — never the full
    # 10-segment run — just stretched across the whole span rather than
    # left as a 1s island followed by a 9s hole.
    assert result.segments == [
        {"start": 0.0, "end": 10.0, "text": loop_text, "low_confidence": True}
    ]


@pytest.mark.asyncio
async def test_transcribe_audio_collapses_across_chunk_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The chunked path merges first, THEN collapses — a hallucination loop
    that straddles a chunk split (identical trailing/leading segments) is
    still caught because collapse runs on the merged, offset-shifted list."""
    audio = tmp_path / "big.opus"
    audio.write_bytes(b"x" * (2 * 1024 * 1024))

    cfg = transcribe.get_config()
    monkeypatch.setattr(cfg.whisper, "max_upload_mb", 1)
    monkeypatch.setattr(transcribe, "get_config", lambda: cfg)

    c0, c1 = tmp_path / "c0.opus", tmp_path / "c1.opus"
    c0.write_bytes(b"0")
    c1.write_bytes(b"1")
    monkeypatch.setattr(
        transcribe, "_split_audio", lambda *a, **k: [(c0, 0.0), (c1, 10.0)]
    )
    monkeypatch.setattr(transcribe, "_probe_duration", lambda _p: 20.0)

    loop_text = "Ja."

    async def fake_post(path: Path, **_kwargs: object) -> dict:
        if path == c0:
            # Chunk 0 ends with a loop tail.
            segs = [{"start": 5.0 + i, "end": 6.0 + i, "text": loop_text} for i in range(3)]
        else:
            # Chunk 1 starts with more of the same loop, offset by the chunk
            # boundary — after merge this becomes one long consecutive run.
            segs = [{"start": i * 1.0, "end": i * 1.0 + 1.0, "text": loop_text} for i in range(4)]
        return _payload(segs)

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result = await transcribe.transcribe_audio(audio, total_duration=None)
    assert len(result.segments) == 1
    assert result.segments[0]["text"] == loop_text


# ---------------------------------------------------------------------------
# Coverage check + bounded retry (job 3IXBfawKZrj7: Whisper decode-loops
# near the end of a chunk and never recovers — the collapse above correctly
# folds the repeat-run down to one segment, but that alone silently drops
# everything the run ate, which was real speech, not noise). See
# transcribe.py's module docstring / .claude/llm.md's "Transcript coverage"
# section for the full mechanism.
# ---------------------------------------------------------------------------

_LOOP_SENTENCE = "I'm not sure if I'm doing that right, he kept saying."


@pytest.mark.asyncio
async def test_legit_trailing_gap_within_threshold_is_not_flagged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A few seconds of trailing music/silence/credits — well under the
    threshold — must not trigger a retry or a missing_seconds flag."""
    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x" * 1024)

    async def fake_post(_path: Path, **_kwargs: object) -> dict:
        return _payload([{"start": 0.0, "end": 95.0, "text": "the whole talk"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)
    monkeypatch.setattr(
        transcribe, "_cut_audio_segment",
        lambda *a, **k: pytest.fail("must not attempt a retry cut"),
    )

    # 100s total, transcript covers to 95s -> 5s gap, under the 90s default.
    result = await transcribe.transcribe_audio(audio, total_duration=100.0)
    assert result.missing_seconds == 0.0
    assert result.segments == [{"start": 0.0, "end": 95.0, "text": "the whole talk"}]


@pytest.mark.asyncio
async def test_chunk_degenerate_retry_is_not_confirmed_silent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Regression guard for a real incident: chunk 2 gives normal speech,
    then decode-loops on one sentence to the chunk's own end. The recheck
    re-transcribes that span and gets the SAME kind of degenerate repeat
    back. An earlier (buggy) version of this code treated "the recheck
    also looped" as proof of non-speech and zeroed out missing_seconds —
    which is backwards: a loop means we still don't know what's there, not
    that we've confirmed nothing is. This must NOT happen: missing_seconds
    stays positive, and the outcome is never logged as "not lost content".
    Whatever collapse's own first-occurrence rule recognized before the
    loop still survives in the transcript (partial recovery), and the SAME
    window is still never re-asked twice within one call."""
    # Sized (with max_bytes=1024 below) so the internal chunk-count math
    # lands on exactly 2 chunks of 20s each (40s / 2) — matching the two
    # fixed offsets _split_audio is mocked to return.
    audio = tmp_path / "big.opus"
    audio.write_bytes(b"x" * 1300)

    # Not testing the leading-edge prefix-distrust window here — pin it to
    # 0 so the splice boundary stays exactly at gap_start, matching this
    # test's intent (see test_prefix_distrust_window_* for that feature).
    monkeypatch.setattr(transcribe, "_PREFIX_DISTRUST_SECONDS", 0.0)
    monkeypatch.setattr(transcribe, "_RETRY_BACKOFF_SECONDS", 1.0)

    c0, c1 = tmp_path / "c0.opus", tmp_path / "c1.opus"
    c0.write_bytes(b"0")
    c1.write_bytes(b"1")
    monkeypatch.setattr(
        transcribe, "_split_audio", lambda *a, **k: [(c0, 0.0), (c1, 20.0)]
    )

    retry_cut_calls: list[tuple[float, float]] = []

    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        retry_cut_calls.append((start, duration))
        return src_path.parent / f"retry{len(retry_cut_calls)}.opus"

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)

    def _loop_forever_from(local_start: float, local_end: float) -> list[dict]:
        segs = []
        t = local_start
        while t < local_end:
            segs.append({"start": t, "end": t + 1.0, "text": _LOOP_SENTENCE})
            t += 1.0
        return segs

    async def fake_post(path: Path, **_kwargs: object) -> dict:
        if path == c0:
            # Clean chunk, full coverage.
            return _payload([{"start": 0.0, "end": 20.0, "text": "clean chunk"}])
        if path == c1:
            # Real speech 0-8s, then decode-loops for the rest of the chunk.
            segs = [{"start": 0.0, "end": 8.0, "text": "real speech here"}]
            segs += _loop_forever_from(8.0, 20.0)
            return _payload(segs)
        # The recheck re-transcribes the same span and ALSO decode-loops —
        # reproducing the original failure, not confirming silence.
        return _payload(_loop_forever_from(0.0, 20.0 - retry_cut_calls[0][0]))

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    with caplog.at_level("INFO", logger="src.workers.transcribe"):
        result = await transcribe._transcribe_chunked(
            audio, total_duration=40.0, max_bytes=1024, per_job_lock=asyncio.Semaphore(1)
        )

    # One recheck settles this exact window — never asked twice within the
    # same call — but that's a dedup/efficiency property, not a verdict.
    assert len(retry_cut_calls) == 1
    # The real content from BOTH the clean chunk and chunk 1's pre-loop
    # speech survives — nothing before the loop is silently dropped.
    texts = [s["text"] for s in result.segments]
    assert "clean chunk" in texts
    assert "real speech here" in texts
    # The core regression check: a degenerate recheck must NEVER zero out
    # missing_seconds or be reported as clean content.
    assert result.missing_seconds is not None
    assert result.missing_seconds > 0
    assert not any("not lost content" in rec.message for rec in caplog.records)
    assert any("decode-loop again" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_chunk_retry_recovers_full_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When a retry's re-cut DOES get past the loop, the recovered tail is
    spliced onto the confirmed-good prefix and no shortfall is reported."""
    # Same 1300-byte / max_bytes=1024 sizing trick as the test above: forces
    # exactly 2 chunks internally, so with total_duration=40 each chunk's
    # own expected coverage is 20s — matching the single fixed chunk offset
    # (0.0) _split_audio is mocked to return here.
    audio = tmp_path / "big.opus"
    audio.write_bytes(b"x" * 1300)

    # Not testing the leading-edge prefix-distrust window here — pin it to
    # 0 so the splice boundary stays exactly at gap_start, matching this
    # test's intent (see test_prefix_distrust_window_* for that feature).
    monkeypatch.setattr(transcribe, "_PREFIX_DISTRUST_SECONDS", 0.0)
    monkeypatch.setattr(transcribe, "_RETRY_BACKOFF_SECONDS", 1.0)

    c0 = tmp_path / "c0.opus"
    c0.write_bytes(b"0")
    monkeypatch.setattr(transcribe, "_split_audio", lambda *a, **k: [(c0, 0.0)])

    retry_cut_calls: list[tuple[float, float]] = []

    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        retry_cut_calls.append((start, duration))
        return src_path.parent / "retry.opus"

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)

    async def fake_post(path: Path, **_kwargs: object) -> dict:
        if path == c0:
            segs = [{"start": 0.0, "end": 8.0, "text": "real speech here"}]
            t = 8.0
            while t < 20.0:
                segs.append({"start": t, "end": t + 1.0, "text": _LOOP_SENTENCE})
                t += 1.0
            return _payload(segs)
        # The retry cut recovers cleanly, covering its whole (local) window.
        start, duration = retry_cut_calls[-1]
        return _payload([{"start": 0.0, "end": duration, "text": "recovered tail"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result = await transcribe._transcribe_chunked(
        audio, total_duration=40.0, max_bytes=1024, per_job_lock=asyncio.Semaphore(1)
    )

    assert len(retry_cut_calls) == 1  # recovered on the first attempt
    assert result.missing_seconds == 0.0
    texts = [s["text"] for s in result.segments]
    assert "real speech here" in texts
    assert "recovered tail" in texts


@pytest.mark.asyncio
async def test_single_shot_hallucination_loop_to_end_flags_incomplete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same failure mode, no chunking involved — a small file that decode-
    loops to its own end must also be caught, not just the chunked path."""
    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x" * 1024)

    # Not testing the leading-edge prefix-distrust window here — pin it to
    # 0 so the splice boundary stays exactly at gap_start, matching this
    # test's intent (see test_prefix_distrust_window_* for that feature).
    monkeypatch.setattr(transcribe, "_PREFIX_DISTRUST_SECONDS", 0.0)
    monkeypatch.setattr(transcribe, "_RETRY_BACKOFF_SECONDS", 1.0)
    monkeypatch.setattr(transcribe, "_cut_audio_segment", lambda *a, **k: None)

    async def fake_post(_path: Path, **_kwargs: object) -> dict:
        segs = [{"start": 0.0, "end": 5.0, "text": "intro speech"}]
        t = 5.0
        while t < 30.0:
            segs.append({"start": t, "end": t + 1.0, "text": _LOOP_SENTENCE})
            t += 1.0
        return _payload(segs)

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result = await transcribe.transcribe_audio(audio, total_duration=30.0)

    assert result.missing_seconds > 0
    assert any(s["text"] == "intro speech" for s in result.segments)


# ---------------------------------------------------------------------------
# Real regression (job Y7odGFeN7agb, same video, re-run after the first
# version of this fix, single-request upload — no chunking involved at
# all): normal speech up to 728.9s, then NOTHING until one stray one-word
# segment ("2025.") at 1290.9s, 1.6s of a 1291.6s duration. A check that
# only compares the LAST segment's end against the known duration sees
# ~100% coverage and reports nothing wrong — the hole is in the middle, not
# the tail. These tests reproduce that exact shape on synthetic segments.
# ---------------------------------------------------------------------------


def _real_speech_then_gap_then_anchor(
    speech_end: float, window_duration: float
) -> list[dict]:
    """Normal speech [0, speech_end), a large gap, then ONE trailing
    segment landing right at the end of the window — the shape that
    fooled a tail-only coverage check."""
    return [
        {"start": 0.0, "end": speech_end, "text": "real speech here"},
        {"start": window_duration - 1.0, "end": window_duration, "text": "2025."},
    ]


@pytest.mark.asyncio
async def test_single_shot_internal_gap_with_trailing_anchor_flags_incomplete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A trailing one-word segment reaching the audio's real duration must
    NOT be read as "fully covered" when everything between it and the
    prior real speech is missing — the last segment's end alone is not
    proof of coverage."""
    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x" * 1024)

    # Not testing the leading-edge prefix-distrust window here — pin it to
    # 0 so the splice boundary stays exactly at gap_start, matching this
    # test's intent (see test_prefix_distrust_window_* for that feature).
    monkeypatch.setattr(transcribe, "_PREFIX_DISTRUST_SECONDS", 0.0)
    monkeypatch.setattr(transcribe, "_RETRY_BACKOFF_SECONDS", 1.0)
    monkeypatch.setattr(transcribe, "_cut_audio_segment", lambda *a, **k: None)

    window_duration = 30.0
    segs = _real_speech_then_gap_then_anchor(8.0, window_duration)

    async def fake_post(_path: Path, **_kwargs: object) -> dict:
        return _payload(segs)

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result = await transcribe.transcribe_audio(audio, total_duration=window_duration)

    # Sanity check: the last segment DOES reach the audio's real duration —
    # exactly the shape that fools an end-of-last-segment-only check.
    last_end = max(s["end"] for s in result.segments)
    assert window_duration - last_end < 2.0

    # The internal gap (8s to 29s, ~21s) is still reported as missing.
    assert result.missing_seconds > 0
    texts = [s["text"] for s in result.segments]
    assert "real speech here" in texts
    assert "2025." in texts


@pytest.mark.asyncio
async def test_single_shot_internal_gap_retry_fills_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the re-cut of the gap's own interval DOES recover content, it's
    spliced in between the confirmed speech and the trailing anchor
    segment — neither of which gets touched — and no shortfall remains."""
    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x" * 1024)

    # Not testing the leading-edge prefix-distrust window here — pin it to
    # 0 so the splice boundary stays exactly at gap_start, matching this
    # test's intent (see test_prefix_distrust_window_* for that feature).
    monkeypatch.setattr(transcribe, "_PREFIX_DISTRUST_SECONDS", 0.0)
    monkeypatch.setattr(transcribe, "_RETRY_BACKOFF_SECONDS", 1.0)

    window_duration = 30.0
    segs = _real_speech_then_gap_then_anchor(8.0, window_duration)

    cut_calls: list[tuple[float, float]] = []

    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        cut_calls.append((start, duration))
        return src_path.parent / "retry.opus"

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)

    async def fake_post(path: Path, **_kwargs: object) -> dict:
        if path == audio:
            return _payload(segs)
        # The retry cut recovers the whole gap cleanly.
        start, duration = cut_calls[-1]
        return _payload([{"start": 0.0, "end": duration, "text": "recovered middle"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result = await transcribe.transcribe_audio(audio, total_duration=window_duration)

    assert len(cut_calls) == 1  # recovered on the first attempt
    assert result.missing_seconds == 0.0
    texts = [s["text"] for s in result.segments]
    assert "real speech here" in texts
    assert "2025." in texts
    assert "recovered middle" in texts


# ---------------------------------------------------------------------------
# Splice seam: the retry's backoff context must not survive into the final
# list. Measured live on the same video, after the gap-fill above landed:
#   segments going backward in time: index 334 start 727.9 -> next start
#   723.9 (splice seam); index 536 start 1292.5 -> next start 1290.9 (tail
#   seam). Long lines appearing twice: 'Mann, das ist Luna Gröner.',
#   'Tochter unserer scheiß Vermieterin.' — the backoff context (re-cut a
#   few seconds past each edge of the gap for decode context) was being
#   spliced in unclipped, so that context got transcribed twice (once by
#   the confirmed neighbor, once by the retry) and, worse, could start
#   before an already-confirmed segment.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_coverage_clips_retry_output_no_duplicate_seam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A retry cut that reaches _RETRY_BACKOFF_SECONDS into confirmed audio
    on BOTH sides of the gap must not let that backoff margin's worth of
    content survive into the final list twice — the confirmed neighbors
    are the source of truth at the seam, not the retry."""
    # Not testing the leading-edge prefix-distrust window here — pin it to
    # 0 so the splice boundary stays exactly at gap_start, matching this
    # test's intent (see test_prefix_distrust_window_* for that feature).
    monkeypatch.setattr(transcribe, "_PREFIX_DISTRUST_SECONDS", 0.0)

    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x")

    window_duration = 25.0
    prefix_text = "Mann, das ist Luna Gröner."
    suffix_text = "Tochter unserer scheiß Vermieterin."
    segments = [
        {"start": 0.0, "end": 8.0, "text": prefix_text},
        {"start": 20.0, "end": 25.0, "text": suffix_text},
    ]

    cut_calls: list[tuple[float, float]] = []

    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        cut_calls.append((start, duration))
        return src_path.parent / "retry.opus"

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)

    async def fake_post(_path: Path, **_kwargs: object) -> dict:
        # Re-transcribes [3.0, 25.0) (backoff=5 both edges, clamped to the
        # window on the right) and — the bug shape being guarded against —
        # duplicates a few seconds of BOTH confirmed neighbors at the edges.
        return _payload(
            [
                {"start": 0.0, "end": 5.0, "text": prefix_text},        # -> abs 3-8, dup prefix
                {"start": 5.0, "end": 12.0, "text": "filled part one"},  # -> abs 8-15, real gap content
                {"start": 12.0, "end": 19.0, "text": "filled part two"}, # -> abs 15-22, straddles gap_end
                {"start": 19.0, "end": 22.0, "text": suffix_text},       # -> abs 22-25, dup suffix
            ]
        )

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result_segments, missing = await transcribe._ensure_coverage(
        segments, source_path=audio, window_duration=window_duration, per_job_lock=asyncio.Semaphore(1)
    )

    assert len(cut_calls) == 1
    assert missing == 0.0

    texts = [s["text"] for s in result_segments]
    assert texts.count(prefix_text) == 1
    assert texts.count(suffix_text) == 1
    assert "filled part one" in texts
    assert "filled part two" in texts

    starts = [s["start"] for s in result_segments]
    assert starts == sorted(starts)

    # The segment straddling the gap's right edge got clamped to the gap's
    # own end, not left reaching into the confirmed suffix's territory.
    straddler = next(s for s in result_segments if s["text"] == "filled part two")
    assert straddler["end"] == 20.0


@pytest.mark.asyncio
async def test_ensure_coverage_result_is_monotonic_by_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The segment list _ensure_coverage returns must never go backward in
    start time — every downstream consumer (build_marked_text's timecodes,
    the translator's forward-only marker alignment) assumes this. Checked
    here as its own assertion, not as an incidental side effect of another
    test's assertions."""
    # Not testing the leading-edge prefix-distrust window here — pin it to
    # 0 so the splice boundary stays exactly at gap_start, matching this
    # test's intent (see test_prefix_distrust_window_* for that feature).
    monkeypatch.setattr(transcribe, "_PREFIX_DISTRUST_SECONDS", 0.0)

    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x")

    window_duration = 20.0
    segments = [{"start": 0.0, "end": 8.0, "text": "real speech here"}]

    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        return src_path.parent / "retry.opus"

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)

    async def fake_post(_path: Path, **_kwargs: object) -> dict:
        # Deliberately out of ORDER (not just out of bounds) — the backend
        # is generally well-behaved but nothing guarantees it, and a naive
        # splice would carry that disorder straight into the final list.
        return _payload(
            [
                {"start": 12.2, "end": 14.0, "text": "second half"},
                {"start": 2.0, "end": 14.0, "text": "first half, drifted earlier"},
            ]
        )

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result_segments, _missing = await transcribe._ensure_coverage(
        segments, source_path=audio, window_duration=window_duration, per_job_lock=asyncio.Semaphore(1)
    )

    starts = [s["start"] for s in result_segments]
    assert starts == sorted(starts)


@pytest.mark.asyncio
async def test_ensure_coverage_tail_gap_clips_retry_overrun_past_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same defect at the LAST seam (the measured 'index 536' case): a
    retry filling a TAIL gap must not let a segment starting beyond the
    audio's own duration (a whisper timestamp overrun past what was fed to
    it) survive into the final list — that's exactly the shape that showed
    up as a segment starting later than the one meant to follow it."""
    # Not testing the leading-edge prefix-distrust window here — pin it to
    # 0 so the splice boundary stays exactly at gap_start, matching this
    # test's intent (see test_prefix_distrust_window_* for that feature).
    monkeypatch.setattr(transcribe, "_PREFIX_DISTRUST_SECONDS", 0.0)

    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x")

    window_duration = 20.0
    segments = [{"start": 0.0, "end": 8.0, "text": "real speech here"}]

    cut_calls: list[tuple[float, float]] = []

    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        cut_calls.append((start, duration))
        return src_path.parent / "retry.opus"

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)

    async def fake_post(_path: Path, **_kwargs: object) -> dict:
        # backoff=5 (default) -> cut window [3.0, 20.0), local 0-17.
        return _payload(
            [
                {"start": 0.0, "end": 5.0, "text": "dup prefix tail"},        # -> abs 3-8, dropped
                {"start": 5.0, "end": 17.0, "text": "recovered tail"},         # -> abs 8-20, kept
                {"start": 17.2, "end": 19.0, "text": "hallucinated overrun"},  # -> abs 20.2-22.0, dropped
            ]
        )

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result_segments, missing = await transcribe._ensure_coverage(
        segments, source_path=audio, window_duration=window_duration, per_job_lock=asyncio.Semaphore(1)
    )

    assert len(cut_calls) == 1
    assert missing == 0.0

    texts = [s["text"] for s in result_segments]
    assert "dup prefix tail" not in texts
    assert "hallucinated overrun" not in texts
    assert "recovered tail" in texts
    assert "real speech here" in texts

    # Nothing in the final list starts beyond the audio's own duration.
    assert all(s["start"] < window_duration for s in result_segments)
    starts = [s["start"] for s in result_segments]
    assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# Prefix distrust window: Whisper measurably DRIFTS before it actually falls
# into a hallucination loop, misattributing real speech to earlier, wrong
# timestamps. Measured live on the same real video: the segments immediately
# before a real gap were each marked EXACTLY 1.000s long (a shape real
# Whisper never produces) and carried the SAME dialogue the retry
# re-transcribed with normal, live-sounding timings. Clipping at the gap's
# own boundary alone can't catch this — the drifted prefix has no gap of
# its own to trigger on, so it survives as "formally legal" duplicate
# content right next to the retry's corrected version.
# ---------------------------------------------------------------------------


def _drifted_prefix_before_gap_scenario() -> tuple[list[dict], float]:
    """Segments shaped like the measured real defect: solid confirmed
    content, then a drifted stretch (suspiciously uniform 1.0s segments,
    including a short dialogue) immediately before a gap, then a confirmed
    suffix. Returns (segments, window_duration)."""
    segments = [
        {"start": 0.0, "end": 5.0, "text": "old confirmed content"},
        {"start": 5.0, "end": 6.0, "text": "Mann, das ist Luna Gröner."},
        {"start": 6.0, "end": 7.0, "text": "Tochter unserer scheiß Vermieterin."},
        {"start": 7.0, "end": 15.0, "text": "more drifted filler"},
        # gap: (15.0, 25.0)
        {"start": 25.0, "end": 35.0, "text": "confirmed after gap"},
    ]
    return segments, 35.0


@pytest.mark.asyncio
async def test_prefix_distrust_window_prevents_duplicate_dialogue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Real regression: a drifted prefix right before a gap carries the SAME
    dialogue the retry re-transcribes with corrected timing. Clipping at the
    gap's own boundary alone keeps both copies (the drifted one has no gap
    of its own); the wider prefix-distrust window must discard the drifted
    copy so the dialogue survives exactly once, with the retry's timing."""
    monkeypatch.setattr(transcribe, "_RETRY_BACKOFF_SECONDS", 2.0)
    monkeypatch.setattr(transcribe, "_PREFIX_DISTRUST_SECONDS", 10.0)

    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x")
    segments, window_duration = _drifted_prefix_before_gap_scenario()

    cut_calls: list[tuple[float, float]] = []

    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        cut_calls.append((start, duration))
        return src_path.parent / "retry.opus"

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)

    async def fake_post(_path: Path, **_kwargs: object) -> dict:
        # The retry's second pass gets normal, live-sounding durations
        # instead of the drifted original's suspicious 1.000s segments.
        return _payload(
            [
                {"start": 2.0, "end": 3.4, "text": "Mann, das ist Luna Gröner."},
                {"start": 3.5, "end": 5.8, "text": "Tochter unserer scheiß Vermieterin."},
                {"start": 5.9, "end": 22.0, "text": "filled gap content"},
            ]
        )

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result_segments, missing = await transcribe._ensure_coverage(
        segments, source_path=audio, window_duration=window_duration, per_job_lock=asyncio.Semaphore(1)
    )

    assert len(cut_calls) == 1
    assert missing == 0.0

    texts = [s["text"] for s in result_segments]
    assert texts.count("Mann, das ist Luna Gröner.") == 1
    assert texts.count("Tochter unserer scheiß Vermieterin.") == 1

    # The surviving copy carries the RETRY's timing, not the drifted
    # original's suspicious exactly-1.000s duration.
    dialogue = next(s for s in result_segments if s["text"] == "Mann, das ist Luna Gröner.")
    assert dialogue["end"] - dialogue["start"] != 1.0
    assert dialogue["end"] == 6.4  # retry's own timing (local 3.4 + cut_start 3.0)

    starts = [s["start"] for s in result_segments]
    assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# _is_confirmed_silence — the structural "is this CONFIRMED non-speech?"
# classifier. Whisper's non-speech wording isn't stable across models/
# backends/languages (measured: 172 consecutive "*Musik*" lines from one
# backend vs. a single "*Dramatic music*" from another for the same kind of
# audio), so this never matches specific phrases — only structure. A
# degenerate repeated run is deliberately OUTSIDE this function (it means
# "still unknown", not "confirmed nothing") — see transcribe.py's own
# docstring and the regression tests below for why that distinction is
# load-bearing, not cosmetic.
# ---------------------------------------------------------------------------


def test_is_confirmed_silence_empty_is_confirmed_silence() -> None:
    assert transcribe._is_confirmed_silence([]) is True
    assert (
        transcribe._is_confirmed_silence([{"start": 0.0, "end": 1.0, "text": "   "}]) is True
    )


def test_is_confirmed_silence_punctuation_only_is_confirmed_silence() -> None:
    assert transcribe._is_confirmed_silence([{"start": 0.0, "end": 1.0, "text": "-"}]) is True
    assert transcribe._is_confirmed_silence([{"start": 0.0, "end": 1.0, "text": "..."}]) is True


def test_is_confirmed_silence_bracket_annotation_is_confirmed_silence() -> None:
    for text in ("*Dramatic music*", "*door slams*", "[Musik]", "(laughs)"):
        assert (
            transcribe._is_confirmed_silence([{"start": 0.0, "end": 1.0, "text": text}]) is True
        )


def test_is_confirmed_silence_degenerate_run_is_not_confirmed_silence() -> None:
    # Real words in a hallucination-loop-shaped run: this is emphatically
    # NOT confirmed silence — a loop means we still don't know what's
    # there. Degenerate-run detection is a SEPARATE check the caller
    # (_ensure_coverage) makes directly via collapse_repeated_segments, not
    # folded into this function — see the regression tests below.
    segs = [
        {"start": float(i), "end": float(i + 1), "text": "I'm not sure about this."}
        for i in range(5)
    ]
    assert transcribe._is_confirmed_silence(segs) is False


def test_is_confirmed_silence_real_dialogue_is_not_confirmed_silence() -> None:
    assert transcribe._is_confirmed_silence(
        [{"start": 0.0, "end": 1.0, "text": "- Mmm, uh,"}]
    ) is False
    assert transcribe._is_confirmed_silence(
        [{"start": 0.0, "end": 2.0, "text": "Hello, this is recovered content."}]
    ) is False


def test_is_confirmed_silence_short_legitimate_repeat_is_not_confirmed_silence() -> None:
    # 5 consecutive "Ja." is real call-and-response dialogue, not silence.
    segs = [{"start": float(i), "end": float(i + 1), "text": "Ja."} for i in range(5)]
    assert transcribe._is_confirmed_silence(segs) is False


# ---------------------------------------------------------------------------
# transcript_is_unusable — the public "is there anything worth summarizing
# here?" gate used by runner.py's media page-text fallback. Reuses (does
# not reimplement) _is_confirmed_silence plus a degenerate-repeated-run
# check via timecodes.collapse_repeated_segments — see its own docstring.
# ---------------------------------------------------------------------------


def test_transcript_is_unusable_empty_segments() -> None:
    assert transcribe.transcript_is_unusable([]) is True
    assert (
        transcribe.transcript_is_unusable(
            [{"start": 0.0, "end": 1.0, "text": "   "}]
        )
        is True
    )


def test_transcript_is_unusable_annotation_only() -> None:
    assert (
        transcribe.transcript_is_unusable(
            [{"start": 0.0, "end": 3.0, "text": "[chime]"}]
        )
        is True
    )


def test_transcript_is_unusable_degenerate_repeated_run() -> None:
    # Not confirmed silence (_is_confirmed_silence says False — real words),
    # but a hallucination-loop-shaped repeat that collapse_repeated_segments
    # discards from: transcript_is_unusable must still call this unusable,
    # unlike the weaker "not raw_text.strip()" check it replaces in runner.py.
    segs = [
        {"start": float(i), "end": float(i + 1), "text": "I'm not sure about this."}
        for i in range(5)
    ]
    assert transcribe._is_confirmed_silence(segs) is False
    assert transcribe.transcript_is_unusable(segs) is True


def test_transcript_is_unusable_real_speech_is_usable() -> None:
    segs = [
        {"start": 0.0, "end": 2.0, "text": "Hello, this is a real transcript."},
        {"start": 2.0, "end": 4.5, "text": "It has more than one sentence."},
    ]
    assert transcribe.transcript_is_unusable(segs) is False


# ---------------------------------------------------------------------------
# _ensure_coverage integration: a recheck's verdict decides splice + missing.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_coverage_bracket_annotation_window_not_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A suspicious window that comes back as a bracketed sound annotation
    is not spliced in and does not count toward transcript_missing_seconds
    — it's confirmed music/noise, not lost speech."""
    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x")

    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        return src_path.parent / "retry.opus"

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)

    async def fake_post(_path: Path, **_kwargs: object) -> dict:
        return _payload([{"start": 0.0, "end": 30.0, "text": "*Dramatic music*"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    segments = [{"start": 0.0, "end": 10.0, "text": "intro speech"}]
    result_segments, missing = await transcribe._ensure_coverage(
        segments, source_path=audio, window_duration=40.0, per_job_lock=asyncio.Semaphore(1)
    )

    assert missing == 0.0
    texts = [s["text"] for s in result_segments]
    assert texts == ["intro speech"]
    assert "*Dramatic music*" not in texts


@pytest.mark.asyncio
async def test_ensure_coverage_dialogue_window_spliced_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A suspicious window that comes back as real dialogue IS spliced in,
    with monotonicity preserved and no duplicated content."""
    # Not testing the leading-edge prefix-distrust window here — pin it to
    # 0 so the splice boundary stays exactly at gap_start (otherwise the
    # default 30s distrust margin would reach back before the short intro
    # segment and drop it too; see test_prefix_distrust_window_* for that
    # feature in isolation).
    monkeypatch.setattr(transcribe, "_PREFIX_DISTRUST_SECONDS", 0.0)

    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x")

    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        return src_path.parent / "retry.opus"

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)

    async def fake_post(_path: Path, **_kwargs: object) -> dict:
        return _payload(
            [{"start": 0.0, "end": 30.0, "text": "Hello, this is recovered content."}]
        )

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    segments = [{"start": 0.0, "end": 10.0, "text": "intro speech"}]
    result_segments, missing = await transcribe._ensure_coverage(
        segments, source_path=audio, window_duration=40.0, per_job_lock=asyncio.Semaphore(1)
    )

    assert missing == 0.0
    texts = [s["text"] for s in result_segments]
    assert "intro speech" in texts
    assert "Hello, this is recovered content." in texts
    assert texts.count("Hello, this is recovered content.") == 1
    starts = [s["start"] for s in result_segments]
    assert starts == sorted(starts)


@pytest.mark.asyncio
async def test_ensure_coverage_degenerate_repeat_is_not_confirmed_silent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Regression: a suspicious window whose recheck comes back as a
    degenerate repeated run (real WORDS, but a hallucination loop) must
    NOT be treated as confirmed non-speech. It must not zero out
    ``missing``, and must not be logged as "not lost content" — a loop
    means we still don't know what's there, not that we've confirmed
    there's nothing."""
    monkeypatch.setattr(transcribe, "_PREFIX_DISTRUST_SECONDS", 0.0)

    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x")

    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        return src_path.parent / "retry.opus"

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)

    async def fake_post(_path: Path, **_kwargs: object) -> dict:
        loop_text = "I'm not sure if I'm doing that right."
        return _payload(
            [{"start": float(i), "end": float(i + 1), "text": loop_text} for i in range(6)]
        )

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    segments = [{"start": 0.0, "end": 10.0, "text": "intro speech"}]
    with caplog.at_level("INFO", logger="src.workers.transcribe"):
        result_segments, missing = await transcribe._ensure_coverage(
            segments, source_path=audio, window_duration=40.0, per_job_lock=asyncio.Semaphore(1)
        )

    # The core regression check.
    assert missing > 0
    assert not any("not lost content" in rec.message for rec in caplog.records)
    assert any("decode-loop again" in rec.message for rec in caplog.records)
    # The confirmed prefix survives untouched.
    texts = [s["text"] for s in result_segments]
    assert "intro speech" in texts


@pytest.mark.asyncio
async def test_ensure_coverage_degenerate_repeat_keeps_partial_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When a recheck's own output degenerates into a repeated run,
    whatever ``collapse_repeated_segments`` recognized BEFORE the loop
    (its first-occurrence survivor) is still spliced in — partial,
    recognized content beats discarding the whole slice — while the
    remainder that's still unresolved stays honestly counted as missing.

    Budget pinned to exactly 1 recheck: the fake backend below always
    answers with the same fixed (recovered-then-loop) shape regardless of
    where it's asked to cut, which realistically stands in for "this
    backend can't get past this specific loop" for ONE recheck — but
    would just re-trigger the same shape again indefinitely if the outer
    loop kept re-asking the shrinking remainder, which isn't what this
    test is about (that iteration behavior is covered by the budget/dedup
    tests instead)."""
    monkeypatch.setattr(transcribe, "_PREFIX_DISTRUST_SECONDS", 0.0)
    monkeypatch.setattr(transcribe, "_RETRY_BACKOFF_SECONDS", 0.0)
    cfg = transcribe.get_config()
    monkeypatch.setattr(cfg.whisper, "max_coverage_rechecks", 1)
    monkeypatch.setattr(transcribe, "get_config", lambda: cfg)

    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x")

    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        return src_path.parent / "retry.opus"

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)

    async def fake_post(_path: Path, **_kwargs: object) -> dict:
        # Real recognized dialogue for the first 3s, THEN a decode-loop for
        # the rest of the slice.
        segs = [{"start": 0.0, "end": 3.0, "text": "recovered before the loop"}]
        t = 3.0
        loop_text = "I'm not sure if I'm doing that right."
        while t < 30.0:
            segs.append({"start": t, "end": t + 1.0, "text": loop_text})
            t += 1.0
        return _payload(segs)

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    segments = [{"start": 0.0, "end": 10.0, "text": "intro speech"}]
    result_segments, missing = await transcribe._ensure_coverage(
        segments, source_path=audio, window_duration=40.0, per_job_lock=asyncio.Semaphore(1)
    )

    texts = [s["text"] for s in result_segments]
    assert "intro speech" in texts
    # The recognized portion before the loop survives...
    assert "recovered before the loop" in texts
    # ...and collapse's own first-occurrence rule keeps exactly ONE copy of
    # the loop sentence too (the same rule that already applies everywhere
    # else in this module) — never the full repeated run.
    assert texts.count("I'm not sure if I'm doing that right.") == 1
    # The remainder past the recognized portion is still honestly counted
    # as missing, not silently accepted.
    assert missing > 0
    starts = [s["start"] for s in result_segments]
    assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# Budget: rechecks are bounded and a window is never asked twice.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_coverage_recheck_budget_is_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Three separate suspicious windows, but a budget of 2 — exactly 2
    rechecks happen (not 3, not unbounded), and no window is ever cut
    twice."""
    cfg = transcribe.get_config()
    monkeypatch.setattr(cfg.whisper, "max_coverage_rechecks", 2)
    monkeypatch.setattr(transcribe, "get_config", lambda: cfg)

    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x")

    # Three widely-separated real gaps, each well above the cost cutoff, in
    # a window long enough that they don't interact via distrust/backoff
    # margins.
    segments = [
        {"start": 0.0, "end": 5.0, "text": "block one"},
        {"start": 25.0, "end": 30.0, "text": "block two"},
        {"start": 50.0, "end": 55.0, "text": "block three"},
    ]
    window_duration = 80.0

    cut_calls: list[tuple[float, float]] = []

    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        cut_calls.append((round(start, 3), round(duration, 3)))
        return src_path.parent / f"retry{len(cut_calls)}.opus"

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)

    async def fake_post(_path: Path, **_kwargs: object) -> dict:
        # Non-speech, so nothing gets spliced and the loop can only ever
        # move on to a DIFFERENT window, never re-ask this one.
        return _payload([{"start": 0.0, "end": 1.0, "text": "*noise*"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    _result_segments, _missing = await transcribe._ensure_coverage(
        segments, source_path=audio, window_duration=window_duration, per_job_lock=asyncio.Semaphore(1)
    )

    assert len(cut_calls) == 2
    # No window was cut more than once.
    assert len(set(cut_calls)) == len(cut_calls)


# ---------------------------------------------------------------------------
# Slicing: a single recheck request must never cover more audio than
# _MAX_RECHECK_SLICE_SECONDS. Regression guard for a live incident: a 562s
# recheck reproduced the EXACT decode-loop that created the hole in the
# first place, while a manual 20s check of the same audio came back with
# real dialogue — the size of the ask, not just whether one happens,
# determines whether a recheck can see something different from the
# original failed attempt.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_coverage_splits_long_window_into_slices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A 590s suspicious window (well past the 500s+ shape that reproduced
    the live regression) is split into consecutive
    _MAX_RECHECK_SLICE_SECONDS-sized requests, never sent whole."""
    monkeypatch.setattr(transcribe, "_PREFIX_DISTRUST_SECONDS", 0.0)
    monkeypatch.setattr(transcribe, "_RETRY_BACKOFF_SECONDS", 0.0)

    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x")

    cut_calls: list[tuple[float, float]] = []

    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        cut_calls.append((start, duration))
        return src_path.parent / f"retry{len(cut_calls)}.opus"

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)

    async def fake_post(_path: Path, **_kwargs: object) -> dict:
        # Confirmed non-speech every time, so each slice resolves in
        # exactly one recheck — isolating the SLICING behaviour from the
        # classification logic already covered by other tests.
        return _payload([{"start": 0.0, "end": 1.0, "text": "*noise*"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    segments = [{"start": 0.0, "end": 10.0, "text": "intro speech"}]
    result_segments, missing = await transcribe._ensure_coverage(
        segments, source_path=audio, window_duration=600.0, per_job_lock=asyncio.Semaphore(1)
    )

    # 590s of suspicious audio (10-600) split into ceil(590/90) = 7 slices.
    assert len(cut_calls) == 7
    # No single request ever covers more than the slice cap.
    assert all(
        duration <= transcribe._MAX_RECHECK_SLICE_SECONDS for _start, duration in cut_calls
    )
    # Boundaries: consecutive, non-overlapping, covering the whole interval
    # exactly (distrust/backoff pinned to 0 above so cut bounds equal slice
    # bounds precisely).
    starts = [start for start, _duration in cut_calls]
    assert starts == sorted(starts)
    assert starts[0] == 10.0
    ends = [start + duration for start, duration in cut_calls]
    assert ends[-1] == 600.0
    for i in range(len(cut_calls) - 1):
        assert ends[i] == starts[i + 1]

    # Confirmed non-speech throughout -> nothing spliced in, nothing missing
    # (this also doubles as the "confirmed silence still isn't missing"
    # check at a much larger scale than the single-slice tests above).
    assert missing == 0.0
    texts = [s["text"] for s in result_segments]
    assert texts == ["intro speech"]


# ---------------------------------------------------------------------------
# Regression: the whole point of removing the correctness threshold. A ~40s
# internal loss — well BELOW the old fixed 90s "it's probably fine" cutoff —
# is now detected and flagged with the DEFAULT configuration, no threshold
# monkeypatching required.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_below_old_90s_threshold_loss_now_detected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A 40s internal gap used to be silently accepted (well under the old
    fixed 90s correctness threshold). With that threshold gone, the cost
    cutoff (~5s) queues it for a recheck regardless of size, and here the
    recheck can't even run (ffmpeg unavailable) — so the shortfall is
    conservatively reported rather than the transcript looking complete."""
    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x" * 1024)

    # Recheck cannot happen at all — the conservative fallback.
    monkeypatch.setattr(transcribe, "_cut_audio_segment", lambda *a, **k: None)

    async def fake_post(_path: Path, **_kwargs: object) -> dict:
        # Real speech 0-10s, then NOTHING for the next 40s of a 50s file.
        return _payload([{"start": 0.0, "end": 10.0, "text": "real speech here"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result = await transcribe.transcribe_audio(audio, total_duration=50.0)

    assert result.missing_seconds is not None
    assert result.missing_seconds == pytest.approx(40.0)


@pytest.mark.asyncio
async def test_prefix_distrust_window_does_not_lose_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The distrust window discards the ORIGINAL (presumed-drifted) segments
    it covers, but the audio itself is re-transcribed, not skipped — content
    that fell inside the window must still be present in the result,
    sourced from the retry, and confirmed content on either side of the
    window must survive untouched."""
    monkeypatch.setattr(transcribe, "_RETRY_BACKOFF_SECONDS", 2.0)
    monkeypatch.setattr(transcribe, "_PREFIX_DISTRUST_SECONDS", 10.0)

    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x")
    segments, window_duration = _drifted_prefix_before_gap_scenario()

    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        return src_path.parent / "retry.opus"

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)

    async def fake_post(_path: Path, **_kwargs: object) -> dict:
        return _payload(
            [
                {"start": 2.0, "end": 3.4, "text": "Mann, das ist Luna Gröner."},
                {"start": 3.5, "end": 5.8, "text": "Tochter unserer scheiß Vermieterin."},
                {"start": 5.9, "end": 22.0, "text": "filled gap content"},
            ]
        )

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result_segments, missing = await transcribe._ensure_coverage(
        segments, source_path=audio, window_duration=window_duration, per_job_lock=asyncio.Semaphore(1)
    )

    assert missing == 0.0
    texts = [s["text"] for s in result_segments]
    # Confirmed content well before the distrust window survives untouched.
    assert "old confirmed content" in texts
    # Content that was actually IN the hole comes back via the retry.
    assert "filled gap content" in texts
    # Confirmed content after the gap survives untouched.
    assert "confirmed after gap" in texts
    # The generic drifted filler (no retry counterpart) is gone — dropped,
    # not silently duplicated — but the retry's output covers that same
    # time span, so the underlying audio wasn't skipped, just re-read.
    assert "more drifted filler" not in texts

    starts = [s["start"] for s in result_segments]
    assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# TranscribeDiagnostics — the non-sensitive, persistable record of chunking
# decisions + coverage-recheck verdicts (see storage/migrations.py's v10).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transcribe_audio_records_single_request_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The single-request path records chunked=False plus the backend/model/
    yt-dlp-version fields — no coverage issues here, so coverage_rechecks
    stays empty."""
    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x" * 1024)

    async def fake_post(_path: Path, **_kwargs: object) -> dict:
        return _payload([{"start": 0.0, "end": 2.0, "text": "hi"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)
    monkeypatch.setattr(transcribe, "_cut_audio_segment", lambda *a, **k: None)

    result = await transcribe.transcribe_audio(audio, total_duration=2.5)

    assert result.diagnostics is not None
    assert result.diagnostics.chunking.chunked is False
    assert result.diagnostics.chunking.audio_size_bytes == 1024
    assert result.diagnostics.coverage_rechecks == []
    assert result.diagnostics.final_missing_seconds == result.missing_seconds
    # max_coverage_rechecks records the CONFIGURED ceiling (a summary
    # value), not the smaller per-call budget _coverage_recheck_budget
    # actually enforced for this short 2.5s window — see each recheck
    # entry's own "of" field for that (there are none here to check, since
    # coverage_rechecks is empty above).
    assert (
        result.diagnostics.max_coverage_rechecks
        == transcribe.get_config().whisper.max_coverage_rechecks
    )
    # Backend/model come straight from config; yt-dlp version is whatever
    # the installed package reports (best-effort, never blocks on failure).
    assert result.diagnostics.whisper_model is not None
    assert result.diagnostics.whisper_backend_base_url is not None


@pytest.mark.asyncio
async def test_transcribe_audio_records_chunked_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The chunked path records chunked=True with num_chunks/chunk_seconds,
    and a coverage_recheck entry per recheck across ALL chunks (unit label
    identifies which chunk)."""
    audio = tmp_path / "big.opus"
    # Must exceed whisper.max_upload_mb (test config default 15 MB) so
    # transcribe_audio actually takes the chunked path, not the
    # single-request one.
    audio_size = 16 * 1024 * 1024
    audio.write_bytes(b"x" * audio_size)

    # This test is about the DIAGNOSTICS STRUCTURE for the chunked path, not
    # about the chunk-duration bound (which has its own dedicated tests) —
    # push max_chunk_seconds high enough that it never dominates num_chunks,
    # so chunking stays driven by the byte cap alone, matching the 2 fixed
    # chunks _split_audio is mocked to return below.
    cfg = transcribe.get_config()
    monkeypatch.setattr(cfg.whisper, "max_chunk_seconds", 100_000.0)
    monkeypatch.setattr(transcribe, "get_config", lambda: cfg)

    c0, c1 = tmp_path / "c0.opus", tmp_path / "c1.opus"
    c0.write_bytes(b"0")
    c1.write_bytes(b"1")
    monkeypatch.setattr(transcribe, "_split_audio", lambda *a, **k: [(c0, 0.0), (c1, 600.0)])

    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        return src_path.parent / "retry.opus"

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)

    async def fake_post(path: Path, **_kwargs: object) -> dict:
        if path == c0:
            # Chunk 0 covers only [0, 10) of its ~600s window — leaves a
            # large suspicious gap that will get rechecked. No reported
            # language (None) — this test is about the diagnostics
            # STRUCTURE, not language pinning (see the dedicated
            # confirmed_non_speech_both_ways / recovered_unpinned tests
            # below), so nothing here should pin and trigger the unpinned-
            # retry path.
            return _payload([{"start": 0.0, "end": 10.0, "text": "first"}], language=None)
        if path == c1:
            return _payload([{"start": 0.0, "end": 600.0, "text": "second, fully covered"}])
        # The coverage recheck itself — confirmed non-speech.
        return _payload([{"start": 0.0, "end": 1.0, "text": "*music*"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result = await transcribe.transcribe_audio(audio, total_duration=1200.0)

    assert result.diagnostics is not None
    assert result.diagnostics.chunking.chunked is True
    assert result.diagnostics.chunking.num_chunks == 2
    assert result.diagnostics.chunking.audio_size_bytes == audio_size
    assert len(result.diagnostics.coverage_rechecks) >= 1
    recheck = result.diagnostics.coverage_rechecks[0]
    assert recheck["unit"] == "chunk 1/2"
    assert recheck["verdict"] == "confirmed_non_speech"
    # "of" is the per-unit SCALED budget for this chunk's own ~600s window,
    # not the global ceiling — see _coverage_recheck_budget.
    assert recheck["of"] == transcribe._coverage_recheck_budget(600.0)


@pytest.mark.asyncio
async def test_ensure_coverage_records_each_verdict_kind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Direct unit test of _ensure_coverage's diagnostics param: a confirmed-
    silence window produces a correctly-labelled diagnostics entry."""
    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x")

    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        return src_path.parent / "retry.opus"

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)

    async def fake_post(_path: Path, **_kwargs: object) -> dict:
        return _payload([{"start": 0.0, "end": 30.0, "text": "*Dramatic music*"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    segments = [{"start": 0.0, "end": 10.0, "text": "intro speech"}]
    diagnostics = transcribe.TranscribeDiagnostics()
    _result_segments, _missing = await transcribe._ensure_coverage(
        segments,
        source_path=audio,
        window_duration=40.0,
        per_job_lock=asyncio.Semaphore(1),
        diagnostics=diagnostics,
        unit_label="whole",
    )

    assert len(diagnostics.coverage_rechecks) == 1
    entry = diagnostics.coverage_rechecks[0]
    assert entry["unit"] == "whole"
    assert entry["verdict"] == "confirmed_non_speech"
    assert entry["window_start"] == 10.0
    assert entry["window_end"] == 40.0


# ---------------------------------------------------------------------------
# Phase 3, fix 1: bound chunk DURATION, not just size. Regression test for
# job r56sj0_o1ihv (ZDF, 24:56 episode, 19.2 MB): the byte cap alone produced
# 2 chunks of ~748s each — long enough to reproduce Whisper's decode-loop
# failure mode on the FIRST pass itself, the same failure
# _MAX_RECHECK_SLICE_SECONDS already guards against for rechecks.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chunk_duration_bound_dominates_over_byte_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """whisper.max_chunk_seconds (default 300s) now bounds num_chunks
    together with the byte cap — max(chunks_by_bytes, chunks_by_seconds) —
    so the exact incident shape (19.2 MB / 24:56 -> 2 chunks of ~748s by
    size alone) instead produces enough chunks that none exceeds the
    duration cap."""
    audio = tmp_path / "big.opus"
    audio_size = int(19.2 * 1024 * 1024)
    audio.write_bytes(b"x" * audio_size)
    duration = 1496.0  # 24:56

    async def passthrough_ensure_coverage(
        segments: list[dict], **_kwargs: object
    ) -> tuple[list[dict], float]:
        # Isolates this test to the CHUNKING decision — coverage-recheck
        # behaviour is covered elsewhere.
        return segments, 0.0

    monkeypatch.setattr(transcribe, "_ensure_coverage", passthrough_ensure_coverage)

    def fake_split(
        _path: Path, num_chunks: int, chunk_seconds: float
    ) -> list[tuple[Path, float]]:
        chunks = []
        for i in range(num_chunks):
            c = tmp_path / f"chunk{i}.opus"
            c.write_bytes(b"x")
            chunks.append((c, i * chunk_seconds))
        return chunks

    monkeypatch.setattr(transcribe, "_split_audio", fake_split)

    async def fake_post(_path: Path, **_kwargs: object) -> dict:
        return _payload([{"start": 0.0, "end": 1.0, "text": "x"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    max_bytes = 15 * 1024 * 1024  # whisper.max_upload_mb default
    diagnostics = transcribe.TranscribeDiagnostics()
    result = await transcribe._transcribe_chunked(
        audio,
        total_duration=duration,
        max_bytes=max_bytes,
        per_job_lock=asyncio.Semaphore(1),
        diagnostics=diagnostics,
    )

    chunking = diagnostics.chunking
    assert chunking.chunks_by_bytes == 2
    assert chunking.chunks_by_seconds == 5  # ceil(1496 / 300)
    # The LARGER bound wins.
    assert chunking.num_chunks == 5
    assert chunking.max_chunk_seconds_cap == 300.0
    assert chunking.chunk_seconds is not None
    # Well under both the configured cap and the incident's actual 748s.
    assert chunking.chunk_seconds < 300.0
    assert chunking.chunk_seconds < 748.0 / 2
    assert "seconds" in chunking.reason
    assert result.duration_seconds == duration


@pytest.mark.asyncio
async def test_chunk_duration_bound_is_a_noop_when_bytes_already_dominate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A short, simply-oversized file (bytes cap drives chunking, duration
    cap wouldn't ask for more chunks) is unaffected by the new bound — same
    num_chunks as the byte cap alone would produce."""
    audio = tmp_path / "big.opus"
    audio_size = 20 * 1024 * 1024
    audio.write_bytes(b"x" * audio_size)
    duration = 120.0  # short — well under max_chunk_seconds even as 1 chunk

    async def passthrough_ensure_coverage(
        segments: list[dict], **_kwargs: object
    ) -> tuple[list[dict], float]:
        return segments, 0.0

    monkeypatch.setattr(transcribe, "_ensure_coverage", passthrough_ensure_coverage)

    def fake_split(
        _path: Path, num_chunks: int, chunk_seconds: float
    ) -> list[tuple[Path, float]]:
        chunks = []
        for i in range(num_chunks):
            c = tmp_path / f"chunk{i}.opus"
            c.write_bytes(b"x")
            chunks.append((c, i * chunk_seconds))
        return chunks

    monkeypatch.setattr(transcribe, "_split_audio", fake_split)

    async def fake_post(_path: Path, **_kwargs: object) -> dict:
        return _payload([{"start": 0.0, "end": 1.0, "text": "x"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    max_bytes = 15 * 1024 * 1024
    diagnostics = transcribe.TranscribeDiagnostics()
    await transcribe._transcribe_chunked(
        audio,
        total_duration=duration,
        max_bytes=max_bytes,
        per_job_lock=asyncio.Semaphore(1),
        diagnostics=diagnostics,
    )

    chunking = diagnostics.chunking
    assert chunking.chunks_by_seconds == 1  # ceil(120 / 300)
    assert chunking.num_chunks == chunking.chunks_by_bytes
    assert "bytes" in chunking.reason


# ---------------------------------------------------------------------------
# Phase 3, fix 2: the recheck budget scales with the unit's own duration
# instead of one fixed number shared by every unit regardless of size.
# ---------------------------------------------------------------------------


def test_coverage_recheck_budget_scales_with_window_duration() -> None:
    # Floor: a very short window's formula result (< min_coverage_rechecks)
    # is raised to the floor.
    assert transcribe._coverage_recheck_budget(10.0) == 4
    # Matches the worked example in _coverage_recheck_budget's own
    # docstring: a 300s chunk (the default max_chunk_seconds) needs 4
    # slices end to end; factor 2.0 -> 8.
    assert transcribe._coverage_recheck_budget(300.0) == 8
    # Ceiling: a very long single-request window's formula result would
    # otherwise be unbounded (ceil(3600/90)*2 = 80) — clamped to 20.
    assert transcribe._coverage_recheck_budget(3600.0) == 20
    # Monotonic: a longer window never gets a SMALLER budget.
    assert transcribe._coverage_recheck_budget(600.0) >= transcribe._coverage_recheck_budget(300.0)


# ---------------------------------------------------------------------------
# Phase 3, fix 3: window memoisation is overlap-based, not exact-key.
# Regression test for job r56sj0_o1ihv: three rechecks
# (469s-490s, 468s-490s, 467s-490s) were spent on what was really ONE ~22s
# hole, because splicing shifted the edges by a fraction of a second each
# time and an exact-key "already checked" test never fired again.
# ---------------------------------------------------------------------------


def test_overlaps_checked_matches_incident_shifted_windows() -> None:
    checked = [(469.0, 490.0)]
    # Each of these is the "same" hole, shifted by a second or two — all
    # should read as already checked.
    assert transcribe._overlaps_checked((468.0, 490.0), checked) is True
    assert transcribe._overlaps_checked((467.0, 490.0), checked) is True
    # A genuinely different, non-overlapping hole must NOT be suppressed.
    assert transcribe._overlaps_checked((100.0, 150.0), checked) is False


def test_overlaps_checked_large_candidate_containing_small_checked_is_not_suppressed() -> None:
    """Asymmetric case (review fix): a LARGE new candidate that merely
    *contains* a small already-checked interval must NOT be suppressed —
    most of it (80 of 90s here) was never actually examined. Measuring the
    overlap fraction against ``min(candidate, checked)`` instead of the
    candidate's own length would read this as 100% overlap
    (10s / min(90, 10) = 1.0) and silently drop 80s of never-checked audio
    into ``missing_seconds`` as though it had been tried — the same class
    of bug this memoisation was written to fix, just inverted."""
    checked = [(400.0, 410.0)]
    assert transcribe._overlaps_checked((400.0, 490.0), checked) is False
    # The small checked window itself is, of course, still suppressed.
    assert transcribe._overlaps_checked((400.0, 410.0), checked) is True
    # And the case the fix must NOT break: a small candidate fully inside a
    # much larger already-checked interval still suppresses.
    checked_large = [(400.0, 490.0)]
    assert transcribe._overlaps_checked((405.0, 415.0), checked_large) is True


def test_pending_slices_suppresses_near_duplicate_window() -> None:
    """Direct test of the memoisation used inside _ensure_coverage: a
    candidate slice overlapping something already in ``checked`` beyond
    the dedup threshold is reported as suppressed, not pending."""
    working = [{"start": 0.0, "end": 100.0, "text": "intro"}]
    # A near-duplicate of an already-checked (99, 150) window.
    checked = [(99.0, 150.0)]
    pending, suppressed = transcribe._pending_slices(working, 150.0, checked)
    assert pending == []
    assert suppressed == [(100.0, 150.0)]

    # A genuinely different checked window doesn't suppress this one.
    pending2, suppressed2 = transcribe._pending_slices(
        working, 150.0, [(0.0, 10.0)]
    )
    assert suppressed2 == []
    assert pending2 == [(100.0, 150.0, True)]


@pytest.mark.asyncio
async def test_ensure_coverage_overlap_dedup_stops_near_duplicate_rechecks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end: a recheck that only partially resolves a hole (a
    degenerate repeat, kept via collapse's first-occurrence rule) leaves a
    SLIGHTLY SMALLER remaining hole that overlaps the just-checked window
    almost entirely. Without overlap-based dedup this looks like a "new"
    window and gets rechecked again; with it, the near-duplicate is
    suppressed and the loop stops — exactly the incident's waste this fix
    removes. The budget (4, for this 100s window) is NOT what stops it —
    only ONE actual Whisper call happens despite plenty of budget left."""
    monkeypatch.setattr(transcribe, "_PREFIX_DISTRUST_SECONDS", 0.0)
    monkeypatch.setattr(transcribe, "_RETRY_BACKOFF_SECONDS", 0.0)

    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x")

    monkeypatch.setattr(
        transcribe, "_cut_audio_segment", lambda src, start, duration: src.parent / "retry.opus"
    )

    post_calls: list[Path] = []

    async def fake_post(_path: Path, **_kwargs: object) -> dict:
        post_calls.append(_path)
        # Real content for 0-5s (local to the recheck's own cut), then a
        # decode-loop for the rest — collapse keeps the FIRST occurrence
        # of the loop line, discarding everything after it.
        segs = [{"start": 0.0, "end": 5.0, "text": "recovered"}]
        t = 5.0
        while t < 90.0:
            segs.append({"start": t, "end": t + 1.0, "text": "loop"})
            t += 1.0
        return _payload(segs)

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    segments = [{"start": 0.0, "end": 10.0, "text": "intro"}]
    diagnostics = transcribe.TranscribeDiagnostics()
    _result_segments, missing = await transcribe._ensure_coverage(
        segments,
        source_path=audio,
        window_duration=100.0,
        per_job_lock=asyncio.Semaphore(1),
        diagnostics=diagnostics,
        unit_label="whole",
    )

    # Only the FIRST recheck actually asked Whisper — the near-duplicate
    # remaining hole on the next outer-loop pass was suppressed instead of
    # re-asked.
    assert len(post_calls) == 1
    assert len(diagnostics.coverage_rechecks) == 1
    assert diagnostics.overlap_suppressed
    suppressed_entry = diagnostics.overlap_suppressed[0]
    assert suppressed_entry["unit"] == "whole"
    assert suppressed_entry["window_start"] == 16.0
    assert suppressed_entry["window_end"] == 100.0
    # The unresolved remainder is still honestly counted as missing — the
    # degenerate loop is NEVER treated as confirmed silence.
    assert missing == pytest.approx(84.0)


# ---------------------------------------------------------------------------
# Phase 3, fix 4: language is detected once per call, then pinned on every
# subsequent request (chunks 2+, and every coverage recheck) instead of each
# one auto-detecting independently. Regression test for job r56sj0_o1ihv:
# the opening ~3 minutes came back transcribed in English while
# transcript_language was recorded as "de".
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chunked_pins_language_after_first_chunk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = tmp_path / "big.opus"
    audio.write_bytes(b"x" * (20 * 1024 * 1024))

    # Push the duration bound out of the way — this test is about language
    # pinning, not chunk-duration bounding (which has its own tests above).
    cfg = transcribe.get_config()
    monkeypatch.setattr(cfg.whisper, "max_chunk_seconds", 100_000.0)
    monkeypatch.setattr(transcribe, "get_config", lambda: cfg)

    c0, c1 = tmp_path / "c0.opus", tmp_path / "c1.opus"
    c0.write_bytes(b"0")
    c1.write_bytes(b"1")
    monkeypatch.setattr(transcribe, "_split_audio", lambda *a, **k: [(c0, 0.0), (c1, 600.0)])
    monkeypatch.setattr(transcribe, "_cut_audio_segment", lambda *a, **k: None)

    calls: list[tuple[Path, str | None]] = []

    async def fake_post(path: Path, *, language: str | None = None) -> dict:
        calls.append((path, language))
        # Fully covered — no coverage rechecks triggered either.
        return _payload([{"start": 0.0, "end": 600.0, "text": "voll"}], "de")

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result = await transcribe.transcribe_audio(audio, total_duration=1200.0)

    assert result.language == "de"
    # Chunk 0's own first-pass request auto-detects (language=None sent).
    assert calls[0] == (c0, None)
    # Chunk 1 PINS what chunk 0 detected, instead of auto-detecting again.
    assert calls[1] == (c1, "de")


@pytest.mark.asyncio
async def test_single_request_recheck_pins_detected_language(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The single-request path's own coverage recheck pins whatever the
    first (auto-detecting) request reported, rather than re-guessing on the
    recheck's smaller, more ambiguous slice."""
    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x" * 1024)

    monkeypatch.setattr(
        transcribe, "_cut_audio_segment", lambda src, start, duration: src.parent / "retry.opus"
    )

    calls: list[str | None] = []

    async def fake_post(_path: Path, *, language: str | None = None) -> dict:
        calls.append(language)
        if len(calls) == 1:
            # First pass: real speech for 0-10s of a 20s file, auto-detected.
            return _payload([{"start": 0.0, "end": 10.0, "text": "hello"}], "ru")
        return _payload([{"start": 0.0, "end": 5.0, "text": "bye"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    await transcribe.transcribe_audio(audio, total_duration=20.0)

    assert calls[0] is None  # first pass auto-detects
    assert calls[1] == "ru"  # the recheck pins what was detected


@pytest.mark.asyncio
async def test_post_audio_sends_language_form_field_only_when_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unit test of _post_audio itself: the ``language`` form field is
    included only when a language is actually passed, never as an empty/
    None value — confirmed against the real LocalAI/whisper.cpp backend to
    be genuinely CONSUMED for decoding, not just echoed (see the module
    docstring)."""
    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x")

    captured: dict[str, Any] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"segments": [], "text": "", "duration": 1.0}

    class _FakeClient:
        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def post(self, _url: str, *, headers: dict, data: dict, files: dict) -> _FakeResponse:
            captured.update(data)
            return _FakeResponse()

    monkeypatch.setattr(transcribe.httpx, "AsyncClient", lambda **_k: _FakeClient())

    await transcribe._post_audio(audio, language=None)
    assert "language" not in captured

    captured.clear()
    await transcribe._post_audio(audio, language="de")
    assert captured["language"] == "de"


# ---------------------------------------------------------------------------
# Fix 4, round 2 (severe regression found by a live re-run against the real
# ZDF episode): a single chunk's text-detection is NEVER trusted as a
# language source any more. Chunk 0's own detect_language() returned a
# CONFIDENT "en" for chunk 0's musical, sparse-dialogue opening in a German
# episode (Whisper's own auto-detect already wobbles on exactly that kind
# of audio) — that got pinned and forced onto every other chunk, corrupting
# a transcript that was ~96% correct WITHOUT any pinning at all (measured:
# 9.6 German/English marker-word ratio before this feature existed, 0.0
# after). The fix has two parts:
#
#   1. PRIMARY source: the publisher's own declared language (yt-dlp
#      metadata, threaded in from workers/runner.py as metadata_language) —
#      authoritative, free, and produced independently of anything Whisper
#      says, so it can't be corrupted by a Whisper mis-detection.
#   2. FALLBACK, only when metadata is absent AND the backend doesn't self-
#      report: text-detection is pinned ONLY when a SECOND, independent
#      chunk agrees with the first. Never from one chunk alone.
# ---------------------------------------------------------------------------


def test_normalize_metadata_language_maps_iso_639_2_to_iso_639_1() -> None:
    """yt-dlp's metadata field is often a 3-letter ISO-639-2 code (e.g.
    "deu" for German) — Whisper's own `language` request field expects
    ISO-639-1. Reuses llm.languages.normalize_lang's existing alias table
    rather than duplicating it."""
    assert transcribe._normalize_metadata_language("deu") == "de"
    assert transcribe._normalize_metadata_language("de") == "de"
    assert transcribe._normalize_metadata_language("German") == "de"


def test_normalize_metadata_language_rejects_unrecognised_or_empty() -> None:
    """An unrecognised or missing value comes back None rather than being
    passed through raw — sending Whisper a code it doesn't understand is
    worse than sending none (falls back to auto-detect)."""
    assert transcribe._normalize_metadata_language("not-a-real-language") is None
    assert transcribe._normalize_metadata_language("") is None
    assert transcribe._normalize_metadata_language(None) is None
    assert transcribe._normalize_metadata_language(123) is None


def test_bootstrap_language_prefers_metadata_over_everything() -> None:
    """metadata_language wins even when the backend ALSO reported one — see
    this function's own docstring for why it outranks a backend self-report:
    it doesn't depend on Whisper's output at all."""
    result = transcribe.TranscribeResult(segments=[], language="en", duration_seconds=1.0)
    lang, source = transcribe._bootstrap_language(
        {"text": "irrelevant"}, result, metadata_language="de"
    )
    assert (lang, source) == ("de", "from_metadata")


def test_bootstrap_language_falls_back_to_backend_report_without_metadata() -> None:
    result = transcribe.TranscribeResult(segments=[], language="de", duration_seconds=1.0)
    lang, source = transcribe._bootstrap_language(
        {"text": "irrelevant"}, result, metadata_language=None
    )
    assert (lang, source) == ("de", "reported_by_backend")


def test_bootstrap_language_stays_unpinned_without_metadata_or_backend_report() -> None:
    """_bootstrap_language itself no longer falls back to text-detection —
    see its own docstring for the regression that removed that tier. A
    silent backend with no metadata leaves THIS call's first request
    unpinned; only the chunked path's two-chunk-agreement fallback (tested
    below) can still recover a pin from text, and never from this function
    alone."""
    result = transcribe.TranscribeResult(segments=[], language=None, duration_seconds=1.0)
    lang, source = transcribe._bootstrap_language(
        {"text": "Ey Fred, komm mal her."}, result, metadata_language=None
    )
    assert (lang, source) == (None, "unpinned")


@pytest.mark.asyncio
async def test_single_request_uses_metadata_language_from_the_first_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unlike a backend self-report or text-detection (which both need a
    response first), metadata is known BEFORE any Whisper call happens at
    all — so it pins the very FIRST request too, not just requests after
    it."""
    monkeypatch.setattr(transcribe, "_PREFIX_DISTRUST_SECONDS", 0.0)
    monkeypatch.setattr(transcribe, "_RETRY_BACKOFF_SECONDS", 0.0)
    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x" * 1024)
    monkeypatch.setattr(
        transcribe, "_cut_audio_segment", lambda src, start, duration: src.parent / "retry.opus"
    )

    calls: list[str | None] = []

    async def fake_post(_path: Path, *, language: str | None = None) -> dict:
        calls.append(language)
        # Fully covered — no recheck triggered, so the count of calls is
        # exactly 1; the point of this test is what LANGUAGE it carries.
        return _payload([{"start": 0.0, "end": 20.0, "text": "hallo"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result = await transcribe.transcribe_audio(
        audio, total_duration=20.0, metadata_language="deu"
    )

    # The FIRST (and only, since fully covered) request is pinned —
    # metadata is known before any Whisper call happens at all.
    assert calls == ["de"]
    assert result.diagnostics is not None
    assert result.diagnostics.pinned_language == "de"
    assert result.diagnostics.language_pin_source == "from_metadata"


@pytest.mark.asyncio
async def test_chunked_uses_metadata_language_from_chunk_zero_onward(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = tmp_path / "big.opus"
    audio.write_bytes(b"x" * (20 * 1024 * 1024))

    cfg = transcribe.get_config()
    monkeypatch.setattr(cfg.whisper, "max_chunk_seconds", 100_000.0)
    monkeypatch.setattr(transcribe, "get_config", lambda: cfg)

    c0, c1 = tmp_path / "c0.opus", tmp_path / "c1.opus"
    c0.write_bytes(b"0")
    c1.write_bytes(b"1")
    monkeypatch.setattr(transcribe, "_split_audio", lambda *a, **k: [(c0, 0.0), (c1, 600.0)])
    monkeypatch.setattr(transcribe, "_cut_audio_segment", lambda *a, **k: None)

    calls: list[tuple[Path, str | None]] = []

    async def fake_post(path: Path, *, language: str | None = None) -> dict:
        calls.append((path, language))
        return _payload([{"start": 0.0, "end": 600.0, "text": "voll"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result = await transcribe.transcribe_audio(
        audio, total_duration=1200.0, metadata_language="deu"
    )

    # Every chunk, INCLUDING the first, is pinned — metadata is known
    # before any Whisper call happens at all.
    assert calls == [(c0, "de"), (c1, "de")]
    assert result.diagnostics is not None
    assert result.diagnostics.pinned_language == "de"
    assert result.diagnostics.language_pin_source == "from_metadata"


@pytest.mark.asyncio
async def test_chunked_pins_from_backend_report_without_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = tmp_path / "big.opus"
    audio.write_bytes(b"x" * (20 * 1024 * 1024))

    cfg = transcribe.get_config()
    monkeypatch.setattr(cfg.whisper, "max_chunk_seconds", 100_000.0)
    monkeypatch.setattr(transcribe, "get_config", lambda: cfg)

    c0, c1 = tmp_path / "c0.opus", tmp_path / "c1.opus"
    c0.write_bytes(b"0")
    c1.write_bytes(b"1")
    monkeypatch.setattr(transcribe, "_split_audio", lambda *a, **k: [(c0, 0.0), (c1, 600.0)])
    monkeypatch.setattr(transcribe, "_cut_audio_segment", lambda *a, **k: None)

    calls: list[tuple[Path, str | None]] = []

    async def fake_post(path: Path, *, language: str | None = None) -> dict:
        calls.append((path, language))
        return _payload([{"start": 0.0, "end": 600.0, "text": "voll"}], "de")

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result = await transcribe.transcribe_audio(audio, total_duration=1200.0)

    assert calls[0] == (c0, None)  # chunk 0 itself still auto-detects
    assert calls[1] == (c1, "de")  # pinned from chunk 0's OWN backend report
    assert result.diagnostics is not None
    assert result.diagnostics.language_pin_source == "reported_by_backend"


@pytest.mark.asyncio
async def test_chunked_requires_two_chunks_to_agree_before_pinning_from_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No metadata, silent backend: chunk 0 and chunk 1 both independently
    detect "de" from their own text — only THEN is it pinned, applied to
    chunk 2 onward. Chunk 1 itself is the CONFIRMING sample and must stay
    unpinned (auto-detect) to be independent."""
    audio = tmp_path / "big.opus"
    audio.write_bytes(b"x" * (30 * 1024 * 1024))

    cfg = transcribe.get_config()
    monkeypatch.setattr(cfg.whisper, "max_chunk_seconds", 100_000.0)
    monkeypatch.setattr(transcribe, "get_config", lambda: cfg)

    c0, c1, c2 = (tmp_path / f"c{i}.opus" for i in range(3))
    for c in (c0, c1, c2):
        c.write_bytes(b"x")
    monkeypatch.setattr(
        transcribe,
        "_split_audio",
        lambda *a, **k: [(c0, 0.0), (c1, 600.0), (c2, 1200.0)],
    )
    monkeypatch.setattr(transcribe, "_cut_audio_segment", lambda *a, **k: None)
    monkeypatch.setattr(transcribe.languages, "detect_language", lambda _text: "de")

    calls: list[tuple[Path, str | None]] = []

    async def fake_post(path: Path, *, language: str | None = None) -> dict:
        calls.append((path, language))
        return {
            "segments": [{"start": 0.0, "end": 600.0, "text": "Ey Fred, komm mal her."}],
            "text": "Ey Fred, komm mal her.",
            "duration": 600.0,
        }

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result = await transcribe.transcribe_audio(audio, total_duration=1800.0)

    assert calls[0] == (c0, None)  # chunk 0: nothing to pin from yet
    assert calls[1] == (c1, None)  # chunk 1: the CONFIRMING sample, independent
    assert calls[2] == (c2, "de")  # chunk 2+: pinned, now that 0 and 1 agreed
    assert result.diagnostics is not None
    assert result.diagnostics.pinned_language == "de"
    assert result.diagnostics.language_pin_source == "detected_agreed"


@pytest.mark.asyncio
async def test_chunked_stays_unpinned_when_two_chunks_disagree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """THE regression itself, reproduced directly: chunk 0's own text would
    mis-detect as "en" (a musical/sparse-dialogue opening), chunk 1
    correctly detects "de". Disagreement means NEITHER gets pinned, and
    every chunk keeps auto-detecting independently — exactly like before
    any language-pinning feature existed, and the SAFE outcome: forcing
    either guess onto the rest of the call would have been worse (this is
    what actually happened on job r56sj0_o1ihv before this fix)."""
    audio = tmp_path / "big.opus"
    audio.write_bytes(b"x" * (20 * 1024 * 1024))

    cfg = transcribe.get_config()
    monkeypatch.setattr(cfg.whisper, "max_chunk_seconds", 100_000.0)
    monkeypatch.setattr(transcribe, "get_config", lambda: cfg)

    c0, c1 = tmp_path / "c0.opus", tmp_path / "c1.opus"
    c0.write_bytes(b"0")
    c1.write_bytes(b"1")
    monkeypatch.setattr(transcribe, "_split_audio", lambda *a, **k: [(c0, 0.0), (c1, 600.0)])
    monkeypatch.setattr(transcribe, "_cut_audio_segment", lambda *a, **k: None)

    def fake_detect(text: str) -> str | None:
        return "en" if "opening" in text else "de"

    monkeypatch.setattr(transcribe.languages, "detect_language", fake_detect)

    calls: list[tuple[Path, str | None]] = []

    async def fake_post(path: Path, *, language: str | None = None) -> dict:
        calls.append((path, language))
        if path == c0:
            return {
                "segments": [{"start": 0.0, "end": 600.0, "text": "musical opening"}],
                "text": "musical opening",
                "duration": 600.0,
            }
        return {
            "segments": [{"start": 0.0, "end": 600.0, "text": "Und jetzt ist Schluss"}],
            "text": "Und jetzt ist Schluss",
            "duration": 600.0,
        }

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result = await transcribe.transcribe_audio(audio, total_duration=1200.0)

    assert calls == [(c0, None), (c1, None)]  # BOTH auto-detect, no pin ever applied
    assert result.diagnostics is not None
    assert result.diagnostics.pinned_language is None
    assert result.diagnostics.language_pin_source == "unpinned"


@pytest.mark.asyncio
async def test_chunked_stays_unpinned_when_chunk_zero_text_is_undetectable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = tmp_path / "big.opus"
    audio.write_bytes(b"x" * (20 * 1024 * 1024))

    cfg = transcribe.get_config()
    monkeypatch.setattr(cfg.whisper, "max_chunk_seconds", 100_000.0)
    monkeypatch.setattr(transcribe, "get_config", lambda: cfg)

    c0, c1 = tmp_path / "c0.opus", tmp_path / "c1.opus"
    c0.write_bytes(b"0")
    c1.write_bytes(b"1")
    monkeypatch.setattr(transcribe, "_split_audio", lambda *a, **k: [(c0, 0.0), (c1, 600.0)])
    monkeypatch.setattr(transcribe, "_cut_audio_segment", lambda *a, **k: None)
    # Simulates a musical/near-silent opening: too short/ambiguous to guess.
    monkeypatch.setattr(transcribe.languages, "detect_language", lambda _text: None)

    calls: list[tuple[Path, str | None]] = []

    async def fake_post(path: Path, *, language: str | None = None) -> dict:
        calls.append((path, language))
        return {
            "segments": [{"start": 0.0, "end": 600.0, "text": "hm"}],
            "text": "hm",
            "duration": 600.0,
        }

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result = await transcribe.transcribe_audio(audio, total_duration=1200.0)

    assert calls == [(c0, None), (c1, None)]
    assert result.diagnostics is not None
    assert result.diagnostics.pinned_language is None
    assert result.diagnostics.language_pin_source == "unpinned"


# ---------------------------------------------------------------------------
# Phase 6 (owner ruling, 2026-08-28): "a wrong transcript beats a hole." A
# span never ends up with literally ZERO text once the recheck budget runs
# out, as long as the first pass produced something for it — restored,
# collapsed to at most what collapse_repeated_segments would keep, and
# marked low_confidence. missing_seconds itself is unaffected: restoration
# changes what TEXT the reader sees, not what still counts as uncertain.
# ---------------------------------------------------------------------------


def test_subtract_confirmed_silent_removes_fully_covered_window() -> None:
    assert transcribe._subtract_confirmed_silent((10.0, 50.0), [(10.0, 50.0)]) == []


def test_subtract_confirmed_silent_splits_around_a_middle_chunk() -> None:
    result = transcribe._subtract_confirmed_silent((0.0, 100.0), [(40.0, 60.0)])
    assert result == [(0.0, 40.0), (60.0, 100.0)]


def test_subtract_confirmed_silent_no_overlap_is_a_no_op() -> None:
    assert transcribe._subtract_confirmed_silent((0.0, 10.0), [(50.0, 60.0)]) == [
        (0.0, 10.0)
    ]


@pytest.mark.asyncio
async def test_ensure_coverage_restores_unresolved_span_instead_of_a_hole(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When every recheck for a discarded-run span fails outright
    (recut_unavailable here), the ORIGINAL first-pass text is restored —
    collapsed to ONE instance, never the full repeated run — and marked
    low_confidence, instead of leaving 94s of the window silent.
    missing_seconds stays exactly what it would have been without this
    feature: the span is just as uncertain as before, it simply isn't
    EMPTY in the output any more."""
    monkeypatch.setattr(transcribe, "_cut_audio_segment", lambda *a, **k: None)

    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x")

    loop_text = "I'm not sure if I'm doing that right."
    segments = [{"start": 0.0, "end": 5.0, "text": "intro speech"}]
    t = 5.0
    while t < 90.0:
        segments.append({"start": t, "end": t + 1.0, "text": loop_text})
        t += 1.0

    diagnostics = transcribe.TranscribeDiagnostics()
    result_segments, missing = await transcribe._ensure_coverage(
        segments,
        source_path=audio,
        window_duration=100.0,
        per_job_lock=asyncio.Semaphore(1),
        diagnostics=diagnostics,
        unit_label="whole",
    )

    assert missing == pytest.approx(94.0)

    loop_segments = [s for s in result_segments if s["text"] == loop_text]
    assert len(loop_segments) == 1  # collapse's one-instance rule still holds
    assert loop_segments[0]["low_confidence"] is True
    # Stretched all the way to the window's end — no residual hole after it.
    assert loop_segments[0]["end"] == pytest.approx(100.0)
    starts = [s["start"] for s in result_segments]
    assert starts == sorted(starts)

    assert diagnostics.backfill_count == 1
    assert diagnostics.backfill_total_seconds == pytest.approx(94.0)
    assert diagnostics.backfilled_spans == [
        {"unit": "whole", "window_start": 6.0, "window_end": 100.0}
    ]


@pytest.mark.asyncio
async def test_restore_never_touches_confirmed_silent_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A window a recheck CONFIRMED as non-speech (music/silence) must NOT
    be backfilled with the original, already-discarded hallucinated text —
    that would reintroduce KNOWN-WRONG content over a span independently
    verified to have no speech in it. Confirmed-silent stays exactly as
    before Phase 6: no text, not counted as missing, no backfill."""
    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        return src_path.parent / "retry.opus"

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)

    async def fake_post(_path: Path, **_kwargs: object) -> dict:
        return _payload([{"start": 0.0, "end": 1.0, "text": "*music*"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x")

    loop_text = "I'm not sure if I'm doing that right."
    segments = [{"start": 0.0, "end": 5.0, "text": "intro speech"}]
    t = 5.0
    while t < 30.0:
        segments.append({"start": t, "end": t + 1.0, "text": loop_text})
        t += 1.0

    diagnostics = transcribe.TranscribeDiagnostics()
    result_segments, missing = await transcribe._ensure_coverage(
        segments,
        source_path=audio,
        window_duration=40.0,
        per_job_lock=asyncio.Semaphore(1),
        diagnostics=diagnostics,
        unit_label="whole",
    )

    assert missing == 0.0  # confirmed non-speech — not lost content
    assert diagnostics.backfill_count == 0
    assert diagnostics.backfilled_spans == []
    assert not any(s.get("low_confidence") for s in result_segments)


@pytest.mark.asyncio
async def test_restore_does_nothing_when_first_pass_produced_no_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pure arithmetic gap — the first pass produced NOTHING for the
    span, not even a discarded repeat — has nothing to restore. This is
    the one case Phase 6 deliberately leaves open: "never delete", not
    "invent"."""
    monkeypatch.setattr(transcribe, "_cut_audio_segment", lambda *a, **k: None)

    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x")

    segments = [{"start": 0.0, "end": 10.0, "text": "real speech here"}]
    diagnostics = transcribe.TranscribeDiagnostics()
    result_segments, missing = await transcribe._ensure_coverage(
        segments,
        source_path=audio,
        window_duration=50.0,
        per_job_lock=asyncio.Semaphore(1),
        diagnostics=diagnostics,
        unit_label="whole",
    )

    assert missing == pytest.approx(40.0)
    assert diagnostics.backfill_count == 0
    assert diagnostics.backfilled_spans == []
    assert result_segments == segments


@pytest.mark.asyncio
async def test_restore_tracks_multiple_independent_unresolved_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two SEPARATE hallucination runs, each never resolved, each collapse
    to their OWN one instance and get restored independently — diagnostics
    tracks both as distinct backfills, not merged into one."""
    monkeypatch.setattr(transcribe, "_cut_audio_segment", lambda *a, **k: None)

    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x")

    text_a = "first hallucination phrase repeating over and over here"
    text_b = "second, different hallucination phrase repeating here too"
    segments = []
    t = 0.0
    while t < 15.0:  # run A: 0-15s
        segments.append({"start": t, "end": t + 1.0, "text": text_a})
        t += 1.0
    while t < 30.0:  # run B: immediately adjacent, 15-30s
        segments.append({"start": t, "end": t + 1.0, "text": text_b})
        t += 1.0

    diagnostics = transcribe.TranscribeDiagnostics()
    result_segments, missing = await transcribe._ensure_coverage(
        segments,
        source_path=audio,
        window_duration=40.0,
        per_job_lock=asyncio.Semaphore(1),
        diagnostics=diagnostics,
        unit_label="whole",
    )

    assert missing > 0
    restored = [s for s in result_segments if s.get("low_confidence")]
    assert len(restored) == 2  # each run collapsed to its OWN one instance
    a_seg = next(s for s in restored if s["text"] == text_a)
    b_seg = next(s for s in restored if s["text"] == text_b)
    assert a_seg["start"] == pytest.approx(0.0)
    assert a_seg["end"] == pytest.approx(15.0)
    assert b_seg["start"] == pytest.approx(15.0)
    assert b_seg["end"] == pytest.approx(40.0)
    assert diagnostics.backfill_count == 2


# ---------------------------------------------------------------------------
# Second regression found by the same live re-run: diagnostics reported 19
# backfilled spans (240.5s), but zero low_confidence segments actually made
# it into the output. Root cause: the CHUNKED path's merge loop rebuilt each
# offset segment as a fresh {"start", "end", "text"} literal, silently
# dropping any OTHER key — including Phase 6's low_confidence flag — before
# handing segments back to transcribe_audio. _clip_to_gap (used both by a
# normal recheck splice and by _restore_unresolved_windows itself) had the
# same defect. Both were fixed to copy the segment (dict(seg, ...)) instead
# of reconstructing a 3-key literal. This test asserts the flag survives ALL
# THE WAY out of transcribe_audio via the CHUNKED path specifically — an
# _ensure_coverage-level assertion alone would not have caught this, since
# the stripping happened one layer above it.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_low_confidence_flag_survives_out_of_transcribe_audio_chunked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end: a chunk whose recheck never resolves gets a
    low_confidence backfill (Phase 6) that must still be present on the
    segment transcribe_audio() finally returns — after the chunk-merge
    offset step AND the whole-transcript collapse_repeated_segments pass
    that runs at the very end of transcribe_audio."""
    monkeypatch.setattr(transcribe, "_cut_audio_segment", lambda *a, **k: None)

    audio = tmp_path / "big.opus"
    audio.write_bytes(b"x" * (20 * 1024 * 1024))

    cfg = transcribe.get_config()
    monkeypatch.setattr(cfg.whisper, "max_chunk_seconds", 100_000.0)
    monkeypatch.setattr(transcribe, "get_config", lambda: cfg)

    c0, c1 = tmp_path / "c0.opus", tmp_path / "c1.opus"
    c0.write_bytes(b"0")
    c1.write_bytes(b"1")
    monkeypatch.setattr(transcribe, "_split_audio", lambda *a, **k: [(c0, 0.0), (c1, 600.0)])

    loop_text = "I'm not sure if I'm doing that right."

    async def fake_post(path: Path, **_kwargs: object) -> dict:
        if path == c0:
            # Chunk 0: real content for 5s, then a hallucination loop for
            # the rest of its ~600s window — recut fails, so it's never
            # resolved and must be backfilled, not left silent.
            segs = [{"start": 0.0, "end": 5.0, "text": "intro speech"}]
            t = 5.0
            while t < 90.0:
                segs.append({"start": t, "end": t + 1.0, "text": loop_text})
                t += 1.0
            return _payload(segs)
        # Chunk 1: fully covered, ordinary content — nothing to backfill.
        return _payload([{"start": 0.0, "end": 600.0, "text": "second chunk, fine"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result = await transcribe.transcribe_audio(audio, total_duration=1200.0)

    assert result.diagnostics is not None
    assert result.diagnostics.backfill_count >= 1

    low_confidence_segments = [s for s in result.segments if s.get("low_confidence")]
    assert low_confidence_segments, (
        "low_confidence flag did not survive out of transcribe_audio "
        "(chunked path) — check the chunk-merge offset loop and "
        "_clip_to_gap for a fresh {'start','end','text'} literal that "
        "drops extra keys"
    )
    assert any(s["text"] == loop_text for s in low_confidence_segments)
    # Still exactly ONE instance of the loop text — Phase 6 restoration
    # must not have reintroduced the full collapsed run.
    assert sum(1 for s in result.segments if s["text"] == loop_text) == 1


# ---------------------------------------------------------------------------
# Third language-pinning regression (owner ruling follow-up): a PINNED
# language can itself cost content. Measured live: forcing language="de"
# collapsed a stretch with real dialogue down to bare punctuation, which
# _is_confirmed_silence correctly read as "nothing here" for THAT decode,
# while an auto-detected pass over the EXACT SAME audio recovered real
# text. Fix: before trusting a confirmed-silence verdict under a pin,
# retry the SAME cut once with language=None; only if that ALSO comes back
# confirmed-silent is the span treated as genuinely non-speech.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_coverage_retries_unpinned_when_pinned_decode_confirms_silence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(transcribe, "_PREFIX_DISTRUST_SECONDS", 0.0)
    monkeypatch.setattr(transcribe, "_RETRY_BACKOFF_SECONDS", 0.0)

    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        return src_path.parent / "retry.opus"

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)

    calls: list[str | None] = []

    async def fake_post(_path: Path, *, language: str | None = None) -> dict:
        calls.append(language)
        if language == "de":
            # The PINNED decode collapses to bare punctuation.
            return _payload([{"start": 0.0, "end": 1.0, "text": "."}])
        # The auto-detected retry recovers real text for the SAME audio.
        return _payload([{"start": 0.0, "end": 30.0, "text": "Ich komme gleich!"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    segments = [{"start": 0.0, "end": 10.0, "text": "intro"}]
    diagnostics = transcribe.TranscribeDiagnostics()
    result_segments, missing = await transcribe._ensure_coverage(
        segments,
        source_path=tmp_path / "a.opus",
        window_duration=40.0,
        per_job_lock=asyncio.Semaphore(1),
        diagnostics=diagnostics,
        unit_label="whole",
        language="de",
    )

    # Pinned attempt first, THEN the unpinned retry of the SAME cut.
    assert calls == ["de", None]
    texts = [s["text"] for s in result_segments]
    assert "Ich komme gleich!" in texts
    assert "." not in texts
    assert len(diagnostics.coverage_rechecks) == 1
    assert diagnostics.coverage_rechecks[0]["verdict"] == "recovered_unpinned"
    assert missing == 0.0  # fully recovered — nothing left uncertain


@pytest.mark.asyncio
async def test_ensure_coverage_confirms_silence_both_ways_when_unpinned_retry_also_silent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When even the unpinned retry comes back empty/punctuation-only, the
    span really is confirmed non-speech — not a pin artifact."""
    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        return src_path.parent / "retry.opus"

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)

    calls: list[str | None] = []

    async def fake_post(_path: Path, *, language: str | None = None) -> dict:
        calls.append(language)
        return _payload([{"start": 0.0, "end": 1.0, "text": "*music*"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    segments = [{"start": 0.0, "end": 10.0, "text": "intro"}]
    diagnostics = transcribe.TranscribeDiagnostics()
    _result_segments, missing = await transcribe._ensure_coverage(
        segments,
        source_path=tmp_path / "a.opus",
        window_duration=40.0,
        per_job_lock=asyncio.Semaphore(1),
        diagnostics=diagnostics,
        unit_label="whole",
        language="de",
    )

    assert calls == ["de", None]  # both decodes attempted
    assert missing == 0.0  # confirmed non-speech, not lost content
    assert len(diagnostics.coverage_rechecks) == 1
    assert diagnostics.coverage_rechecks[0]["verdict"] == "confirmed_non_speech_both_ways"


@pytest.mark.asyncio
async def test_ensure_coverage_does_not_retry_unpinned_when_no_language_pinned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Re-asking with language=None when NOTHING was pinned would just be
    an identical duplicate request — must never happen."""
    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        return src_path.parent / "retry.opus"

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)

    calls: list[str | None] = []

    async def fake_post(_path: Path, *, language: str | None = None) -> dict:
        calls.append(language)
        return _payload([{"start": 0.0, "end": 1.0, "text": "*music*"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    segments = [{"start": 0.0, "end": 10.0, "text": "intro"}]
    diagnostics = transcribe.TranscribeDiagnostics()
    await transcribe._ensure_coverage(
        segments,
        source_path=tmp_path / "a.opus",
        window_duration=40.0,
        per_job_lock=asyncio.Semaphore(1),
        diagnostics=diagnostics,
        unit_label="whole",
        language=None,
    )

    assert calls == [None]  # exactly one call — no unpinned retry attempted
    assert diagnostics.coverage_rechecks[0]["verdict"] == "confirmed_non_speech"


@pytest.mark.asyncio
async def test_unpinned_retry_does_not_consume_extra_recheck_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The unpinned retry verifies a recheck slot's OWN verdict — it does
    not spend an additional one. With budget=2 and TWO separate windows
    that each need an unpinned retry, both still get fully resolved (4
    Whisper calls total, but only 2 recheck slots) — if the retry
    incorrectly cost a second slot, only one window would ever be
    reached."""
    cfg = transcribe.get_config()
    monkeypatch.setattr(cfg.whisper, "max_coverage_rechecks", 2)
    monkeypatch.setattr(transcribe, "get_config", lambda: cfg)
    monkeypatch.setattr(transcribe, "_PREFIX_DISTRUST_SECONDS", 0.0)
    monkeypatch.setattr(transcribe, "_RETRY_BACKOFF_SECONDS", 0.0)

    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        return src_path.parent / "retry.opus"

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)

    calls: list[str | None] = []

    async def fake_post(_path: Path, *, language: str | None = None) -> dict:
        calls.append(language)
        if language == "de":
            return _payload([{"start": 0.0, "end": 1.0, "text": "."}])
        return _payload([{"start": 0.0, "end": 20.0, "text": "recovered"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    # Two widely-separated real gaps, each well above the cost cutoff.
    segments = [
        {"start": 0.0, "end": 5.0, "text": "block one"},
        {"start": 25.0, "end": 30.0, "text": "block two"},
    ]
    diagnostics = transcribe.TranscribeDiagnostics()
    await transcribe._ensure_coverage(
        segments,
        source_path=tmp_path / "a.opus",
        window_duration=50.0,
        per_job_lock=asyncio.Semaphore(1),
        diagnostics=diagnostics,
        unit_label="whole",
        language="de",
    )

    # 2 recheck SLOTS (matches the budget), each costing 2 Whisper calls
    # (the pinned attempt + its unpinned retry) = 4 calls total.
    assert len(calls) == 4
    assert len(diagnostics.coverage_rechecks) == 2
    assert all(r["verdict"] == "recovered_unpinned" for r in diagnostics.coverage_rechecks)
