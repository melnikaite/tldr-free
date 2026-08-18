"""Tests for workers.transcribe — single-shot vs chunked transcription."""

from __future__ import annotations

from pathlib import Path

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

    async def fake_post(path: Path) -> dict:
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

    async def fake_post(path: Path) -> dict:
        if path == c0:
            return _payload([{"start": 5.0, "end": 10.0, "text": "first"}], "ru")
        return _payload([{"start": 3.0, "end": 8.0, "text": "second"}], "en")

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result = await transcribe._transcribe_chunked(
        audio, total_duration=1200.0, max_bytes=1024
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

    async def fake_post(path: Path) -> dict:
        return _payload([{"start": 0.0, "end": 1.0, "text": "whole"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)
    monkeypatch.setattr(
        transcribe, "_split_audio", lambda *a, **k: pytest.fail("should not split")
    )

    result = await transcribe._transcribe_chunked(
        audio, total_duration=None, max_bytes=1024
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
    """The single-request path collapses a Whisper repetition-loop run."""
    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x" * 1024)  # under the cap → single-request path

    loop_text = "I'm not sure if I'm doing that right."
    segments = [
        {"start": float(i), "end": float(i + 1), "text": loop_text} for i in range(10)
    ]

    async def fake_post(_path: Path) -> dict:
        return _payload(segments)

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result = await transcribe.transcribe_audio(audio, total_duration=10.0)
    assert result.segments == [segments[0]]


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

    async def fake_post(path: Path) -> dict:
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

    async def fake_post(_path: Path) -> dict:
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

    async def fake_post(path: Path) -> dict:
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
            audio, total_duration=40.0, max_bytes=1024
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

    async def fake_post(path: Path) -> dict:
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
        audio, total_duration=40.0, max_bytes=1024
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

    async def fake_post(_path: Path) -> dict:
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

    async def fake_post(_path: Path) -> dict:
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

    async def fake_post(path: Path) -> dict:
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

    async def fake_post(_path: Path) -> dict:
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
        segments, source_path=audio, window_duration=window_duration
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

    async def fake_post(_path: Path) -> dict:
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
        segments, source_path=audio, window_duration=window_duration
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

    async def fake_post(_path: Path) -> dict:
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
        segments, source_path=audio, window_duration=window_duration
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

    async def fake_post(_path: Path) -> dict:
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
        segments, source_path=audio, window_duration=window_duration
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

    async def fake_post(_path: Path) -> dict:
        return _payload([{"start": 0.0, "end": 30.0, "text": "*Dramatic music*"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    segments = [{"start": 0.0, "end": 10.0, "text": "intro speech"}]
    result_segments, missing = await transcribe._ensure_coverage(
        segments, source_path=audio, window_duration=40.0
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

    async def fake_post(_path: Path) -> dict:
        return _payload(
            [{"start": 0.0, "end": 30.0, "text": "Hello, this is recovered content."}]
        )

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    segments = [{"start": 0.0, "end": 10.0, "text": "intro speech"}]
    result_segments, missing = await transcribe._ensure_coverage(
        segments, source_path=audio, window_duration=40.0
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

    async def fake_post(_path: Path) -> dict:
        loop_text = "I'm not sure if I'm doing that right."
        return _payload(
            [{"start": float(i), "end": float(i + 1), "text": loop_text} for i in range(6)]
        )

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    segments = [{"start": 0.0, "end": 10.0, "text": "intro speech"}]
    with caplog.at_level("INFO", logger="src.workers.transcribe"):
        result_segments, missing = await transcribe._ensure_coverage(
            segments, source_path=audio, window_duration=40.0
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
    monkeypatch.setattr(transcribe, "_MAX_COVERAGE_RECHECKS", 1)

    audio = tmp_path / "a.opus"
    audio.write_bytes(b"x")

    def fake_cut(src_path: Path, start: float, duration: float) -> Path:
        return src_path.parent / "retry.opus"

    monkeypatch.setattr(transcribe, "_cut_audio_segment", fake_cut)

    async def fake_post(_path: Path) -> dict:
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
        segments, source_path=audio, window_duration=40.0
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
    monkeypatch.setattr(transcribe, "_MAX_COVERAGE_RECHECKS", 2)

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

    async def fake_post(_path: Path) -> dict:
        # Non-speech, so nothing gets spliced and the loop can only ever
        # move on to a DIFFERENT window, never re-ask this one.
        return _payload([{"start": 0.0, "end": 1.0, "text": "*noise*"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    _result_segments, _missing = await transcribe._ensure_coverage(
        segments, source_path=audio, window_duration=window_duration
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

    async def fake_post(_path: Path) -> dict:
        # Confirmed non-speech every time, so each slice resolves in
        # exactly one recheck — isolating the SLICING behaviour from the
        # classification logic already covered by other tests.
        return _payload([{"start": 0.0, "end": 1.0, "text": "*noise*"}])

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    segments = [{"start": 0.0, "end": 10.0, "text": "intro speech"}]
    result_segments, missing = await transcribe._ensure_coverage(
        segments, source_path=audio, window_duration=600.0
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

    async def fake_post(_path: Path) -> dict:
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

    async def fake_post(_path: Path) -> dict:
        return _payload(
            [
                {"start": 2.0, "end": 3.4, "text": "Mann, das ist Luna Gröner."},
                {"start": 3.5, "end": 5.8, "text": "Tochter unserer scheiß Vermieterin."},
                {"start": 5.9, "end": 22.0, "text": "filled gap content"},
            ]
        )

    monkeypatch.setattr(transcribe, "_post_audio", fake_post)

    result_segments, missing = await transcribe._ensure_coverage(
        segments, source_path=audio, window_duration=window_duration
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
