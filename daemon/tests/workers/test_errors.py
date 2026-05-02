"""Tests for workers.errors — code attribute matches DeferredReason values."""

from __future__ import annotations

from src.api.schemas import DeferredReason
from src.workers.errors import (
    ExhaustedRetriesError,
    NetworkTranscriptError,
    PermanentTranscriptError,
    TranscriptError,
    TransientTranscriptError,
)


def test_codes_match_deferred_reason_enum() -> None:
    # Each error class's `code` is one of DeferredReason's values.
    valid = {r.value for r in DeferredReason}
    for cls in (PermanentTranscriptError, TransientTranscriptError, NetworkTranscriptError):
        assert cls.code in valid, f"{cls.__name__}.code={cls.code!r} not a DeferredReason"


def test_permanent_code() -> None:
    err = PermanentTranscriptError("disabled")
    assert err.code == DeferredReason.TRANSCRIPT_UNAVAILABLE.value


def test_transient_code() -> None:
    err = TransientTranscriptError("blocked")
    assert err.code == DeferredReason.TRANSCRIPT_BLOCKED.value


def test_network_code() -> None:
    err = NetworkTranscriptError("timeout")
    assert err.code == DeferredReason.NETWORK_ERROR.value


def test_exhausted_retries_inherits_code_from_wrapped() -> None:
    last = TransientTranscriptError("rate limited")
    wrapped = ExhaustedRetriesError(last)
    assert wrapped.code == last.code
    assert wrapped.code == DeferredReason.TRANSCRIPT_BLOCKED.value


def test_exhausted_retries_inherits_code_from_network() -> None:
    last = NetworkTranscriptError("conn refused")
    wrapped = ExhaustedRetriesError(last)
    assert wrapped.code == DeferredReason.NETWORK_ERROR.value


def test_inheritance_hierarchy() -> None:
    # All concrete classes share TranscriptError as a base.
    assert issubclass(PermanentTranscriptError, TranscriptError)
    assert issubclass(TransientTranscriptError, TranscriptError)
    assert issubclass(NetworkTranscriptError, TranscriptError)
    assert issubclass(ExhaustedRetriesError, TranscriptError)
