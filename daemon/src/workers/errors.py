"""Domain error types for the transcript / transcription pipeline.

Class hierarchy:

    TranscriptError                         # base
        PermanentTranscriptError            # disabled / not found / private → defer to whisper
        TransientTranscriptError            # blocked / 429 → retry, then defer
        NetworkTranscriptError              # connection / timeout → retry, then defer
        ExhaustedRetriesError               # tenacity gave up; wraps the last attempt error

Each subclass carries a ``code`` attribute that matches one of the
``api.schemas.DeferredReason`` string values
("transcript_unavailable", "transcript_blocked", "network_error") so the
HTTP layer can echo it back without a translation table.
"""

from __future__ import annotations


class TranscriptError(Exception):
    """Base class for the transcript pipeline. ``code`` matches DeferredReason."""

    code: str = "transcript_unavailable"


class PermanentTranscriptError(TranscriptError):
    """Permanent error: transcript disabled, not found, video unavailable, age-restricted.

    The fast path immediately gives up and the job is deferred to Whisper.
    """

    code: str = "transcript_unavailable"


class TransientTranscriptError(TranscriptError):
    """Transient error: rate limited / IP blocked. Retry with backoff, then defer."""

    code: str = "transcript_blocked"


class NetworkTranscriptError(TranscriptError):
    """Transient network error: connection refused / timeout / DNS. Retry with backoff."""

    code: str = "network_error"


class ExhaustedRetriesError(TranscriptError):
    """Raised after tenacity gives up. Inherits ``code`` from the wrapped last attempt."""

    def __init__(self, last: TranscriptError) -> None:
        super().__init__(str(last))
        # Carry the wrapped error's code forward so the API layer can keep using
        # ``DeferredReason(err.code)`` uniformly.
        self.code = last.code
        self.last = last


__all__ = [
    "ExhaustedRetriesError",
    "NetworkTranscriptError",
    "PermanentTranscriptError",
    "TranscriptError",
    "TransientTranscriptError",
]
