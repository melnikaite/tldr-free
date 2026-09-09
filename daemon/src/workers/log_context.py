"""``job_id`` on every worker log line, via contextvars + a logging filter.

The problem this fixes: worker log lines (``transcribe:``, ``runner:``, …)
never carried any indication of WHICH job produced them. Diagnosing a bad
Whisper result meant reading ``<data_dir>/logs/daemon.log`` by hand and
correlating lines to a job by wall-clock timestamp — something only a
developer sitting at the machine could do, and something rotation
eventually makes impossible anyway (see ``storage.repo``'s per-job
diagnostic record, migration v10, for the durable half of this fix).

Deliberately NOT a parameter threaded through every worker function
signature — that would touch dozens of call sites across
``workers/pipeline.py``, ``workers/runner.py``, ``workers/transcribe.py``,
for a value that's the same for the whole duration of processing one job.
A ``contextvars.ContextVar`` set ONCE where a job starts being processed
(``pipeline.run_pipeline``, ``runner._process_one``) is instead picked up
automatically by every ``logging`` call made anywhere below that point in
the same task — including across ``await`` boundaries and
``asyncio.to_thread`` calls (contextvars propagate into a thread started
via ``to_thread`` because it runs inside a copy of the calling context).

Mechanism: ``JobIdLogFilter`` is attached to the daemon's own rotating file
handler (``logging_setup.configure_logging``), not to any individual
logger — a filter attached to a HANDLER sees every record that reaches
that handler regardless of which logger emitted it (including ones that
merely propagate up from a child logger), which is what makes this work
uniformly for ``src.workers.*``, ``src.llm.*``, etc. without listing every
module by name.

``get_job_id()`` defaults to ``None`` when nothing has set it (e.g. a
request-handling code path outside any job's processing) — the filter
renders that as ``"-"`` so the log line's shape stays fixed-width and
grep-able either way.
"""

from __future__ import annotations

import contextvars
import logging

_current_job_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tldr_current_job_id", default=None
)


def set_job_id(job_id: str | None) -> contextvars.Token[str | None]:
    """Bind ``job_id`` as "currently being processed" for the calling task.
    Returns a token — pass it to ``reset_job_id`` in a ``finally`` block so
    the binding never leaks past the job it was set for (e.g. into the next
    iteration of a persistent worker loop like ``runner.whisper_worker``)."""
    return _current_job_id.set(job_id)


def reset_job_id(token: contextvars.Token[str | None]) -> None:
    """Undo a prior ``set_job_id`` call. Always call this in a ``finally``,
    matched with the token ``set_job_id`` returned — see its docstring."""
    _current_job_id.reset(token)


def get_job_id() -> str | None:
    """The job id bound by the nearest enclosing ``set_job_id`` call in this
    task's context, or ``None`` if nothing has bound one."""
    return _current_job_id.get()


class JobIdLogFilter(logging.Filter):
    """Stamps ``record.job_id`` from the current contextvar onto every
    record that reaches the handler this is attached to. Always returns
    True (never actually filters anything out) — its only job is to
    annotate, matching ``logging.Filter``'s documented use as a mutate-and-
    keep hook, same pattern as ``logging_setup._AccessLogDropQueryStringFilter``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = _current_job_id.get() or "-"
        return True


__all__ = ["JobIdLogFilter", "get_job_id", "reset_job_id", "set_job_id"]
