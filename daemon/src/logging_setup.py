"""Where the daemon's own logs go, and how they're kept from growing forever.

Background (see ``.claude/runbook.md`` for the user-facing version): under
launchd (native/uv install), ``dev.tldr.daemon.plist`` sets
``StandardOutPath``/``StandardErrorPath`` to plain files launchd itself opens
and holds a file descriptor to (``daemon.out.log``/``daemon.err.log`` next to
the SQLite DB). Nothing in this process can rotate those — renaming the file
out from under launchd's fd just means it keeps writing to the now-unlinked
inode, invisibly, forever. Left alone they grow without bound (observed:
8+ MB / 500+ KB with zero traffic filtering) and — worse — accumulate whatever
the app logs, including full URLs of processed pages/videos.

The fix: give the daemon its OWN rotating log file, driven by the stdlib
``logging`` module (which is what both ``src.main``'s own ``log.info(...)``
calls and uvicorn's access/error logging go through), and — on a native
install only — detach the root/uvicorn loggers from stdout/stderr entirely
so launchd's files stop receiving anything logging-based. What's left
flowing to stdout/stderr after that is exactly what DOESN'T go through
`logging`: an interpreter traceback on a hard crash, a stray ``print()``.
That's a fine, tiny thing for those files to keep catching — it's the one
case the file handler below can't help with anyway (the process is dying).

Docker is different on purpose: ``docker compose logs`` / ``task logs`` is
how this project's own runbook says to tail the daemon, and `docker`'s log
driver already rotates the container's stdout. So in a container we leave
uvicorn's/`logging`'s stdout/stderr handlers exactly as they were — nothing
here is allowed to break that path — and ADD the rotating file on top
(purely additive) so ``GET /diagnostics`` still has a log tail to read
inside the container's own data volume.

One more thing this module enforces regardless of Docker vs native:
``uvicorn.access`` logs the full request target INCLUDING the query
string (``uvicorn.protocols.utils.get_path_with_query_string`` —
``path + "?" + query_string``). For this daemon that query string routinely
IS the page/video URL the user is looking at (``GET /jobs?url=<page>``) —
an access log with query strings left in is a browsing-history log, not a
diagnostic one, and no diagnostic question this daemon needs to answer
("was this endpoint hit, did it 200") needs more than the path. So
``configure_logging`` also installs a ``logging.Filter`` on the
``uvicorn.access`` logger that drops everything from ``?`` onward in the
request-target argument BEFORE it's formatted — this runs unconditionally
(container or native), because it's a data-minimization policy, not a
Docker/native rotation concern: it's just as wrong for a query string to
sit in `docker compose logs` as in the native rotating file. This is the
PRIMARY fix for the same problem ``GET /diagnostics``'s log-tail scrubbing
(``src/api/diagnostics.py``) exists for — data that's never written can't
leak from a copy-pasted log file either, which is why this is preferred
over relying on the scrub alone.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src import paths
from src.config import Config

log = logging.getLogger(__name__)

# 5 MB * 4 backups = 20 MB ceiling for the daemon's own log directory. Small
# enough not to matter on any modern disk, generous enough to cover a couple
# of days of a single-user, single-machine daemon.
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 4

# Marker attribute stamped on every handler this module installs, so a
# repeat call (the FastAPI lifespan re-runs `configure_logging` once per
# process start in production, and once per `with TestClient(app)` block in
# tests — see test_logging_setup.py) can find-and-replace its OWN handlers
# instead of stacking duplicates (which would leak file descriptors and
# double-log everything under pytest, where the app starts up repeatedly in
# one process).
_MANAGED_MARKER = "_tldr_managed_handler"

_LOGGER_NAMES = ("", "uvicorn", "uvicorn.error", "uvicorn.access")


def log_dir(config: Config) -> Path:
    return Path(config.storage.data_dir) / "logs"


def log_file_path(config: Config) -> Path:
    return log_dir(config) / "daemon.log"


def _is_container() -> bool:
    """Structural container detection — same signal ``src/paths.py`` already
    uses (compose mounts ``/data``), so this module doesn't need its own env
    var or heuristic."""
    return paths.CONTAINER_DATA.is_dir()


def _is_stdlib_stream_handler(handler: logging.Handler) -> bool:
    """A plain ``logging.StreamHandler`` writing to stdout/stderr — the kind
    ``logging.basicConfig()`` and uvicorn's default logging config install.

    Deliberately NOT ``isinstance(handler, logging.StreamHandler)`` — both
    ``logging.FileHandler`` and our own ``RotatingFileHandler`` subclass
    ``StreamHandler``, and we must never sweep those up here (that would
    strip a previous call's own file handler, or — if some other part of
    the process ever attaches a FileHandler for its own reasons — that
    handler too). ``type(...) is StreamHandler`` (exact type, not
    inheritance) plus a stream check keeps this scoped to exactly the
    stdout/stderr handlers we mean to detach in native mode.
    """
    return type(handler) is logging.StreamHandler and getattr(handler, "stream", None) in (
        sys.stdout,
        sys.stderr,
    )


class _AccessLogDropQueryStringFilter(logging.Filter):
    """Strips everything from ``?`` onward out of ``uvicorn.access``'s
    request-target argument, in place, before the record is formatted by
    ANY handler. See the module docstring for why this exists and why it's
    unconditional (not gated on Docker vs native).

    Mechanism: uvicorn logs each access line as
    ``access_logger.info('%s - "%s %s HTTP/%s" %d', client_addr, method,
    path_with_query_string, http_version, status_code)`` — i.e. the
    request-target (index 2 of the five positional args) is handed to
    ``logging`` as a lazy ``%``-format argument, not a pre-rendered string.
    Mutating ``record.args`` here (a `logging.Filter` is explicitly allowed
    to mutate the record, not just decide whether to keep it) rewrites the
    argument BEFORE any handler's formatter ever sees it, so the query
    string never reaches a rendered string at all — this is the actual
    "don't write it" fix, not a post-hoc scrub of already-formatted text.
    Attached to the LOGGER (not a handler), so it applies to every handler
    that logger has, including uvicorn's own stdout access handler in a
    Docker/additive setup.

    Deliberately narrow: only touches ``uvicorn.access``, and only its
    known 5-arg shape (falls through untouched if uvicorn ever changes
    that, rather than guessing at a different shape).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) == 5:
            path_with_query = args[2]
            if isinstance(path_with_query, str) and "?" in path_with_query:
                new_args = list(args)
                new_args[2] = path_with_query.split("?", 1)[0]
                record.args = tuple(new_args)
        return True


def configure_logging(config: Config) -> None:
    """Point app + uvicorn logging at a rotating file, per the module
    docstring's Docker-vs-native split. Safe to call more than once (see
    ``_MANAGED_MARKER``) — each call fully replaces this module's own
    handlers rather than adding to them.
    """
    level = getattr(logging, config.logging.level.upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s")

    directory = log_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_file_path(config), maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
    )
    file_handler.setFormatter(formatter)
    setattr(file_handler, _MANAGED_MARKER, True)

    in_container = _is_container()

    for name in _LOGGER_NAMES:
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Drop any handler a PREVIOUS call to this function installed, so
        # repeat calls (see docstring) don't stack duplicate file handlers /
        # leak file descriptors.
        for handler in list(logger.handlers):
            if getattr(handler, _MANAGED_MARKER, False):
                logger.removeHandler(handler)
                handler.close()
            elif not in_container and _is_stdlib_stream_handler(handler):
                # Native install: detach the stdout/stderr handler that
                # `logging.basicConfig()` (root) or uvicorn's own default
                # logging config (uvicorn/uvicorn.error/uvicorn.access)
                # installed, so launchd's daemon.out.log/daemon.err.log
                # stop receiving anything that goes through `logging` —
                # only the rotating file below does from now on.
                logger.removeHandler(handler)

        logger.addHandler(file_handler)

    access_logger = logging.getLogger("uvicorn.access")
    for existing in list(access_logger.filters):
        if isinstance(existing, _AccessLogDropQueryStringFilter):
            access_logger.removeFilter(existing)
    access_logger.addFilter(_AccessLogDropQueryStringFilter())


# ---------------------------------------------------------------------------
# One-time cleanup of the pre-existing, unbounded launchd log files.
# ---------------------------------------------------------------------------

# Names launchd is configured to write to — see src/service.py's
# launchd_plist(), which points StandardOutPath/StandardErrorPath at these
# two files directly under storage.data_dir (NOT the new logs/ subdirectory
# above, so the two never collide).
_LEGACY_LOG_NAMES = ("daemon.out.log", "daemon.err.log")

# Written next to the new log directory once the legacy files have been
# truncated, so this only ever happens once per install — see
# `truncate_legacy_launchd_logs_once`'s docstring for why a repeat truncation
# would be actively harmful.
_LEGACY_TRUNCATED_MARKER = ".legacy_logs_truncated"


def truncate_legacy_launchd_logs_once(config: Config) -> None:
    """Native installs only: truncate the old unbounded
    ``daemon.{out,err}.log`` launchd was writing to before this module
    existed, so an existing install doesn't keep carrying an 8+ MB file
    around indefinitely with no way to shrink it.

    Called ONLY from ``cli.py``'s ``_serve()`` (the real ``tldr-daemon``
    entrypoint) — deliberately NOT from ``src.main``'s FastAPI ``lifespan``,
    even though that's where ``configure_logging`` above lives. A lifespan
    runs every time something starts the app, including
    ``with TestClient(app)`` in the test suite — that's the whole point of
    a lifespan hook, and tests are RIGHT to exercise it. But this function
    reaches outside the app's own data (real files at ``storage.data_dir``
    that may not even be the test's own tmp dir if a test forgot to
    override it) — a one-time migration like this belongs at the "real
    native daemon is starting for the first time on this machine" layer,
    not the "an ASGI app object was constructed" layer. Keeping it here
    instead of inline in ``cli.py`` is only for the constants/sentinel
    logic to live next to ``configure_logging``; the CALL SITE is what
    makes this test-safe by construction, not careful fixture hygiene (see
    ``tests/test_main_lifespan.py`` and ``.claude/ops.md``).

    Truncate-in-place (``os.truncate(path, 0)``), never delete: launchd
    opened these files with ``O_APPEND`` and holds that fd for the whole
    life of the LaunchAgent. Truncating the underlying file is safe under
    O_APPEND (every write still seeks to EOF first, so it just resumes
    appending to a now-empty file) — deleting it would instead orphan
    launchd's fd on an unlinked, invisible inode that keeps consuming disk
    until the process is restarted, which defeats the entire point.

    No backup copy is kept: the whole reason this file needs to shrink is
    that it accumulated privacy-sensitive content (processed URLs — see
    ``.claude/runbook.md``) with no rotation; copying that content
    somewhere else before deleting it would just relocate the same problem.

    Guarded by a sentinel file so this runs AT MOST ONCE per install. Without
    that guard, truncating on every daemon start would keep destroying the
    one thing stdout/stderr are left to catch post-rotation: a crash
    traceback from the previous run. A user stuck in a crash loop needs that
    traceback to survive at least until the NEXT start, not to have it wiped
    out from under them before they get to read it.
    """
    if _is_container():
        return  # Docker doesn't use launchd; nothing to clean up.

    directory = log_dir(config)
    marker = directory / _LEGACY_TRUNCATED_MARKER
    if marker.exists():
        return

    data_dir = Path(config.storage.data_dir)
    for name in _LEGACY_LOG_NAMES:
        path = data_dir / name
        if path.is_file():
            try:
                os.truncate(path, 0)
            except OSError:
                log.warning("logging_setup: failed to truncate legacy log %s", path, exc_info=True)

    directory.mkdir(parents=True, exist_ok=True)
    marker.write_text("")
