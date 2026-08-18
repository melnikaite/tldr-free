"""Tests for src/logging_setup.py: rotation, the container/native split, and
the one-time legacy-launchd-log truncation.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from src import logging_setup
from src.config import validate_full_config


def _cfg(tmp_path: Path):
    return validate_full_config({"storage": {"data_dir": str(tmp_path)}})


def _cleanup_managed_handlers() -> None:
    """Undo configure_logging()'s handler surgery so one test can't leak
    file handlers (and their open fds) into the next."""
    for name in logging_setup._LOGGER_NAMES:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            if getattr(handler, logging_setup._MANAGED_MARKER, False):
                logger.removeHandler(handler)
                handler.close()

    access_logger = logging.getLogger("uvicorn.access")
    for f in list(access_logger.filters):
        if isinstance(f, logging_setup._AccessLogDropQueryStringFilter):
            access_logger.removeFilter(f)


@pytest.fixture(autouse=True)
def _restore_logging() -> None:
    yield
    _cleanup_managed_handlers()


def test_configure_logging_creates_log_file(tmp_path: Path) -> None:
    config = _cfg(tmp_path)
    logging_setup.configure_logging(config)
    logging.getLogger("src.somewhere").info("hello world")
    for h in logging.getLogger().handlers:
        h.flush()

    log_path = logging_setup.log_file_path(config)
    assert log_path.is_file()
    assert "hello world" in log_path.read_text()


def test_configure_logging_rotates_past_max_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(logging_setup, "LOG_MAX_BYTES", 500)
    monkeypatch.setattr(logging_setup, "LOG_BACKUP_COUNT", 2)
    config = _cfg(tmp_path)
    logging_setup.configure_logging(config)

    logger = logging.getLogger("src.rotation_test")
    for i in range(200):
        logger.info("padding line number %d to exceed the tiny rotation cap", i)
    for h in logging.getLogger().handlers:
        h.flush()

    log_path = logging_setup.log_file_path(config)
    assert log_path.is_file()
    assert log_path.stat().st_size <= 500 * 2  # generous slack for the last write
    assert (log_path.parent / "daemon.log.1").is_file()


def test_configure_logging_is_idempotent_no_handler_leak(tmp_path: Path) -> None:
    config = _cfg(tmp_path)
    for _ in range(5):
        logging_setup.configure_logging(config)

    for name in logging_setup._LOGGER_NAMES:
        logger = logging.getLogger(name)
        managed = [
            h for h in logger.handlers if getattr(h, logging_setup._MANAGED_MARKER, False)
        ]
        assert len(managed) == 1, f"{name!r} accumulated {len(managed)} managed handlers"


def test_configure_logging_native_detaches_stdio_stream_handlers(tmp_path: Path) -> None:
    config = _cfg(tmp_path)
    root = logging.getLogger()
    stream_handler = logging.StreamHandler(sys.stderr)
    root.addHandler(stream_handler)
    logging_setup.configure_logging(config)
    # configure_logging() itself removed it (native mode) — nothing left to
    # clean up here beyond the autouse fixture's managed-handler sweep.
    assert stream_handler not in root.handlers


def test_configure_logging_container_mode_keeps_stdio_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(logging_setup, "_is_container", lambda: True)
    config = _cfg(tmp_path)
    root = logging.getLogger()
    stream_handler = logging.StreamHandler(sys.stderr)
    root.addHandler(stream_handler)
    try:
        logging_setup.configure_logging(config)
        assert stream_handler in root.handlers
    finally:
        root.removeHandler(stream_handler)


def test_truncate_legacy_launchd_logs_once(tmp_path: Path) -> None:
    config = _cfg(tmp_path)
    out_log = tmp_path / "daemon.out.log"
    err_log = tmp_path / "daemon.err.log"
    out_log.write_bytes(b"x" * 10_000)
    err_log.write_bytes(b"y" * 5_000)

    logging_setup.truncate_legacy_launchd_logs_once(config)

    assert out_log.stat().st_size == 0
    assert err_log.stat().st_size == 0

    # Simulate the file growing again after truncation (e.g. a fresh crash
    # traceback written straight to stderr) — a SECOND call must NOT wipe
    # it, since that's the one thing left for these files to catch.
    out_log.write_bytes(b"fresh crash traceback")
    logging_setup.truncate_legacy_launchd_logs_once(config)
    assert out_log.read_bytes() == b"fresh crash traceback"


def test_truncate_legacy_launchd_logs_noop_in_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(logging_setup, "_is_container", lambda: True)
    config = _cfg(tmp_path)
    out_log = tmp_path / "daemon.out.log"
    out_log.write_bytes(b"x" * 10_000)

    logging_setup.truncate_legacy_launchd_logs_once(config)

    assert out_log.stat().st_size == 10_000


def test_configure_logging_drops_query_string_from_access_log(tmp_path: Path) -> None:
    """Regression test for the live-daemon leak: `uvicorn.access` logs the
    full request target INCLUDING the query string, which for this daemon
    routinely IS the page/video URL the user is looking at
    (`GET /jobs?url=<page>`). This is the PRIMARY fix — the data must never
    be written at all, not merely scrubbed back out of a report later (see
    `tests/test_diagnostics.py` for the scrub-side belt-and-suspenders
    tests, which cover logs that predate this fix)."""
    config = _cfg(tmp_path)
    logging_setup.configure_logging(config)

    access_logger = logging.getLogger("uvicorn.access")
    access_logger.info(
        '%s - "%s %s HTTP/%s" %d',
        "127.0.0.1:65404",
        "GET",
        "/jobs?url=https://example.com/a-private-page&limit=1",
        "1.1",
        200,
    )
    for h in logging.getLogger().handlers:
        h.flush()

    content = logging_setup.log_file_path(config).read_text()
    assert "example.com" not in content
    assert "?" not in content
    assert "/jobs" in content
    assert "200" in content


def test_configure_logging_leaves_non_access_loggers_alone(tmp_path: Path) -> None:
    """The query-string filter is scoped to `uvicorn.access` only — a
    regular app log line that happens to contain a literal "?" must not be
    mangled by it."""
    config = _cfg(tmp_path)
    logging_setup.configure_logging(config)

    logging.getLogger("src.somewhere").info("did this work? yes")
    for h in logging.getLogger().handlers:
        h.flush()

    content = logging_setup.log_file_path(config).read_text()
    assert "did this work? yes" in content
