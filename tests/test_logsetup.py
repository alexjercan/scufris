"""Tests for the central logging configuration."""

from __future__ import annotations

import logging

from scufris.logsetup import (
    _RequestIdFilter,
    configure_logging,
    new_request_id,
    set_request_id,
    truncate,
)


def test_configure_logging_sets_the_level() -> None:
    configure_logging("WARNING", force=True)
    assert logging.getLogger("scufris").level == logging.WARNING
    configure_logging("DEBUG", force=True)
    assert logging.getLogger("scufris").level == logging.DEBUG
    assert logging.getLogger("uvicorn.access").level == logging.DEBUG


def test_unforced_configure_is_a_noop_after_first() -> None:
    configure_logging("DEBUG", force=True)  # configured + DEBUG
    configure_logging("ERROR")  # un-forced -> ignored
    assert logging.getLogger("scufris").level == logging.DEBUG


def test_bad_level_falls_back_to_info() -> None:
    configure_logging("NONSENSE", force=True)
    assert logging.getLogger("scufris").level == logging.INFO


def test_request_id_filter_attaches_req() -> None:
    set_request_id("abc12345")
    record = logging.LogRecord("n", logging.INFO, "f", 1, "m", None, None)
    _RequestIdFilter().filter(record)
    assert record.req == " [abc12345]"
    set_request_id("")
    _RequestIdFilter().filter(record)
    assert record.req == ""


def test_new_request_id_is_short_and_unique() -> None:
    a, b = new_request_id(), new_request_id()
    assert len(a) == 8
    assert a != b


def test_truncate_bounds_long_text() -> None:
    assert truncate("short", 10) == "short"
    out = truncate("x" * 300, 100)
    assert out.startswith("x" * 100)
    assert "(+200 chars)" in out
