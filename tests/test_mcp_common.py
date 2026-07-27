"""Tests for the shared MCP helpers (scufris.mcp_common).

``_run`` is the curated-command shell wrapper every server shares; the HTTP
bridge ``_api_call`` is exercised through the control tools (test_mcp_server) and
the callback tools (test_agent_mcp_server) that use it.
"""

from __future__ import annotations

import sys

import pytest

from scufris.mcp_common import _run


def test_run_reports_missing_binary() -> None:
    assert "not found on PATH" in _run(["scufris-no-such-binary-xyz"])


def test_run_captures_stdout() -> None:
    assert _run([sys.executable, "-c", "print('hi', end='')"]) == "hi"


def test_run_reports_nonzero_exit() -> None:
    out = _run(
        [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(2)"]
    )
    assert "boom" in out


def test_run_logs_the_command(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.DEBUG, logger="scufris.mcp_common"):
        _run([sys.executable, "-c", "print('hi', end='')"])
    assert any("exit=0" in record.getMessage() for record in caplog.records)
