"""Central logging configuration for scufris.

Named ``logsetup`` (not ``logging``) so it does not shadow the stdlib module that
every scufris module imports. Provides one entry point, ``configure_logging``,
plus a per-request id that rides a ``contextvars`` value into every log line, and
a ``truncate`` redaction helper for logging bounded/safe values.

The default is INFO; ``--debug``/``SCUFRIS_LOG_LEVEL=DEBUG`` turns on the verbose
in-depth logs (subprocess argv, per-request lines, streamed events, ...).
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

_LOG_FORMAT = "%(asctime)s %(levelname)-5s %(name)s%(req)s %(message)s"
_DATE_FORMAT = "%H:%M:%S"

# The current request's short id, injected into every log record by the filter
# below so a whole request's logs can be grouped. Empty outside a request.
_request_id: ContextVar[str] = ContextVar("scufris_request_id", default="")

_configured = False


class _RequestIdFilter(logging.Filter):
    """Attach the current request id (if any) to each record as ``req``."""

    def filter(self, record: logging.LogRecord) -> bool:
        rid = _request_id.get()
        record.req = f" [{rid}]" if rid else ""
        return True


def new_request_id() -> str:
    """A short random id for one HTTP request."""
    return uuid.uuid4().hex[:8]


def set_request_id(rid: str) -> None:
    _request_id.set(rid)


def _coerce_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    return logging.getLevelName(level.upper().strip())  # type: ignore[return-value]


def configure_logging(level: str | int = "INFO", *, force: bool = False) -> None:
    """Install the scufris log format + level. First call wins unless ``force``.

    The CLI resolves the effective level (``--debug`` beats the setting) and calls
    this with ``force=True`` before dispatching; ``run_server`` calls it un-forced
    so a direct, non-CLI launch still configures without clobbering the CLI choice.
    Idempotent: repeated un-forced calls are no-ops.
    """
    global _configured
    if _configured and not force:
        return
    resolved = _coerce_level(level)
    if not isinstance(resolved, int):
        resolved = logging.INFO

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler()  # stderr
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    handler.addFilter(_RequestIdFilter())
    root.addHandler(handler)
    root.setLevel(resolved)

    # scufris + uvicorn share the level; uvicorn keeps its own access logger.
    for name in ("scufris", "uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).setLevel(resolved)

    _configured = True


def truncate(text: str, limit: int = 200) -> str:
    """Bound a value for logging: long text is cut with a `(+N chars)` marker."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...(+{len(text) - limit} chars)"
