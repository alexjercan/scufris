"""Bounded journal reads.

The journal is the one inspection that can trivially produce megabytes, so it is
bounded twice: a LINE cap passed to journalctl (so the data never crosses the
process boundary) and a BYTE cap applied to what comes back (so a handful of
enormous lines cannot blow the budget the line cap appeared to set). Either cap
firing sets ``truncated``, and the renderer prints the marker.

Parsing is over ``-o json``, which emits one JSON object per LINE - not a JSON
array. Lines that do not parse are counted rather than dropped silently, because
"we could not read 40 of these entries" is information.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from .models import Availability, Bounded, Report, clamp
from .run import DEFAULT_TIMEOUT, Runner
from .units import Scope

DEFAULT_JOURNAL_LINES = 100
MAX_JOURNAL_LINES = 1000
# Bytes of rendered message text kept. A single journal line can be huge (a
# stack trace, a base64 blob), so the line cap alone does not bound the output.
MAX_JOURNAL_BYTES = 40_000

# journalctl priority words, most severe first. Passing one filters to it AND
# everything more severe, which is journalctl's own semantics.
PRIORITIES = (
    "emerg",
    "alert",
    "crit",
    "err",
    "warning",
    "notice",
    "info",
    "debug",
)


class JournalEntry(BaseModel):
    """One journal record, reduced to the fields worth a model's attention."""

    timestamp: datetime | None = None
    unit: str = ""
    priority: int = 6
    message: str = ""
    pid: int | None = None

    @property
    def severe(self) -> bool:
        """err (3) or worse - the entries that answer "did anything break"."""
        return self.priority <= 3


class JournalQuery(BaseModel):
    """The query a ``JournalReport`` answers, echoed back so the caller can see
    what was actually asked (including any clamping applied to it)."""

    unit: str = ""
    scope: Scope = Scope.SYSTEM
    priority: str = ""
    since: str = ""
    until: str = ""
    lines: int = DEFAULT_JOURNAL_LINES


class JournalReport(Report, Bounded):
    entries: list[JournalEntry] = Field(default_factory=list)
    query: JournalQuery = Field(default_factory=JournalQuery)
    unparsed: int = 0
    bytes_truncated: bool = False

    @property
    def severe_count(self) -> int:
        return sum(1 for e in self.entries if e.severe)


def _timestamp(raw: object) -> datetime | None:
    """``__REALTIME_TIMESTAMP`` is microseconds since the epoch, as a string."""
    try:
        micros = int(str(raw))
    except (TypeError, ValueError):
        return None
    try:
        return datetime.fromtimestamp(micros / 1_000_000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _message(raw: object) -> str:
    """MESSAGE is usually a string but is a byte ARRAY when it is not valid
    UTF-8, which journalctl represents as a list of ints. Render both."""
    if isinstance(raw, list):
        try:
            return bytes(int(b) for b in raw).decode("utf-8", "replace")
        except (TypeError, ValueError):
            return ""
    return "" if raw is None else str(raw)


def read_journal(
    runner: Runner,
    *,
    unit: str = "",
    scope: Scope = Scope.SYSTEM,
    priority: str = "",
    since: str = "",
    until: str = "",
    lines: int = DEFAULT_JOURNAL_LINES,
    timeout: float = DEFAULT_TIMEOUT,
) -> JournalReport:
    """Read a bounded window of the journal.

    ``since``/``until`` take journalctl's own time vocabulary ("1 hour ago",
    "yesterday", "2026-07-29 12:00"). ``priority`` filters to that severity and
    worse. ``lines`` is clamped to ``MAX_JOURNAL_LINES``; asking for more returns
    the cap with ``truncated`` set, never the full firehose.
    """
    capped = clamp(lines, default=DEFAULT_JOURNAL_LINES, maximum=MAX_JOURNAL_LINES)
    query = JournalQuery(
        unit=unit.strip(),
        scope=scope,
        priority=priority.strip(),
        since=since.strip(),
        until=until.strip(),
        lines=capped,
    )
    report = JournalReport(query=query, limit=capped)
    if query.priority and query.priority not in PRIORITIES:
        report.available = Availability.unavailable(
            f"unknown priority {query.priority!r}; use one of: " + ", ".join(PRIORITIES)
        )
        return report

    argv = [
        "journalctl",
        *scope.args,
        "--no-pager",
        "-o",
        "json",
        # One more than the cap, so "there was more" is observed rather than
        # inferred from a full page.
        "-n",
        str(capped + 1),
    ]
    if query.unit:
        argv += ["-u", query.unit]
    if query.priority:
        argv += ["-p", query.priority]
    if query.since:
        argv += ["--since", query.since]
    if query.until:
        argv += ["--until", query.until]

    result = runner(argv, timeout=timeout)
    report.available = Availability.of(result)
    if not result.ok:
        return report

    raw_lines = [line for line in result.stdout.splitlines() if line.strip()]
    report.total_seen = len(raw_lines)
    budget = MAX_JOURNAL_BYTES
    for line in raw_lines:
        if len(report.entries) >= capped:
            report.truncated = True
            break
        try:
            row = json.loads(line)
        except ValueError:
            report.unparsed += 1
            continue
        if not isinstance(row, dict):
            report.unparsed += 1
            continue
        message = _message(row.get("MESSAGE"))
        budget -= len(message.encode("utf-8", "replace"))
        if budget < 0:
            report.truncated = True
            report.bytes_truncated = True
            if report.entries:
                break
            # A single message bigger than the whole budget would otherwise
            # return zero entries, which reads as "nothing happened" about data
            # that WAS read. Keep it, cut to the budget, and let the marker say
            # so - a clipped line is information, an empty report is not.
            clipped = message.encode("utf-8", "replace")[:MAX_JOURNAL_BYTES]
            # Cut by BYTES, not characters: the budget is counted in bytes, and
            # slicing a multi-byte message by character count could retain
            # several times the budget. `errors="ignore"` drops the partial
            # code point a byte-slice can leave at the end.
            message = clipped.decode("utf-8", "ignore") + " ...[message cut to fit]"
        try:
            priority_value = int(str(row.get("PRIORITY", 6)))
        except ValueError:
            priority_value = 6
        pid_raw = row.get("_PID")
        try:
            pid = int(str(pid_raw)) if pid_raw is not None else None
        except ValueError:
            pid = None
        report.entries.append(
            JournalEntry(
                timestamp=_timestamp(row.get("__REALTIME_TIMESTAMP")),
                unit=str(
                    row.get("_SYSTEMD_USER_UNIT") or row.get("_SYSTEMD_UNIT") or ""
                ),
                priority=priority_value,
                message=message,
                pid=pid,
            )
        )
    # `-n capped+1` was requested, so more raw lines than the cap PROVES there
    # were more matching entries - even if some of them failed to parse and so
    # never consumed a slot. Deriving truncation only from the entry count would
    # present a full page as the complete set whenever a line was unparseable.
    report.truncated = report.truncated or report.total_seen > capped
    # journalctl returns newest-last already; keep that order (a reader wants the
    # tail at the bottom, like the command they would have typed).
    return report
