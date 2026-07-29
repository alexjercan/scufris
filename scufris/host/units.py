"""systemd unit inspection: what is running, what failed, what one unit is doing.

Everything here parses ``systemctl``'s MACHINE-readable output - ``-o json`` for
listings, ``--property=`` key=value for a single unit - never the human table,
whose column widths and truncation are not a contract.

Scope is explicit on every call. A user unit and a system unit can share a name
(``scufris.service`` is a user unit on this host), so "did it fail" has two
different answers and the caller must say which one it means.
"""

from __future__ import annotations

import json
from enum import StrEnum

from pydantic import BaseModel, Field

from .models import Availability, Bounded, Report, clamp, rejects_option
from .run import DEFAULT_TIMEOUT, Runner

# A listing this long is already unreadable; the cap protects the model context.
DEFAULT_UNIT_LIMIT = 50
MAX_UNIT_LIMIT = 400

# The properties one unit's status is built from. An explicit list keeps the
# output bounded and stable - `systemctl show` with no filter emits ~200 fields.
_UNIT_PROPERTIES = (
    "Id",
    "Description",
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "MainPID",
    "Result",
    "NRestarts",
    "ExecMainStartTimestamp",
    "ExecMainExitTimestamp",
    "ExecMainStatus",
    "ActiveEnterTimestamp",
    "MemoryCurrent",
    "StatusText",
)


class Scope(StrEnum):
    """Which systemd manager to ask.

    ``SYSTEM`` is pid-1's units, ``USER`` is the operator's session manager. Both
    are readable as ``alex`` on this host - the system journal because ``wheel``
    carries the journal ACL.
    """

    SYSTEM = "system"
    USER = "user"

    @property
    def args(self) -> list[str]:
        """The systemctl/journalctl flag selecting this manager."""
        return ["--user"] if self is Scope.USER else ["--system"]


class UnitSummary(BaseModel):
    """One row of a unit listing."""

    name: str
    load: str
    active: str
    sub: str
    description: str = ""

    @property
    def failed(self) -> bool:
        return self.active == "failed"


class UnitList(Report, Bounded):
    """A listing of units in one scope."""

    scope: Scope = Scope.SYSTEM
    state_filter: str = ""
    units: list[UnitSummary] = Field(default_factory=list)

    @property
    def failed_count(self) -> int:
        return sum(1 for u in self.units if u.failed)


class UnitStatus(Report):
    """One unit's current state and the result of its last run."""

    scope: Scope = Scope.SYSTEM
    name: str
    description: str = ""
    load_state: str = ""
    active_state: str = ""
    sub_state: str = ""
    unit_file_state: str = ""
    result: str = ""
    main_pid: int = 0
    restarts: int = 0
    exit_status: int | None = None
    started_at: str = ""
    exited_at: str = ""
    memory_bytes: int | None = None
    status_text: str = ""

    @property
    def loaded(self) -> bool:
        """False when systemd has no such unit (``LoadState=not-found``)."""
        return self.load_state == "loaded"


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def list_units(
    runner: Runner,
    *,
    scope: Scope = Scope.SYSTEM,
    state: str = "",
    pattern: str = "",
    limit: int = DEFAULT_UNIT_LIMIT,
    timeout: float = DEFAULT_TIMEOUT,
) -> UnitList:
    """List units in ``scope``, optionally filtered by state and name pattern.

    ``state`` is a systemd state word (``failed``, ``running``, ``active``,
    ``inactive``); ``pattern`` is a systemd unit glob (``nginx*``). Both are
    passed to systemctl rather than filtered here, so the semantics are
    systemd's own.
    """
    capped = clamp(limit, default=DEFAULT_UNIT_LIMIT, maximum=MAX_UNIT_LIMIT)
    argv = [
        "systemctl",
        *scope.args,
        "list-units",
        "--all",
        "--no-pager",
        "--no-legend",
        "-o",
        "json",
    ]
    if state:
        # Inside one --state= token, so it cannot become a separate option.
        argv.append(f"--state={state}")
    report = UnitList(scope=scope, state_filter=state, limit=capped)
    if pattern:
        if rejects_option(pattern):
            report.available = Availability.unavailable(
                f"a unit pattern may not start with '-' (got {pattern!r}); "
                "systemctl would read it as an option, not a unit glob"
            )
            return report
        # After `--`, so a pattern is an operand even if the screen above is
        # ever loosened.
        argv += ["--", pattern]
    result = runner(argv, timeout=timeout)
    report.available = Availability.of(result)
    if not result.ok:
        return report
    try:
        rows = json.loads(result.stdout or "[]")
    except ValueError as exc:
        report.available = Availability.unavailable(
            f"systemctl returned output this build cannot parse: {exc}"
        )
        return report
    if not isinstance(rows, list):
        report.available = Availability.unavailable(
            "systemctl returned a non-list where a unit array was expected"
        )
        return report
    report.total_seen = len(rows)
    for row in rows[:capped]:
        if not isinstance(row, dict):
            continue
        report.units.append(
            UnitSummary(
                name=str(row.get("unit", "")),
                load=str(row.get("load", "")),
                active=str(row.get("active", "")),
                sub=str(row.get("sub", "")),
                description=str(row.get("description", "")),
            )
        )
    report.truncated = len(rows) > capped
    return report


def failed_units(
    runner: Runner,
    *,
    scope: Scope = Scope.SYSTEM,
    limit: int = DEFAULT_UNIT_LIMIT,
    timeout: float = DEFAULT_TIMEOUT,
) -> UnitList:
    """The units in ``scope`` that are in the failed state.

    An EMPTY result here is a real answer ("nothing failed"), which is exactly
    why it must not be confused with an unavailable one - see the renderer.
    """
    return list_units(runner, scope=scope, state="failed", limit=limit, timeout=timeout)


def unit_status(
    runner: Runner,
    name: str,
    *,
    scope: Scope = Scope.SYSTEM,
    timeout: float = DEFAULT_TIMEOUT,
) -> UnitStatus:
    """One unit's state, last exit status and restart count.

    Uses ``systemctl show`` (stable key=value) rather than ``systemctl status``
    (a human display that also pipes the journal through a pager).
    """
    name = name.strip()
    if not name:
        return UnitStatus(
            available=Availability.unavailable("no unit name was given"),
            scope=scope,
            name="",
        )
    if rejects_option(name):
        return UnitStatus(
            available=Availability.unavailable(
                f"a unit name may not start with '-' (got {name!r}); systemctl "
                "would read it as an option, not a unit"
            ),
            scope=scope,
            name=name,
        )
    argv = [
        "systemctl",
        *scope.args,
        "show",
        "--no-pager",
        *[f"--property={prop}" for prop in _UNIT_PROPERTIES],
        "--",
        name,
    ]
    result = runner(argv, timeout=timeout)
    status = UnitStatus(available=Availability.of(result), scope=scope, name=name)
    if not result.ok:
        return status
    fields: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields[key.strip()] = value.strip()
    status.name = fields.get("Id") or name
    status.description = fields.get("Description", "")
    status.load_state = fields.get("LoadState", "")
    status.active_state = fields.get("ActiveState", "")
    status.sub_state = fields.get("SubState", "")
    status.unit_file_state = fields.get("UnitFileState", "")
    status.result = fields.get("Result", "")
    status.main_pid = _to_int(fields.get("MainPID", "0"))
    status.restarts = _to_int(fields.get("NRestarts", "0"))
    status.started_at = fields.get("ExecMainStartTimestamp", "") or fields.get(
        "ActiveEnterTimestamp", ""
    )
    status.exited_at = fields.get("ExecMainExitTimestamp", "")
    status.status_text = fields.get("StatusText", "")
    raw_exit = fields.get("ExecMainStatus", "")
    status.exit_status = _to_int(raw_exit, default=-1) if raw_exit else None
    # systemd reports "[not set]" for an unbounded/unaccounted cgroup.
    memory = fields.get("MemoryCurrent", "")
    status.memory_bytes = _to_int(memory, default=-1) if memory.isdigit() else None
    if status.load_state == "not-found":
        # A real answer, not a failure: systemd was reachable and says there is
        # no such unit. Saying so beats an empty status block.
        status.available = Availability(
            ok=True, caveat=f"systemd has no unit named {name!r} in the {scope} scope"
        )
    return status
