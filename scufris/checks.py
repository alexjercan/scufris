"""The host checks: bounded reads with explicit thresholds, judged in code.

Each check answers one question about this machine and says how sure it is. No
model is involved - `tasks/20260729-125046/DECISION.md` section 1: the digest's
value is that it can be trusted and that the boring case is one line, and both are
guarantees code gives rather than approximates.

Three properties every check here has, because the digest is only as honest as its
weakest one:

- **A threshold, from settings.** A check never decides what "too full" means; it
  reads the number the operator configured. Changing the line is a settings edit,
  not a code edit.
- **Unavailable is not OK.** A read that could not be performed reports
  ``UNAVAILABLE`` with the reason, never a passing verdict. The `scufris.host`
  package already answers this way (every report carries an `Availability`); this
  layer must not flatten it into silence - a blank that reads as fine is the exact
  failure the host package was built to avoid.
- **A raise is a result.** `run_checks` turns an exception or a timeout into a
  ``FAILED`` result naming the check, so one broken check degrades the digest
  instead of suppressing it.

The escalation field is the only place a check may ask for something to HAPPEN, and
it can only ask: the proposal it names goes through the ordinary approval queue.
`ESCALATABLE` is the allowlist, and it holds the R2 disposable-cleanup verbs only -
a threshold is not a thing that should be able to restart a service or activate a
configuration (DECISION.md section 4).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Awaitable

from pydantic import BaseModel, Field

from .hostd.actions import ActionKind

if TYPE_CHECKING:
    from .config import Settings
    from .health import AgentHealth
    from .host import HostInspector

logger = logging.getLogger(__name__)

# Per-check wall clock. Generous, because these are subprocess reads (systemctl,
# nix-store) whose own timeouts are shorter; this is the backstop that keeps ONE
# slow check from holding the digest.
CHECK_TIMEOUT_SECONDS = 45.0

# The only verbs a threshold may ever propose. R2 disposable cleanup and nothing
# else: an automatic proposal is a claim that the situation is unambiguous, which is
# true of "these store paths are unreachable" and false of everything that stops a
# service or switches the system. Enforced by `escalation_for`, pinned by a test.
ESCALATABLE: frozenset[ActionKind] = frozenset({ActionKind.GC_STORE})


class CheckState(StrEnum):
    """How a check came out. Ordered by how much it wants attention."""

    OK = "ok"
    WARN = "warn"
    CRIT = "crit"
    # The read could not be performed (no sensors, a command missing, a permission
    # denied). NOT a pass, and never rendered as one.
    UNAVAILABLE = "unavailable"
    # The check itself raised or timed out. The digest names it and carries on.
    FAILED = "failed"


# What the digest leads with, in order.
_ATTENTION: tuple[CheckState, ...] = (
    CheckState.CRIT,
    CheckState.WARN,
    CheckState.FAILED,
    CheckState.UNAVAILABLE,
)


class Escalation(BaseModel):
    """A host action a breached check would like proposed.

    Nothing here applies anything: the scheduler hands this to the approval service,
    which previews it and leaves it for the operator like any other proposal.
    """

    kind: ActionKind
    args: dict[str, object] = Field(default_factory=dict)
    # Why this is being proposed, in the words the operator will read.
    because: str


class CheckResult(BaseModel):
    """One check's verdict."""

    name: str
    state: CheckState
    # One line, the thing worth knowing. This is what the digest prints.
    headline: str
    # Optional supporting lines (the two fullest filesystems, the failed units).
    detail: list[str] = Field(default_factory=list)
    # The numbers judged, so a later digest can say what CHANGED without re-reading.
    facts: dict[str, float | int | str | None] = Field(default_factory=dict)
    escalation: Escalation | None = None

    @property
    def wants_attention(self) -> bool:
        return self.state in _ATTENTION


class CheckRun(BaseModel):
    """Every check's verdict from one pass, plus how long it took."""

    at: float
    results: list[CheckResult] = Field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def attention(self) -> list[CheckResult]:
        """The results worth leading with, worst first."""
        order = {state: index for index, state in enumerate(_ATTENTION)}
        return sorted(
            (r for r in self.results if r.wants_attention),
            key=lambda r: order.get(r.state, len(order)),
        )

    def by_name(self) -> dict[str, CheckResult]:
        return {result.name: result for result in self.results}


def escalation_for(kind: ActionKind, args: dict[str, object], because: str) -> Escalation:
    """Build an escalation, refusing anything outside the allowlist.

    A `ValueError` here is a programming error rather than a runtime condition: it
    means a check tried to propose something a threshold has no business proposing,
    and it should fail loudly in tests rather than quietly widen what automation can
    ask for.
    """
    if kind not in ESCALATABLE:
        raise ValueError(
            f"{kind} is not escalatable: a threshold may only propose "
            f"{sorted(ESCALATABLE)}"
        )
    return Escalation(kind=kind, args=dict(args), because=because)


# --- the checks -------------------------------------------------------------
#
# Each takes the inspector and the settings and returns ONE result. They are
# synchronous (the inspector is), and `run_checks` is what moves them off the loop.


def check_disk(inspector: "HostInspector", settings: "Settings") -> CheckResult:
    """Is any real filesystem close to full?"""
    report = inspector.storage().filesystems
    if not report.available.ok:
        return CheckResult(
            name="disk",
            state=CheckState.UNAVAILABLE,
            headline=f"disk usage could not be read: {report.available.reason}",
        )
    fullest = report.fullest
    if fullest is None:
        return CheckResult(
            name="disk",
            state=CheckState.UNAVAILABLE,
            headline="no real filesystem was reported",
        )
    warn = settings.check_disk_warn_percent
    crit = settings.check_disk_crit_percent
    state = (
        CheckState.CRIT
        if fullest.percent >= crit
        else CheckState.WARN
        if fullest.percent >= warn
        else CheckState.OK
    )
    headline = (
        f"{fullest.mountpoint} is {fullest.percent:.0f}% full"
        if state is not CheckState.OK
        else f"disks are fine (fullest: {fullest.mountpoint} at {fullest.percent:.0f}%)"
    )
    detail = [
        f"{fs.mountpoint}: {fs.percent:.0f}% ({fs.used / 1e9:.1f}/{fs.total / 1e9:.1f} GB)"
        for fs in sorted(report.filesystems, key=lambda f: f.percent, reverse=True)[:3]
    ]
    return CheckResult(
        name="disk",
        state=state,
        headline=headline,
        detail=detail if state is not CheckState.OK else [],
        facts={"fullest_percent": round(fullest.percent, 1), "mount": fullest.mountpoint},
    )


def check_failed_units(inspector: "HostInspector", settings: "Settings") -> CheckResult:
    """Did anything break? Both scopes, because scufris itself is a USER unit."""
    from .host import Scope

    system = inspector.failed_units(scope=Scope.SYSTEM)
    user = inspector.failed_units(scope=Scope.USER)
    unreadable = [
        report.available.reason
        for report in (system, user)
        if not report.available.ok
    ]
    names = [f"{u.name} (system)" for u in system.units if u.failed]
    names += [f"{u.name} (user)" for u in user.units if u.failed]
    if unreadable and not names:
        return CheckResult(
            name="units",
            state=CheckState.UNAVAILABLE,
            headline=f"failed units could not be read: {unreadable[0]}",
        )
    if not names:
        return CheckResult(
            name="units",
            state=CheckState.OK,
            headline="nothing is in a failed state",
            facts={"failed": 0},
        )
    return CheckResult(
        name="units",
        # A failed unit is not a warning: something the machine was told to run is
        # not running.
        state=CheckState.CRIT,
        headline=f"{len(names)} unit(s) failed: {', '.join(names[:3])}"
        + (" ..." if len(names) > 3 else ""),
        detail=names,
        facts={"failed": len(names)},
    )


def check_thermal(inspector: "HostInspector", settings: "Settings") -> CheckResult:
    """Is it hot, and has it been held back?

    This host is a DESKTOP (chassis_type 3, no battery, no fan sensors - corrected
    during 20260729-125024), so the answer comes from coretemp plus the CPU's
    thermal_throttle counters. The counters are the part that settles it: they are
    cumulative, so they show throttling that already happened even when the current
    temperature looks fine.
    """
    report = inspector.thermal()
    hottest = report.hottest
    if not report.available.ok and hottest is None:
        return CheckResult(
            name="thermal",
            state=CheckState.UNAVAILABLE,
            headline=f"temperatures could not be read: {report.available.reason}",
        )
    limit = settings.check_temp_warn_celsius
    throttle = report.throttling
    facts: dict[str, float | int | str | None] = {
        "hottest_celsius": round(hottest.celsius, 1) if hottest else None,
        "core_events": throttle.core_events,
        "package_events": throttle.package_events,
    }
    if throttle.throttled:
        return CheckResult(
            name="thermal",
            state=CheckState.WARN,
            headline=(
                f"the CPU has been thermally throttled ({throttle.core_events} "
                f"per-core, {throttle.package_events} whole-package events)"
            ),
            detail=[f"hottest sensor: {hottest.label} at {hottest.celsius:.0f}C"]
            if hottest
            else [],
            facts=facts,
        )
    if hottest is not None and hottest.celsius >= limit:
        return CheckResult(
            name="thermal",
            state=CheckState.WARN,
            headline=f"{hottest.label} is at {hottest.celsius:.0f}C (limit {limit:.0f}C)",
            facts=facts,
        )
    return CheckResult(
        name="thermal",
        state=CheckState.OK,
        headline=(
            f"temperatures are fine (hottest: {hottest.celsius:.0f}C)"
            if hottest
            else "no throttling recorded"
        ),
        facts=facts,
    )


def check_store(inspector: "HostInspector", settings: "Settings") -> CheckResult:
    """How much of the Nix store is dead, and is the store's filesystem tight?

    This is the one check that may ESCALATE, and only to `gc_store` - collecting
    unreachable paths, which touches no generation. Whether it does is the
    operator's switch (`check_escalate_gc`, off by default).
    """
    storage = inspector.storage()
    reclaimable = inspector.reclaimable_space()
    store_fs = storage.nix_store
    dead = reclaimable.dead_paths
    facts: dict[str, float | int | str | None] = {
        "dead_paths": dead,
        "store_percent": round(store_fs.percent, 1) if store_fs else None,
    }
    if not reclaimable.available.ok and dead is None:
        return CheckResult(
            name="store",
            state=CheckState.UNAVAILABLE,
            headline=f"the store could not be measured: {reclaimable.available.reason}",
            facts=facts,
        )
    threshold = settings.check_store_dead_paths
    tight = store_fs is not None and store_fs.percent >= settings.check_disk_warn_percent
    over = dead is not None and dead >= threshold
    if not (over and tight):
        return CheckResult(
            name="store",
            state=CheckState.OK,
            headline=(
                f"the store has {dead} unreachable path(s)"
                if dead is not None
                else "the store looks fine"
            ),
            facts=facts,
        )
    # Both conditions: there is something to collect AND the space is wanted. A
    # store full of dead paths on a half-empty disk is not a problem to wake anyone
    # for.
    headline = (
        f"the store holds {dead} unreachable path(s) and its filesystem is "
        f"{store_fs.percent:.0f}% full"
        if store_fs
        else f"the store holds {dead} unreachable path(s)"
    )
    escalation = None
    if settings.check_escalate_gc:
        escalation = escalation_for(
            ActionKind.GC_STORE,
            {},
            because=headline,
        )
    return CheckResult(
        name="store",
        state=CheckState.WARN,
        headline=headline,
        detail=[
            "collecting them frees space and touches no system generation",
            "note: this is a path COUNT, not a byte total - nix reports no size here",
        ],
        facts=facts,
        escalation=escalation,
    )


def check_flake(inspector: "HostInspector", settings: "Settings") -> CheckResult:
    """How old are the config flake's pinned inputs?

    Age, never "behind": proving a newer commit exists needs a network fetch, which
    this does not do. The headline says age for that reason.
    """
    report = inspector.flake_status()
    if not report.available.ok:
        return CheckResult(
            name="flake",
            state=CheckState.UNAVAILABLE,
            headline=f"the flake lock could not be read: {report.available.reason}",
        )
    oldest = report.oldest()
    age = oldest.age_days() if oldest is not None else None
    if oldest is None or age is None:
        return CheckResult(
            name="flake",
            state=CheckState.UNAVAILABLE,
            headline="no dated inputs in the flake lock",
        )
    limit = settings.check_flake_age_days
    state = CheckState.WARN if age >= limit else CheckState.OK
    return CheckResult(
        name="flake",
        state=state,
        headline=(
            f"{oldest.name} was pinned {age} days ago (limit {limit})"
            if state is CheckState.WARN
            else f"the oldest pinned input is {age} days old"
        ),
        facts={"oldest_days": age, "input": oldest.name},
    )


def check_scufris(health: "AgentHealth") -> CheckResult:
    """Is Scufris itself well? Its own health card, judged.

    Takes the already-collected health rather than an inspector: the same reader the
    dashboard and `/settings health` use, so the digest cannot disagree with them.
    """
    bad = [check for check in health.checks if check.status == "error"]
    warn = [check for check in health.checks if check.status == "warn"]
    if bad:
        return CheckResult(
            name="scufris",
            state=CheckState.CRIT,
            headline=f"scufris is degraded: {bad[0].name} - {bad[0].detail}",
            detail=[f"{check.name}: {check.detail}" for check in bad + warn],
            facts={"errors": len(bad), "warnings": len(warn)},
        )
    if warn:
        return CheckResult(
            name="scufris",
            state=CheckState.WARN,
            headline=f"scufris has {len(warn)} warning(s): {warn[0].name}",
            detail=[f"{check.name}: {check.detail}" for check in warn],
            facts={"errors": 0, "warnings": len(warn)},
        )
    return CheckResult(
        name="scufris",
        state=CheckState.OK,
        headline=f"scufris {health.scufris_version} is healthy",
        facts={"errors": 0, "warnings": 0},
    )


# The set, in the order a digest reads them: the machine first, ourselves last.
HostCheck = Callable[["HostInspector", "Settings"], CheckResult]

HOST_CHECKS: tuple[tuple[str, HostCheck], ...] = (
    ("disk", check_disk),
    ("units", check_failed_units),
    ("thermal", check_thermal),
    ("store", check_store),
    ("flake", check_flake),
)


async def run_checks(
    inspector: "HostInspector",
    settings: "Settings",
    *,
    health: "Callable[[], Awaitable[AgentHealth]] | None" = None,
    timeout: float = CHECK_TIMEOUT_SECONDS,
    clock: Callable[[], float] = time.time,
) -> CheckRun:
    """Run every check off the event loop, and never raise.

    Each check is a synchronous subprocess-backed read, so it goes to a thread -
    otherwise one `systemctl` call stalls the poll loop and every concurrent stream
    with it (the ledger's `sync-read-inline-on-a-latency-loop-stalls-it`).

    A check that raises or exceeds ``timeout`` becomes a FAILED result naming it, so
    the digest degrades instead of vanishing. Be honest about the limit: a thread
    handed a hung read CANNOT be cancelled - the timeout bounds this function and the
    digest, while the thread stays parked until the inspector's own subprocess
    timeout releases it. That is why the bound here is generous rather than tight:
    tripping it means something is genuinely stuck, not merely slow.
    """
    started = clock()
    results: list[CheckResult] = []
    for name, check in HOST_CHECKS:
        results.append(await _one(name, check, inspector, settings, timeout))
    if health is not None:
        results.append(await _own_health(health, timeout))
    return CheckRun(
        at=started, results=results, duration_seconds=max(0.0, clock() - started)
    )


async def _one(
    name: str,
    check: HostCheck,
    inspector: "HostInspector",
    settings: "Settings",
    timeout: float,
) -> CheckResult:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(check, inspector, settings), timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.warning("host check %s timed out after %.0fs", name, timeout)
        return CheckResult(
            name=name,
            state=CheckState.FAILED,
            headline=f"the {name} check timed out after {timeout:.0f}s",
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - a broken check is a result, not a crash
        logger.exception("host check %s failed", name)
        return CheckResult(
            name=name,
            state=CheckState.FAILED,
            headline=f"the {name} check failed: {type(exc).__name__}: {exc}",
        )


async def _own_health(
    health: "Callable[[], Awaitable[AgentHealth]]", timeout: float
) -> CheckResult:
    try:
        snapshot = await asyncio.wait_for(health(), timeout=timeout)
    except asyncio.TimeoutError:
        return CheckResult(
            name="scufris",
            state=CheckState.FAILED,
            headline=f"the scufris check timed out after {timeout:.0f}s",
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("the scufris health check failed")
        return CheckResult(
            name="scufris",
            state=CheckState.FAILED,
            headline=f"the scufris check failed: {type(exc).__name__}: {exc}",
        )
    return check_scufris(snapshot)
