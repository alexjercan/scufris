"""The fakes the domain-router suite drives, and the rig that wires them.

Lifted out of `test_domain_routers.py` verbatim when carving the privileged
helper into its own distribution pushed that file over the 900-line test cap.
Nothing here asserts anything: it is the object graph the routers are handed,
kept in one place so the suite that uses it stays about what the routers DO.
"""

from __future__ import annotations

from typing import Any, cast

from conftest import make_fixture_stats
from fastapi import FastAPI

from scufris.api.host import HostDeps, build_host_router
from scufris.api.hostconfig import HostConfigDeps, build_hostconfig_router
from scufris.config import Settings
from scufris.digest import Digest
from scufris.eventbus import EventBus
from scufris.host_actions import (
    Confirmation,
    ConfirmationStyle,
    HostActionRecord,
    UnknownAction,
)
from scufris.hostconfig import ConfigChange, Resolved, UnknownChange
from scufris.scheduler import DAILY, WATCH, ScheduleState
from scufris_host import Availability, HostOverview, ProcessList
from scufris_hostd import (
    ActionKind,
    AuditRecord,
    Fingerprint,
    Preview,
    PreviewKind,
    ProposalView,
    Requester,
    Reversal,
    RiskClass,
    Step,
)


class _Sampler:
    def __init__(self, value: Any) -> None:
        self._value = value

    def sample(self) -> Any:
        return self._value


ACTION_ID = "act-1"
CHANGE_ID = "chg-1"

ORDINARY_CONFIRMATION = Confirmation(
    style=ConfirmationStyle.ORDINARY,
    risk=RiskClass.R1,
    risk_label="service control",
    undo="stop it again",
)
ONE_WAY_CONFIRMATION = Confirmation(
    style=ConfirmationStyle.ONE_WAY,
    risk=RiskClass.R3,
    risk_label="destructive",
    undo="this cannot be undone",
    no_undo=True,
    acknowledge="store.collect",
)


def record(
    action_id: str = ACTION_ID, *, run_id: str | None = None
) -> HostActionRecord:
    """The smallest real record: enough for the response model to serialize."""
    return HostActionRecord(
        proposal=ProposalView(
            id=action_id,
            kind=ActionKind.UNIT_RESTART,
            risk=RiskClass.R1,
            steps=[Step(argv=["systemctl", "restart", "--", "nginx.service"])],
            summary="restart nginx",
            preview=Preview(
                kind=PreviewKind.STATE,
                headline="h",
                label="l",
                available=Availability(),
                lines=["x"],
            ),
            reversal=Reversal(possible=True, summary="stop it"),
            fingerprint=Fingerprint(value="f", describes="d"),
            created_at=0.0,
            expires_at=1e12,
        ),
        run_id=run_id,
    )


def _change(change_id: str = CHANGE_ID) -> ConfigChange:
    return ConfigChange(
        id=change_id,
        resolved=Resolved(repo="/srv/nixos", ref="HEAD", rev="deadbeef"),
        attr="box",
        run_id=f"config:{change_id}",
    )


class FakeGate:
    """The identity answers, without a session store or a database.

    It records nothing but what it was asked, because the thing worth proving
    about a route is that it asks the GATE who is calling rather than reading the
    request body - the whole point of `SessionGate` existing.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def operator_identity(self, request: Any) -> str:
        self.calls.append(("operator", "", ""))
        return "operator:abcd1234"

    async def requester_identity(
        self, request: Any, *, agent: str = "", run: str = ""
    ) -> Requester:
        self.calls.append(("requester", agent, run))
        return Requester(actor="agent", agent=agent or "orchestrator", run=run)


class FakeApprovals:
    """The decision path, recording calls and raising whatever a test asks for."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.raises: dict[str, Exception] = {}

    def _maybe_raise(self, name: str) -> None:
        exc = self.raises.get(name)
        if exc is not None:
            raise exc

    async def propose(
        self, kind: ActionKind, args: dict[str, Any], requester: Requester
    ) -> HostActionRecord:
        self.calls.append(("propose", (kind, args, requester)))
        self._maybe_raise("propose")
        return record()

    async def refresh_pending(self, *, min_interval: float = 0.0) -> int:
        self.calls.append(("refresh_pending", min_interval))
        self._maybe_raise("refresh_pending")
        return 0

    async def approve(
        self, action_id: str, *, actor: str, acknowledge: str = ""
    ) -> tuple[HostActionRecord, str]:
        self.calls.append(("approve", (action_id, actor, acknowledge)))
        self._maybe_raise("approve")
        return record(action_id, run_id="run-1"), "run-1"

    async def confirmation(self, action_id: str) -> Confirmation:
        self.calls.append(("confirmation", action_id))
        self._maybe_raise("confirmation")
        return ORDINARY_CONFIRMATION

    async def cancel(self, action_id: str) -> HostActionRecord:
        self.calls.append(("cancel", action_id))
        self._maybe_raise("cancel")
        return record(action_id)

    async def deny(
        self, action_id: str, *, actor: str, reason: str = ""
    ) -> HostActionRecord:
        self.calls.append(("deny", (action_id, actor, reason)))
        self._maybe_raise("deny")
        return record(action_id)

    async def revert(self, action_id: str, *, actor: str) -> HostActionRecord:
        self.calls.append(("revert", (action_id, actor)))
        self._maybe_raise("revert")
        return record("act-2")


class FakeActions:
    def __init__(self) -> None:
        self.records = {ACTION_ID: record(run_id="run-1")}

    def get(self, action_id: str) -> HostActionRecord:
        try:
            return self.records[action_id]
        except KeyError:
            raise UnknownAction(action_id) from None

    def list(self) -> list[HostActionRecord]:
        return list(self.records.values())


class FakeScheduler:
    def __init__(self) -> None:
        self.started: list[str] = []

    async def states(self) -> list[ScheduleState]:
        return [ScheduleState(name=WATCH), ScheduleState(name=DAILY)]

    def start_now(self, name: str) -> None:
        if name not in (WATCH, DAILY):
            raise ValueError(f"no such schedule: {name}")
        self.started.append(name)


class FakeDigests:
    def list(self) -> list[Digest]:
        return [Digest(at=1.0, schedule=WATCH, verdict="ok", text="all quiet")]


class FakeHostd:
    def __init__(self) -> None:
        self.audit_limits: list[int] = []
        self.raises: Exception | None = None

    async def audit_tail(self, limit: int) -> list[AuditRecord]:
        self.audit_limits.append(limit)
        if self.raises is not None:
            raise self.raises
        return []


class FakeRuns:
    def __init__(self) -> None:
        self.buses: dict[str, EventBus[Any]] = {}

    def bus(self, run_id: str) -> EventBus[Any] | None:
        return self.buses.get(run_id)


class FakeOverview:
    def __init__(self) -> None:
        self.calls = 0

    async def overview(self) -> HostOverview:
        self.calls += 1
        return HostOverview()


class FakeChanges:
    """The configuration-change flow, recording what the router handed it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.raises: dict[str, Exception] = {}
        self.live_bus: EventBus[Any] | None = None

    def _maybe_raise(self, name: str) -> None:
        exc = self.raises.get(name)
        if exc is not None:
            raise exc

    async def list(self) -> list[ConfigChange]:
        self.calls.append(("list", None))
        return [_change()]

    async def get(self, change_id: str) -> ConfigChange:
        self.calls.append(("get", change_id))
        if change_id != CHANGE_ID:
            raise UnknownChange(change_id)
        return _change(change_id)

    def bus(self, change: ConfigChange) -> EventBus[Any] | None:
        self.calls.append(("bus", change.id))
        return self.live_bus

    async def start(
        self,
        requester: Requester,
        *,
        ref: str = "",
        repo: str = "",
        attr: str = "",
    ) -> ConfigChange:
        self.calls.append(("start", (requester, ref, repo, attr)))
        self._maybe_raise("start")
        return _change()

    async def cancel(self, change_id: str) -> ConfigChange:
        self.calls.append(("cancel", change_id))
        self._maybe_raise("cancel")
        return _change(change_id)


class Rig:
    """The two routers on a bare app, with every fake reachable by name."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.gate = FakeGate()
        self.approvals = FakeApprovals()
        self.actions = FakeActions()
        self.scheduler = FakeScheduler()
        self.digests = FakeDigests()
        self.hostd = FakeHostd()
        self.runs = FakeRuns()
        self.overview = FakeOverview()
        self.changes = FakeChanges()
        self.stats = make_fixture_stats()
        self.processes = ProcessList(groups=[], total=0)
        self.app = FastAPI()
        self.app.include_router(
            build_host_router(
                HostDeps(
                    settings=settings,
                    gate=cast(Any, self.gate),
                    collector=cast(Any, _Sampler(self.stats)),
                    processes=cast(Any, _Sampler(self.processes)),
                    overview=cast(Any, self.overview),
                    hostd=cast(Any, self.hostd),
                    actions=cast(Any, self.actions),
                    approvals=cast(Any, self.approvals),
                    runs=cast(Any, self.runs),
                    scheduler=cast(Any, self.scheduler),
                    digests=cast(Any, self.digests),
                )
            )
        )
        self.app.include_router(
            build_hostconfig_router(
                HostConfigDeps(
                    gate=cast(Any, self.gate), changes=cast(Any, self.changes)
                )
            )
        )
