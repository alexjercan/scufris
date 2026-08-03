"""The auth and host routers, driven over fakes on a bare app.

This is the test the extraction exists to make possible. Before it, exercising
`/api/host/actions/{id}/approve` meant calling `create_app`: a state directory, a
SQLite database, an `AgentStore`, a `ProjectStore`, a settings object read from
the environment, a supervisor and a lifespan - all to prove that a route calls
one method and maps one exception onto one status.

The auth router gets the same treatment for the same reason: the login/logout
round trip and the session probe are driven over an in-memory `SessionStore`, so
proving what they do costs no database at all.

Two things are asserted here, and they are the two halves of the same claim:

- `test_host_routes_delegate_to_domain_services` - a route TRANSLATES. Every
  decision the routes used to hold inline (may an `activate` be proposed, is that
  a real schedule, is a second build of this repository allowed) now belongs to a
  service, and what the router does with it is turn its refusal into a status.
  The fakes record what they were called with, so a rule quietly reappearing in a
  route body shows up as a call that never happened;
- `test_domain_router_dependency_isolation` - a router reaches for NOTHING. The
  database, the two stores and the environment-derived settings object are booby
  trapped for the duration, so a router that constructs one fails loudly here
  rather than by being slow and untestable somewhere else. It sweeps the auth
  router as well as the host ones - `SessionGate` is the piece most likely to
  reach for a database, and here it does not have one.
"""

from __future__ import annotations

from typing import Any, Iterator, cast

import pytest
from conftest import make_fixture_stats
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_auth import ORIGIN, PASSWORD
from test_auth import _settings as auth_settings

import scufris.db
from scufris.agent_store import AgentStore
from scufris.api.auth import SessionGate, build_auth_router
from scufris.api.host import HostDeps, build_host_router
from scufris.api.hostconfig import HostConfigDeps, build_hostconfig_router
from scufris.auth import CSRF_COOKIE, SESSION_COOKIE, LoginThrottle, Session
from scufris.config import Settings
from scufris.digest import Digest
from scufris.eventbus import EventBus
from scufris.host import HostOverview
from scufris.host.models import Availability
from scufris.host_actions import (
    Confirmation,
    ConfirmationStyle,
    HostActionRecord,
    UnknownAction,
)
from scufris.host_approvals import (
    AlreadyDecided,
    CannotPropose,
    CannotUndo,
    ConfirmationRequired,
    NoLiveRun,
    NotApplied,
    ProposalExpired,
)
from scufris.hostclient import HostdError, HostdUnavailable
from scufris.hostconfig import (
    ChangeInFlight,
    ConfigChange,
    ConfigChangeRefused,
    NoRunningBuild,
    Resolved,
    UnknownChange,
)
from scufris.hostd.actions import ActionKind, RiskClass, Step
from scufris.hostd.audit import AuditRecord, Requester
from scufris.hostd.preview import Fingerprint, Preview, PreviewKind, Reversal
from scufris.hostd.protocol import ErrorCode, ProposalView
from scufris.processes import ProcessList
from scufris.projects import ProjectStore
from scufris.scheduler import DAILY, WATCH, ScheduleState
from scufris_core import Database

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


def _record(
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
        return _record()

    async def refresh_pending(self, *, min_interval: float = 0.0) -> int:
        self.calls.append(("refresh_pending", min_interval))
        self._maybe_raise("refresh_pending")
        return 0

    async def approve(
        self, action_id: str, *, actor: str, acknowledge: str = ""
    ) -> tuple[HostActionRecord, str]:
        self.calls.append(("approve", (action_id, actor, acknowledge)))
        self._maybe_raise("approve")
        return _record(action_id, run_id="run-1"), "run-1"

    async def confirmation(self, action_id: str) -> Confirmation:
        self.calls.append(("confirmation", action_id))
        self._maybe_raise("confirmation")
        return ORDINARY_CONFIRMATION

    async def cancel(self, action_id: str) -> HostActionRecord:
        self.calls.append(("cancel", action_id))
        self._maybe_raise("cancel")
        return _record(action_id)

    async def deny(
        self, action_id: str, *, actor: str, reason: str = ""
    ) -> HostActionRecord:
        self.calls.append(("deny", (action_id, actor, reason)))
        self._maybe_raise("deny")
        return _record(action_id)

    async def revert(self, action_id: str, *, actor: str) -> HostActionRecord:
        self.calls.append(("revert", (action_id, actor)))
        self._maybe_raise("revert")
        return _record("act-2")


class FakeActions:
    def __init__(self) -> None:
        self.records = {ACTION_ID: _record(run_id="run-1")}

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


class FakeSessions:
    """A `SessionStore` in a dict: mint, read, revoke, prune.

    Expiry is enforced on READ: `get` drops a session past either the idle or the
    absolute window, because the gate hands the store those windows and a store
    that ignored them would make the gate's contract untestable. Nothing else is
    modelled - `get` does not renew `last_seen`, and `prune` only records that it
    was called. No database, which is the point.
    """

    def __init__(self) -> None:
        self.records: dict[str, Session] = {}
        self.pruned_at: list[float] = []
        self._next = 0

    def create(self, *, now: float) -> Session:
        self._next += 1
        session = Session(
            id=f"sess-{self._next}",
            csrf=f"csrf-{self._next}",
            created_at=now,
            last_seen=now,
        )
        self.records[session.id] = session
        return session

    def get(
        self, session_id: str | None, *, now: float, idle: float, absolute: float
    ) -> Session | None:
        if not session_id:
            return None
        session = self.records.get(session_id)
        if session is None:
            return None
        if now - session.last_seen > idle or now - session.created_at > absolute:
            del self.records[session_id]
            return None
        return session

    def revoke(self, session_id: str | None) -> None:
        if session_id:
            self.records.pop(session_id, None)

    def prune(self, *, now: float, idle: float, absolute: float) -> int:
        self.pruned_at.append(now)
        return 0


class AuthRig:
    """The auth router on a bare app, over a real gate and a fake store.

    The gate is REAL here - it is the thing under test, and its whole job is to
    be the one place a session is read, minted and revoked. What is faked is the
    store beneath it, which is where the database used to be.
    """

    def __init__(self, settings: Settings) -> None:
        self.sessions = FakeSessions()
        gate = SessionGate(settings, cast(Any, self.sessions))
        throttle = LoginThrottle(max_failures=3, window_seconds=60.0)
        self.app = FastAPI()
        self.app.include_router(build_auth_router(gate, throttle))


class _Sampler:
    def __init__(self, value: Any) -> None:
        self._value = value

    def sample(self) -> Any:
        return self._value


@pytest.fixture
def rig(tmp_path: Any) -> Rig:
    # Built HERE, from an explicit state dir, and handed to the routers. The
    # routers never construct one - `test_domain_router_dependency_isolation`
    # is what holds them to that.
    return Rig(Settings(state_dir=tmp_path / "state"))


@pytest.fixture
def client(rig: Rig) -> Iterator[TestClient]:
    with TestClient(rig.app) as started:
        yield started


@pytest.fixture
def auth_rig(tmp_path: Any) -> AuthRig:
    # `test_auth._settings` is the auth-domain-local helper the other auth suites
    # share: password configured, `_env_file=None` so a developer's real .env
    # cannot decide the posture (lesson settings-test-must-disable-env-file).
    return AuthRig(auth_settings(tmp_path))


@pytest.fixture
def auth_client(auth_rig: AuthRig) -> Iterator[TestClient]:
    with TestClient(auth_rig.app) as started:
        yield started


def test_host_routes_delegate_to_domain_services(rig: Rig, client: TestClient) -> None:
    """Every host route calls its service and reports what came back.

    Driven against a bare app: no middleware, no auth, no lifespan - so what is
    under test is the route, not the stack around it.
    """
    assert client.get("/api/stats").status_code == 200
    assert client.get("/api/processes").status_code == 200

    assert client.get("/api/host/overview").status_code == 200
    assert rig.overview.calls == 1

    # A proposal names the caller from the GATE, and the body's `agent` is only
    # a hint the gate is told about - never the actor.
    proposed = client.post(
        "/api/host/actions",
        json={"kind": "unit_restart", "args": {"unit": "nginx"}, "agent": "coder"},
    )
    assert proposed.status_code == 201
    kind, args, requester = rig.approvals.calls[0][1]
    assert (kind, args) == (ActionKind.UNIT_RESTART, {"unit": "nginx"})
    assert requester.actor == "agent"
    assert ("requester", "coder", "") in rig.gate.calls

    # The queue reconciles with the helper before it answers, at the configured
    # throttle rather than one the route decides.
    assert client.get("/api/host/actions").status_code == 200
    assert dict(rig.approvals.calls)["refresh_pending"] == (
        rig.settings.host_queue_refresh_seconds
    )

    assert client.get(f"/api/host/actions/{ACTION_ID}").json()["proposal"]["id"] == (
        ACTION_ID
    )

    approved = client.post(f"/api/host/actions/{ACTION_ID}/approve", json={})
    assert approved.status_code == 200
    assert approved.json()["run_id"] == "run-1"
    action_id, actor, acknowledge = dict(rig.approvals.calls)["approve"]
    assert (action_id, actor, acknowledge) == (ACTION_ID, "operator:abcd1234", "")

    assert client.get(f"/api/host/actions/{ACTION_ID}/confirmation").status_code == 200

    denied = client.post(
        f"/api/host/actions/{ACTION_ID}/deny", json={"reason": "not now"}
    )
    assert denied.status_code == 200
    assert dict(rig.approvals.calls)["deny"] == (
        ACTION_ID,
        "operator:abcd1234",
        "not now",
    )

    assert client.post(f"/api/host/actions/{ACTION_ID}/cancel").status_code == 200
    assert dict(rig.approvals.calls)["cancel"] == ACTION_ID
    assert client.post(f"/api/host/actions/{ACTION_ID}/revert").status_code == 201
    assert dict(rig.approvals.calls)["revert"] == (ACTION_ID, "operator:abcd1234")

    # The checks: reading is the scheduler's states plus the digest store, and
    # "run it now" is one call the scheduler owns - including the task that keeps
    # the fire-and-forget run alive, which the route used to hold.
    digests = client.get("/api/host/digests")
    assert digests.status_code == 200
    assert [s["name"] for s in digests.json()["schedules"]] == [WATCH, DAILY]
    assert client.post("/api/host/digests/run?schedule=daily").status_code == 202
    assert rig.scheduler.started == [DAILY]

    assert client.get("/api/host/audit?limit=900").status_code == 200
    # Clamped by the route, because the helper's tail is not a client's to size.
    assert rig.hostd.audit_limits == [500]

    # The configuration flow: the router hands the service a ref and a requester,
    # and holds no build rule of its own.
    built = client.post(
        "/api/host/config/changes", json={"ref": "HEAD~1", "agent": "coder"}
    )
    assert built.status_code == 201
    requester, ref, repo, attr = dict(rig.changes.calls)["start"]
    assert (requester.agent, ref, repo, attr) == ("coder", "HEAD~1", "", "")

    assert client.get("/api/host/config/changes").status_code == 200
    assert client.get(f"/api/host/config/changes/{CHANGE_ID}").status_code == 200
    assert (
        client.post(f"/api/host/config/changes/{CHANGE_ID}/cancel").status_code == 200
    )
    assert dict(rig.changes.calls)["cancel"] == CHANGE_ID


def test_auth_routes_run_over_a_fake_session_store(
    auth_rig: AuthRig, auth_client: TestClient
) -> None:
    """The login round trip, with no database under it.

    Same claim as the host half: the router takes its collaborators explicitly,
    so what used to need `create_app` (a state dir, a SQLite file, a lifespan) is
    a dict. Everything asserted here is behaviour the middleware depends on -
    that a session is MINTED on success and on nothing else, that logout revokes
    server-side rather than only clearing a cookie, and that the probe reports
    posture without naming anyone.
    """
    probe = auth_client.get("/api/auth/session").json()
    assert probe == {"authenticated": False, "required": True}

    refused = auth_client.post(
        "/api/auth/login", json={"password": "wrong"}, headers={"Origin": ORIGIN}
    )
    assert refused.status_code == 401
    # Nothing minted: a failed login must not leave a record behind.
    assert auth_rig.sessions.records == {}

    ok = auth_client.post(
        "/api/auth/login", json={"password": PASSWORD}, headers={"Origin": ORIGIN}
    )
    assert ok.status_code == 200
    assert set(ok.cookies) == {SESSION_COOKIE, CSRF_COOKIE}
    [session] = auth_rig.sessions.records.values()
    assert ok.cookies[SESSION_COOKIE] == session.id
    assert ok.cookies[CSRF_COOKIE] == session.csrf
    # Swept on login - the one moment a record is added is the one moment the
    # store is worth sweeping.
    assert auth_rig.sessions.pruned_at

    assert auth_client.get("/api/auth/session").json() == {
        "authenticated": True,
        "required": True,
    }

    # Rotation: a second login revokes the first id rather than reusing it, which
    # is what closes session fixation.
    first_id = session.id
    assert (
        auth_client.post(
            "/api/auth/login", json={"password": PASSWORD}, headers={"Origin": ORIGIN}
        ).status_code
        == 200
    )
    assert first_id not in auth_rig.sessions.records
    assert len(auth_rig.sessions.records) == 1

    assert auth_client.post("/api/auth/logout").status_code == 200
    # Revoked SERVER-SIDE, not merely un-cookied: a stolen id is dead.
    assert auth_rig.sessions.records == {}
    assert auth_client.get("/api/auth/session").json()["authenticated"] is False


def test_a_cross_origin_login_is_refused_before_it_can_burn_the_throttle(
    auth_rig: AuthRig, auth_client: TestClient
) -> None:
    """Login is public, so without the origin check any page the operator visits
    could fire logins at the dashboard until the lockout window burns and deny
    the real operator their own login. The check runs BEFORE the throttle."""
    for _ in range(5):
        blocked = auth_client.post(
            "/api/auth/login",
            json={"password": "wrong"},
            headers={"Origin": "http://evil.example"},
        )
        assert blocked.status_code == 403

    assert (
        auth_client.post(
            "/api/auth/login", json={"password": PASSWORD}, headers={"Origin": ORIGIN}
        ).status_code
        == 200
    )


def test_the_login_throttle_locks_out_after_repeated_failures(
    auth_client: TestClient,
) -> None:
    """Same-origin failures DO count, and the lockout says when to come back."""
    for _ in range(3):
        assert (
            auth_client.post(
                "/api/auth/login",
                json={"password": "wrong"},
                headers={"Origin": ORIGIN},
            ).status_code
            == 401
        )
    locked = auth_client.post(
        "/api/auth/login", json={"password": PASSWORD}, headers={"Origin": ORIGIN}
    )
    assert locked.status_code == 429
    assert int(locked.headers["Retry-After"]) > 0


def test_activate_is_refused_by_the_service_not_the_route(
    rig: Rig, client: TestClient
) -> None:
    """The one rule most likely to be re-inlined: what may be proposed at all.

    The route reaches the service even for `activate` - it does not shortcut -
    and the service's refusal is what becomes the 422. That is what keeps the MCP
    tool surface from being able to reach a different answer.
    """
    rig.approvals.raises["propose"] = CannotPropose("activate is not proposed directly")
    refused = client.post("/api/host/actions", json={"kind": "activate", "args": {}})
    assert refused.status_code == 422
    assert "not proposed directly" in refused.json()["detail"]
    assert [name for name, _ in rig.approvals.calls] == ["propose"]


@pytest.mark.parametrize(
    ("route", "method", "call", "raises", "status"),
    [
        ("/api/host/actions", "POST", "propose", HostdUnavailable("down"), 503),
        (
            "/api/host/actions",
            "POST",
            "propose",
            HostdError(ErrorCode.REFUSED, "no"),
            422,
        ),
        (
            f"/api/host/actions/{ACTION_ID}/approve",
            "POST",
            "approve",
            ConfirmationRequired(ONE_WAY_CONFIRMATION),
            422,
        ),
        (
            f"/api/host/actions/{ACTION_ID}/approve",
            "POST",
            "approve",
            AlreadyDecided("decided"),
            409,
        ),
        (
            f"/api/host/actions/{ACTION_ID}/approve",
            "POST",
            "approve",
            ProposalExpired("expired"),
            409,
        ),
        (
            f"/api/host/actions/{ACTION_ID}/approve",
            "POST",
            "approve",
            UnknownAction("gone"),
            404,
        ),
        (
            f"/api/host/actions/{ACTION_ID}/cancel",
            "POST",
            "cancel",
            NoLiveRun("nothing running"),
            409,
        ),
        (
            f"/api/host/actions/{ACTION_ID}/revert",
            "POST",
            "revert",
            CannotUndo("one way"),
            422,
        ),
        (
            f"/api/host/actions/{ACTION_ID}/revert",
            "POST",
            "revert",
            NotApplied("never ran"),
            409,
        ),
    ],
)
def test_host_router_maps_service_refusals_onto_statuses(
    rig: Rig,
    client: TestClient,
    route: str,
    method: str,
    call: str,
    raises: Exception,
    status: int,
) -> None:
    """A route translates. Each domain refusal has one status, and it is here."""
    rig.approvals.raises[call] = raises
    body = {"kind": "unit_restart", "args": {}} if call == "propose" else {}
    assert client.request(method, route, json=body).status_code == status


@pytest.mark.parametrize(
    ("call", "raises", "status"),
    [
        ("start", ConfigChangeRefused("bad ref"), 422),
        ("start", ChangeInFlight("already building"), 409),
        ("cancel", NoRunningBuild("nothing building"), 409),
        ("cancel", UnknownChange("gone"), 404),
    ],
)
def test_hostconfig_router_maps_service_refusals_onto_statuses(
    rig: Rig, client: TestClient, call: str, raises: Exception, status: int
) -> None:
    rig.changes.raises[call] = raises
    route = (
        "/api/host/config/changes"
        if call == "start"
        else f"/api/host/config/changes/{CHANGE_ID}/cancel"
    )
    assert client.post(route, json={}).status_code == status


def test_a_missing_action_or_change_is_a_404(client: TestClient) -> None:
    """The store's "no such thing" is a 404, on the read routes as well as the
    decision ones - a client polling a swept action must not see a 500."""
    assert client.get("/api/host/actions/nope").status_code == 404
    assert client.get("/api/host/actions/nope/events").status_code == 404
    assert client.get("/api/host/config/changes/nope").status_code == 404
    assert client.get("/api/host/config/changes/nope/events").status_code == 404


def test_a_stream_with_no_live_run_is_a_404(rig: Rig, client: TestClient) -> None:
    """An action that exists but has nothing running is not a stream.

    Distinct from the case above on purpose: the record is real, so the answer
    cannot be "no such action" - it is "there is nothing to attach to".
    """
    rig.actions.records[ACTION_ID] = _record(run_id=None)
    assert client.get(f"/api/host/actions/{ACTION_ID}/events").status_code == 404


def test_an_unknown_schedule_never_reaches_the_scheduler(
    rig: Rig, client: TestClient
) -> None:
    """422 and nothing started, so a typo cannot report a run that is not
    happening. The refusal is the scheduler's, raised before it starts anything."""
    refused = client.post("/api/host/digests/run?schedule=hourly")
    assert refused.status_code == 422
    assert "no such schedule" in refused.json()["detail"]
    assert rig.scheduler.started == []


def test_the_audit_tail_reports_the_helper_being_down(
    rig: Rig, client: TestClient
) -> None:
    """503, not 500: the helper is a separate process and its absence is a
    condition a client can act on."""
    rig.hostd.raises = HostdUnavailable("no socket")
    assert client.get("/api/host/audit").status_code == 503


def test_the_queue_still_answers_when_the_helper_is_unreachable(
    rig: Rig, client: TestClient
) -> None:
    """Reconciling is best-effort. A dead helper must not hide the proposals this
    process already knows about - that is the queue the operator is looking at."""
    rig.approvals.raises["refresh_pending"] = HostdUnavailable("no socket")
    listed = client.get("/api/host/actions")
    assert listed.status_code == 200
    assert [record["proposal"]["id"] for record in listed.json()] == [ACTION_ID]


def _forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError(
        "a router constructed a store, a database or a settings object; the whole "
        "point of the deps dataclass is that it does not"
    )


def test_domain_router_dependency_isolation(
    rig: Rig,
    client: TestClient,
    auth_rig: AuthRig,
    auth_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A router reaches for nothing. Booby trap the four things it must not touch,
    then drive the whole surface through them.

    The traps go on ``__init__`` rather than on the name in the defining module,
    and that is not a detail: `from ..config import Settings` binds the class into
    the importing module at import time, so replacing `scufris.config.Settings`
    leaves the router calling the real one and the trap never fires (measured -
    the first version of this test passed against a route that called
    `Settings()`). Patching the class itself catches every import spelling.
    """
    for target in (Settings, AgentStore, ProjectStore, Database):
        monkeypatch.setattr(target, "__init__", _forbidden)
    monkeypatch.setattr(scufris.db, "state_database", _forbidden)

    for path in (
        "/api/stats",
        "/api/processes",
        "/api/config",
        "/api/host/overview",
        "/api/host/actions",
        f"/api/host/actions/{ACTION_ID}",
        f"/api/host/actions/{ACTION_ID}/confirmation",
        "/api/host/digests",
        "/api/host/audit",
        "/api/host/config/changes",
        f"/api/host/config/changes/{CHANGE_ID}",
    ):
        assert client.get(path).status_code == 200, path

    assert (
        client.post(
            "/api/host/actions", json={"kind": "unit_restart", "args": {}}
        ).status_code
        == 201
    )
    assert client.post(
        f"/api/host/actions/{ACTION_ID}/approve", json={}
    ).status_code == (200)
    assert client.post("/api/host/digests/run").status_code == 202
    assert client.post("/api/host/config/changes", json={}).status_code == 201

    # The auth router under the same traps. `SessionGate` is the one that used to
    # own a database handle, so this is where a reintroduced one would surface.
    assert auth_client.get("/api/auth/session").status_code == 200
    assert (
        auth_client.post(
            "/api/auth/login", json={"password": PASSWORD}, headers={"Origin": ORIGIN}
        ).status_code
        == 200
    )
    assert auth_client.post("/api/auth/logout").status_code == 200
    assert auth_rig.sessions.records == {}


def test_the_app_config_route_reports_the_floored_overview_interval(
    client: TestClient,
) -> None:
    """The client polls at the cadence the server actually refreshes at.

    The cache floors its TTL; reporting the unfloored setting would have the
    dashboard poll faster than the cache will ever answer differently.
    """
    from scufris.host.overview import MIN_HOST_OVERVIEW_TTL

    reported = client.get("/api/config").json()["host_overview_seconds"]
    assert reported >= MIN_HOST_OVERVIEW_TTL
