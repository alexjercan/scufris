"""The whole app's state on ONE transactional boundary, proven at the boundary.

This module is where the epic's cross-store claims live. `test_projects.py` keeps
the projects store's own tests; the claim that EVERY app-owned store shares one
database - and that a burst across two of them survives a restart - is not a
projects test, and was only ever there because projects was the first store to
land (DECISION.md 1).

The two claims that discriminate rather than restate:

- ``test_post_host_state_uses_declared_persistence_boundary`` fails the moment a
  new store is wired with a path instead of the handle, which is the way this
  boundary would erode - one store at a time, each looking local and harmless.
- ``test_privileged_audit_remains_an_external_boundary`` fails if the audit log is
  ever pulled INTO the app's database. It is root-owned and append-only precisely
  because the app must not be able to rewrite it, and a migration that tidied it
  into `scufris.db` would delete that property while every other test still passed.
"""

from __future__ import annotations

import json
import stat
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from conftest import ORIGIN, PASSWORD, _Helper, _login, _propose, _settings
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from scufris.app import create_app
from scufris.auth import CSRF_HEADER, SESSION_COOKIE
from scufris.db import Database, database_path
from scufris.db.models import AuthSessionRow, HostActionRow, LegacyImportRow
from scufris.enums import AgentState
from scufris_core import FILE_MODE, SIDECAR_SUFFIXES
from scufris_host import Collector

MakeClient = Callable[[Any], TestClient]

# Carried over from the projects-only proof: wide enough that the JSON store this
# replaces lost a record (measured in 20260729-102146), short enough to stay under
# a second.
BURST = 24


def _app(tmp_path: Path, fake_collector: Collector, helper: _Helper) -> Any:
    return create_app(collector=fake_collector, settings=_settings(tmp_path, helper))


def _project_and_agents(app: Any, tmp_path: Path, count: int) -> list[str]:
    """Two agents in a real project, created through the app's own stores."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    project = app.state.projects.create(name="Boundary", cwd=str(workspace))
    return [
        app.state.agents.create(name=f"Worker {n}", project_id=project.id).id
        for n in range(count)
    ]


# --- the epic's headline proof, widened across stores ------------------------


def test_concurrent_state_mutations_survive_restart(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: MakeClient,
) -> None:
    """Two agent completions, a host proposal and a write burst, all concurrent.

    The mutations are deliberately in DIFFERENT stores: that is the property the
    one boundary buys and the property several separate persistence mechanisms
    cannot have. On the base this fails on the host proposal - it lived in an
    ``OrderedDict`` and went with the process.

    The burst of project creates is carried over from the projects-only version
    of this proof: 20260729-102146 measured record loss at this width under the
    JSON store, so the concurrency here is the concurrency that was shown to
    break something rather than a number chosen to look busy.

    The action is read back through the STORE rather than through
    ``GET /api/host/actions``, because that route reconciles with the helper
    first: the helper still holds this proposal, so the route would answer
    truthfully from the socket whether or not anything had been persisted.
    """
    app = _app(tmp_path, fake_collector, helper)
    client = make_client(app)
    csrf = _login(client)
    agent_ids = _project_and_agents(app, tmp_path, 2)
    workspace = tmp_path / "workspace"
    proposed: dict[str, str] = {}
    codes: list[int] = []

    def finish(agent_id: str) -> None:
        app.state.agents.mark_finished(
            agent_id,
            state=AgentState.DONE,
            session_id=f"session-{agent_id}",
            message=f"{agent_id} finished",
            run_id=f"run-{agent_id}",
        )

    def propose() -> None:
        proposed["id"] = _propose(client, csrf)["proposal"]["id"]

    def create(n: int) -> None:
        resp = client.post(
            "/api/projects",
            json={"name": f"Project {n:02d}", "cwd": str(workspace)},
            headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
        )
        codes.append(resp.status_code)

    work: list[Callable[[], None]] = [
        lambda: finish(agent_ids[0]),
        lambda: finish(agent_ids[1]),
        propose,
    ]
    work += [(lambda n=n: create(n)) for n in range(BURST)]  # type: ignore[misc]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in [pool.submit(job) for job in work]:
            future.result()

    assert [code for code in codes if code != 200] == []
    action_id = proposed["id"]

    restarted = _app(tmp_path, fake_collector, helper)
    with TestClient(restarted):
        for agent_id in agent_ids:
            outcome = restarted.state.agents.outcome(agent_id)
            assert outcome is not None, f"{agent_id}'s completion did not survive"
            assert outcome.message == f"{agent_id} finished"
            assert outcome.session_id == f"session-{agent_id}"
        record = restarted.state.host_actions.get(action_id)
        assert record.proposal.id == action_id
        assert record.proposal.summary
        names = {p.name for p in restarted.state.projects.list()}
    assert {f"Project {n:02d}" for n in range(BURST)} <= names


# --- the boundary itself -----------------------------------------------------


def _discover_stores(app: FastAPI) -> dict[str, object]:
    """Every app-owned store reachable from ``app.state``, by attribute path.

    A store is a ``scufris``-defined object whose class name ends in ``Store``.
    One level of nesting is walked because ``SchedulerStore`` hangs off
    ``HostScheduler`` rather than off ``app.state`` directly, and a store owned by
    the thing that uses it is on the same boundary as one owned by the app.
    """

    def is_store(value: object) -> bool:
        cls = type(value)
        return cls.__module__.startswith("scufris.") and cls.__name__.endswith("Store")

    found: dict[str, object] = {}
    # Starlette's `State` keeps its attributes in `_state`, not in `__dict__`.
    for name, value in app.state._state.items():
        if is_store(value):
            found[name] = value
            continue
        if type(value).__module__.startswith("scufris."):
            for inner in vars(value).values():
                if is_store(inner):
                    found[f"{name}.store"] = inner
    return found


def test_post_host_state_uses_declared_persistence_boundary(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: MakeClient,
) -> None:
    """Every app-owned store holds the app's ONE handle, and none writes JSON.

    Both halves are needed. The first catches a store constructed with its own
    ``Database``; the second catches one that takes the handle and keeps a JSON
    sidecar anyway - which is how a "migrated" store still loses a write.

    The store set is DISCOVERED from ``app.state`` rather than listed, so a
    seventh store has to hold the boundary rather than be silently unchecked by a
    test that only knows the six that existed when it was written (review round 1,
    R1.2). The names below are asserted as a floor on what discovery found: a
    walk that silently stopped finding stores would otherwise pass with nothing
    to check.
    """
    app = _app(tmp_path, fake_collector, helper)
    client = make_client(app)
    csrf = _login(client)

    db = app.state.db
    assert isinstance(db, Database)
    stores = _discover_stores(app)
    assert {
        "agents",
        "config_changes",
        "digests",
        "host_actions",
        "host_scheduler.store",
        "projects",
        "sessions",
    } <= set(stores)
    for name, store in sorted(stores.items()):
        held = getattr(store, "_db", None)
        assert held is db, f"{name} does not hold the app's Database ({held!r})"

    # Then MOVE every one of them, so a store that kept a file has written it.
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    app.state.projects.create(name="Boundary", cwd=str(workspace))
    _propose(client, csrf)
    app.state.host_scheduler.store.all()
    client.post(
        "/api/host/digests/run",
        params={"schedule": "watch"},
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )

    strays = sorted(p.name for p in Path(tmp_path).glob("*.json"))
    assert strays == [], (
        f"a runtime store still writes JSON under the state dir: {strays}"
    )


def test_state_database_is_private_with_a_live_session(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: MakeClient,
) -> None:
    """A logged-in operator's session id is in a file only this uid can read.

    ``-wal`` and ``-shm`` are asserted alongside the database because SQLite
    writes committed data into them before the next checkpoint folds it back: a
    0600 database with a 0644 write-ahead log publishes the same session id.
    """
    app = _app(tmp_path, fake_collector, helper)
    client = make_client(app)
    _login(client)
    session_id = client.cookies[SESSION_COOKIE]

    with app.state.db.transaction() as conn:
        live = set(conn.scalars(select(AuthSessionRow.id)).all())
    assert session_id in live

    path = database_path(Path(tmp_path).resolve())
    for candidate in (path, *(Path(f"{path}{suffix}") for suffix in SIDECAR_SUFFIXES)):
        assert candidate.is_file(), f"{candidate.name} is missing"
        mode = stat.S_IMODE(candidate.stat().st_mode)
        assert mode == FILE_MODE, f"{candidate.name} is {oct(mode)}, not 0600"


def test_privileged_audit_remains_an_external_boundary(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: MakeClient,
) -> None:
    """The audit log stays the helper's file; the app's journal is a SECOND record.

    Three things, and the third is the point: an applied action writes BOTH the
    helper's audit line and the app's ``host_action`` row. They are two records of
    one act, kept by two owners, and the app can rewrite only its own.
    """
    audit_source = (Path(__file__).parent.parent / "scufris" / "hostd").resolve()
    offenders = [
        path.name
        for path in sorted(audit_source.glob("*.py"))
        if "scufris.db" in path.read_text() or "from ..db" in path.read_text()
    ]
    assert offenders == [], (
        f"the privileged helper imports the app database: {offenders}"
    )

    app = _app(tmp_path, fake_collector, helper)
    client = make_client(app)
    csrf = _login(client)
    action = _propose(client, csrf)
    action_id = action["proposal"]["id"]
    resp = client.post(
        f"/api/host/actions/{action_id}/approve",
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )
    assert resp.status_code == 200, resp.text
    _settle(client, action_id)

    assert helper.audit.path.is_file()
    applied = [
        entry
        for entry in helper.audit.path.read_text().splitlines()
        if action_id in entry
    ]
    assert applied, "the helper recorded nothing about the applied action"

    with app.state.db.transaction() as conn:
        row = conn.execute(
            select(HostActionRow.decision, HostActionRow.decided_by).where(
                HostActionRow.id == action_id
            )
        ).first()
    assert row is not None, "the app kept no row for the action it applied"
    assert row.decision == "approved"
    assert row.decided_by.startswith("operator")


def _settle(client: TestClient, action_id: str, tries: int = 200) -> dict[str, Any]:
    """Poll one action until the apply has produced a result or an error."""
    for _ in range(tries):
        record = client.get(f"/api/host/actions/{action_id}").json()
        if record.get("result") is not None or record.get("error"):
            return record
    raise AssertionError(f"host action {action_id} never settled")


# --- the decision journal ----------------------------------------------------


def test_host_proposal_decisions_survive_restart(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: MakeClient,
) -> None:
    """A denial - decision, operator and reason - is still there after a restart,
    and ``refresh_pending`` still reconciles the PENDING set from the helper.

    The two halves are one property: the journal is the app's, the pending set is
    the helper's, and a restart must not let either answer for the other. A denied
    action is exactly where they differ - the helper has burned it, so nothing
    would bring it back if the app had not kept it.
    """
    app = _app(tmp_path, fake_collector, helper)
    client = make_client(app)
    csrf = _login(client)
    denied = _propose(client, csrf)["proposal"]["id"]
    # A different unit, so the helper issues a SECOND proposal rather than
    # returning the first one's id.
    still_pending = _propose(client, csrf, args={"unit": "postgresql"})["proposal"][
        "id"
    ]

    resp = client.post(
        f"/api/host/actions/{denied}/deny",
        json={"reason": "not during the release"},
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )
    assert resp.status_code == 200, resp.text

    restarted = _app(tmp_path, fake_collector, helper)
    with TestClient(restarted) as fresh:
        record = restarted.state.host_actions.get(denied)
        assert record.decision == "denied"
        assert record.decided_by.startswith("operator")
        assert record.reason == "not during the release"

        _login(fresh)
        listed = fresh.get("/api/host/actions").json()
    by_id = {item["proposal"]["id"]: item for item in listed}
    assert by_id[still_pending]["decision"] == "pending"
    assert by_id[denied]["decision"] == "denied"


# --- the legacy state directory ---------------------------------------------


def _legacy_state_dir(tmp_path: Path) -> dict[str, Any]:
    """An operator's pre-database state directory, one file per source.

    The session's timestamps are LIVE: the app prunes expired sessions at startup
    with the operator's own idle and absolute windows, so a 1970 record would be
    correctly deleted before anything could present it, and this fixture would be
    testing the sweep rather than the import.
    """
    session_id = "legacy-session-id"
    now = time.time()
    (tmp_path / "auth_sessions.json").write_text(
        json.dumps(
            {
                "sessions": {
                    session_id: {
                        "csrf": "legacy-csrf-token",
                        "created_at": now - 60.0,
                        "last_seen": now - 30.0,
                    }
                }
            }
        )
    )
    (tmp_path / "schedules.json").write_text(
        json.dumps(
            {
                "schedules": {
                    "watch": {
                        "name": "watch",
                        # Due in an hour, for the same reason: a due time in the
                        # past is a window the live scheduler MISSED, and its tick
                        # would rewrite the counters this asserts on.
                        "next_due": now + 3600.0,
                        "last_run": now - 600.0,
                        "last_result": "ran (ok), delivered",
                        "missed": 3,
                        "runs": 7,
                    }
                }
            }
        )
    )
    (tmp_path / "digests.json").write_text(
        json.dumps(
            {
                "digests": [
                    {
                        "at": 500.0,
                        "schedule": "daily",
                        "verdict": "ok",
                        "text": "09:00 - all clear on 4 check(s)",
                        "delivered": True,
                        "delivery_error": "",
                        "states": {"disk": "ok"},
                    }
                ]
            }
        )
    )
    return {"session_id": session_id}


def test_post_host_state_migrates_transactionally(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: MakeClient,
) -> None:
    """Auth, schedule and digest JSON land together, each with its own gate row.

    A live legacy session AUTHENTICATES after the import - the strongest form of
    "the login survived the upgrade" available, and one an assertion about rows
    would not give.
    """
    legacy = _legacy_state_dir(tmp_path)

    app = _app(tmp_path, fake_collector, helper)
    client = make_client(app)
    client.cookies.set(SESSION_COOKIE, legacy["session_id"])
    assert client.get("/api/auth/session").json()["authenticated"] is True

    schedules = {state.name: state for state in app.state.host_scheduler.store.all()}
    assert schedules["watch"].runs == 7
    assert schedules["watch"].missed == 3
    assert schedules["watch"].last_result == "ran (ok), delivered"

    digests = app.state.digests.list()
    assert [d.text for d in digests] == ["09:00 - all clear on 4 check(s)"]
    assert digests[0].delivered is True
    assert digests[0].states == {"disk": "ok"}

    with app.state.db.transaction() as conn:
        gates = set(conn.scalars(select(LegacyImportRow.source)).all())
    assert {"auth_sessions.json", "schedules.json", "digests.json"} <= gates

    # A host action has no legacy file: the store was memory-only. Nothing is
    # imported for it, and nothing pretends to have been.
    assert "host_actions.json" not in gates


def test_a_legacy_password_login_still_works_after_the_import(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: MakeClient,
) -> None:
    """The ordinary path still mints a session after a legacy import ran."""
    _legacy_state_dir(tmp_path)
    app = _app(tmp_path, fake_collector, helper)
    client = make_client(app)
    resp = client.post(
        "/api/auth/login", json={"password": PASSWORD}, headers={"Origin": ORIGIN}
    )
    assert resp.status_code == 200, resp.text
    assert client.get("/api/auth/session").json()["authenticated"] is True
    assert app.state.db is not None
