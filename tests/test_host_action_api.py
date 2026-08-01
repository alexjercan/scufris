"""The approval boundary, driven over HTTP against a real hostd socket.

These tests run the actual helper - its socket, its protocol, its proposal
registry, its audit log - in a background thread, and drive the app through
``TestClient``. Nothing is mocked between the HTTP request and the helper's
refusals, because the property under test is exactly what happens at that
boundary.

The load-bearing one is ``test_machine_token_cannot_approve_a_host_action``.
The app's own MCP tool subprocesses hold a bearer token, and the auth
middleware short-circuits on it BEFORE the session and CSRF checks - so an
agent could otherwise approve its own proposal, and the whole
propose/preview/approve contract would be a description of something the code
does not do (LESSONS.md, enforcement-point-not-the-decision-record).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from conftest import ORIGIN, _Helper, _login, _propose, _settings
from fastapi.testclient import TestClient

from scufris.app import create_app
from scufris.auth import CSRF_HEADER
from scufris.enums import AuthPolicy
from scufris.hostd import (
    AuditEvent,
)
from scufris.metrics import Collector


def test_host_action_requires_preview_and_approval(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    """Proposing runs nothing; approving is what runs it, and only once.

    "Regardless of the path it was requested through" is what the route sweep
    at the end asserts: no route other than approve reaches execution.
    """
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    client = make_client(app)
    csrf = _login(client)

    action = _propose(client, csrf)

    # A proposal is inert, and it arrives with a preview the operator can read.
    assert helper.executor.calls == []
    assert action["decision"] == "pending"
    assert action["proposal"]["preview"]["lines"]
    assert "not a prediction" in action["proposal"]["preview"]["label"]
    assert [step["argv"] for step in action["proposal"]["steps"]] == [
        ["systemctl", "restart", "--", "nginx.service"],
    ]

    # Reading it, listing it and auditing it all leave it unexecuted.
    assert (
        client.get(f"/api/host/actions/{action['proposal']['id']}").status_code == 200
    )
    assert client.get("/api/host/actions").status_code == 200
    assert client.get("/api/host/audit").status_code == 200
    assert helper.executor.calls == []

    resp = client.post(
        f"/api/host/actions/{action['proposal']['id']}/approve",
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )
    assert resp.status_code == 200, resp.text
    _settle(client, action)
    assert helper.executor.calls == [["systemctl", "restart", "--", "nginx.service"]]


def test_machine_token_cannot_approve_a_host_action(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    """The credential the agent's tool subprocesses hold is refused at approve.

    Asserted at the middleware that enforces it, with the token that actually
    exists in production - not inferred from a decision record.
    """
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    client = make_client(app)
    token = app.state.api_token
    machine = {"Authorization": f"Bearer {token}", "Origin": ORIGIN}

    # The machine token works for proposing: an agent is allowed to ask.
    proposed = client.post(
        "/api/host/actions",
        json={"kind": "unit_restart", "args": {"unit": "nginx"}},
        headers=machine,
    )
    assert proposed.status_code == 201, proposed.text
    action_id = proposed.json()["proposal"]["id"]

    # And it is refused for every operator decision, with no CSRF token needed
    # to get that far - the refusal is BEFORE the bearer short-circuit.
    for verb in ("approve", "deny", "revert", "cancel"):
        resp = client.post(f"/api/host/actions/{action_id}/{verb}", headers=machine)
        assert resp.status_code == 403, f"{verb} answered {resp.status_code}"
        assert "operator session" in resp.json()["detail"]

    assert helper.executor.calls == []


def test_an_anonymous_caller_cannot_approve_with_auth_off(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    """No credential at all must not be a way past the approval gate.

    The earlier version of the check asked "is a bearer token present?", so a
    caller that sent NO Authorization header sailed through the auth-off
    short-circuit and executed a root command anonymously. On loopback that is
    any process on this machine - including the shell the model runs its own
    commands in, which needs only
    `curl -XPOST http://127.0.0.1:8000/api/host/actions/<id>/approve`.

    The test that was supposed to cover this sent `Bearer anything` and so
    stepped over the hole. This one sends nothing (review round 1, R1.1).
    """
    app = create_app(
        collector=fake_collector,
        settings=_settings(tmp_path, helper, auth_mode=AuthPolicy.DISABLED),
    )
    client = make_client(app)

    proposed = client.post(
        "/api/host/actions",
        json={"kind": "unit_restart", "args": {"unit": "nginx"}},
        headers={"Origin": ORIGIN},
    )
    assert proposed.status_code == 201, proposed.text
    action_id = proposed.json()["proposal"]["id"]

    for verb in ("approve", "deny", "revert", "cancel"):
        resp = client.post(f"/api/host/actions/{action_id}/{verb}")
        assert resp.status_code in (401, 403), (
            f"{verb} answered {resp.status_code} to a caller with no credential at all"
        )
    assert helper.executor.calls == []


def test_host_agency_without_an_operator_credential_refuses_to_start(
    tmp_path: Path, fake_collector: Collector, helper: _Helper
) -> None:
    """Fail closed at construction: no human to approve means no host agency.

    The middleware refuses these paths anyway; this is the layer that stops the
    deployment existing in the first place, so the guarantee does not rest on a
    single check.
    """
    from scufris.auth import AuthConfigError

    with pytest.raises(AuthConfigError) as refused:
        create_app(
            collector=fake_collector,
            settings=_settings(
                tmp_path,
                helper,
                auth_mode=AuthPolicy.DISABLED,
                auth_password_hash=None,
            ),
        )
    assert "no human to be" in str(refused.value)


def test_every_mutating_host_route_is_operator_only(
    tmp_path: Path, fake_collector: Collector, helper: _Helper
) -> None:
    """The sweep auth.py's comment promised, which did not exist.

    Enumerated from `app.routes`, so a host route added later is covered by
    existing rather than by someone remembering to extend a list.

    The exception is drawn at PRIVILEGE, not at the HTTP verb: an agent may ask
    for something (propose an action, build a configuration) and may stop its own
    asking, because none of that runs as root or changes the system. Approving,
    denying, reverting and cancelling an apply are the acts that reach the root
    helper, and those are the operator's alone.
    """
    from starlette.routing import Route

    from scufris.auth import operator_only

    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    may_be_machine_driven = {
        "/api/host/actions",  # propose: an agent may ask
        # Build a committed configuration, and stop a build. Unprivileged: it
        # runs as this process's user, writes only to the nix store, and produces
        # a PROPOSAL that still needs the operator.
        "/api/host/config/changes",
        "/api/host/config/changes/{change_id}/cancel",
    }

    checked = 0
    for route in app.routes:
        if not isinstance(route, Route) or not route.path.startswith("/api/host/"):
            continue
        for method in sorted((route.methods or set()) - {"HEAD", "OPTIONS", "GET"}):
            if route.path in may_be_machine_driven:
                continue
            assert operator_only(route.path), (
                f"{method} {route.path} changes host state but is not "
                "operator-only; a machine token would be accepted"
            )
            checked += 1
    assert checked >= 4, f"the sweep found only {checked} mutating host routes"


def test_a_machine_token_cannot_approve_even_with_auth_off(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    """An agent approving its own proposal has nothing to do with the bind address.

    On loopback the dashboard runs open by design, which is a statement about
    the NETWORK. It is not a statement about the process the app itself spawned
    with a credential in its hand.
    """
    # A credential IS configured (host agency without one is refused at
    # construction, see the test above); authentication is merely switched off
    # for the network. The machine token must still not approve.
    app = create_app(
        collector=fake_collector,
        settings=_settings(tmp_path, helper, auth_mode=AuthPolicy.DISABLED),
    )
    client = make_client(app)

    proposed = client.post(
        "/api/host/actions",
        json={"kind": "unit_restart", "args": {"unit": "nginx"}},
        headers={"Origin": ORIGIN},
    )
    assert proposed.status_code == 201, proposed.text
    action_id = proposed.json()["proposal"]["id"]

    refused = client.post(
        f"/api/host/actions/{action_id}/approve",
        headers={"Authorization": "Bearer anything", "Origin": ORIGIN},
    )
    assert refused.status_code == 403
    assert helper.executor.calls == []


def test_an_approval_is_not_replayed_over_http(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    """Approving twice runs once - refused by the app AND by the helper."""
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    client = make_client(app)
    csrf = _login(client)
    action = _propose(client, csrf)
    action_id = action["proposal"]["id"]
    headers = {"Origin": ORIGIN, CSRF_HEADER: csrf}

    first = client.post(f"/api/host/actions/{action_id}/approve", headers=headers)
    assert first.status_code == 200
    _settle(client, action)

    second = client.post(f"/api/host/actions/{action_id}/approve", headers=headers)
    assert second.status_code == 409
    assert len(helper.executor.calls) == 1


def test_a_denied_action_cannot_then_be_approved(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    client = make_client(app)
    csrf = _login(client)
    action_id = _propose(client, csrf)["proposal"]["id"]
    headers = {"Origin": ORIGIN, CSRF_HEADER: csrf}

    denied = client.post(
        f"/api/host/actions/{action_id}/deny",
        json={"reason": "not now"},
        headers=headers,
    )
    assert denied.status_code == 200
    assert denied.json()["decision"] == "denied"

    resp = client.post(f"/api/host/actions/{action_id}/approve", headers=headers)
    assert resp.status_code == 409
    assert helper.executor.calls == []


def test_a_refused_action_never_becomes_a_proposal(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    """A deny-listed unit has no approvable proposal to begin with."""
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    client = make_client(app)
    csrf = _login(client)

    resp = client.post(
        "/api/host/actions",
        json={"kind": "unit_restart", "args": {"unit": "sshd"}},
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )
    assert resp.status_code == 422
    assert "deny-list" in resp.json()["detail"]
    assert client.get("/api/host/actions").json() == []


def test_an_unknown_verb_is_refused_before_it_reaches_the_helper(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    client = make_client(app)
    csrf = _login(client)

    resp = client.post(
        "/api/host/actions",
        json={"kind": "run_shell", "args": {"cmd": "rm -rf /"}},
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )
    assert resp.status_code == 422
    assert helper.executor.calls == []


def test_host_actions_are_audited_through_the_api(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    """The audit the dashboard reads is the helper's own root-written log."""
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    client = make_client(app)
    csrf = _login(client)
    action = _propose(client, csrf, agent="ops-1", run="run-7")
    action_id = action["proposal"]["id"]
    client.post(
        f"/api/host/actions/{action_id}/approve",
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )
    _settle(client, action)

    records = client.get("/api/host/audit?limit=50").json()
    events = [record["event"] for record in records]
    assert AuditEvent.REQUESTED in events
    assert AuditEvent.APPROVED in events
    assert AuditEvent.APPLIED in events

    requested = next(r for r in records if r["event"] == AuditEvent.REQUESTED)
    assert requested["requester"]["agent"] == "ops-1"
    assert requested["requester"]["run"] == "run-7"
    approved = next(r for r in records if r["event"] == AuditEvent.APPROVED)
    assert approved["requester"]["actor"].startswith("operator:")


def test_a_machine_proposal_is_never_audited_as_the_operator(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    """Who asked is derived from the CREDENTIAL, not from the request body.

    The first version read `actor = "agent" if body.agent else operator`, and the
    MCP tool sent no `agent` field - so every agent-originated proposal was
    written into the root-owned audit as the operator's. That is the one field
    the audit exists to answer (review round 1, R1.6).
    """
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    client = make_client(app)
    machine = {
        "Authorization": f"Bearer {app.state.api_token}",
        "Origin": ORIGIN,
    }

    # No `agent` field at all - exactly what the MCP tool used to send.
    proposed = client.post(
        "/api/host/actions",
        json={"kind": "unit_restart", "args": {"unit": "nginx"}},
        headers=machine,
    )
    assert proposed.status_code == 201, proposed.text

    requested = [
        record
        for record in helper.audit.tail(20)
        if record.event is AuditEvent.REQUESTED
    ]
    assert requested, "the proposal was not audited at all"
    actor = requested[-1].requester.actor
    assert actor == "agent", f"a machine proposal was audited as {actor!r}"
    assert not actor.startswith("operator")
    # It still names WHICH agent, so the record is useful.
    assert requested[-1].requester.agent

    # A claim in the body cannot promote a machine caller to the operator.
    client.post(
        "/api/host/actions",
        json={
            "kind": "unit_stop",
            "args": {"unit": "nginx"},
            "agent": "operator:deadbeef",
        },
        headers=machine,
    )
    actors = {r.requester.actor for r in helper.audit.tail(20)}
    assert not any(a.startswith("operator") for a in actors), actors


def test_an_operator_proposal_is_audited_as_the_operator(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    """The paired guard: the fix must not label everything an agent."""
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    client = make_client(app)
    csrf = _login(client)

    _propose(client, csrf)

    requested = [
        record
        for record in helper.audit.tail(20)
        if record.event is AuditEvent.REQUESTED
    ]
    assert requested[-1].requester.actor.startswith("operator:")


def test_the_helper_being_absent_is_a_503_not_a_half_working_surface(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    """With no secret configured there is no privileged surface at all."""
    app = create_app(
        collector=fake_collector,
        settings=_settings(tmp_path, helper, hostd_secret=""),
    )
    client = make_client(app)
    csrf = _login(client)

    resp = client.post(
        "/api/host/actions",
        json={"kind": "unit_restart", "args": {"unit": "nginx"}},
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )
    assert resp.status_code == 503
    assert "not configured" in resp.json()["detail"]


def test_a_wrong_secret_reaches_nothing(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    """The socket authenticates every frame, and records the attempt."""
    app = create_app(
        collector=fake_collector,
        settings=_settings(tmp_path, helper, hostd_secret="not-the-secret"),
    )
    client = make_client(app)
    csrf = _login(client)

    resp = client.post(
        "/api/host/actions",
        json={"kind": "unit_restart", "args": {"unit": "nginx"}},
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )
    assert resp.status_code == 502
    refusals = [
        record for record in helper.audit.tail(20) if record.event is AuditEvent.REFUSED
    ]
    assert refusals and "no valid secret" in refusals[-1].detail


def test_reverting_an_applied_action_proposes_its_inverse(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    """An undo is itself a host action: it gets a preview and an approval."""
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    client = make_client(app)
    csrf = _login(client)
    headers = {"Origin": ORIGIN, CSRF_HEADER: csrf}
    action = _propose(client, csrf, kind="unit_stop", args={"unit": "nginx"})
    action_id = action["proposal"]["id"]
    assert action["proposal"]["reversal"]["possible"]

    client.post(f"/api/host/actions/{action_id}/approve", headers=headers)
    _settle(client, action)

    reverted = client.post(f"/api/host/actions/{action_id}/revert", headers=headers)
    assert reverted.status_code == 201, reverted.text
    inverse = reverted.json()
    assert inverse["decision"] == "pending"  # it still needs an approval
    assert inverse["proposal"]["steps"][0]["argv"] == [
        "systemctl",
        "start",
        "--",
        "nginx.service",
    ]
    assert len(helper.executor.calls) == 1  # nothing ran on the revert call


def test_a_one_way_action_refuses_to_be_reverted(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    client = make_client(app)
    csrf = _login(client)
    headers = {"Origin": ORIGIN, CSRF_HEADER: csrf}
    action = _propose(client, csrf, kind="gc_store", args={})
    action_id = action["proposal"]["id"]
    # A one-way action takes the STRONG path: the ordinary approve is refused, so
    # this one carries the acknowledgement token the service requires.
    client.post(
        f"/api/host/actions/{action_id}/approve",
        headers=headers,
        json={"acknowledge": "gc_store"},
    )
    _settle(client, action)

    resp = client.post(f"/api/host/actions/{action_id}/revert", headers=headers)
    assert resp.status_code == 422
    assert "ONE-WAY" in resp.json()["detail"]


def test_cancelling_a_live_apply_is_recorded(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    """The stop button reaches root, and the helper writes down what happened.

    This exercises the path that is easy to get wrong: a supervisor cancel
    closes the client's async generator, which arrives as GeneratorExit at the
    yield rather than as CancelledError. The cancel frame is written from the
    generator's finally for exactly that reason.
    """
    helper.executor.hang = True
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    client = make_client(app)
    csrf = _login(client)
    headers = {"Origin": ORIGIN, CSRF_HEADER: csrf}
    action_id = _propose(client, csrf)["proposal"]["id"]

    started = client.post(f"/api/host/actions/{action_id}/approve", headers=headers)
    assert started.status_code == 200

    # Wait for the apply to be genuinely in flight before cancelling: a cancel
    # that lands before the command started would prove nothing. The wait is on
    # the executor having been CALLED, which is the fact that matters, rather
    # than on a duration.
    _until(lambda: bool(helper.executor.calls), "the apply never reached the executor")

    cancelled = client.post(f"/api/host/actions/{action_id}/cancel", headers=headers)
    assert cancelled.status_code == 200

    def _cancelled() -> list[Any]:
        return [
            record
            for record in helper.audit.tail(20)
            if record.event is AuditEvent.CANCELLED
        ]

    _until(lambda: bool(_cancelled()), "the helper did not record the cancellation")
    records = _cancelled()
    assert records[-1].action_id == action_id
    assert "already done stands" in records[-1].detail

    record = client.get(f"/api/host/actions/{action_id}").json()
    assert "cancelled mid-apply" in record["error"]


def test_cancelling_an_action_with_no_run_is_refused(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    client = make_client(app)
    csrf = _login(client)
    action_id = _propose(client, csrf)["proposal"]["id"]

    resp = client.post(
        f"/api/host/actions/{action_id}/cancel",
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )
    assert resp.status_code == 409


def _until(
    condition: Callable[[], bool], message: str, *, timeout: float = 10.0
) -> None:
    """Wait for ``condition``, or fail with ``message``.

    A generous timeout on a condition beats a fixed number of short sleeps: the
    latter fails under load while looking like a real defect, which is what made
    the cancellation test read as flaky (review round 1, R1.13).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError(message)


def _settle(client: TestClient, action: dict[str, Any], tries: int = 200) -> None:
    """Wait for an approved action's background run to reach a terminal state.

    The approval endpoint STARTS the apply and returns; the run outlives the
    request by design (ADR-001), so a test that asserts on the outcome has to
    wait for it rather than assume the response implied it.
    """
    action_id = action["proposal"]["id"]
    for _ in range(tries):
        record = client.get(f"/api/host/actions/{action_id}").json()
        if record["result"] is not None or record["error"]:
            return
        time.sleep(0.02)
    raise AssertionError("the approved action never settled")
