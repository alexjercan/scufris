"""The decision core: the rules one service applies to every host action.

One service decides host actions for both surfaces, so these drive the WEB
surface and assert on rules that belong to the service rather than to the
route. ``test_the_web_surface_owns_no_decision_rule_of_its_own`` is what pins
that: the same methods answer a chat-derived actor identically.

Covers confirmation strength, the approval race, expiry, durability across a
restart, operator binding, delivery of the outcome back to the requesting
agent, and the queue row the page renders.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from conftest import ORIGIN, _Helper, _login, _propose, _settings, patch_get_backend
from fastapi.testclient import TestClient
from test_host_action_api import _settle

from scufris.agent import AgentReply, StreamDone
from scufris.agent_store import HOST_AGENT_ID
from scufris.app import create_app
from scufris.auth import CSRF_HEADER
from scufris.config import Settings
from scufris.host_approvals import ConfirmationRequired
from scufris.hostd import (
    AuditEvent,
)
from scufris_host import Collector


def test_one_way_action_requires_stronger_confirmation(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    """A one-way action cannot be approved through the ordinary confirmation.

    `gc_store` frees store paths that do not come back, so the ordinary approve is
    refused (422) and NOTHING runs; the strong path carries the acknowledgement the
    service names. A reversible R1 restart still approves with no ceremony - the
    strong path has to stay rare or it stops meaning anything.
    """
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    client = make_client(app)
    csrf = _login(client)
    headers = {"Origin": ORIGIN, CSRF_HEADER: csrf}

    one_way = _propose(client, csrf, kind="gc_store", args={})
    one_way_id = one_way["proposal"]["id"]

    # What the surfaces render, computed once by the service.
    confirmation = client.get(f"/api/host/actions/{one_way_id}/confirmation").json()
    assert confirmation["style"] == "one_way"
    assert confirmation["acknowledge"] == "gc_store"
    assert confirmation["no_undo"] is True
    assert "ONE-WAY" in confirmation["risk_label"]

    # The ordinary approve: refused, and nothing reached the executor.
    ordinary = client.post(f"/api/host/actions/{one_way_id}/approve", headers=headers)
    assert ordinary.status_code == 422, ordinary.text
    assert "cannot be undone" in ordinary.json()["detail"]
    assert "gc_store" in ordinary.json()["detail"]
    assert helper.executor.calls == []

    # A WRONG acknowledgement is not a confirmation either.
    wrong = client.post(
        f"/api/host/actions/{one_way_id}/approve",
        headers=headers,
        json={"acknowledge": "yes"},
    )
    assert wrong.status_code == 422
    assert helper.executor.calls == []

    # The strong path runs it, and the action is still pending until then.
    strong = client.post(
        f"/api/host/actions/{one_way_id}/approve",
        headers=headers,
        json={"acknowledge": "gc_store"},
    )
    assert strong.status_code == 200, strong.text
    _settle(client, one_way)
    assert helper.executor.calls != []

    # A reversible action needs no acknowledgement: an R1 restart's confirmation is
    # ordinary even though restarting a running unit has no undo, because that is
    # the NORMAL answer for service control rather than the alarming one.
    reversible = _propose(client, csrf)
    reversible_id = reversible["proposal"]["id"]
    conf = client.get(f"/api/host/actions/{reversible_id}/confirmation").json()
    assert conf["style"] == "ordinary"
    assert conf["acknowledge"] == ""
    assert conf["no_undo"] is True  # stated, but not gated
    assert (
        client.post(
            f"/api/host/actions/{reversible_id}/approve", headers=headers
        ).status_code
        == 200
    )


def test_approval_race_yields_one_execution(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    """Two approvals of one proposal produce one execution and one refusal.

    This is the cross-surface race written as the two calls it reduces to: the
    decision is CLAIMED before the apply starts, so whichever arrives second is
    refused by the app - and even if it were not, the helper's own APPLYING state
    would refuse it. Both layers are checked, because "approve twice" must not
    become "run twice" at any of them.
    """
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    client = make_client(app)
    csrf = _login(client)
    headers = {"Origin": ORIGIN, CSRF_HEADER: csrf}
    action = _propose(client, csrf)
    action_id = action["proposal"]["id"]

    first = client.post(f"/api/host/actions/{action_id}/approve", headers=headers)
    second = client.post(f"/api/host/actions/{action_id}/approve", headers=headers)
    assert first.status_code == 200, first.text
    assert second.status_code == 409, second.text
    assert "already" in second.json()["detail"]

    _settle(client, action)
    assert helper.executor.calls == [["systemctl", "restart", "--", "nginx.service"]]

    # And a denial after an approval is refused too: one decision per proposal.
    denied = client.post(
        f"/api/host/actions/{action_id}/deny", headers=headers, json={"reason": "no"}
    )
    assert denied.status_code == 409


def test_an_expired_proposal_is_refused_with_its_reason(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    """An expired proposal is refused BEFORE the helper is asked, and the operator
    is told which of the four things went wrong rather than getting a generic
    failure. The preview no longer describes a decision that can be made."""
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    client = make_client(app)
    csrf = _login(client)
    headers = {"Origin": ORIGIN, CSRF_HEADER: csrf}
    action = _propose(client, csrf)
    action_id = action["proposal"]["id"]

    # Move the app's clock past the proposal's own expiry, leaving the helper's
    # copy alone: this asserts the app refuses on its own, not that the helper did.
    record = app.state.host_actions.get(action_id)
    record.proposal.expires_at = time.time() - 1
    # Written back: the store hands out DETACHED records now, so an edit to one is
    # a local edit until it is persisted.
    app.state.host_actions.refresh(action_id, record.proposal)

    resp = client.post(f"/api/host/actions/{action_id}/approve", headers=headers)
    assert resp.status_code == 409, resp.text
    assert "expired" in resp.json()["detail"]
    assert helper.executor.calls == []


def test_approval_queue_survives_restart(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    """A restart does not strand a live approval.

    The app's registry is in-memory by design - the HELPER holds every proposal -
    so a second app over the same socket recovers the queue from it (the read-only
    `list_pending` verb) rather than from a state file of its own. Proposal,
    expiry and audit all come back, and the recovered action is approvable exactly
    once.
    """
    settings = _settings(tmp_path, helper)
    first_app = create_app(collector=fake_collector, settings=settings)
    first = make_client(first_app)
    csrf = _login(first)
    action = _propose(first, csrf)
    action_id = action["proposal"]["id"]

    # A brand new app: nothing in memory, same helper on the same socket.
    second_app = create_app(
        collector=fake_collector, settings=_settings(tmp_path, helper)
    )
    second = make_client(second_app)
    csrf2 = _login(second)
    headers = {"Origin": ORIGIN, CSRF_HEADER: csrf2}

    queue = second.get("/api/host/actions").json()
    assert [row["proposal"]["id"] for row in queue] == [action_id]
    recovered = queue[0]
    # Not a stub: the preview, the commands, the expiry and the requester survive,
    # because they are the helper's own record of the proposal.
    assert recovered["decision"] == "pending"
    assert recovered["proposal"]["preview"]["lines"]
    assert [s["argv"] for s in recovered["proposal"]["steps"]] == [
        ["systemctl", "restart", "--", "nginx.service"],
    ]
    assert recovered["proposal"]["expires_at"] == action["proposal"]["expires_at"]

    # The audit is the helper's, so it was never at risk - and it still names the
    # request the first app made.
    audit = second.get("/api/host/audit").json()
    assert any(row["action_id"] == action_id for row in audit)

    # And the recovered action is really approvable: once.
    approved = second.post(f"/api/host/actions/{action_id}/approve", headers=headers)
    assert approved.status_code == 200, approved.text
    _settle(second, action)
    assert helper.executor.calls == [["systemctl", "restart", "--", "nginx.service"]]
    again = second.post(f"/api/host/actions/{action_id}/approve", headers=headers)
    assert again.status_code == 409


def test_a_proposal_made_by_another_client_appears_in_the_queue(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    """The queue shows what the HELPER holds, not only what this process created.

    The same reconcile that recovers a restart also means a proposal made by
    another client of the socket - the example script, a second process - is not
    invisible to the operator who has to decide it.
    """
    # No reconcile throttle here: the property under test is that the queue asks
    # the helper at all, not how often it is willing to.
    app = create_app(
        collector=fake_collector,
        settings=_settings(tmp_path, helper, host_queue_refresh_seconds=0.0),
    )
    client = make_client(app)
    _login(client)
    assert client.get("/api/host/actions").json() == []

    other = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    other_client = make_client(other)
    csrf = _login(other_client)
    action_id = _propose(other_client, csrf)["proposal"]["id"]

    # The first app never saw that request. It asks the helper.
    queue = client.get("/api/host/actions").json()
    assert [row["proposal"]["id"] for row in queue] == [action_id]


class _RecordingBackend:
    """A backend that records the prompts turns are launched with.

    The property these tests assert is that a DECISION reaches the agent that asked
    for the action - which means a real turn, resuming that agent's session, whose
    prompt carries the outcome. Recording the prompt is how you see that from
    outside; asserting on the store would only prove a note was written somewhere.
    """

    name = "recording"

    def __init__(self) -> None:
        self.turns: list[tuple[str, str]] = []  # (agent_id, prompt)

    async def stream(
        self,
        settings: Settings,
        prompt: str,
        *,
        session_id: str | None = None,
        cwd: str | None = None,
        image_paths: list[str] | None = None,
        permission_mode: str = "manual",
        is_orchestrator: bool = False,
        agent_id: str = "",
    ) -> Any:
        self.turns.append((agent_id, prompt))
        yield StreamDone(
            reply=AgentReply(text="ok", status="completed"),
            session_id=session_id or "sess-host",
        )

    def read_status(self, settings: Settings, session_id: str | None) -> None:
        return None

    def read_transcript(self, settings: Settings, session_id: str | None) -> list[Any]:
        return []


def _wait_for_turn(backend: _RecordingBackend, agent_id: str, tries: int = 200) -> str:
    """The prompt of the next turn launched for ``agent_id``.

    A decision launches the turn in the background (the supervisor), so a test has
    to wait for it rather than assume the response implied it.
    """
    for _ in range(tries):
        for launched_id, prompt in backend.turns:
            if launched_id == agent_id:
                return prompt
        time.sleep(0.02)
    raise AssertionError(f"no turn was launched for {agent_id}")


def _propose_as_the_host_agent(client: TestClient, app: Any, **body: Any) -> str:
    """Propose the way the host agent really does: with the machine bearer token its
    MCP subprocess carries, naming itself. Returns the action id."""
    payload: dict[str, Any] = {
        "kind": "unit_restart",
        "args": {"unit": "nginx"},
        "agent": HOST_AGENT_ID,
    }
    payload.update(body)
    resp = client.post(
        "/api/host/actions",
        json=payload,
        headers={"Authorization": f"Bearer {app.state.api_token}", "Origin": ORIGIN},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["proposal"]["id"]


def test_pending_approval_is_operator_bound(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proposal leaves the requesting agent BLOCKED, and only the operator can
    end that.

    BLOCKED rather than WAITING is the whole point: the orchestrator SEES that a
    delegated host change is sitting with the operator, and is refused when it tries
    to answer it. Without the refusal the orchestrator would resume the agent with
    "approved, go ahead" - an answer it has no authority to give, on a machine where
    only a session or an allowlisted chat can reach `apply`.
    """
    backend = _RecordingBackend()
    patch_get_backend(monkeypatch, backend)
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    client = make_client(app)
    # A second client for the machine caller, so it has no session cookie: the app
    # derives "who is asking" from the credential, and a test that sent both would
    # be testing an operator with an Authorization header rather than an agent.
    agent_client = make_client(app)
    machine = {"Authorization": f"Bearer {app.state.api_token}", "Origin": ORIGIN}
    action_id = _propose_as_the_host_agent(agent_client, app)

    csrf = _login(client)
    pending = client.get("/api/agents/pending").json()
    row = next(r for r in pending if r["agent_id"] == HOST_AGENT_ID)
    assert row["state"] == "blocked"
    assert action_id in row["message"]

    # The orchestrator cannot answer it, and is told why in terms it can act on.
    answered = agent_client.post(
        f"/api/agents/{HOST_AGENT_ID}/chat",
        json={"message": "approved, go ahead"},
        headers=machine,
    )
    assert answered.status_code == 409, answered.text
    assert "OPERATOR" in answered.json()["detail"]
    assert backend.turns == []

    # Nor can it clear the signal by acknowledging it.
    acked = agent_client.post(
        f"/api/agents/{HOST_AGENT_ID}/acknowledge", headers=machine
    )
    assert acked.json()["acknowledged"] is False
    assert client.get("/api/agents/pending").json()

    # The OPERATOR may still talk to the agent: reading its own chat is not deciding.
    operator = {"Origin": ORIGIN, CSRF_HEADER: csrf}
    with client.stream(
        "POST",
        f"/api/agents/{HOST_AGENT_ID}/chat",
        json={"message": "what are you waiting for?"},
        headers=operator,
    ) as streamed:
        assert streamed.status_code == 200
        streamed.read()
    assert _wait_for_turn(backend, HOST_AGENT_ID)


def test_denial_reaches_the_requesting_agent(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A denial resumes the agent that asked, carrying the reason.

    An agent that never hears why it was refused proposes the same thing again; that
    is the failure this closes. The reason travels from the operator's deny call,
    through the helper (which burns the proposal), into a turn on the agent's own
    session.
    """
    backend = _RecordingBackend()
    patch_get_backend(monkeypatch, backend)
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    client = make_client(app)
    action_id = _propose_as_the_host_agent(client, app)
    csrf = _login(client)

    denied = client.post(
        f"/api/host/actions/{action_id}/deny",
        json={"reason": "nginx is serving the demo right now; ask me after 18:00"},
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )
    assert denied.status_code == 200, denied.text

    prompt = _wait_for_turn(backend, HOST_AGENT_ID)
    assert "DENIED" in prompt
    assert "ask me after 18:00" in prompt
    assert action_id in prompt
    # And it is told not to simply retry, which is the point of carrying the reason.
    assert "Do not propose the same action again" in prompt
    assert helper.executor.calls == []


def test_an_applied_result_reaches_the_requesting_agent(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of the round trip: an approved action's OUTCOME comes back.

    The agent is resumed once the apply is terminal, not when the approval is
    granted - an agent told "approved, starting" would spend a turn learning
    nothing, and would report "I proposed it" as though that were the result.
    """
    backend = _RecordingBackend()
    patch_get_backend(monkeypatch, backend)
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    client = make_client(app)
    action_id = _propose_as_the_host_agent(client, app)
    csrf = _login(client)

    approved = client.post(
        f"/api/host/actions/{action_id}/approve",
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )
    assert approved.status_code == 200, approved.text

    prompt = _wait_for_turn(backend, HOST_AGENT_ID)
    assert "APPLIED" in prompt
    assert action_id in prompt


def test_the_web_surface_owns_no_decision_rule_of_its_own(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    """The route is a translator, not a decider.

    "Telegram enforces the same gate as the web path" is only true if there is one
    gate, so this pins the shape the Telegram surface will rely on: the app exposes
    ONE approval service, and its methods are what the routes call. A second decision
    path added later fails this by making the service's own approve unable to produce
    the same effect the route does.
    """
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    client = make_client(app)
    csrf = _login(client)
    action = _propose(client, csrf, kind="gc_store", args={})
    action_id = action["proposal"]["id"]
    approvals = app.state.host_approvals

    # Approving through the SERVICE, with a chat-shaped actor and the same
    # acknowledgement rule, has the same effect the route would have had - the
    # ordinary call is refused first.
    def decide(**kwargs: Any) -> Any:
        # On the app's OWN event loop, which is where the Telegram bot's callbacks
        # run too - the supervisor it starts needs that loop.
        return client.portal.call(  # type: ignore[union-attr]
            functools.partial(
                approvals.approve, action_id, actor="operator:telegram:42", **kwargs
            )
        )

    with pytest.raises(ConfirmationRequired):
        decide()
    decide(acknowledge="gc_store")
    _settle(client, action)
    assert helper.executor.calls != []
    # And the audit records WHICH surface decided, because the actor is the only
    # thing a surface supplies.
    approved = [r for r in helper.audit.tail(50) if r.event == AuditEvent.APPROVED]
    assert approved and approved[-1].requester.actor == "operator:telegram:42"


def test_an_undecided_approval_does_not_strand_the_agent(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    """A proposal the operator never answers must not lock the agent out for good.

    Review round 1, R1.1. The refusals that protect a pending decision - the
    orchestrator cannot message the agent, and cannot acknowledge the signal - were
    keyed on the agent's BLOCKED state, and nothing clears that state when a
    proposal merely EXPIRES: a decision clears it by resuming the agent, an expiry
    resumes nobody. So one abandoned proposal left the host agent unreachable AND
    unacknowledgeable, with the approval that would have freed it no longer
    approvable. Both refusals now depend on the approval being LIVE.
    """
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    client = make_client(app)
    agent_client = make_client(app)
    machine = {"Authorization": f"Bearer {app.state.api_token}", "Origin": ORIGIN}
    action_id = _propose_as_the_host_agent(agent_client, app)
    csrf = _login(client)

    # While the approval is live, both refusals hold - and the refusal now names
    # the action, so the orchestrator can report WHAT is waiting.
    blocked = agent_client.post(
        f"/api/agents/{HOST_AGENT_ID}/chat",
        json={"message": "approved, go ahead"},
        headers=machine,
    )
    assert blocked.status_code == 409
    assert action_id in blocked.json()["detail"]
    assert (
        agent_client.post(
            f"/api/agents/{HOST_AGENT_ID}/acknowledge", headers=machine
        ).json()["acknowledged"]
        is False
    )

    # The operator never decides, and the window closes.
    expired = app.state.host_actions.get(action_id).proposal
    expired.expires_at = time.time() - 1
    app.state.host_actions.refresh(action_id, expired)
    assert (
        client.post(
            f"/api/host/actions/{action_id}/approve",
            headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
        ).status_code
        == 409
    )

    # Now there is nothing to protect: the orchestrator can reach the agent again
    # and clear the stale signal, so delegation keeps working.
    assert (
        agent_client.post(
            f"/api/agents/{HOST_AGENT_ID}/acknowledge", headers=machine
        ).json()["acknowledged"]
        is True
    )
    assert not [
        row
        for row in client.get("/api/agents/pending").json()
        if row["agent_id"] == HOST_AGENT_ID
    ]


def test_a_queue_row_carries_everything_the_page_renders(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
) -> None:
    """The contract between this API and the /host/ page.

    The page renders one row per proposal from the LIST response alone - the risk
    class and its acknowledgement requirement included - so a field dropped here
    does not fail a backend test, it silently empties part of a decision the
    operator is making. These are the fields `renderHost` reads.
    """
    app = create_app(collector=fake_collector, settings=_settings(tmp_path, helper))
    client = make_client(app)
    csrf = _login(client)
    _propose(client, csrf, kind="gc_store", args={})

    rows = client.get("/api/host/actions").json()
    assert len(rows) == 1
    row = rows[0]
    proposal = row["proposal"]

    # What it would run, in order, and what it would change.
    assert proposal["id"] and proposal["kind"] == "gc_store"
    assert [step["argv"] for step in proposal["steps"]]
    assert proposal["summary"]
    assert set(proposal["preview"]) >= {"kind", "label", "available", "lines"}
    assert proposal["expires_at"] > proposal["created_at"]
    assert proposal["state"] == "pending"
    # Who asked, derived from the credential.
    assert set(proposal["requester"]) >= {"actor", "agent", "run"}
    # And the confirmation requirement, INLINE - the page needs the risk of every
    # row to render the row at all, so it must not cost a request per row.
    confirmation = row["confirmation"]
    assert confirmation["style"] == "one_way"
    assert confirmation["acknowledge"] == "gc_store"
    assert confirmation["no_undo"] is True
    assert confirmation["risk"] == "r2" and confirmation["risk_label"]
    assert confirmation["undo"]
    # The decision fields the page renders for a decided row.
    assert set(row) >= {"decision", "decided_by", "reason", "run_id", "result", "error"}
