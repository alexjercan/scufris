"""The host agent: its audience, its record, and the operator-bound pending state.

Three properties, each asserted where it is actually enforced rather than where it
is described:

- the AUDIENCE is physical - the host agent's turn wires up the `host` server (the
  only one carrying the mutating tools) plus the `agent` callbacks, and the
  orchestrator's turn wires up neither;
- the RECORD is reserved and synthetic - it exists without an agents.json row, is
  listed so a delegation target is visible, and refuses every CRUD mutation;
- a pending approval is BLOCKED, not WAITING, so the orchestrator can see it and
  cannot answer it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from scufris.agent import _mcp_overrides, _steer, scufris_mcp_servers
from scufris.agent_store import (
    HOST_AGENT_ID,
    ORCHESTRATOR_ID,
    AgentStore,
    InvalidAgent,
    ReservedAgent,
)
from scufris.config import Settings
from scufris.db import Database
from scufris.enums import AgentState, Audience, PermissionMode, audience_for
from scufris.projects import ProjectStore
from scufris.sessions import (
    AGENT_STEERING_PREAMBLE,
    HOST_STEERING_PREAMBLE,
    STEERING_PREAMBLE,
    strip_steering,
)


def _settings(tmp_path: Path, **kwargs: Any) -> Settings:
    base: dict[str, Any] = {
        "state_dir": tmp_path,
        "agent_tools_enabled": True,
        "agent_backend": "mock",
        "enable_mock_backend": True,
        "_env_file": None,
    }
    base.update(kwargs)
    return Settings(**base)


def _store(tmp_path: Path, database: Database, **kwargs: Any) -> AgentStore:
    settings = _settings(tmp_path, **kwargs)
    return AgentStore(settings, ProjectStore(settings, database), database)


# --- the audience -------------------------------------------------------------


def test_audience_is_derived_from_the_identity_in_one_place() -> None:
    assert audience_for(is_orchestrator=True) is Audience.ORCHESTRATOR
    # is_orchestrator wins: a stray id must not demote the landing orchestrator.
    assert (
        audience_for(is_orchestrator=True, agent_id=HOST_AGENT_ID)
        is Audience.ORCHESTRATOR
    )
    assert audience_for(agent_id=HOST_AGENT_ID) is Audience.HOST
    assert audience_for(agent_id="builder") is Audience.AGENT
    assert audience_for() is Audience.NONE


def test_host_audience_holds_the_mutating_tools(
    tmp_path: Path, database: Database
) -> None:
    """The host agent's turn is the ONLY one with the propose tools on it.

    Asserted from both ends: the servers the turn registers, and the tools those
    servers advertise. A runtime filter would pass the first half and fail the
    second - the guarantee is that the orchestrator's turn does not HAVE the server.
    """
    settings = _settings(tmp_path)

    host = scufris_mcp_servers(settings, agent_id=HOST_AGENT_ID)
    assert [s.server_id for s in host] == ["host", "agent"]
    host_server = host[0]
    assert host_server.args == ("-m", "scufris.host_mcp_server")
    # It calls the dashboard's API to propose and to read the queue, so it carries
    # the base and its own id - the id is what the audit records as the asker.
    assert host_server.env["SCUFRIS_AGENT_ID"] == HOST_AGENT_ID
    assert host_server.env["SCUFRIS_API_BASE"].startswith("http://")

    orchestrator = scufris_mcp_servers(settings, is_orchestrator=True)
    assert "host" not in [s.server_id for s in orchestrator]

    regular = scufris_mcp_servers(settings, agent_id="builder")
    assert [s.server_id for s in regular] == ["agent"]


async def test_only_the_host_server_advertises_the_propose_tools() -> None:
    from scufris import host_mcp_server, mcp_server
    from scufris.mcp_host_tools import ACTIONS

    async def names(module: Any) -> set[str]:
        return {t.name for t in await module.mcp.list_tools()}

    mutating = {fn.__name__ for fn in ACTIONS}
    assert mutating <= await names(host_mcp_server)
    assert mutating.isdisjoint(await names(mcp_server))
    # And no audience has an approve tool. Being allowed to ask is not being
    # allowed to decide.
    for module in (host_mcp_server, mcp_server):
        assert not [n for n in await names(module) if "approve" in n]


def test_codex_registers_the_host_server_for_the_host_agent(
    tmp_path: Path, database: Database
) -> None:
    """The codex wiring follows the same audience core, so the two cannot drift."""
    joined = " ".join(_mcp_overrides(_settings(tmp_path), agent_id=HOST_AGENT_ID))
    assert "mcp_servers.host.command" in joined
    assert "scufris.host_mcp_server" in joined
    assert "mcp_servers.agent.command" in joined
    orchestrator = " ".join(_mcp_overrides(_settings(tmp_path), is_orchestrator=True))
    assert "mcp_servers.host." not in orchestrator


def test_the_host_agent_turn_is_steered_by_the_host_preamble(
    tmp_path: Path, database: Database
) -> None:
    steered = _steer(_settings(tmp_path), "restart nginx", agent_id=HOST_AGENT_ID)
    assert steered.startswith("[scufris-tools]")
    assert "propose_host_action(action, unit, days, generation)" in steered
    assert "propose_nixos_change(ref, repo, attr)" in steered
    # It must say plainly that it cannot approve, and that the shell is not a way
    # around that.
    assert "cannot approve" in steered
    assert steered.endswith("restart nginx")
    # A regular sub-agent still gets the sub-agent preamble, not this one.
    other = _steer(_settings(tmp_path), "go", agent_id="builder")
    assert AGENT_STEERING_PREAMBLE in other
    assert HOST_STEERING_PREAMBLE not in other


def test_host_steering_stays_a_single_block() -> None:
    # Ledger invariant (orchestrator-steering-is-one-block-two-clauses): each
    # preamble is ONE [scufris-tools] block, because strip_steering removes only
    # the first leading block (regex count=1). A second block would survive
    # uncleaned in titles and transcripts.
    assert HOST_STEERING_PREAMBLE.count("[scufris-tools]") == 1
    assert HOST_STEERING_PREAMBLE.count("[/scufris-tools]") == 1
    assert (
        strip_steering(f"{HOST_STEERING_PREAMBLE}\n\nrestart nginx") == "restart nginx"
    )


async def test_host_steering_names_tools_that_exist() -> None:
    """Every tool the host preamble names is a real tool on a server that turn
    registers.

    The ledger's `ground-steering-text-in-the-real-tool-signatures`: a typo'd name
    steers the model to a call that cannot succeed, which is worse than no steering.
    Checked against the LIVE registries of both servers a host turn wires up
    (`host` and `agent`), not against a hand-kept list.
    """
    import re

    from scufris import agent_mcp_server, host_mcp_server

    real = {t.name for t in await host_mcp_server.mcp.list_tools()}
    real |= {t.name for t in await agent_mcp_server.mcp.list_tools()}
    named = set(re.findall(r"\b([a-z_]{4,})\(", HOST_STEERING_PREAMBLE))
    assert named, "the preamble names no tools at all"
    assert named <= real, f"the preamble names unknown tools: {sorted(named - real)}"
    # And it names the ones that carry the contract, not just some of them.
    assert {"propose_host_action", "propose_nixos_change", "report_back"} <= named


def test_the_orchestrator_is_steered_to_delegate_a_host_change(
    tmp_path: Path, database: Database
) -> None:
    """It kept the read-only tools and lost the propose ones, so its steering has
    to send a CHANGE somewhere - otherwise the model reaches for the shell, which
    cannot do it either."""
    steered = _steer(_settings(tmp_path), "restart nginx", is_orchestrator=True)
    assert 'run_agent("host", goal)' in steered
    assert "needs root" in steered
    assert STEERING_PREAMBLE.count("[scufris-tools]") == 1


# --- the record ---------------------------------------------------------------


def test_the_host_agent_is_reserved_synthetic_and_listed(
    tmp_path: Path, database: Database
) -> None:
    store = _store(tmp_path, database)
    agent = store.get(HOST_AGENT_ID)
    assert agent.id == HOST_AGENT_ID
    # Bound to the MACHINE: no project, so it runs in the server cwd.
    assert agent.project_id == ""
    # Read-only by construction. Its power is proposing, and file-write or
    # unattended-command access would only widen a prompt-injected turn.
    assert agent.permission_mode is PermissionMode.MANUAL
    # Listed (unlike the hidden orchestrator), because the orchestrator delegates
    # to it by id and the operator should see the agent that can propose changes.
    assert HOST_AGENT_ID in [a.id for a in store.list()]
    assert ORCHESTRATOR_ID not in [a.id for a in store.list()]


def test_the_host_agent_refuses_every_crud_mutation(
    tmp_path: Path, database: Database
) -> None:
    store = _store(tmp_path, database, settings_writable=True)
    with pytest.raises(ReservedAgent):
        store.update(HOST_AGENT_ID, model="gpt-x")
    with pytest.raises(ReservedAgent):
        store.delete(HOST_AGENT_ID)
    # And the id cannot be taken by a created agent either.
    projects = ProjectStore(store._settings, database)
    projects.create(name="Host", cwd=str(tmp_path))
    with pytest.raises(InvalidAgent):
        store.create(name="host", project_id="host")


def test_the_host_agents_run_state_lives_in_memory(
    tmp_path: Path, database: Database
) -> None:
    """It has no agents.json row, so its lifecycle is tracked like the
    orchestrator's - and its session id still persists through the registry, so a
    restart does not lose the conversation."""
    store = _store(tmp_path, database)
    store.mark_running(HOST_AGENT_ID)
    assert store.get(HOST_AGENT_ID).state is AgentState.RUNNING
    store.mark_finished(
        HOST_AGENT_ID, state=AgentState.DONE, session_id="host-sess-1", backend="mock"
    )
    assert store.get(HOST_AGENT_ID).state is AgentState.DONE
    fresh = _store(tmp_path, database)
    assert fresh.get(HOST_AGENT_ID).session_id == "host-sess-1"
    assert fresh.get(HOST_AGENT_ID).state is AgentState.IDLE


# --- the operator-bound pending state -----------------------------------------


def test_a_pending_approval_is_blocked_not_waiting(
    tmp_path: Path, database: Database
) -> None:
    """BLOCKED is the state the enum already reserved for "waiting on an approval",
    and the distinction is the DECIDER: a WAITING agent is one the orchestrator
    answers, a BLOCKED one is waiting for a human."""
    store = _store(tmp_path, database)
    outcome = store.awaiting_approval(HOST_AGENT_ID, "waiting on host action abc")
    assert outcome.state is AgentState.BLOCKED
    # It is visible to the orchestrator's poll - a delegated change sitting with the
    # operator must not look like an agent that went quiet.
    pending = store.pending_outcomes()
    assert pending[HOST_AGENT_ID].state is AgentState.BLOCKED
    # Whether it may be CLEARED is not this store's question: it depends on the
    # approval still being live, which only the approval service can answer, so the
    # store stays dumb and the route holds the policy (review round 1, R1.1 -
    # `test_an_undecided_approval_does_not_strand_the_agent` covers it end to end).
    assert store.acknowledge(HOST_AGENT_ID) is True


def test_a_blocked_signal_survives_the_turn_ending(
    tmp_path: Path, database: Database
) -> None:
    """The agent proposes and then ends its turn; the natural DONE that follows must
    not erase the fact that it is waiting on the operator."""
    store = _store(tmp_path, database)
    store.awaiting_approval(HOST_AGENT_ID, "waiting on host action abc", run_id="run-1")
    store.mark_finished(
        HOST_AGENT_ID, state=AgentState.DONE, run_id="run-1", backend="mock"
    )
    assert store.outcome(HOST_AGENT_ID) is not None
    assert store.outcome(HOST_AGENT_ID).state is AgentState.BLOCKED  # type: ignore[union-attr]


def test_only_the_host_agent_may_signal_without_a_record(
    tmp_path: Path, database: Database
) -> None:
    """The host agent is synthetic and must still be able to signal. The
    orchestrator is equally synthetic and must NOT: it registers no callback server,
    so a route accepting its id would be accepting a caller that cannot exist."""
    from scufris.agent_store import AgentNotFound

    store = _store(tmp_path, database)
    store.request_input(HOST_AGENT_ID, "should I?")
    with pytest.raises(AgentNotFound):
        store.request_input(ORCHESTRATOR_ID, "should I?")
    with pytest.raises(AgentNotFound):
        store.report_back("ghost", "done")


# --- delegation ---------------------------------------------------------------


def test_orchestrator_delegates_to_the_host_agent(
    tmp_path: Path, fake_collector: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The orchestrator reaches the host agent through the machinery it already has.

    No parallel communication path: it runs the host agent by id like any agent
    (`run_agent`), the turn goes out on the HOST audience so the propose tools are on
    it, and the agent's report comes back through the ordinary outcome/pending
    route the orchestrator already polls.
    """
    from fastapi.testclient import TestClient

    from scufris.app import create_app

    class _Backend:
        name = "fake"

        def __init__(self) -> None:
            self.turns: list[tuple[str, bool, str, str | None]] = []

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
            from scufris.agent import AgentReply, StreamDone

            self.turns.append((agent_id, is_orchestrator, prompt, cwd))
            yield StreamDone(
                reply=AgentReply(text="proposed it", status="completed"),
                session_id="host-sess",
            )

        def read_status(self, settings: Settings, session_id: str | None) -> None:
            return None

        def read_transcript(
            self, settings: Settings, session_id: str | None
        ) -> list[Any]:
            return []

    backend = _Backend()
    monkeypatch.setattr("scufris.app.get_backend", lambda _name: backend)
    settings = _settings(tmp_path, web_dist=tmp_path / "absent")
    with TestClient(create_app(collector=fake_collector, settings=settings)) as client:
        # It is discoverable by id and in the list the orchestrator's `list_agents`
        # tool reads - a delegation target nobody can see is not one.
        assert client.get(f"/api/agents/{HOST_AGENT_ID}").status_code == 200
        assert HOST_AGENT_ID in [a["id"] for a in client.get("/api/agents").json()]

        started = client.post(
            f"/api/agents/{HOST_AGENT_ID}/run",
            json={"goal": "restart nginx, it is not answering"},
        )
        assert started.status_code == 200, started.text

        for _ in range(200):
            if backend.turns:
                break
            time.sleep(0.02)
        assert backend.turns, "the host agent never ran"
        agent_id, is_orchestrator, prompt, cwd = backend.turns[0]
        # The turn carries the HOST identity, which is what puts the `host` server -
        # and only that turn's `host` server - on it.
        assert agent_id == HOST_AGENT_ID
        assert is_orchestrator is False
        assert "restart nginx" in prompt
        # No project: it is bound to the machine, so it runs in the server cwd.
        assert cwd is None

        # Its report comes back the ordinary way, so the orchestrator's existing
        # poll finds it.
        reported = client.post(
            f"/api/agents/{HOST_AGENT_ID}/report_back",
            json={"summary": "proposed the restart; waiting on you"},
        )
        assert reported.status_code == 200, reported.text
        pending = client.get("/api/agents/pending").json()
        row = next(r for r in pending if r["agent_id"] == HOST_AGENT_ID)
        assert row["state"] == "reported"
        # And THAT one the orchestrator may clear: it is a report, not a decision.
        assert (
            client.post(f"/api/agents/{HOST_AGENT_ID}/acknowledge").json()[
                "acknowledged"
            ]
            is True
        )


def test_the_tool_listings_report_the_audience_they_wire(
    tmp_path: Path, fake_collector: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The endpoints that TELL the operator what an agent can call must match what
    the turn actually registers.

    Review round 1, R1.4. The wiring is pinned above at the module level, but these
    two routes are what the settings page renders, and they resolve the audience
    through a different path (`agent_diagnostics.mcp_servers_for_audience` ->
    `mcp_health.servers_for_audience`). The ledger's
    `tool-reachable-by-two-runners-needs-a-test-per-runner` is this exact shape: the
    listing can drift from the wiring while each looks right on its own.
    """
    from fastapi.testclient import TestClient

    from scufris.app import create_app
    from scufris.mcp_host_tools import ACTIONS, INSPECTION

    settings = _settings(tmp_path, web_dist=tmp_path / "absent", agent_backend="codex")
    mutating = {fn.__name__ for fn in ACTIONS}
    inspection = {fn.__name__ for fn in INSPECTION}
    with TestClient(create_app(collector=fake_collector, settings=settings)) as client:
        host_tools = {
            t["name"]: t["server"]
            for t in client.get(f"/api/agents/{HOST_AGENT_ID}/tools").json()["value"]
        }
        assert mutating <= set(host_tools)
        assert {host_tools[name] for name in mutating} == {"host"}
        # It also holds the callbacks, so it reports back like any sub-agent.
        assert host_tools.get("report_back") == "agent"

        # The console is orchestrator-scoped: inspection yes, propose no.
        console = {t["name"] for t in client.get("/api/agent/tools").json()}
        assert inspection <= console
        assert mutating.isdisjoint(console)

        # And a regular project agent sees neither.
        client.post("/api/projects", json={"name": "My App", "cwd": str(tmp_path)})
        created = client.post(
            "/api/agents",
            json={"name": "Builder", "project_id": "my-app", "backend": "codex"},
        )
        builder = created.json()["id"]
        regular = {
            t["name"]
            for t in client.get(f"/api/agents/{builder}/tools").json()["value"]
        }
        assert regular == {"request_input", "report_back"}
