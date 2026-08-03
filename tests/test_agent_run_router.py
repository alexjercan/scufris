"""The agent-run router's booby-trap pass, on a bare app over fakes.

A fourth file in the `test_domain_routers.py` / `test_orchestrator_routers.py` /
`test_chat_router.py` family, for the same reason the others exist: the shared
file is at its line cap.

`test_orchestrator_routers.py` proves the project and agent-record routers
construct nothing ambient, and the chat and legacy files do the same for theirs.
This one carries the claim for `api/agent_runs.py` - the largest of the five, 16
routes - so a `Settings()` or a `ProjectStore(...)` creeping back into a run
route body fails here rather than in production.

The rig's fakes are imported from `test_orchestrator_routers`; the diagnostics
fake and the run-service fake are redefined, because the run surface asks
diagnostics for `health`, `tools` and `mcp` as well as the three the shared one
answers, and asks the run service for `fork_seed`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_orchestrator_routers import (
    AGENT_ID,
    FakeAgentGate,
    FakeAgentStore,
    FakeApprovalsForAgents,
    FakeRunService,
    FakeSupervisor,
)

import scufris.db
from scufris.agent_diagnostics import AccountInfo
from scufris.agent_store import AgentRecord, AgentStore
from scufris.api.agent_runs import AgentRunDeps, build_agent_run_router
from scufris.backends import Capability
from scufris.config import Settings
from scufris.db.engine import Database
from scufris.health import AgentHealth
from scufris.mcp_models import AgentTool, McpServerHealth
from scufris.projects import ProjectStore
from scufris.sessions import MemoryFootprint, TranscriptMessage, UsageQuota


class FullDiagnostics:
    """Every reader the run surface asks for, all answering trivially: what this
    file proves is which object the route asks, not what the answer says."""

    def usage(self, agent: AgentRecord) -> Capability[UsageQuota]:
        return Capability(supported=False)

    def memory(self, agent: AgentRecord) -> Capability[MemoryFootprint]:
        return Capability(supported=False)

    def account(self, agent: AgentRecord) -> AccountInfo:
        return AccountInfo(
            model="m",
            auth_mode=None,
            enabled=True,
            quota=Capability(supported=False),
        )

    async def health(self, agent: AgentRecord) -> AgentHealth:
        return AgentHealth(scufris_version="test", backend=agent.backend, checks=[])

    async def tools(self, agent: AgentRecord) -> Capability[list[AgentTool]]:
        return Capability(supported=False)

    async def mcp(self, agent: AgentRecord) -> list[McpServerHealth]:
        return []


class FakeTranscriptBackend:
    """`get_backend`'s answer for the transcript route, which reads the history
    off the backend rather than off a service."""

    def read_transcript(
        self, settings: Settings, session_id: str | None
    ) -> list[TranscriptMessage]:
        return []


class ForkingRunService(FakeRunService):
    """`FakeRunService` plus the one answer only the fork route asks for.

    Redefined here rather than on the shared fake for the same reason
    `FullDiagnostics` is: `/fork` is part of THIS file's surface, and
    `test_orchestrator_routers.py` is at its line cap.
    """

    def fork_seed(
        self, agent: AgentRecord, session_id: str | None, message_index: int, text: str
    ) -> str:
        self.calls.append(("fork_seed", (agent.id, session_id, message_index, text)))
        return "seeded: go"


class RunTrapRig:
    """The agent-run router on a bare `FastAPI()`, over the fakes above."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.runs = ForkingRunService()
        self.approvals = FakeApprovalsForAgents()
        self.gate = FakeAgentGate()
        self.agents = FakeAgentStore()
        self.app = FastAPI()
        self.app.include_router(
            build_agent_run_router(
                AgentRunDeps(
                    settings=settings,
                    agents=cast(Any, self.agents),
                    runs=cast(Any, self.runs),
                    diagnostics=cast(Any, FullDiagnostics()),
                    gate=cast(Any, self.gate),
                    approvals=cast(Any, self.approvals),
                    supervisor=cast(Any, FakeSupervisor()),
                )
            )
        )


@pytest.fixture
def trap_rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunTrapRig:
    monkeypatch.setattr(
        "scufris.api.agent_runs.get_backend",
        lambda name: FakeTranscriptBackend(),
    )
    return RunTrapRig(
        Settings(
            state_dir=tmp_path / "state",
            web_dist=tmp_path / "dist",
            _env_file=None,  # type: ignore[call-arg]
        )
    )


@pytest.fixture
def trap_client(trap_rig: RunTrapRig) -> Iterator[TestClient]:
    with TestClient(trap_rig.app) as client:
        yield client


def _forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError(
        "the agent-run router constructed a store, a database or a settings "
        "object; the whole point of the deps dataclass is that it does not"
    )


def test_the_agent_run_router_reaches_for_nothing(
    trap_rig: RunTrapRig, trap_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Booby trap the four things a router must not touch, then drive the surface.

    The traps go on ``__init__`` rather than on the name in the defining module:
    `from ..config import Settings` binds the class into the importing module at
    import time, so replacing `scufris.config.Settings` would leave the router
    calling the real one and the trap would never fire. Patching the class itself
    catches every import spelling. The rig is built by its fixture BEFORE the
    traps go down, so the one `Settings` the router answers from is the one
    handed to `AgentRunDeps`.

    ``/events`` is the one route left out: it relays a live bus, and subscribing
    to a bus nobody closes blocks the request forever.
    """
    for target in (Settings, AgentStore, ProjectStore, Database):
        monkeypatch.setattr(target, "__init__", _forbidden)
    monkeypatch.setattr(scufris.db, "state_database", _forbidden)

    for path in (
        "status",
        "transcript",
        "usage",
        "memory",
        "health",
        "account",
        "tools",
        "mcp",
    ):
        response = trap_client.get(f"/api/agents/{AGENT_ID}/{path}")
        assert response.status_code == 200, path

    assert trap_client.post(f"/api/agents/{AGENT_ID}/run", json={}).status_code == 200
    assert trap_client.post(f"/api/agents/{AGENT_ID}/cancel").status_code == 200
    assert (
        trap_client.post(
            f"/api/agents/{AGENT_ID}/request_input", json={"question": "which?"}
        ).status_code
        == 200
    )
    assert (
        trap_client.post(
            f"/api/agents/{AGENT_ID}/report_back", json={"summary": "done"}
        ).status_code
        == 200
    )
    assert trap_client.post(f"/api/agents/{AGENT_ID}/acknowledge").status_code == 200
    assert (
        trap_client.post(
            f"/api/agents/{AGENT_ID}/chat", json={"message": "go"}
        ).status_code
        == 200
    )
    assert (
        trap_client.post(
            f"/api/agents/{AGENT_ID}/fork", json={"message_index": 0, "text": "go"}
        ).status_code
        == 200
    )
