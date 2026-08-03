"""The orchestrator services, driven without the application they front.

Four heights, and the point of all four is that the turn path is no longer
trapped inside `create_app`:

- the three transports that start an orchestrator turn reach ONE service;
- a run's whole lifecycle - launch, the one-run-per-agent guard, cancel, status,
  events, the sub-agent signals, fork and the completion fan-out - is exercised
  against real stores and a real supervisor with no FastAPI app anywhere;
- the `/api/chat/stream` frames a browser sees are byte-for-byte what they were
  before the extraction, captured from the tree it was taken out of;
- and the services themselves import no transport, which is what keeps the
  first three true.
"""

from __future__ import annotations

import ast
import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TypeVar, cast

import pytest
from fastapi.testclient import TestClient
from test_agent_store import _projects_with_one, _settings

from scufris.agent import AgentReply, StreamDone, StreamEvent
from scufris.agent_store import ORCHESTRATOR_ID, AgentStore
from scufris.app import create_app
from scufris.config import Settings
from scufris.db import Database
from scufris.enums import AgentState, Backend
from scufris.orchestrator import (
    AgentDisabled,
    AgentProjectMissing,
    AgentRunService,
    NoActiveRun,
    OrchestratorTurnService,
    RunAlreadyActive,
)
from scufris.projects import ProjectStore
from scufris.reasoning_store import ReasoningStore
from scufris.supervisor import AgentSupervisor, agent_supervisor
from scufris.telegram.orchestrator import build_telegram_callbacks
from scufris.wake import WakeBridge, wake_prompt

_T = TypeVar("_T")

# --- 1. one service behind all three transports ------------------------------


class _RecordingRuns:
    """An `AgentRunService` stand-in that records every launch it is handed.

    Only the three members the turn service actually uses, so a launch that
    started routing around it would fail here rather than pass quietly.
    """

    def __init__(self) -> None:
        self.launched: list[tuple[Any, str]] = []

    def active(self, agent_id: str) -> bool:
        return False

    def run_id(self, agent_id: str) -> str | None:
        return None

    async def launch(
        self,
        agent: Any,
        project: Any,
        prompt: str,
        *,
        image_paths: list[str] | None = None,
        on_done: Any = None,
    ) -> tuple[str, Any]:
        self.launched.append((agent, prompt))
        return f"run{len(self.launched)}", _ClosedBus()


class _BusyRuns(_RecordingRuns):
    """A run service that always refuses: the orchestrator is already running."""

    async def launch(
        self,
        agent: Any,
        project: Any,
        prompt: str,
        *,
        image_paths: list[str] | None = None,
        on_done: Any = None,
    ) -> tuple[str, Any]:
        raise RunAlreadyActive()


class _ClosedBus:
    """A bus that replays one finished turn and closes, like a mock backend's."""

    async def subscribe(
        self, after_seq: int = 0
    ) -> AsyncIterator[tuple[int, StreamEvent]]:
        yield 1, StreamDone(reply=AgentReply(text="ok"), session_id="s1")


class _Outcome:
    def __init__(self) -> None:
        self.state = AgentState.WAITING
        self.message = "should I merge?"
        self.acknowledged = False


class _FakeAgents:
    """The store calls the turn service and the wake bridge make."""

    def __init__(self) -> None:
        self.orchestrator_session: str | None = "s0"

    def get(self, agent_id: str) -> str:
        return f"agent:{agent_id}"

    def outcome(self, agent_id: str) -> _Outcome:
        return _Outcome()

    def set_orchestrator_session(self, session_id: str | None) -> None:
        self.orchestrator_session = session_id


class _FakeSupervisor:
    def __init__(self) -> None:
        self.serialized_keys: list[str] = []

    def serialized(self, key: str) -> Any:
        self.serialized_keys.append(key)

        @asynccontextmanager
        async def _cm() -> AsyncIterator[None]:
            yield

        return _cm()


def _turn_over(
    runs: Any,
    settings: Settings,
    supervisor: Any | None = None,
    agents: _FakeAgents | None = None,
) -> OrchestratorTurnService:
    return OrchestratorTurnService(
        settings=settings,
        agents=cast(Any, agents or _FakeAgents()),
        supervisor=cast(Any, supervisor or _FakeSupervisor()),
        runs=cast(Any, runs),
    )


async def test_orchestrator_transports_share_turn_service(tmp_path: Path) -> None:
    """The landing chat, the Telegram bot and the wake bridge launch through the
    SAME run service - one recording fake, three callers, three launches.

    Each of the three used to reach the turn path on its own terms, which is how
    they came to disagree about what a launch failure means. Sharing the service
    is what makes the answer to "is the orchestrator busy" one answer.
    """
    settings = Settings(
        state_dir=tmp_path,
        agent_enabled=True,
        auto_wake=True,
        _env_file=None,  # type: ignore[call-arg]
    )
    runs = _RecordingRuns()
    turn = _turn_over(runs, settings)

    # 1. the landing chat (/api/chat/stream translates for exactly this call).
    await turn.stream("from the web")

    # 2. the Telegram bot, through its real callbacks.
    on_message, _on_reset, _on_cancel = build_telegram_callbacks(turn)
    events = [event async for event in on_message("from telegram")]
    assert [type(event).__name__ for event in events] == ["StreamDone"]

    # 3. the wake bridge, through its real drain.
    bridge = WakeBridge(
        agents=cast(Any, _FakeAgents()),
        settings=settings,
        is_orchestrator_busy=turn.busy,
        launch=turn.wake,
    )
    await bridge.on_run_complete("builder")

    assert [prompt for _agent, prompt in runs.launched] == [
        "from the web",
        "from telegram",
        wake_prompt({"builder": (AgentState.WAITING, "should I merge?")}),
    ]
    # ...and every one of them against the reserved orchestrator record, resolved
    # by the service rather than by each transport.
    assert {agent for agent, _prompt in runs.launched} == {f"agent:{ORCHESTRATOR_ID}"}

    # A wake that loses the race is a False, not a raise: the bridge backs off
    # and waits for the next completion rather than failing a callback.
    busy = _BusyRuns()
    assert await _turn_over(busy, settings).wake("wake up") is False
    assert busy.launched == []


async def test_the_turn_service_owns_the_disabled_and_reset_rules(
    tmp_path: Path,
) -> None:
    """The two rules the three transports used to each repeat.

    A disabled agent refuses before any store read, and a reset takes the
    orchestrator's serialize key so it cannot interleave with an in-flight turn.
    """
    runs = _RecordingRuns()
    supervisor = _FakeSupervisor()
    agents = _FakeAgents()
    disabled = _turn_over(
        runs,
        Settings(
            state_dir=tmp_path,
            agent_enabled=False,
            _env_file=None,  # type: ignore[call-arg]
        ),
        supervisor,
        agents,
    )

    with pytest.raises(AgentDisabled):
        await disabled.send("hi")
    with pytest.raises(AgentDisabled):
        await disabled.stream("hi")
    assert runs.launched == []

    await disabled.reset()
    assert supervisor.serialized_keys == [ORCHESTRATOR_ID]
    assert agents.orchestrator_session is None


# --- 2. the run lifecycle, with no app ---------------------------------------


async def _off(call: Callable[[], _T]) -> _T:
    """Run a synchronous store-touching call off the event loop thread.

    `scufris.db` refuses to open a transaction on a thread with a running loop,
    so a synchronous service method reaching a store has to be offloaded here
    for the same reason Starlette runs a `def` route in its threadpool. In the
    app this is the framework's job; driving the service directly, it is ours.
    """
    return await asyncio.to_thread(call)


def _run_service(
    tmp_path: Path, database: Database
) -> tuple[Settings, AgentStore, ProjectStore, AgentSupervisor, AgentRunService]:
    """Real stores, a real supervisor, the mock backend - and no FastAPI.

    The settings say `agent_backend=mock` so the reserved orchestrator record -
    which tracks the landing config rather than a stored row - resolves to the
    mock backend like the created agents do.
    """
    settings = _settings(tmp_path).model_copy(update={"agent_backend": Backend.MOCK})
    projects = _projects_with_one(tmp_path, settings, database)
    agents = AgentStore(settings, projects, database)
    supervisor = agent_supervisor(max_concurrent=2)
    runs = AgentRunService(
        settings=settings,
        agents=agents,
        projects=projects,
        reasoning_store=ReasoningStore(database),
        supervisor=supervisor,
    )
    return settings, agents, projects, supervisor, runs


async def test_agent_run_lifecycle_is_owned_by_the_run_service(
    tmp_path: Path, database: Database
) -> None:
    """Launch, the busy guard, cancel, status, events, the signals, fork and the
    completion hooks - all of it driven directly, with no app and no lifespan.

    That this test can exist at all is the extraction's whole point: the same
    path used to cost a state directory, a database, two stores, a supervisor
    and a `create_app` before a single turn could be observed.
    """
    _settings_, agents, projects, _supervisor, runs = await _off(
        lambda: _run_service(tmp_path, database)
    )
    agent = await _off(
        lambda: agents.create(name="Builder", project_id="my-app", backend="mock")
    )
    project = await _off(lambda: runs.require_agent_project(agent))
    assert project is not None

    # An idle agent has no run, and every reader says so the same way.
    assert runs.active(agent.id) is False
    assert runs.run_id(agent.id) is None
    with pytest.raises(NoActiveRun):
        runs.cancel(agent.id)
    with pytest.raises(NoActiveRun):
        runs.bus(agent.id)

    # The completion fan-out, in registration order, at the end of the run.
    fired: list[str] = []
    finished = asyncio.Event()

    async def first(agent_id: str) -> None:
        fired.append(f"first:{agent_id}")

    async def second(agent_id: str) -> None:
        fired.append(f"second:{agent_id}")
        finished.set()

    runs.on_complete(first)
    runs.on_complete(second)

    run_id, bus = await runs.launch(agent, project, "ping")
    assert runs.run_id(agent.id) == run_id
    assert runs.active(agent.id) is True
    assert runs.bus(agent.id) is bus
    # The one-run-per-agent guard lives HERE, not in a route: it and the claim
    # are one synchronous step, which is what closes the launch race.
    with pytest.raises(RunAlreadyActive):
        await runs.launch(agent, project, "again")

    done = await runs.drain(bus)
    assert done.reply.text == "[mock reply] ping"

    await asyncio.wait_for(finished.wait(), timeout=5)
    assert fired == [f"first:{agent.id}", f"second:{agent.id}"]
    assert (await _off(lambda: agents.get(agent.id))).state == AgentState.DONE

    # Status merges the (now finished) run state with the backend's progress.
    status = await _off(lambda: runs.status(agents.get(agent.id)))
    assert status.agent_id == agent.id
    assert status.session_id == "mock-session"
    assert status.progress is not None and status.progress.turns == 1
    assert status.prompt is None  # only exposed while the run is live

    # The sub-agent signals, keyed to the agent's current run.
    assert (
        await _off(lambda: runs.request_input(agents.get(agent.id), "merge?"))
    ) == AgentState.WAITING
    assert (
        await _off(lambda: runs.report_back(agents.get(agent.id), "shipped"))
    ) == AgentState.REPORTED
    assert await runs.acknowledge(agent.id) is True
    assert await runs.acknowledge(agent.id) is False

    # Fork seeds a fresh turn from the transcript before the edited message.
    assert "edited" in await _off(
        lambda: runs.fork_seed(agent, "mock-session", 0, "edited")
    )

    # Cancelling a live run is the service's job too.
    _cancel_id, _cancel_bus = await runs.launch(
        await _off(lambda: agents.get(agent.id)), project, "again"
    )
    runs.cancel(agent.id)

    # An agent whose project has been deleted is unprocessable, not missing.
    orphan = await _off(
        lambda: agents.create(name="Orphan", project_id="my-app", backend="mock")
    )
    await _off(lambda: projects.delete("my-app"))
    with pytest.raises(AgentProjectMissing):
        await _off(lambda: runs.require_agent_project(orphan))

    await runs.aclose()


async def test_the_turn_service_cancels_only_a_live_orchestrator_turn(
    tmp_path: Path, database: Database
) -> None:
    """`cancel()` is False when the orchestrator is idle and True once it is not.

    The Telegram `/cancel` reads that bool as its answer, so "there was nothing
    to stop" has to stay a False rather than a refusal. Driven over the REAL run
    service: the invariant used to be pinned through the callback's supervisor
    and moved in here with the extraction.
    """
    settings, agents, _projects, supervisor, runs = await _off(
        lambda: _run_service(tmp_path, database)
    )
    turn = OrchestratorTurnService(
        settings=settings, agents=agents, supervisor=supervisor, runs=runs
    )

    assert await turn.cancel() is False
    assert turn.busy() is False

    orchestrator = await _off(lambda: agents.get(ORCHESTRATOR_ID))
    await runs.launch(orchestrator, None, "ping")
    assert turn.busy() is True
    assert await turn.cancel() is True

    await runs.aclose()


# --- 3. the SSE surface, unchanged -------------------------------------------
#
# Captured from the tree this extraction was taken out of (98650f0), by driving
# `/api/chat/stream` against the mock backend. Byte-for-byte: the 2KB padding
# comment that defeats proxy buffering, the `id:` sequence a reconnect replays
# from, and the JSON of each frame.

_PADDING = ":" + " " * 2048 + "\n\n"
_EXPECTED_STREAM = (
    _PADDING
    + 'id: 1\ndata: {"kind":"text_delta","delta":"[mock] ping"}\n\n'
    + "id: 2\ndata: "
    + '{"kind":"done","reply":{"text":"[mock reply] ping","status":null,'
    + '"tool_calls":[],"usage":null},"session_id":"mock-session"}\n\n'
)
# A bad attachment never launches a turn: ONE error frame, and deliberately no
# `id:` - there is no bus and so no seq for a client to reconnect from.
_EXPECTED_IMAGE_ERROR = (
    _PADDING + 'data: {"kind": "error", "detail": "attachment is not valid base64"}\n\n'
)


def test_chat_stream_events_are_unchanged(tmp_path: Path) -> None:
    """The frames and the `id:` sequence a browser sees, pinned to the capture.

    The turn now starts in `OrchestratorTurnService` and the image decode stayed
    in the route; neither may show up on the wire.
    """
    settings = Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path,
        agent_backend=Backend.MOCK,
        enable_mock_backend=True,
        _env_file=None,  # type: ignore[call-arg]
    )
    app = create_app(settings=settings)
    with TestClient(app) as client:
        with client.stream("POST", "/api/chat/stream", json={"message": "ping"}) as r:
            assert r.status_code == 200
            assert "".join(r.iter_text()) == _EXPECTED_STREAM

        # The route reached the PUBLISHED service, not a private copy of it.
        assert isinstance(app.state.runs, AgentRunService)
        assert app.state.runs.run_id(ORCHESTRATOR_ID) is not None

        with client.stream(
            "POST",
            "/api/chat/stream",
            json={
                "message": "ping",
                "image": {"data_base64": "!!!not base64!!!", "mime": "image/png"},
            },
        ) as r:
            assert r.status_code == 200
            assert "".join(r.iter_text()) == _EXPECTED_IMAGE_ERROR


# --- 4. no transport inside the services -------------------------------------

_TRANSPORTS = ("fastapi", "starlette", "telegram", "httpx")


def test_the_services_carry_no_transport_imports() -> None:
    """`scufris/orchestrator/` imports nothing that speaks a wire protocol.

    The imports are read out of the AST rather than grepped, so a module aliased
    or imported inside a function is caught too; the raw-text check after it is
    the same proof the definition of done runs from a shell.
    """
    package = Path(__file__).resolve().parent.parent / "scufris" / "orchestrator"
    sources = sorted(package.glob("*.py"))
    assert sources, "the orchestrator package must exist"

    for source in sources:
        text = source.read_text()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert name.split(".")[0] not in _TRANSPORTS, (
                    f"{source.name} imports the transport {name}"
                )
        for transport in ("fastapi", "telegram"):
            assert transport not in text, f"{source.name} mentions {transport}"
