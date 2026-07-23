"""Unit tests for the WakeBridge defer/batch/absorb logic (BC4).

The bridge's collaborators (is-busy, launch) are injected, so the enqueue filter,
deferral, batching and 409-absorption are tested in isolation against a real
AgentStore. The end-to-end wiring (persist callback -> bridge -> a real
orchestrator turn) is exercised by an integration test in test_app.py.
"""

from __future__ import annotations

from pathlib import Path

from scufris.agent_store import ORCHESTRATOR_ID, AgentStore
from scufris.config import Settings
from scufris.enums import AgentState
from scufris.projects import ProjectStore
from scufris.wake import WakeBridge, wake_prompt


def _store(tmp_path: Path) -> AgentStore:
    settings = Settings(state_dir=tmp_path / "state", enable_mock_backend=True)
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    projects = ProjectStore(settings)
    projects.create(name="P", cwd=str(proj))
    return AgentStore(settings, projects)


def _agent(store: AgentStore, name: str) -> str:
    return store.create(name=name, project_id="p", backend="mock").id


class _Launcher:
    """A recording launch callback whose return (granted vs 409-busy) is settable."""

    def __init__(self, granted: bool = True) -> None:
        self.granted = granted
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> bool:
        self.prompts.append(prompt)
        return self.granted


def _bridge(
    store: AgentStore,
    launcher: _Launcher,
    *,
    auto_wake: bool = True,
    busy: bool = False,
) -> tuple[WakeBridge, dict[str, bool]]:
    state = {"busy": busy}
    bridge = WakeBridge(
        agents=store,
        settings=Settings(auto_wake=auto_wake),
        is_orchestrator_busy=lambda: state["busy"],
        launch=launcher,
    )
    return bridge, state


def test_auto_wake_off_no_launch(tmp_path: Path) -> None:
    """With auto_wake off, a WAITING sub-agent never wakes the orchestrator (it is
    polled via pending_agents instead)."""
    store = _store(tmp_path)
    _agent(store, "Waiter")
    store.request_input("waiter", "merge?", run_id="waiter:r1")
    launcher = _Launcher()
    bridge, _ = _bridge(store, launcher, auto_wake=False)
    bridge.on_run_complete("waiter")
    assert launcher.prompts == []


def test_wake_bridge_defers_and_batches(tmp_path: Path) -> None:
    """While the orchestrator is busy, WAITING sub-agents are DEFERRED (no launch);
    when it goes idle its own completion drains them as ONE batched wake turn."""
    store = _store(tmp_path)
    for n in ("Waiter", "Waiter2"):
        _agent(store, n)
    store.request_input("waiter", "merge to master?", run_id="waiter:r1")
    store.request_input("waiter2", "delete the temp dir?", run_id="waiter2:r1")

    launcher = _Launcher()
    bridge, state = _bridge(store, launcher, busy=True)

    bridge.on_run_complete("waiter")  # orchestrator busy -> deferred
    bridge.on_run_complete("waiter2")  # busy -> deferred
    assert launcher.prompts == []  # nothing launched while busy

    # The orchestrator's own turn ends -> not busy -> drain fires ONE wake.
    state["busy"] = False
    bridge.on_run_complete(ORCHESTRATOR_ID)
    assert len(launcher.prompts) == 1
    prompt = launcher.prompts[0]
    assert "waiter" in prompt and "merge to master?" in prompt
    assert "waiter2" in prompt and "delete the temp dir?" in prompt

    # Drained: a further completion does not re-fire.
    bridge.on_run_complete(ORCHESTRATOR_ID)
    assert len(launcher.prompts) == 1


def test_error_outcome_wakes_too(tmp_path: Path) -> None:
    """An errored sub-agent (empty message) also wakes the orchestrator."""
    store = _store(tmp_path)
    _agent(store, "Crasher")
    store.mark_finished("crasher", state=AgentState.ERROR, run_id="crasher:r1")
    launcher = _Launcher()
    bridge, _ = _bridge(store, launcher)
    bridge.on_run_complete("crasher")
    assert len(launcher.prompts) == 1
    assert "crasher" in launcher.prompts[0]


def test_done_or_acknowledged_not_enqueued(tmp_path: Path) -> None:
    """A cleanly DONE agent, or an already-acknowledged WAITING one, does not wake."""
    store = _store(tmp_path)
    _agent(store, "Doner")
    _agent(store, "Acked")
    store.mark_finished("doner", state=AgentState.DONE, message="done", run_id="d:r1")
    store.request_input("acked", "merge?", run_id="acked:r1")
    store.acknowledge("acked")

    launcher = _Launcher()
    bridge, _ = _bridge(store, launcher)
    bridge.on_run_complete("doner")
    bridge.on_run_complete("acked")
    assert launcher.prompts == []


def test_launch_409_keeps_pending(tmp_path: Path) -> None:
    """If launch reports the orchestrator busy (409 race, returns False), the wake
    stays pending and a later completion retries it - never dropped."""
    store = _store(tmp_path)
    _agent(store, "Waiter")
    store.request_input("waiter", "merge?", run_id="waiter:r1")

    launcher = _Launcher(granted=False)  # first launch loses the race
    bridge, _ = _bridge(store, launcher)
    bridge.on_run_complete("waiter")
    assert len(launcher.prompts) == 1  # attempted, but not granted

    launcher.granted = True  # orchestrator now free
    bridge.on_run_complete(ORCHESTRATOR_ID)
    assert len(launcher.prompts) == 2  # retried and granted
    assert "waiter" in launcher.prompts[1]


def test_wake_prompt_lists_each_agent() -> None:
    out = wake_prompt({"a": "q1", "b": "q2"})
    assert "a: q1" in out and "b: q2" in out
    assert "pending_agents" in out and "acknowledge" in out
