"""Tests for the first-class AgentStore: CRUD, persistence, validation, gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from scufris.agent_store import (
    ORCHESTRATOR_ID,
    AgentNotFound,
    AgentsReadOnly,
    AgentStore,
    InvalidAgent,
    ReservedAgent,
    SessionRegistry,
)
from scufris.config import Settings
from scufris.enums import AgentState, Backend
from scufris.projects import ProjectStore


def _settings(tmp_path: Path, *, writable: bool = True) -> Settings:
    return Settings(
        state_dir=tmp_path / "state",
        settings_writable=writable,
        enable_mock_backend=True,  # tests create mock-backed agents
    )


def _projects_with_one(tmp_path: Path, settings: Settings) -> ProjectStore:
    """A ProjectStore holding a single project 'my-app' at a real dir."""
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir(exist_ok=True)
    projects = ProjectStore(settings)
    projects.create(name="My App", cwd=str(proj_dir))
    return projects


def test_agent_store_round_trip(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)

    created = store.create(
        name="Builder", project_id="my-app", backend="mock", goal="do the thing"
    )
    assert created.id == "builder"
    assert created.project_id == "my-app"
    assert created.backend == "mock"
    assert created.state == "idle"
    assert created.permission_mode == "manual"  # safe default

    # A fresh store over the same state dir sees it (projects reloaded too).
    fresh_projects = ProjectStore(settings)
    fresh = AgentStore(settings, fresh_projects)
    got = fresh.get("builder")
    assert got.name == "Builder"
    assert got.goal == "do the thing"

    # Update persists.
    fresh.update("builder", permission_mode="edit", model="gpt-x")
    reloaded = AgentStore(settings, ProjectStore(settings)).get("builder")
    assert reloaded.permission_mode == "edit"
    assert reloaded.model == "gpt-x"

    # Delete persists (the reserved orchestrator is always present besides it).
    fresh.delete("builder")
    reloaded_list = AgentStore(settings, ProjectStore(settings)).list()
    assert [a.id for a in reloaded_list if a.id != "orchestrator"] == []


def test_create_agent_rejects_unknown_project(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    with pytest.raises(InvalidAgent, match="no such project"):
        store.create(name="Ghost", project_id="does-not-exist")


def test_create_agent_validates_name_and_backend(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    with pytest.raises(InvalidAgent, match="name must not be empty"):
        store.create(name="   ", project_id="my-app")
    with pytest.raises(InvalidAgent, match="unknown or disabled backend"):
        store.create(name="Bad", project_id="my-app", backend="nope")


def test_agent_ids_dedup(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    a = store.create(name="Worker", project_id="my-app")
    b = store.create(name="Worker", project_id="my-app")
    assert a.id == "worker"
    assert b.id == "worker-2"


def test_agent_writes_gated_when_read_only(tmp_path: Path) -> None:
    settings = _settings(tmp_path, writable=False)
    # Seed a project with a writable store first, then a read-only agent store.
    writable = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, writable)
    store = AgentStore(settings, ProjectStore(settings))
    # The project persisted by the writable store is visible read-only.
    assert projects.get("my-app").id == "my-app"
    with pytest.raises(AgentsReadOnly):
        store.create(name="Nope", project_id="my-app")


def test_agent_store_tolerates_a_corrupt_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "agents.json").write_text("{ this is not valid json ]")
    # Load does not raise; the store has no persisted agents. The reserved
    # orchestrator is synthetic + HIDDEN from the list (still reachable via get).
    store = AgentStore(settings, ProjectStore(settings))
    assert store.list() == []
    assert store.get(ORCHESTRATOR_ID).id == ORCHESTRATOR_ID


def test_get_unknown_agent_raises(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = AgentStore(settings, ProjectStore(settings))
    with pytest.raises(AgentNotFound):
        store.get("ghost")


def test_orchestrator_reserved_and_undeletable(tmp_path: Path) -> None:
    """The orchestrator is a synthetic reserved agent: reachable via get but
    HIDDEN from the list (a hidden default), not in agents.json, undeletable,
    un-creatable by id, and not store-editable (its config lives in settings)."""
    settings = _settings(tmp_path)
    store = AgentStore(settings, ProjectStore(settings))

    # Reachable via get, even with no persisted agents...
    orch = store.get(ORCHESTRATOR_ID)
    assert orch.id == ORCHESTRATOR_ID
    assert orch.name == "Orchestrator"
    assert orch.project_id == ""  # no project -> server cwd
    assert orch.backend == "codex"  # from settings.agent_backend (default codex)
    # The orchestrator DEFAULTS to auto (full write posture): it must do write
    # work unattended (Bash tatr, create projects/agents). Regular agents still
    # default to manual on create - only the settings-derived orchestrator changed.
    assert orch.permission_mode == "auto"
    # ...but HIDDEN from the list (it is reached via `/`, not the /agents grid).
    assert ORCHESTRATOR_ID not in [a.id for a in store.list()]

    # It is NOT written to agents.json (a fresh store still synthesizes it, and
    # the file does not exist because nothing real was persisted).
    assert not (settings.state_dir / "agents.json").exists()

    # Undeletable, un-updatable via the store, and its id is reserved for create.
    with pytest.raises(ReservedAgent):
        store.delete(ORCHESTRATOR_ID)
    with pytest.raises(ReservedAgent):
        store.update(ORCHESTRATOR_ID, model="gpt-x")
    projects = _projects_with_one(tmp_path, settings)
    reserving = AgentStore(settings, projects)
    with pytest.raises(InvalidAgent, match="reserved"):
        reserving.create(name="Orchestrator", project_id="my-app")


def test_orchestrator_backend_follows_settings(tmp_path: Path) -> None:
    """Its backend/model come from the landing settings, not agents.json."""
    settings = Settings(state_dir=tmp_path / "state", claude_model="claude-opus-4-8")
    settings.agent_backend = Backend.MOCK  # in-place mutation (validate_assignment)
    store = AgentStore(settings, ProjectStore(settings))
    assert store.get(ORCHESTRATOR_ID).backend == "mock"


def test_orchestrator_run_state_is_in_memory(tmp_path: Path) -> None:
    """mark_finished on the orchestrator updates in-memory state, never
    agents.json."""
    settings = _settings(tmp_path)
    store = AgentStore(settings, ProjectStore(settings))
    store.mark_running(ORCHESTRATOR_ID)
    assert store.get(ORCHESTRATOR_ID).state == "running"
    store.mark_finished(ORCHESTRATOR_ID, state=AgentState.DONE, session_id="orch-1")
    assert store.get(ORCHESTRATOR_ID).state == "done"
    assert store.get(ORCHESTRATOR_ID).session_id == "orch-1"
    assert not (settings.state_dir / "agents.json").exists()


def test_mock_backend_gated_by_flag(tmp_path: Path) -> None:
    """A mock agent is creatable only when enable_mock_backend is on."""
    off = Settings(state_dir=tmp_path / "state")  # flag defaults off
    projects = _projects_with_one(tmp_path, off)
    with pytest.raises(InvalidAgent, match="disabled backend 'mock'"):
        AgentStore(off, projects).create(name="M", project_id="my-app", backend="mock")


def test_backend_canonicalized_and_claude_default_model(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)

    # Legacy codex mode names collapse to "codex" and get the codex model.
    codex = store.create(name="Cx", project_id="my-app", backend="app_server")
    assert codex.backend == "codex"
    assert codex.model == settings.agent_model  # gpt-5.5

    # A claude agent gets the claude default model, NOT the codex "gpt-5.5".
    claude = store.create(name="Cl", project_id="my-app", backend="claude")
    assert claude.backend == "claude"
    assert claude.model == settings.claude_model
    assert claude.model != "gpt-5.5"


def test_update_backend_redefaults_model(tmp_path: Path) -> None:
    """Switching an agent's backend without sending a model re-stamps the model
    to the new backend's default (the reported gpt-5.5-sticks-on-claude bug)."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)

    codex = store.create(name="Builder", project_id="my-app", backend="codex")
    assert codex.model == settings.agent_model  # gpt-5.5

    # Switch to claude WITHOUT sending a model: the model must follow.
    switched = store.update("builder", backend="claude")
    assert switched.backend == "claude"
    assert switched.model == settings.claude_model
    assert switched.model != "gpt-5.5"

    # Switching back re-defaults to the codex model.
    back = store.update("builder", backend="codex")
    assert back.model == settings.agent_model


def test_update_backend_change_clears_session(tmp_path: Path) -> None:
    """A backend switch drops the stale (backend-specific) session id and resets
    the run state - so the next turn starts a fresh conversation instead of
    resuming a session the new backend cannot find (the reported claude
    error_during_execution). Regression pin for 20260721-152034."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)

    store.create(name="Builder", project_id="my-app", backend="codex")
    # A finished codex turn persisted a session id + a terminal state.
    store.mark_finished("builder", state=AgentState.DONE, session_id="codex-sess-1")
    assert store.get("builder").session_id == "codex-sess-1"

    # Switching to claude must NOT carry the codex session across.
    switched = store.update("builder", backend="claude")
    assert switched.backend == "claude"
    assert switched.session_id is None
    assert switched.state == "idle"

    # A no-op update (no backend change) leaves an existing session alone.
    store.mark_finished("builder", state=AgentState.DONE, session_id="claude-sess-2")
    same = store.update("builder", description="still claude")
    assert same.session_id == "claude-sess-2"


def test_update_explicit_model_wins_over_default(tmp_path: Path) -> None:
    """An explicit non-empty model on a backend switch is kept; a blank model
    falls back to the (effective) backend's default."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="codex")

    # Explicit model + backend switch: the explicit model wins.
    a = store.update("builder", backend="claude", model="claude-sonnet-4-6")
    assert a.model == "claude-sonnet-4-6"

    # A blank model re-defaults to the current (claude) backend's default.
    b = store.update("builder", model="   ")
    assert b.model == settings.claude_model


def test_update_model_only_no_backend_change_keeps_backend(tmp_path: Path) -> None:
    """Editing only the model (no backend change) does not touch the backend
    and keeps the given model."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="claude")

    updated = store.update("builder", model="claude-opus-4-8-custom")
    assert updated.backend == "claude"
    assert updated.model == "claude-opus-4-8-custom"


def test_legacy_backend_normalized_on_load(tmp_path: Path) -> None:
    """A persisted record with a legacy 'app_server' backend loads as 'codex'."""
    settings = _settings(tmp_path)
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "agents.json").write_text(
        '[{"id": "legacy", "name": "Legacy", "project_id": "p", '
        '"backend": "app_server", "model": "gpt-5.5"}]'
    )
    store = AgentStore(settings, ProjectStore(settings))
    assert store.get("legacy").backend == "codex"


def test_agent_store_permission_mode_default(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    a = store.create(name="A", project_id="my-app", permission_mode="auto")
    assert a.permission_mode == "auto"
    # An unknown mode folds to the safe default.
    b = store.create(name="B", project_id="my-app", permission_mode="nonsense")
    assert b.permission_mode == "manual"


def test_legacy_write_enabled_migrates_to_edit(tmp_path: Path) -> None:
    """A persisted legacy record with write_enabled=true loads as mode 'edit'."""
    settings = _settings(tmp_path)
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "agents.json").write_text(
        '[{"id": "w", "name": "W", "project_id": "p", "backend": "codex", '
        '"write_enabled": true}, '
        '{"id": "r", "name": "R", "project_id": "p", "backend": "codex", '
        '"write_enabled": false}]'
    )
    store = AgentStore(settings, ProjectStore(settings))
    assert store.get("w").permission_mode == "edit"
    assert store.get("r").permission_mode == "manual"


def test_orchestrator_and_subagent_sessions_stay_distinct_across_restart(
    tmp_path: Path,
) -> None:
    """The mixing reproduction (20260723-001251): an orchestrator turn and a
    codex sub-agent turn each record a session id; after a restart (a fresh
    store over the same state dir) BOTH ids must still be there and distinct.
    Before the registry, the orchestrator's id was in-memory only, so the
    restart lost it - the first step toward latching onto the wrong session."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="codex")

    # One finished turn each (the supervisor's persist path calls mark_finished).
    store.mark_finished(ORCHESTRATOR_ID, state=AgentState.DONE, session_id="orch-sess")
    store.mark_finished("builder", state=AgentState.DONE, session_id="sub-sess")
    assert store.orchestrator_session_id() == "orch-sess"
    assert store.get("builder").session_id == "sub-sess"

    # Simulated restart.
    fresh = AgentStore(settings, ProjectStore(settings))
    assert fresh.orchestrator_session_id() == "orch-sess"
    assert fresh.get(ORCHESTRATOR_ID).session_id == "orch-sess"
    assert fresh.get("builder").session_id == "sub-sess"
    assert fresh.orchestrator_session_id() != fresh.get("builder").session_id


def test_mark_finished_keys_session_by_run_backend_not_current(
    tmp_path: Path,
) -> None:
    """A backend switch that races an in-flight turn must not mislabel the
    finishing session: mark_finished keys the id by the backend the run
    executed under (passed explicitly), so a codex session that finishes AFTER
    a switch to claude is still recorded under codex - and stays unreachable
    from the now-current claude backend instead of being resumed by it."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="codex")

    # The record switched to claude mid-run; the codex turn now finishes.
    store.update("builder", backend="claude")
    store.mark_finished(
        "builder", state=AgentState.DONE, session_id="codex-sess-late", backend="codex"
    )

    # The claude record cannot see the codex session (backend-mismatch guard)...
    assert store.get("builder").session_id is None
    # ...but it is not lost: switching back to codex resumes it.
    switched_back = store.update("builder", backend="codex")
    assert switched_back.session_id is None  # the switch itself cleared codex
    # A fresh codex turn's id lands under codex and reads back.
    store.mark_finished("builder", state=AgentState.DONE, session_id="codex-sess-2")
    assert store.get("builder").session_id == "codex-sess-2"


def test_delete_removes_session_mapping(tmp_path: Path) -> None:
    """Deleting an agent removes its registry mapping: a NEW agent that happens
    to reuse the freed id must not inherit the old session."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="codex")
    store.mark_finished("builder", state=AgentState.DONE, session_id="old-sess")
    store.delete("builder")

    recreated = store.create(name="Builder", project_id="my-app", backend="codex")
    assert recreated.id == "builder"  # the freed id is reused
    assert recreated.session_id is None
    # And a fresh store agrees (the removal was persisted).
    fresh = AgentStore(settings, ProjectStore(settings))
    assert fresh.get("builder").session_id is None


def test_backend_switch_clears_session_mapping(tmp_path: Path) -> None:
    """A backend switch clears the persisted mapping too: after a restart the
    stale wrong-backend id must not resurface. (The in-record clearing is
    pinned by test_update_backend_change_clears_session; this pins the
    registry/persistence side.)"""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="codex")
    store.mark_finished("builder", state=AgentState.DONE, session_id="codex-sess-1")

    store.update("builder", backend="claude")
    fresh = AgentStore(settings, ProjectStore(settings))
    assert fresh.get("builder").session_id is None
    # Switching back to codex must NOT resurrect the old codex id either.
    fresh.update("builder", backend="codex")
    assert fresh.get("builder").session_id is None


def test_legacy_agents_json_session_id_migrates_to_registry(tmp_path: Path) -> None:
    """A pre-registry agents.json that still carries a session_id seeds the
    registry on load, so an upgrade does not drop live conversations."""
    settings = _settings(tmp_path)
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "agents.json").write_text(
        '[{"id": "legacy", "name": "Legacy", "project_id": "p", '
        '"backend": "codex", "session_id": "legacy-sess"}]'
    )
    store = AgentStore(settings, ProjectStore(settings))
    assert store.get("legacy").session_id == "legacy-sess"
    # And it survives another restart via the registry (not agents.json).
    fresh = AgentStore(settings, ProjectStore(settings))
    assert fresh.get("legacy").session_id == "legacy-sess"


# --- SessionRegistry: multi-session history + ownership (part 1) --------------


def test_registry_add_accumulates_history(tmp_path: Path) -> None:
    reg = SessionRegistry(_settings(tmp_path))
    reg.add("a", "codex", "s1")
    reg.add("a", "codex", "s2")
    assert reg.get("a", "codex") == "s2"  # current is the latest
    assert reg.sessions_for("a", "codex") == ["s1", "s2"]
    # Re-adding a known id does not duplicate it, just re-currents.
    reg.add("a", "codex", "s1")
    assert reg.get("a", "codex") == "s1"
    assert reg.sessions_for("a", "codex") == ["s1", "s2"]


def test_registry_set_current_preserves_history(tmp_path: Path) -> None:
    reg = SessionRegistry(_settings(tmp_path))
    reg.add("a", "codex", "s1")
    reg.set_current("a", "codex", None)  # "new chat"
    assert reg.get("a", "codex") is None
    assert reg.sessions_for("a", "codex") == ["s1"]  # history kept


def test_registry_set_current_appends_unseen(tmp_path: Path) -> None:
    reg = SessionRegistry(_settings(tmp_path))
    reg.add("a", "codex", "s1")
    reg.set_current("a", "codex", "s2")  # switch to an id we had not recorded
    assert reg.get("a", "codex") == "s2"
    assert reg.sessions_for("a", "codex") == ["s1", "s2"]


def test_registry_remove_drops_one_session(tmp_path: Path) -> None:
    reg = SessionRegistry(_settings(tmp_path))
    reg.add("a", "codex", "s1")
    reg.add("a", "codex", "s2")
    reg.remove("a", "codex", "s1")
    assert reg.sessions_for("a", "codex") == ["s2"]
    assert reg.get("a", "codex") == "s2"
    reg.remove("a", "codex", "s2")  # removing the current one clears current
    assert reg.sessions_for("a", "codex") == []
    assert reg.get("a", "codex") is None


def test_registry_backend_switch_resets_history(tmp_path: Path) -> None:
    reg = SessionRegistry(_settings(tmp_path))
    reg.add("a", "codex", "s1")
    reg.add("a", "codex", "s2")
    reg.add("a", "claude", "c1")  # a different backend starts fresh
    assert reg.sessions_for("a", "claude") == ["c1"]
    assert reg.sessions_for("a", "codex") == []  # old-backend history unreachable
    assert reg.get("a", "codex") is None


def test_legacy_session_entry_loads_as_single_history(tmp_path: Path) -> None:
    """A pre-multi-session sessions.json entry ({backend, session_id}) loads as a
    one-element history so an upgrade keeps that session listed."""
    settings = _settings(tmp_path)
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "sessions.json").write_text(
        '{"orchestrator": {"backend": "codex", "session_id": "leg-sess"}}'
    )
    reg = SessionRegistry(settings)
    assert reg.get("orchestrator", "codex") == "leg-sess"
    assert reg.sessions_for("orchestrator", "codex") == ["leg-sess"]


def test_registry_records_and_preserves_spawn_parent(tmp_path: Path) -> None:
    """set_parent records who/which-chat spawned a child - even before the child
    has a session - and _fresh preserves it when the child later runs (parent is a
    backend-independent fact)."""
    reg = SessionRegistry(_settings(tmp_path))
    # No entry yet -> a minimal placeholder is created.
    reg.set_parent("builder", "orchestrator", "chat-1")
    assert reg.parent_of("builder") == ("orchestrator", "chat-1")
    # The child then runs (mints a session) -> parent is preserved through _fresh.
    reg.add("builder", "codex", "sess-b")
    assert reg.parent_of("builder") == ("orchestrator", "chat-1")
    assert reg.get("builder", "codex") == "sess-b"
    # A backend switch (fresh history) still preserves the parent.
    reg.add("builder", "claude", "sess-c")
    assert reg.parent_of("builder") == ("orchestrator", "chat-1")
    # Survives a reload.
    fresh = SessionRegistry(_settings(tmp_path))
    assert fresh.parent_of("builder") == ("orchestrator", "chat-1")
    # Unknown agent -> (None, None).
    assert fresh.parent_of("nobody") == (None, None)


def test_store_record_spawn_parent_round_trips(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="codex")
    store.record_spawn_parent("builder", ORCHESTRATOR_ID, "chat-9")
    assert store.parent_of("builder") == (ORCHESTRATOR_ID, "chat-9")
    # Persisted.
    fresh = AgentStore(settings, ProjectStore(settings))
    assert fresh.parent_of("builder") == (ORCHESTRATOR_ID, "chat-9")


def test_orchestrator_session_history_accumulates(tmp_path: Path) -> None:
    """Each finished orchestrator turn with a new id appends to its history."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    store.mark_finished(ORCHESTRATOR_ID, state=AgentState.DONE, session_id="o1")
    store.set_orchestrator_session(None)  # new chat
    store.mark_finished(ORCHESTRATOR_ID, state=AgentState.DONE, session_id="o2")
    assert store.orchestrator_session_id() == "o2"
    assert store.orchestrator_sessions() == ["o1", "o2"]
    # Forgetting one (a session delete) drops it from the switcher history.
    store.forget_orchestrator_session("o1")
    assert store.orchestrator_sessions() == ["o2"]


def test_agent_description_round_trips(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    a = store.create(name="Doc", project_id="my-app", description="  a helpful agent  ")
    assert a.description == "a helpful agent"  # stripped
    # Persists and is updatable.
    store.update(a.id, description="updated blurb")
    reloaded = AgentStore(settings, ProjectStore(settings)).get(a.id)
    assert reloaded.description == "updated blurb"
    # goal is optional and defaults empty (retired from the create flow).
    assert reloaded.goal == ""


# --- BC1: durable run-outcome record (bidirectional-comms substrate) ----------


def test_waiting_state_is_distinct() -> None:
    """AgentState.WAITING ('ended a turn awaiting a decision') is a real member,
    distinct from BLOCKED (waiting on an approval) and DONE."""
    assert AgentState.WAITING == "waiting"
    assert AgentState.WAITING != AgentState.BLOCKED
    assert AgentState.WAITING != AgentState.DONE


def test_run_outcome_persists_and_survives_restart(tmp_path: Path) -> None:
    """A finished run leaves a durable outcome (final message + terminal state)
    readable from a fresh store over the same state_dir - the substrate that
    outlives the ephemeral per-run EventBus (BC1, spike 20260723-001256)."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="mock")

    store.mark_finished(
        "builder",
        state=AgentState.WAITING,
        session_id="sess-1",
        message="should I merge to master?",
        run_id="builder:run-1",
    )

    # Same-process read.
    outcome = store.outcome("builder")
    assert outcome is not None
    assert outcome.state == AgentState.WAITING
    assert outcome.message == "should I merge to master?"
    assert outcome.session_id == "sess-1"
    assert outcome.run_id == "builder:run-1"
    assert outcome.acknowledged is False
    assert outcome.ts > 0
    assert "builder" in store.outcomes()

    # Survives a simulated restart (fresh store over the same state_dir).
    fresh = AgentStore(settings, ProjectStore(settings))
    reloaded = fresh.outcome("builder")
    assert reloaded is not None
    assert reloaded.state == AgentState.WAITING
    assert reloaded.message == "should I merge to master?"
    assert reloaded.session_id == "sess-1"


def test_delete_removes_outcome(tmp_path: Path) -> None:
    """Deleting an agent drops its outcome, and it does not resurrect on
    restart - a reused id can never inherit a stale outcome."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="mock")
    store.mark_finished("builder", state=AgentState.DONE, message="done")
    assert store.outcome("builder") is not None

    store.delete("builder")
    assert store.outcome("builder") is None
    fresh = AgentStore(settings, ProjectStore(settings))
    assert fresh.outcome("builder") is None


def test_delete_then_mark_finished_does_not_resurrect_outcome(tmp_path: Path) -> None:
    """A run that finishes AFTER its agent was deleted mid-run (the persist
    callback firing post-delete - an anticipated path, per app.py) must not
    resurrect a stale outcome: mark_finished raises AgentNotFound and writes
    nothing. Regression for review R1.1."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="mock")
    store.mark_finished("builder", state=AgentState.WAITING, message="merge?")
    assert store.outcome("builder") is not None

    store.delete("builder")
    # The racing completion callback fires after the delete.
    with pytest.raises(AgentNotFound):
        store.mark_finished("builder", state=AgentState.DONE, message="late")

    assert store.outcome("builder") is None
    fresh = AgentStore(settings, ProjectStore(settings))
    assert fresh.outcome("builder") is None


def test_error_terminal_outcome_recorded(tmp_path: Path) -> None:
    """An error turn (no final reply, so no message) records an ERROR outcome
    with an empty message, not a crash (review R1.3)."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="mock")
    store.mark_finished("builder", state=AgentState.ERROR)
    outcome = store.outcome("builder")
    assert outcome is not None
    assert outcome.state == AgentState.ERROR
    assert outcome.message == ""


def test_outcome_store_tolerates_a_corrupt_file(tmp_path: Path) -> None:
    """A garbled outcomes.json loads as empty, like the other stores."""
    settings = _settings(tmp_path)
    state = settings.state_dir
    state.mkdir(parents=True, exist_ok=True)
    (state / "outcomes.json").write_text("{ not json")
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    assert store.outcomes() == {}


# --- BC2: request_input needs-input signal ------------------------------------


def test_request_input_sets_waiting_outcome(tmp_path: Path) -> None:
    """A sub-agent's request_input records a WAITING outcome carrying the
    question, keyed to the current run, unacknowledged (BC2)."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="mock")

    store.request_input(
        "builder", "should I merge to master?", run_id="builder:r1", session_id="s1"
    )
    outcome = store.outcome("builder")
    assert outcome is not None
    assert outcome.state == AgentState.WAITING
    assert outcome.message == "should I merge to master?"
    assert outcome.run_id == "builder:r1"
    assert outcome.session_id == "s1"
    assert outcome.acknowledged is False


def test_waiting_survives_same_run_completion(tmp_path: Path) -> None:
    """request_input fires mid-turn; the turn then ends DONE. The natural
    completion must NOT clobber the WAITING outcome for the SAME run - it keeps
    WAITING + the question, and refreshes the now-finalized session id (BC2)."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="mock")

    store.request_input("builder", "merge?", run_id="builder:r1")
    # The turn ends normally right after; the completion callback fires for r1.
    store.mark_finished(
        "builder", state=AgentState.DONE, session_id="s1", run_id="builder:r1"
    )
    outcome = store.outcome("builder")
    assert outcome is not None
    assert outcome.state == AgentState.WAITING  # preserved, not DONE
    assert outcome.message == "merge?"
    assert outcome.session_id == "s1"  # refreshed from the finished run


def test_stale_waiting_overwritten_by_a_new_run(tmp_path: Path) -> None:
    """A WAITING outcome from a PRIOR run does not stick forever: a new run that
    finishes DONE (different run_id) overwrites it (BC2 - run-id-keyed)."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="mock")

    store.request_input("builder", "merge?", run_id="builder:r1")
    # The orchestrator resumed and the agent finished a LATER run without asking.
    store.mark_finished(
        "builder", state=AgentState.DONE, message="done", run_id="builder:r2"
    )
    outcome = store.outcome("builder")
    assert outcome is not None
    assert outcome.state == AgentState.DONE
    assert outcome.message == "done"


def test_error_after_request_input_wins(tmp_path: Path) -> None:
    """If the run ERRORs after a request_input, the error terminal state wins
    over the WAITING signal (the agent did not cleanly wait, it crashed)."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="mock")

    store.request_input("builder", "merge?", run_id="builder:r1")
    store.mark_finished("builder", state=AgentState.ERROR, run_id="builder:r1")
    outcome = store.outcome("builder")
    assert outcome is not None
    assert outcome.state == AgentState.ERROR


def test_request_input_on_deleted_agent_raises(tmp_path: Path) -> None:
    """request_input on a missing agent raises AgentNotFound and writes nothing,
    like mark_finished."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    with pytest.raises(AgentNotFound):
        store.request_input("ghost", "merge?", run_id="ghost:r1")
    assert store.outcome("ghost") is None


# --- report_back: finished-my-task signal (sibling of request_input) ----------


def test_report_back_sets_reported_outcome(tmp_path: Path) -> None:
    """A sub-agent's report_back records a REPORTED outcome carrying the summary,
    keyed to the current run, unacknowledged."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="mock")

    store.report_back(
        "builder", "implemented X; tests green", run_id="builder:r1", session_id="s1"
    )
    outcome = store.outcome("builder")
    assert outcome is not None
    assert outcome.state == AgentState.REPORTED
    assert outcome.message == "implemented X; tests green"
    assert outcome.run_id == "builder:r1"
    assert outcome.session_id == "s1"
    assert outcome.acknowledged is False


def test_reported_survives_same_run_completion(tmp_path: Path) -> None:
    """report_back fires mid-turn; the turn then ends DONE. The natural completion
    must NOT clobber the REPORTED outcome for the SAME run - it keeps REPORTED + the
    summary, and refreshes the now-finalized session id (mirrors WAITING)."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="mock")

    store.report_back("builder", "done: X shipped", run_id="builder:r1")
    store.mark_finished(
        "builder", state=AgentState.DONE, session_id="s1", run_id="builder:r1"
    )
    outcome = store.outcome("builder")
    assert outcome is not None
    assert outcome.state == AgentState.REPORTED  # preserved, not DONE
    assert outcome.message == "done: X shipped"
    assert outcome.session_id == "s1"  # refreshed from the finished run


def test_stale_reported_overwritten_by_a_new_run(tmp_path: Path) -> None:
    """A REPORTED outcome from a PRIOR run does not stick forever: a new run that
    finishes DONE (different run_id) overwrites it (run-id-keyed)."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="mock")

    store.report_back("builder", "done: X shipped", run_id="builder:r1")
    store.mark_finished(
        "builder", state=AgentState.DONE, message="done", run_id="builder:r2"
    )
    outcome = store.outcome("builder")
    assert outcome is not None
    assert outcome.state == AgentState.DONE
    assert outcome.message == "done"


def test_error_after_report_back_wins(tmp_path: Path) -> None:
    """If the run ERRORs after a report_back, the error terminal state wins over the
    REPORTED signal (the agent did not cleanly finish, it crashed)."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="mock")

    store.report_back("builder", "done: X shipped", run_id="builder:r1")
    store.mark_finished("builder", state=AgentState.ERROR, run_id="builder:r1")
    outcome = store.outcome("builder")
    assert outcome is not None
    assert outcome.state == AgentState.ERROR


def test_report_back_on_deleted_agent_raises(tmp_path: Path) -> None:
    """report_back on a missing agent raises AgentNotFound and writes nothing,
    like request_input."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    with pytest.raises(AgentNotFound):
        store.report_back("ghost", "done", run_id="ghost:r1")
    assert store.outcome("ghost") is None


# --- BC3: pending outcomes + acknowledge --------------------------------------


def _agent(store: AgentStore, name: str) -> str:
    return store.create(name=name, project_id="my-app", backend="mock").id


def test_pending_outcomes_lists_waiting_reported_and_error_only(
    tmp_path: Path,
) -> None:
    """pending_outcomes surfaces the agents that need the orchestrator: an
    unacknowledged WAITING (needs input), REPORTED (finished + reported) or ERROR
    outcome. A cleanly DONE agent that did NOT report is not pending (BC3)."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    for n in ("Waiter", "Reporter", "Crasher", "Finisher"):
        _agent(store, n)

    store.request_input("waiter", "merge?", run_id="waiter:r1")
    store.report_back("reporter", "shipped X", run_id="reporter:r1")
    store.mark_finished("crasher", state=AgentState.ERROR, run_id="crasher:r1")
    store.mark_finished(
        "finisher", state=AgentState.DONE, message="all done", run_id="finisher:r1"
    )

    pending = store.pending_outcomes()
    assert set(pending) == {"waiter", "reporter", "crasher"}
    assert pending["waiter"].state == AgentState.WAITING
    assert pending["waiter"].message == "merge?"
    assert pending["reporter"].state == AgentState.REPORTED
    assert pending["reporter"].message == "shipped X"
    assert pending["crasher"].state == AgentState.ERROR


def test_pending_outcomes_excludes_the_orchestrator(tmp_path: Path) -> None:
    """The orchestrator is never a member of its OWN 'who needs me' poll (mirrors
    list() hiding it). Its turns now persist an ERROR outcome on a StreamError, so
    without the guard it would self-appear in pending_agents - exclude it."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)

    store.mark_finished(
        ORCHESTRATOR_ID,
        state=AgentState.ERROR,
        message="app-server timed out after 120s",
        run_id="orch:r1",
    )
    # The outcome is recorded (readable directly) but not surfaced as pending.
    assert store.outcome(ORCHESTRATOR_ID) is not None
    assert ORCHESTRATOR_ID not in store.pending_outcomes()


def test_acknowledge_clears_from_pending(tmp_path: Path) -> None:
    """acknowledge marks a pending outcome handled so it drops out of the poll,
    persists, and is idempotent (BC3)."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings)
    store = AgentStore(settings, projects)
    _agent(store, "Waiter")
    store.request_input("waiter", "merge?", run_id="waiter:r1")
    assert "waiter" in store.pending_outcomes()

    assert store.acknowledge("waiter") is True
    assert "waiter" not in store.pending_outcomes()
    # The outcome is retained (still readable), just marked acknowledged.
    outcome = store.outcome("waiter")
    assert outcome is not None
    assert outcome.acknowledged is True
    # Idempotent: a second ack is a no-op returning False.
    assert store.acknowledge("waiter") is False
    # An agent with no outcome (or unknown) acks to False, not an error.
    assert store.acknowledge("ghost") is False

    # Survives a restart.
    fresh = AgentStore(settings, ProjectStore(settings))
    assert "waiter" not in fresh.pending_outcomes()
