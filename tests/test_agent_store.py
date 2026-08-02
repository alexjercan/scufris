"""The first-class AgentStore: CRUD, persistence, validation, and the write gate.

Covers creating, reading, updating and deleting an agent, the validation that
refuses a bad name, backend or project, the read-only gate, the reserved
orchestrator that cannot be deleted, the backend and model defaults - including
what an explicit model overrides - and the migrations applied on load.

The ``_settings`` and ``_projects_with_one`` helpers here are imported by
``tests/test_agent_sessions.py`` and ``tests/test_agent_outcomes.py``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DatabaseError

from scufris.agent_store import (
    HOST_AGENT_ID,
    ORCHESTRATOR_ID,
    AgentNotFound,
    AgentsReadOnly,
    AgentStore,
    InvalidAgent,
    ReservedAgent,
)
from scufris.config import Settings
from scufris.db import Database, open_database
from scufris.enums import AgentState, Backend
from scufris.projects import ProjectStore


def _settings(tmp_path: Path, *, writable: bool = True) -> Settings:
    return Settings(
        state_dir=tmp_path,
        settings_writable=writable,
        enable_mock_backend=True,  # tests create mock-backed agents
    )


def _projects_with_one(
    tmp_path: Path, settings: Settings, database: Database
) -> ProjectStore:
    """A ProjectStore holding a single project 'my-app' at a real dir."""
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir(exist_ok=True)
    projects = ProjectStore(settings, database)
    projects.create(name="My App", cwd=str(proj_dir))
    return projects


def test_agent_store_round_trip(tmp_path: Path, database: Database) -> None:
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects, database)

    created = store.create(
        name="Builder", project_id="my-app", backend="mock", goal="do the thing"
    )
    assert created.id == "builder"
    assert created.project_id == "my-app"
    assert created.backend == "mock"
    assert created.state == "idle"
    assert created.permission_mode == "manual"  # safe default

    # A fresh store over the same state dir sees it (projects reloaded too).
    fresh_projects = ProjectStore(settings, database)
    fresh = AgentStore(settings, fresh_projects, database)
    got = fresh.get("builder")
    assert got.name == "Builder"
    assert got.goal == "do the thing"

    # Update persists.
    fresh.update("builder", permission_mode="edit", model="gpt-x")
    reloaded = AgentStore(settings, ProjectStore(settings, database), database).get(
        "builder"
    )
    assert reloaded.permission_mode == "edit"
    assert reloaded.model == "gpt-x"

    # Delete persists. `list()` still holds the reserved HOST agent, which is
    # synthetic and never in agents.json (the orchestrator stays hidden from it).
    fresh.delete("builder")
    reloaded_list = AgentStore(
        settings, ProjectStore(settings, database), database
    ).list()
    assert [a.id for a in reloaded_list] == [HOST_AGENT_ID]


def test_create_agent_rejects_unknown_project(
    tmp_path: Path, database: Database
) -> None:
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects, database)
    with pytest.raises(InvalidAgent, match="no such project"):
        store.create(name="Ghost", project_id="does-not-exist")


def test_create_agent_validates_name_and_backend(
    tmp_path: Path, database: Database
) -> None:
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects, database)
    with pytest.raises(InvalidAgent, match="name must not be empty"):
        store.create(name="   ", project_id="my-app")
    with pytest.raises(InvalidAgent, match="unknown or disabled backend"):
        store.create(name="Bad", project_id="my-app", backend="nope")


def test_agent_ids_dedup(tmp_path: Path, database: Database) -> None:
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects, database)
    a = store.create(name="Worker", project_id="my-app")
    b = store.create(name="Worker", project_id="my-app")
    assert a.id == "worker"
    assert b.id == "worker-2"


def test_agent_writes_gated_when_read_only(tmp_path: Path, database: Database) -> None:
    settings = _settings(tmp_path, writable=False)
    # Seed a project with a writable store first, then a read-only agent store.
    writable = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, writable, database)
    store = AgentStore(settings, ProjectStore(settings, database), database)
    # The project persisted by the writable store is visible read-only.
    assert projects.get("my-app").id == "my-app"
    with pytest.raises(AgentsReadOnly):
        store.create(name="Nope", project_id="my-app")


def test_agent_store_tolerates_a_corrupt_file(
    tmp_path: Path, database: Database
) -> None:
    settings = _settings(tmp_path)
    state = tmp_path
    state.mkdir(parents=True, exist_ok=True)
    (state / "agents.json").write_text("{ this is not valid json ]")
    # Load does not raise; the store has no persisted agents. Both reserved agents
    # are synthetic: the orchestrator is HIDDEN from the list (still reachable via
    # get), the host agent is listed so a delegation target is visible.
    store = AgentStore(settings, ProjectStore(settings, database), database)
    assert [a.id for a in store.list()] == [HOST_AGENT_ID]
    assert store.get(ORCHESTRATOR_ID).id == ORCHESTRATOR_ID


def test_get_unknown_agent_raises(tmp_path: Path, database: Database) -> None:
    settings = _settings(tmp_path)
    store = AgentStore(settings, ProjectStore(settings, database), database)
    with pytest.raises(AgentNotFound):
        store.get("ghost")


def test_orchestrator_reserved_and_undeletable(
    tmp_path: Path, database: Database
) -> None:
    """The orchestrator is a synthetic reserved agent: reachable via get but
    HIDDEN from the list (a hidden default), not in agents.json, undeletable,
    un-creatable by id, and not store-editable (its config lives in settings)."""
    settings = _settings(tmp_path)
    store = AgentStore(settings, ProjectStore(settings, database), database)

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
    projects = _projects_with_one(tmp_path, settings, database)
    reserving = AgentStore(settings, projects, database)
    with pytest.raises(InvalidAgent, match="reserved"):
        reserving.create(name="Orchestrator", project_id="my-app")


def test_orchestrator_backend_follows_settings(
    tmp_path: Path, database: Database
) -> None:
    """Its backend/model come from the landing settings, not agents.json."""
    settings = Settings(state_dir=tmp_path, claude_model="claude-opus-4-8")
    settings.agent_backend = Backend.MOCK  # in-place mutation (validate_assignment)
    store = AgentStore(settings, ProjectStore(settings, database), database)
    assert store.get(ORCHESTRATOR_ID).backend == "mock"


def test_orchestrator_run_state_is_in_memory(
    tmp_path: Path, database: Database
) -> None:
    """mark_finished on the orchestrator updates in-memory state, never
    agents.json."""
    settings = _settings(tmp_path)
    store = AgentStore(settings, ProjectStore(settings, database), database)
    store.mark_running(ORCHESTRATOR_ID)
    assert store.get(ORCHESTRATOR_ID).state == "running"
    store.mark_finished(ORCHESTRATOR_ID, state=AgentState.DONE, session_id="orch-1")
    assert store.get(ORCHESTRATOR_ID).state == "done"
    assert store.get(ORCHESTRATOR_ID).session_id == "orch-1"
    assert not (settings.state_dir / "agents.json").exists()


def test_mock_backend_gated_by_flag(tmp_path: Path, database: Database) -> None:
    """A mock agent is creatable only when enable_mock_backend is on."""
    off = Settings(state_dir=tmp_path)  # flag defaults off
    projects = _projects_with_one(tmp_path, off, database)
    with pytest.raises(InvalidAgent, match="disabled backend 'mock'"):
        AgentStore(off, projects, database).create(
            name="M", project_id="my-app", backend="mock"
        )


def test_backend_canonicalized_and_claude_default_model(
    tmp_path: Path, database: Database
) -> None:
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects, database)

    # Legacy codex mode names collapse to "codex" and get the codex model.
    codex = store.create(name="Cx", project_id="my-app", backend="app_server")
    assert codex.backend == "codex"
    assert codex.model == settings.agent_model  # gpt-5.5

    # A claude agent gets the claude default model, NOT the codex "gpt-5.5".
    claude = store.create(name="Cl", project_id="my-app", backend="claude")
    assert claude.backend == "claude"
    assert claude.model == settings.claude_model
    assert claude.model != "gpt-5.5"


def test_update_backend_redefaults_model(tmp_path: Path, database: Database) -> None:
    """Switching an agent's backend without sending a model re-stamps the model
    to the new backend's default (the reported gpt-5.5-sticks-on-claude bug)."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects, database)

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


def test_update_backend_change_clears_session(
    tmp_path: Path, database: Database
) -> None:
    """A backend switch drops the stale (backend-specific) session id and resets
    the run state - so the next turn starts a fresh conversation instead of
    resuming a session the new backend cannot find (the reported claude
    error_during_execution)."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects, database)

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


def test_update_explicit_model_wins_over_default(
    tmp_path: Path, database: Database
) -> None:
    """An explicit non-empty model on a backend switch is kept; a blank model
    falls back to the (effective) backend's default."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects, database)
    store.create(name="Builder", project_id="my-app", backend="codex")

    # Explicit model + backend switch: the explicit model wins.
    a = store.update("builder", backend="claude", model="claude-sonnet-4-6")
    assert a.model == "claude-sonnet-4-6"

    # A blank model re-defaults to the current (claude) backend's default.
    b = store.update("builder", model="   ")
    assert b.model == settings.claude_model


def test_update_model_only_no_backend_change_keeps_backend(
    tmp_path: Path, database: Database
) -> None:
    """Editing only the model (no backend change) does not touch the backend
    and keeps the given model."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects, database)
    store.create(name="Builder", project_id="my-app", backend="claude")

    updated = store.update("builder", model="claude-opus-4-8-custom")
    assert updated.backend == "claude"
    assert updated.model == "claude-opus-4-8-custom"


def test_agent_store_permission_mode_default(
    tmp_path: Path, database: Database
) -> None:
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects, database)
    a = store.create(name="A", project_id="my-app", permission_mode="auto")
    assert a.permission_mode == "auto"
    # An unknown mode folds to the safe default.
    b = store.create(name="B", project_id="my-app", permission_mode="nonsense")
    assert b.permission_mode == "manual"


def test_agent_description_round_trips(tmp_path: Path, database: Database) -> None:
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects, database)
    a = store.create(name="Doc", project_id="my-app", description="  a helpful agent  ")
    assert a.description == "a helpful agent"  # stripped
    # Persists and is updatable.
    store.update(a.id, description="updated blurb")
    reloaded = AgentStore(settings, ProjectStore(settings, database), database).get(
        a.id
    )
    assert reloaded.description == "updated blurb"
    # goal is optional and defaults empty (retired from the create flow).
    assert reloaded.goal == ""


# --- the durability proofs this cutover exists for ---------------------------
#
# Enough concurrent completions to lose a record under the three-file JSON write
# the database replaces (20260729-102146 measured loss and torn updates at this
# width), small enough that the burst stays well under a second.
BURST = 24


def test_concurrent_agent_completions_persist_every_record(
    tmp_path: Path, database: Database
) -> None:
    """The headline proof: simultaneous completions lose nothing.

    A completion writes three things - the agent row's terminal state, the
    session record and the outcome. Under the JSON stores each was a full-file
    rewrite through one shared temp path, so a burst this wide lost records
    outright AND published half-finished completions. The reconstruction is what
    makes it a durability claim rather than a claim about one process's memory:
    a SECOND handle on the same file has no in-memory mirror to be right for the
    wrong reason.
    """
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects, database)
    agents = [
        store.create(name=f"Agent {n:02d}", project_id="my-app", backend="mock").id
        for n in range(BURST)
    ]

    def finish(index: int) -> None:
        store.mark_finished(
            agents[index],
            state=AgentState.DONE,
            session_id=f"sess-{index:02d}",
            message=f"done {index:02d}",
            run_id=f"run-{index:02d}",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(finish, range(BURST)))

    reopened = open_database(tmp_path)
    try:
        restarted = AgentStore(settings, ProjectStore(settings, reopened), reopened)
        outcomes = restarted.outcomes()
        assert set(outcomes) == set(agents)
        for index, agent_id in enumerate(agents):
            record = restarted.get(agent_id)
            assert record.state == AgentState.DONE
            assert record.session_id == f"sess-{index:02d}"
            assert outcomes[agent_id].message == f"done {index:02d}"
    finally:
        reopened.close()


def test_agent_completion_commits_as_one_transaction(
    tmp_path: Path, database: Database
) -> None:
    """A completion whose outcome write fails leaves NO partial record.

    This is the guarantee the orchestrator depends on: it polls outcomes, so an
    outcome without the session record it names is a report of a conversation
    that cannot be resumed - and the opposite, a terminal agent row with no
    outcome, is an agent that finished silently. Dropping `agent_outcome` makes
    the LAST of the three writes fail, which is the ordering that used to leave
    the first two committed.
    """
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects, database)
    agent_id = store.create(name="Builder", project_id="my-app", backend="mock").id
    with database.transaction() as conn:
        conn.execute(text("DROP TABLE agent_outcome"))

    with pytest.raises(DatabaseError):
        store.mark_finished(
            agent_id,
            state=AgentState.DONE,
            session_id="sess-1",
            message="finished",
            run_id="run-1",
        )

    # Neither the terminal state nor the session record survived the rollback.
    assert store.get(agent_id).state == AgentState.IDLE
    assert store.get(agent_id).session_id is None
