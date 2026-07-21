"""Tests for the first-class AgentStore: CRUD, persistence, validation, gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from scufris.agent_store import (
    AgentNotFound,
    AgentsReadOnly,
    AgentStore,
    InvalidAgent,
)
from scufris.config import Settings
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

    # Delete persists.
    fresh.delete("builder")
    assert AgentStore(settings, ProjectStore(settings)).list() == []


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
    # Load does not raise; the store is just empty.
    store = AgentStore(settings, ProjectStore(settings))
    assert store.list() == []


def test_get_unknown_agent_raises(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = AgentStore(settings, ProjectStore(settings))
    with pytest.raises(AgentNotFound):
        store.get("ghost")


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
