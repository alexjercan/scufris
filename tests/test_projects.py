"""Tests for the first-class Project store: CRUD, persistence, validation, gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from scufris.config import Settings
from scufris.projects import (
    InvalidProject,
    ProjectNotFound,
    ProjectsReadOnly,
    ProjectStore,
    read_project_tasks,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(state_dir=tmp_path / "state")


def test_project_store_round_trip(tmp_path: Path) -> None:
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    store = ProjectStore(_settings(tmp_path))
    created = store.create(
        name="My App", cwd=str(proj_dir), language="python", description="a thing"
    )
    assert created.id == "my-app"
    assert created.cwd == str(proj_dir)

    # A fresh store over the same state dir sees it.
    fresh = ProjectStore(_settings(tmp_path))
    got = fresh.get("my-app")
    assert got.name == "My App"
    assert got.language == "python"

    # Update persists.
    fresh.update("my-app", description="updated")
    assert ProjectStore(_settings(tmp_path)).get("my-app").description == "updated"

    # Delete persists.
    fresh.delete("my-app")
    assert ProjectStore(_settings(tmp_path)).list() == []


def test_project_create_validates_cwd_and_name(tmp_path: Path) -> None:
    store = ProjectStore(_settings(tmp_path))
    with pytest.raises(InvalidProject):
        store.create(name="x", cwd=str(tmp_path / "does-not-exist"))
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    with pytest.raises(InvalidProject):
        store.create(name="   ", cwd=str(proj_dir))


def test_duplicate_name_gets_distinct_ids(tmp_path: Path) -> None:
    d = tmp_path / "d"
    d.mkdir()
    store = ProjectStore(_settings(tmp_path))
    a = store.create(name="Same", cwd=str(d))
    b = store.create(name="Same", cwd=str(d))
    assert a.id == "same"
    assert b.id == "same-2"
    assert {p.id for p in store.list()} == {"same", "same-2"}


def test_get_and_delete_unknown_raise(tmp_path: Path) -> None:
    store = ProjectStore(_settings(tmp_path))
    with pytest.raises(ProjectNotFound):
        store.get("ghost")
    with pytest.raises(ProjectNotFound):
        store.delete("ghost")


def test_writes_refused_when_read_only(tmp_path: Path) -> None:
    d = tmp_path / "d"
    d.mkdir()
    ro = Settings(state_dir=tmp_path / "state", settings_writable=False)
    store = ProjectStore(ro)
    with pytest.raises(ProjectsReadOnly):
        store.create(name="x", cwd=str(d))


def test_store_ignores_corrupt_file(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "projects.json").write_text("{not json")
    ProjectStore(_settings(tmp_path))  # must not raise


def _tatr_new(cwd: Path, title: str, priority: int, tags: str) -> None:
    """Create a real tatr task under cwd/tasks (tatr needs the dir to exist)."""
    import subprocess

    (cwd / "tasks").mkdir(exist_ok=True)
    subprocess.run(
        ["tatr", "-r", str(cwd), "new", title, "-p", str(priority), "-t", tags],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.needs_tatr
def test_read_project_tasks_parses_real_tatr(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    _tatr_new(proj, "Implement the widget", 30, "feature,ui")
    tasks = read_project_tasks(str(proj))
    assert len(tasks) == 1
    task = tasks[0]
    assert task.title == "Implement the widget"
    assert task.priority == 30
    assert set(task.tags) == {"feature", "ui"}
    assert task.id  # the task dir name (a timestamp)


@pytest.mark.needs_tatr
def test_read_project_tasks_empty_when_no_tasks_dir(tmp_path: Path) -> None:
    # A project dir with no tasks/ returns [] and does NOT walk up to a parent.
    parent = tmp_path / "parent"
    (parent / "tasks").mkdir(parents=True)
    _tatr_new(parent, "parent task", 5, "feature")
    child = parent / "child"
    child.mkdir()
    assert read_project_tasks(str(child)) == []  # child has no tasks/ of its own
