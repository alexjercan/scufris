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
