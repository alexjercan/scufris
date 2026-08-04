"""Tests for the first-class Project store: CRUD, persistence, validation, gate.

The store reads through the state database, so most of this file takes the
`database` fixture (file-backed, at head) and pairs it with a `Settings` whose
``state_dir`` is the same ``tmp_path`` - that pairing is what "the store and the
app open one database" means in a test.

The durability proofs the epic is for live here too: a concurrent burst through
the API, a failed write, and the cross-process claim (an MCP subprocess and the
app writing to the same file). A leftover `projects.json` is not one of them:
nothing reads it any more, so the Projects page is the database's answer.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection

from scufris.app import create_app
from scufris.config import Settings
from scufris.db import Database, open_database
from scufris.projects import (
    DuplicateProject,
    InvalidProject,
    Project,
    ProjectNotFound,
    ProjectsReadOnly,
    ProjectStore,
    read_project_tasks,
)
from scufris_host import Collector

# Enough concurrent writers to lose a record under the JSON store the database
# replaces (20260729-102146 measured loss at this width), small enough that the
# burst stays under a second.
BURST = 24


def _settings(tmp_path: Path, **kwargs: Any) -> Settings:
    base: dict[str, Any] = {
        "state_dir": tmp_path,
        "web_dist": tmp_path / "absent",
        "_env_file": None,
    }
    base.update(kwargs)
    return Settings(**base)


def test_project_store_round_trip(tmp_path: Path, database: Database) -> None:
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    settings = _settings(tmp_path)
    store = ProjectStore(settings, database)
    created = store.create(
        name="My App", cwd=str(proj_dir), language="python", description="a thing"
    )
    assert created.id == "my-app"
    assert created.cwd == str(proj_dir)

    # A fresh store over the same database sees it.
    fresh = ProjectStore(settings, database)
    got = fresh.get("my-app")
    assert got.name == "My App"
    assert got.language == "python"

    # Update persists.
    fresh.update("my-app", description="updated")
    assert ProjectStore(settings, database).get("my-app").description == "updated"

    # Delete persists.
    fresh.delete("my-app")
    assert ProjectStore(settings, database).list() == []


def test_project_store_survives_reopening_the_database(
    tmp_path: Path, database: Database
) -> None:
    """The committed row is in the FILE, not in the handle that wrote it."""
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    settings = _settings(tmp_path)
    ProjectStore(settings, database).create(name="My App", cwd=str(proj_dir))

    reopened = open_database(tmp_path)
    try:
        assert ProjectStore(settings, reopened).get("my-app").name == "My App"
    finally:
        reopened.close()


def test_project_create_validates_cwd_and_name(
    tmp_path: Path, database: Database
) -> None:
    store = ProjectStore(_settings(tmp_path), database)
    with pytest.raises(InvalidProject):
        store.create(name="x", cwd=str(tmp_path / "does-not-exist"))
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    with pytest.raises(InvalidProject):
        store.create(name="   ", cwd=str(proj_dir))


def test_duplicate_name_gets_distinct_ids(tmp_path: Path, database: Database) -> None:
    d = tmp_path / "d"
    d.mkdir()
    store = ProjectStore(_settings(tmp_path), database)
    a = store.create(name="Same", cwd=str(d))
    b = store.create(name="Same", cwd=str(d))
    assert a.id == "same"
    assert b.id == "same-2"
    assert {p.id for p in store.list()} == {"same", "same-2"}


def test_list_orders_by_lowercased_name(tmp_path: Path, database: Database) -> None:
    d = tmp_path / "d"
    d.mkdir()
    store = ProjectStore(_settings(tmp_path), database)
    for name in ("zebra", "Apple", "mango"):
        store.create(name=name, cwd=str(d))
    assert [p.name for p in store.list()] == ["Apple", "mango", "zebra"]


def test_get_and_delete_unknown_raise(tmp_path: Path, database: Database) -> None:
    store = ProjectStore(_settings(tmp_path), database)
    with pytest.raises(ProjectNotFound):
        store.get("ghost")
    with pytest.raises(ProjectNotFound):
        store.delete("ghost")
    with pytest.raises(ProjectNotFound):
        store.update("ghost", description="x")
    # An unknown id is NOT FOUND whatever else is wrong with the request: the
    # route maps these two to 404 and 422, so the order they are checked in is
    # the status an operator sees.
    with pytest.raises(ProjectNotFound):
        store.update("ghost", name="   ")
    with pytest.raises(ProjectNotFound):
        store.update("ghost", cwd=str(tmp_path / "does-not-exist"))


def test_writes_refused_when_read_only(tmp_path: Path, database: Database) -> None:
    d = tmp_path / "d"
    d.mkdir()
    ro = _settings(tmp_path, settings_writable=False)
    store = ProjectStore(ro, database)
    with pytest.raises(ProjectsReadOnly):
        store.create(name="x", cwd=str(d))
    with pytest.raises(ProjectsReadOnly):
        store.update("anything", description="x")
    with pytest.raises(ProjectsReadOnly):
        store.delete("anything")


def test_duplicate_project_id_raises_the_domain_error(
    tmp_path: Path, database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A colliding id is a `DuplicateProject`, never a raw database error.

    The id is now a real ``PRIMARY KEY``, so a collision that dedup did not catch
    arrives as an ``IntegrityError`` from the driver. Dedup and the insert share
    one transaction, so nothing outside this process can produce that collision -
    which is why the dedup is neutralized here rather than raced against.
    """
    from sqlalchemy.exc import IntegrityError

    import scufris.projects as projects_module

    d = tmp_path / "d"
    d.mkdir()
    store = ProjectStore(_settings(tmp_path), database)
    store.create(name="Same", cwd=str(d))
    monkeypatch.setattr(projects_module, "_unique_id", lambda conn, base: base)
    with pytest.raises(DuplicateProject):
        store.create(name="Same", cwd=str(d))
    # And the raw error never escapes.
    try:
        store.create(name="Same", cwd=str(d))
    except IntegrityError:  # pragma: no cover - the assertion above already fails
        pytest.fail("the driver's IntegrityError escaped the store")
    except DuplicateProject:
        pass


def test_project_store_returns_detached_pydantic_records(
    tmp_path: Path, database: Database
) -> None:
    """Nothing bound to a Session escapes: the routes serialize these directly.

    Every field is read AFTER the transaction that produced the record closed, so
    an ORM instance that lazy-loads on attribute access would fail here rather
    than in a response.
    """
    d = tmp_path / "d"
    d.mkdir()
    store = ProjectStore(_settings(tmp_path), database)
    created = store.create(name="My App", cwd=str(d), language="python")
    assert type(created) is Project
    assert created.model_dump() == {
        "id": "my-app",
        "cwd": str(d),
        "name": "My App",
        "language": "python",
        "description": "",
    }
    updated = store.update("my-app", description="a thing")
    for record in (updated, store.get("my-app"), *store.list()):
        assert type(record) is Project
        assert record.model_dump() == {
            "id": "my-app",
            "cwd": str(d),
            "name": "My App",
            "language": "python",
            "description": "a thing",
        }


class _FailsBeforeCommit(Database):
    """A database whose units of work do the work and then fail, as a full disk would.

    Wraps a real one rather than faking it: the statements really run, and the
    failure lands where a persist failure lands - after the mutation, before the
    commit.
    """

    def __init__(self, db: Database) -> None:
        super().__init__(db.engine, db.path)

    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        with super().transaction() as conn:
            yield conn
            raise RuntimeError("simulated write failure")


def test_failed_project_write_leaves_nothing_live_in_memory(
    tmp_path: Path, database: Database
) -> None:
    """A refused write is invisible to this store, a fresh one, and the file.

    20260729-102146 measured 97 of 97 failed writes staying live in the process
    under the JSON store and being published by the next successful write. There
    is no in-memory mirror to hold them now.
    """
    d = tmp_path / "d"
    d.mkdir()
    settings = _settings(tmp_path)
    failing = ProjectStore(settings, _FailsBeforeCommit(database))
    with pytest.raises(RuntimeError):
        failing.create(name="Ghost", cwd=str(d))

    healthy = ProjectStore(settings, database)
    assert healthy.list() == []

    # And the next SUCCESSFUL write does not publish it.
    healthy.create(name="Real", cwd=str(d))
    assert [p.id for p in ProjectStore(settings, database).list()] == ["real"]


# --- the app on the database -------------------------------------------------


def _client(fake_collector: Collector, tmp_path: Path) -> TestClient:
    return TestClient(
        create_app(collector=fake_collector, settings=_settings(tmp_path))
    )


# The cross-store durability proof - a concurrent burst that also moves the agent
# and host-action stores, then a restart - is NOT here. It is
# `test_db_state_boundary.py::test_concurrent_state_mutations_survive_restart`:
# one claim, one test, and a projects-only version of it would report the epic
# green off a proof that never learned about the other stores (DECISION.md 1 of
# 20260801-100413).


# What the MCP subprocess does: open the store the way an MCP server does, write
# one project, and report every project it can see. Run as a real child process
# because the claim under test is about two PROCESSES sharing one database.
_MCP_CHILD = """
import json, sys
from scufris.config import Settings
from scufris.mcp_stores import project_store

store = project_store(Settings())
store.create(name="From Mcp", cwd=sys.argv[1])
print(json.dumps([p.id for p in store.list()]))
"""


def test_mcp_and_app_share_one_project_store(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """The cross-process claim the whole mechanism was chosen for.

    SPIKE.md scenario 5 measured the JSON alternative losing 150 of 300
    cross-process writes with ``raised=0``.
    """
    proj = tmp_path / "proj"
    proj.mkdir()
    with _client(fake_collector, tmp_path) as client:
        assert (
            client.post(
                "/api/projects", json={"name": "From App", "cwd": str(proj)}
            ).status_code
            == 200
        )

        env = dict(os.environ)
        env["SCUFRIS_STATE_DIR"] = str(tmp_path)
        proc = subprocess.run(
            [sys.executable, "-c", _MCP_CHILD, str(proj)],
            capture_output=True,
            text=True,
            env=env,
            cwd=Path(__file__).resolve().parent.parent,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        # The subprocess sees the app's write...
        assert set(json.loads(proc.stdout.strip().splitlines()[-1])) == {
            "from-app",
            "from-mcp",
        }
        # ...and the running app sees the subprocess's, with no restart.
        assert {p["id"] for p in client.get("/api/projects").json()} == {
            "from-app",
            "from-mcp",
        }


# --- tatr task discovery (unchanged by the cutover) --------------------------


def _tatr_new(cwd: Path, title: str, priority: int, tags: str) -> None:
    """Create a real tatr task under cwd/tasks (tatr needs the dir to exist)."""
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
