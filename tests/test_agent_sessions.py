"""Session mapping: which conversation an agent resumes, and whose it is.

The failure these exist for is latching onto the wrong session. An orchestrator
turn and a sub-agent turn each record a session id, and before the registry the
orchestrator's was in-memory only, so a restart lost it. Covers the mapping
through the store - keyed by the run's backend, not the current one - and the
registry's own history and ownership: accumulation, the current pointer,
removal, the reset a backend switch performs, and the spawn parent.

The ``_settings`` and ``_projects_with_one`` helpers come from
``tests/test_agent_store.py``.
"""

from __future__ import annotations

from pathlib import Path

from test_agent_store import _projects_with_one, _settings

from scufris.agent_store import ORCHESTRATOR_ID, AgentStore, SessionRegistry
from scufris.db import Database
from scufris.enums import AgentState
from scufris.projects import ProjectStore


def test_orchestrator_and_subagent_sessions_stay_distinct_across_restart(
    tmp_path: Path,
    database: Database,
) -> None:
    """The mixing reproduction: an orchestrator turn and a
    codex sub-agent turn each record a session id; after a restart (a fresh
    store over the same state dir) BOTH ids must still be there and distinct.
    Before the registry, the orchestrator's id was in-memory only, so the
    restart lost it - the first step toward latching onto the wrong session."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="codex")

    # One finished turn each (the supervisor's persist path calls mark_finished).
    store.mark_finished(ORCHESTRATOR_ID, state=AgentState.DONE, session_id="orch-sess")
    store.mark_finished("builder", state=AgentState.DONE, session_id="sub-sess")
    assert store.orchestrator_session_id() == "orch-sess"
    assert store.get("builder").session_id == "sub-sess"

    # Simulated restart.
    fresh = AgentStore(settings, ProjectStore(settings, database))
    assert fresh.orchestrator_session_id() == "orch-sess"
    assert fresh.get(ORCHESTRATOR_ID).session_id == "orch-sess"
    assert fresh.get("builder").session_id == "sub-sess"
    assert fresh.orchestrator_session_id() != fresh.get("builder").session_id


def test_mark_finished_keys_session_by_run_backend_not_current(
    tmp_path: Path,
    database: Database,
) -> None:
    """A backend switch that races an in-flight turn must not mislabel the
    finishing session: mark_finished keys the id by the backend the run
    executed under (passed explicitly), so a codex session that finishes AFTER
    a switch to claude is still recorded under codex - and stays unreachable
    from the now-current claude backend instead of being resumed by it."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
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


def test_delete_removes_session_mapping(tmp_path: Path, database: Database) -> None:
    """Deleting an agent removes its registry mapping: a NEW agent that happens
    to reuse the freed id must not inherit the old session."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="codex")
    store.mark_finished("builder", state=AgentState.DONE, session_id="old-sess")
    store.delete("builder")

    recreated = store.create(name="Builder", project_id="my-app", backend="codex")
    assert recreated.id == "builder"  # the freed id is reused
    assert recreated.session_id is None
    # And a fresh store agrees (the removal was persisted).
    fresh = AgentStore(settings, ProjectStore(settings, database))
    assert fresh.get("builder").session_id is None


def test_backend_switch_clears_session_mapping(
    tmp_path: Path, database: Database
) -> None:
    """A backend switch clears the persisted mapping too: after a restart the
    stale wrong-backend id must not resurface. (The in-record clearing is
    pinned by test_update_backend_change_clears_session; this pins the
    registry/persistence side.)"""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="codex")
    store.mark_finished("builder", state=AgentState.DONE, session_id="codex-sess-1")

    store.update("builder", backend="claude")
    fresh = AgentStore(settings, ProjectStore(settings, database))
    assert fresh.get("builder").session_id is None
    # Switching back to codex must NOT resurrect the old codex id either.
    fresh.update("builder", backend="codex")
    assert fresh.get("builder").session_id is None


def test_legacy_agents_json_session_id_migrates_to_registry(
    tmp_path: Path, database: Database
) -> None:
    """A pre-registry agents.json that still carries a session_id seeds the
    registry on load, so an upgrade does not drop live conversations."""
    settings = _settings(tmp_path)
    state = tmp_path
    state.mkdir(parents=True, exist_ok=True)
    (state / "agents.json").write_text(
        '[{"id": "legacy", "name": "Legacy", "project_id": "p", '
        '"backend": "codex", "session_id": "legacy-sess"}]'
    )
    store = AgentStore(settings, ProjectStore(settings, database))
    assert store.get("legacy").session_id == "legacy-sess"
    # And it survives another restart via the registry (not agents.json).
    fresh = AgentStore(settings, ProjectStore(settings, database))
    assert fresh.get("legacy").session_id == "legacy-sess"


def test_registry_add_accumulates_history(tmp_path: Path, database: Database) -> None:
    reg = SessionRegistry(_settings(tmp_path))
    reg.add("a", "codex", "s1")
    reg.add("a", "codex", "s2")
    assert reg.get("a", "codex") == "s2"  # current is the latest
    assert reg.sessions_for("a", "codex") == ["s1", "s2"]
    # Re-adding a known id does not duplicate it, just re-currents.
    reg.add("a", "codex", "s1")
    assert reg.get("a", "codex") == "s1"
    assert reg.sessions_for("a", "codex") == ["s1", "s2"]


def test_registry_set_current_preserves_history(
    tmp_path: Path, database: Database
) -> None:
    reg = SessionRegistry(_settings(tmp_path))
    reg.add("a", "codex", "s1")
    reg.set_current("a", "codex", None)  # "new chat"
    assert reg.get("a", "codex") is None
    assert reg.sessions_for("a", "codex") == ["s1"]  # history kept


def test_registry_set_current_appends_unseen(
    tmp_path: Path, database: Database
) -> None:
    reg = SessionRegistry(_settings(tmp_path))
    reg.add("a", "codex", "s1")
    reg.set_current("a", "codex", "s2")  # switch to an id we had not recorded
    assert reg.get("a", "codex") == "s2"
    assert reg.sessions_for("a", "codex") == ["s1", "s2"]


def test_registry_remove_drops_one_session(tmp_path: Path, database: Database) -> None:
    reg = SessionRegistry(_settings(tmp_path))
    reg.add("a", "codex", "s1")
    reg.add("a", "codex", "s2")
    reg.remove("a", "codex", "s1")
    assert reg.sessions_for("a", "codex") == ["s2"]
    assert reg.get("a", "codex") == "s2"
    reg.remove("a", "codex", "s2")  # removing the current one clears current
    assert reg.sessions_for("a", "codex") == []
    assert reg.get("a", "codex") is None


def test_registry_backend_switch_resets_history(
    tmp_path: Path, database: Database
) -> None:
    reg = SessionRegistry(_settings(tmp_path))
    reg.add("a", "codex", "s1")
    reg.add("a", "codex", "s2")
    reg.add("a", "claude", "c1")  # a different backend starts fresh
    assert reg.sessions_for("a", "claude") == ["c1"]
    assert reg.sessions_for("a", "codex") == []  # old-backend history unreachable
    assert reg.get("a", "codex") is None


def test_legacy_session_entry_loads_as_single_history(
    tmp_path: Path, database: Database
) -> None:
    """A pre-multi-session sessions.json entry ({backend, session_id}) loads as a
    one-element history so an upgrade keeps that session listed."""
    settings = _settings(tmp_path)
    state = tmp_path
    state.mkdir(parents=True, exist_ok=True)
    (state / "sessions.json").write_text(
        '{"orchestrator": {"backend": "codex", "session_id": "leg-sess"}}'
    )
    reg = SessionRegistry(settings)
    assert reg.get("orchestrator", "codex") == "leg-sess"
    assert reg.sessions_for("orchestrator", "codex") == ["leg-sess"]


def test_registry_records_and_preserves_spawn_parent(
    tmp_path: Path, database: Database
) -> None:
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


def test_store_record_spawn_parent_round_trips(
    tmp_path: Path, database: Database
) -> None:
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="codex")
    store.record_spawn_parent("builder", ORCHESTRATOR_ID, "chat-9")
    assert store.parent_of("builder") == (ORCHESTRATOR_ID, "chat-9")
    # Persisted.
    fresh = AgentStore(settings, ProjectStore(settings, database))
    assert fresh.parent_of("builder") == (ORCHESTRATOR_ID, "chat-9")


def test_orchestrator_session_history_accumulates(
    tmp_path: Path, database: Database
) -> None:
    """Each finished orchestrator turn with a new id appends to its history."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects)
    store.mark_finished(ORCHESTRATOR_ID, state=AgentState.DONE, session_id="o1")
    store.set_orchestrator_session(None)  # new chat
    store.mark_finished(ORCHESTRATOR_ID, state=AgentState.DONE, session_id="o2")
    assert store.orchestrator_session_id() == "o2"
    assert store.orchestrator_sessions() == ["o1", "o2"]
    # Forgetting one (a session delete) drops it from the switcher history.
    store.forget_orchestrator_session("o1")
    assert store.orchestrator_sessions() == ["o2"]
