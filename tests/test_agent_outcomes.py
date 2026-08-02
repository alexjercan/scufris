"""The durable run outcome: what a finished or stuck run leaves behind.

A run's final message and terminal state must be readable from a fresh store
over the same state dir, because the outcome outlives the ephemeral per-run
EventBus. Covers the record itself and its corrupt-file tolerance, the two
signals an agent raises mid-run - ``request_input`` (needs a decision) and
``report_back`` (finished my task) - the run-id keying that stops a stale
signal outliving its run, and the pending list and acknowledge that clear them.

The ``_settings`` and ``_projects_with_one`` helpers come from
``tests/test_agent_store.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_agent_store import _projects_with_one, _settings

from scufris.agent_store import ORCHESTRATOR_ID, AgentNotFound, AgentStore
from scufris.db import Database
from scufris.enums import AgentState
from scufris.projects import ProjectStore


def test_waiting_state_is_distinct() -> None:
    """AgentState.WAITING ('ended a turn awaiting a decision') is a real member,
    distinct from BLOCKED (waiting on an approval) and DONE."""
    assert AgentState.WAITING == "waiting"
    assert AgentState.WAITING != AgentState.BLOCKED
    assert AgentState.WAITING != AgentState.DONE


def test_run_outcome_persists_and_survives_restart(
    tmp_path: Path, database: Database
) -> None:
    """A finished run leaves a durable outcome (final message + terminal state)
    readable from a fresh store over the same state_dir - the substrate that
    outlives the ephemeral per-run EventBus."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
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
    fresh = AgentStore(settings, ProjectStore(settings, database))
    reloaded = fresh.outcome("builder")
    assert reloaded is not None
    assert reloaded.state == AgentState.WAITING
    assert reloaded.message == "should I merge to master?"
    assert reloaded.session_id == "sess-1"


def test_delete_removes_outcome(tmp_path: Path, database: Database) -> None:
    """Deleting an agent drops its outcome, and it does not resurrect on
    restart - a reused id can never inherit a stale outcome."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="mock")
    store.mark_finished("builder", state=AgentState.DONE, message="done")
    assert store.outcome("builder") is not None

    store.delete("builder")
    assert store.outcome("builder") is None
    fresh = AgentStore(settings, ProjectStore(settings, database))
    assert fresh.outcome("builder") is None


def test_delete_then_mark_finished_does_not_resurrect_outcome(
    tmp_path: Path, database: Database
) -> None:
    """A run that finishes AFTER its agent was deleted mid-run (the persist
    callback firing post-delete - an anticipated path, per app.py) must not
    resurrect a stale outcome: mark_finished raises AgentNotFound and writes
    nothing. Regression for review R1.1."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="mock")
    store.mark_finished("builder", state=AgentState.WAITING, message="merge?")
    assert store.outcome("builder") is not None

    store.delete("builder")
    # The racing completion callback fires after the delete.
    with pytest.raises(AgentNotFound):
        store.mark_finished("builder", state=AgentState.DONE, message="late")

    assert store.outcome("builder") is None
    fresh = AgentStore(settings, ProjectStore(settings, database))
    assert fresh.outcome("builder") is None


def test_error_terminal_outcome_recorded(tmp_path: Path, database: Database) -> None:
    """An error turn (no final reply, so no message) records an ERROR outcome
    with an empty message, not a crash (review R1.3)."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="mock")
    store.mark_finished("builder", state=AgentState.ERROR)
    outcome = store.outcome("builder")
    assert outcome is not None
    assert outcome.state == AgentState.ERROR
    assert outcome.message == ""


def test_outcome_store_tolerates_a_corrupt_file(
    tmp_path: Path, database: Database
) -> None:
    """A garbled outcomes.json loads as empty, like the other stores."""
    settings = _settings(tmp_path)
    state = settings.state_dir
    state.mkdir(parents=True, exist_ok=True)
    (state / "outcomes.json").write_text("{ not json")
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects)
    assert store.outcomes() == {}


# --- request_input: the needs-input signal -----------------------------------


def test_request_input_sets_waiting_outcome(tmp_path: Path, database: Database) -> None:
    """A sub-agent's request_input records a WAITING outcome carrying the
    question, keyed to the current run, unacknowledged."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
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


def test_waiting_survives_same_run_completion(
    tmp_path: Path, database: Database
) -> None:
    """request_input fires mid-turn; the turn then ends DONE. The natural
    completion must NOT clobber the WAITING outcome for the SAME run - it keeps
    WAITING + the question, and refreshes the now-finalized session id."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
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


def test_stale_waiting_overwritten_by_a_new_run(
    tmp_path: Path, database: Database
) -> None:
    """A WAITING outcome from a PRIOR run does not stick forever: a new run that
    finishes DONE (different run_id) overwrites it - the outcome is run-id-keyed."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
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


def test_error_after_request_input_wins(tmp_path: Path, database: Database) -> None:
    """If the run ERRORs after a request_input, the error terminal state wins
    over the WAITING signal (the agent did not cleanly wait, it crashed)."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="mock")

    store.request_input("builder", "merge?", run_id="builder:r1")
    store.mark_finished("builder", state=AgentState.ERROR, run_id="builder:r1")
    outcome = store.outcome("builder")
    assert outcome is not None
    assert outcome.state == AgentState.ERROR


def test_request_input_on_deleted_agent_raises(
    tmp_path: Path, database: Database
) -> None:
    """request_input on a missing agent raises AgentNotFound and writes nothing,
    like mark_finished."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects)
    with pytest.raises(AgentNotFound):
        store.request_input("ghost", "merge?", run_id="ghost:r1")
    assert store.outcome("ghost") is None


# --- report_back: finished-my-task signal (sibling of request_input) ----------


def test_report_back_sets_reported_outcome(tmp_path: Path, database: Database) -> None:
    """A sub-agent's report_back records a REPORTED outcome carrying the summary,
    keyed to the current run, unacknowledged."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
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


def test_reported_survives_same_run_completion(
    tmp_path: Path, database: Database
) -> None:
    """report_back fires mid-turn; the turn then ends DONE. The natural completion
    must NOT clobber the REPORTED outcome for the SAME run - it keeps REPORTED + the
    summary, and refreshes the now-finalized session id (mirrors WAITING)."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
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


def test_stale_reported_overwritten_by_a_new_run(
    tmp_path: Path, database: Database
) -> None:
    """A REPORTED outcome from a PRIOR run does not stick forever: a new run that
    finishes DONE (different run_id) overwrites it (run-id-keyed)."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
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


def test_error_after_report_back_wins(tmp_path: Path, database: Database) -> None:
    """If the run ERRORs after a report_back, the error terminal state wins over the
    REPORTED signal (the agent did not cleanly finish, it crashed)."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects)
    store.create(name="Builder", project_id="my-app", backend="mock")

    store.report_back("builder", "done: X shipped", run_id="builder:r1")
    store.mark_finished("builder", state=AgentState.ERROR, run_id="builder:r1")
    outcome = store.outcome("builder")
    assert outcome is not None
    assert outcome.state == AgentState.ERROR


def test_report_back_on_deleted_agent_raises(
    tmp_path: Path, database: Database
) -> None:
    """report_back on a missing agent raises AgentNotFound and writes nothing,
    like request_input."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
    store = AgentStore(settings, projects)
    with pytest.raises(AgentNotFound):
        store.report_back("ghost", "done", run_id="ghost:r1")
    assert store.outcome("ghost") is None


# --- pending outcomes + acknowledge -------------------------------------------


def _agent(store: AgentStore, name: str) -> str:
    return store.create(name=name, project_id="my-app", backend="mock").id


def test_pending_outcomes_lists_waiting_reported_and_error_only(
    tmp_path: Path,
    database: Database,
) -> None:
    """pending_outcomes surfaces the agents that need the orchestrator: an
    unacknowledged WAITING (needs input), REPORTED (finished + reported) or ERROR
    outcome. A cleanly DONE agent that did NOT report is not pending."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
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


def test_pending_outcomes_excludes_the_orchestrator(
    tmp_path: Path, database: Database
) -> None:
    """The orchestrator is never a member of its OWN 'who needs me' poll (mirrors
    list() hiding it). Its turns now persist an ERROR outcome on a StreamError, so
    without the guard it would self-appear in pending_agents - exclude it."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
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


def test_acknowledge_clears_from_pending(tmp_path: Path, database: Database) -> None:
    """acknowledge marks a pending outcome handled so it drops out of the poll,
    persists, and is idempotent."""
    settings = _settings(tmp_path)
    projects = _projects_with_one(tmp_path, settings, database)
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
    fresh = AgentStore(settings, ProjectStore(settings, database))
    assert "waiter" not in fresh.pending_outcomes()
