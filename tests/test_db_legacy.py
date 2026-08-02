"""The legacy JSON import: once, backed up, all-or-nothing, never tolerant.

Each proof here discriminates against the behaviour the store cutover is
REMOVING rather than restating the happy path. `ProjectStore._load` logs and
skips an invalid record and returns silently on a damaged file; an import that
did either would leave an operator with a database quietly missing the records
they still see in their JSON, which is the failure this epic exists to end.
"""

from __future__ import annotations

import json
import re
import shutil
import stat
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path

import pytest
from sqlalchemy import func, select

from scufris.db import Database, open_database, upgrade_to_head
from scufris.db.legacy import (
    BACKUP_SUFFIX,
    LegacyImportRefused,
    backup_path,
    import_agent_state,
    import_projects,
)
from scufris.db.models import LegacyImportRow, ProjectRow
from scufris.enums import AgentState

RECORDS = [
    {"id": "scufris", "cwd": "/tmp", "name": "Scufris", "language": "python"},
    {"id": "den", "cwd": "/tmp", "name": "Den", "description": "the journal"},
]


def _write(state_dir: Path, payload: object) -> Path:
    source = state_dir / "projects.json"
    source.write_text(json.dumps(payload, indent=2))
    return source


def _write_text(state_dir: Path, text: str) -> Path:
    source = state_dir / "projects.json"
    source.write_text(text)
    return source


def _ids(db: Database) -> list[str]:
    with db.transaction() as conn:
        return list(
            conn.execute(select(ProjectRow.id).order_by(ProjectRow.id)).scalars()
        )


def _completed(db: Database) -> list[str]:
    with db.transaction() as conn:
        return list(conn.execute(select(LegacyImportRow.source)).scalars())


@pytest.fixture
def make_state(tmp_path: Path) -> Iterator[Callable[[str], tuple[Path, Database]]]:
    """A second (third, ...) migrated state directory, each with its own database.

    The gate is keyed by the source file's NAME, so a test that needs a source
    file imported for the first time twice needs two databases, not two
    directories.
    """
    with ExitStack() as stack:

        def make(name: str) -> tuple[Path, Database]:
            state_dir = tmp_path / name
            state_dir.mkdir()
            db = open_database(state_dir)
            stack.callback(db.close)
            upgrade_to_head(db)
            return state_dir, db

        yield make


def test_legacy_projects_import_is_idempotent_and_refuses_damage(
    database: Database,
    tmp_path: Path,
    make_state: Callable[[str], tuple[Path, Database]],
) -> None:
    """One import, a second run that does nothing, and damage that is refused.

    The idempotency half fails without the `legacy_import` gate (four rows, or
    an IntegrityError, on the second run). The damage half fails against a
    tolerant loader, which would report success having imported nothing.
    """
    _write(tmp_path, RECORDS)

    assert import_projects(database, tmp_path) is True
    assert _ids(database) == ["den", "scufris"]
    assert _completed(database) == ["projects.json"]

    assert import_projects(database, tmp_path) is False
    assert _ids(database) == ["den", "scufris"]

    damaged_dir, damaged_db = make_state("damaged")
    source = _write_text(damaged_dir, json.dumps(RECORDS) + "\n{trailing}")

    with pytest.raises(LegacyImportRefused) as excinfo:
        import_projects(damaged_db, damaged_dir)

    message = str(excinfo.value)
    assert str(source) in message
    assert "line 2 col 1" in message
    assert "Extra data" in message
    # The `.bak` beside a REFUSED file is a copy of the damage this run just
    # took, so the remedy must not send the operator back to it: restoring it is
    # a no-op at best, and overwrites their own repair at worst.
    assert "not a repair" in message
    assert "move it aside" in message
    # Refused, not half-done: nothing imported and the gate is NOT closed, so an
    # operator who repairs the file gets a retry rather than a skipped store.
    assert _ids(damaged_db) == []
    assert _completed(damaged_db) == []
    # The backup is taken BEFORE the file is read, so it survives the refusal.
    assert backup_path(source).read_text() == source.read_text()


def test_legacy_import_backs_up_and_never_deletes_the_source(
    database: Database, tmp_path: Path
) -> None:
    """The source is copied to `<name>.pre-sqlite.bak` and left exactly as it was."""
    source = _write(tmp_path, RECORDS)
    original = source.read_text()

    assert import_projects(database, tmp_path) is True

    assert source.is_file(), "the import deleted the legacy file"
    assert source.read_text() == original, "the import rewrote the legacy file"

    backup = tmp_path / f"projects.json{BACKUP_SUFFIX}"
    assert backup.read_text() == original
    # 0600 from creation, like the database and its own backups: the later store
    # migrations copy auth sessions through this same function.
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600


def test_legacy_import_rolls_back_on_an_invalid_record(
    database: Database, tmp_path: Path
) -> None:
    """One invalid record fails the whole import and leaves no rows behind.

    Two discriminations in one: the record BEFORE the invalid one is inserted
    and then rolled back, so dropping the transaction fails this; and skipping
    the invalid record - what `ProjectStore._load` does today - fails it too.
    """
    invalid = {"id": "broken", "name": "No cwd"}
    source = _write(tmp_path, [RECORDS[0], invalid])

    with pytest.raises(LegacyImportRefused) as excinfo:
        import_projects(database, tmp_path)

    message = str(excinfo.value)
    assert str(source) in message
    assert "cwd" in message

    assert _ids(database) == []
    assert _completed(database) == []

    # The operator repairs the file; the retry imports the whole store.
    _write(tmp_path, RECORDS)
    assert import_projects(database, tmp_path) is True
    assert _ids(database) == ["den", "scufris"]


def test_a_symlinked_backup_target_is_refused(
    database: Database, tmp_path: Path
) -> None:
    """The backup is never written THROUGH a symlink, as elsewhere in this package.

    Refused as `LegacyImportRefused`, like every other legacy file that cannot be
    trusted, rather than as a bare `RuntimeError`: the cutover's call site
    catches the documented exception, and this condition means the same to it.
    """
    source = _write(tmp_path, RECORDS)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_text("untouched")
    backup_path(source).symlink_to(elsewhere)

    with pytest.raises(LegacyImportRefused, match="symlink"):
        import_projects(database, tmp_path)

    assert elsewhere.read_text() == "untouched"
    assert _ids(database) == []


def test_an_absent_legacy_file_is_not_an_import(
    database: Database, tmp_path: Path
) -> None:
    """A fresh install has no legacy file, and that is not a failure or an import."""
    assert import_projects(database, tmp_path) is False
    assert _completed(database) == []


# --- the agent-state sources -------------------------------------------------
#
# `tests/fixtures/legacy_state` holds one POPULATED example of each of the five
# shapes an operator upgrading from the JSON stores actually has on disk,
# including the two legacy spellings the agent loader has to migrate
# (`write_enabled`, an `app_server` backend) and the pre-multi-session
# `{backend, session_id}` entry.

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "legacy_state"

# What `import_agent_state` writes a `legacy_import` row for, given the fixtures.
AGENT_STATE_SOURCES = {
    "agents.json",
    "sessions.json",
    "outcomes.json",
    "settings.json",
    "reasoning/orch-1.json",
    "reasoning/builder-1.json",
}


def _seed(state_dir: Path) -> Path:
    """Copy the legacy fixtures into a state directory, as an upgrade would find them."""
    shutil.copytree(FIXTURES, state_dir, dirs_exist_ok=True)
    return state_dir


def test_legacy_agent_state_migrates_idempotently(
    database: Database, tmp_path: Path
) -> None:
    """Every supported field survives, and a second start imports nothing again.

    The idempotence half is the one that matters operationally: `open_state_database`
    runs the import on EVERY start, so a gate that did not hold would duplicate an
    operator's whole agent list on their second boot.
    """
    from scufris.agent_store.outcomes import RunOutcome
    from scufris.db.models import (
        AgentOutcomeRow,
        AgentRow,
        AgentSessionHistoryRow,
        AgentSessionRow,
        ReasoningTurnRow,
        SettingsOverrideRow,
    )

    _seed(tmp_path)
    import_agent_state(database, tmp_path)

    with database.transaction() as conn:
        agents = {row.id: row for row in conn.execute(select(AgentRow.__table__)).all()}
        sessions = {
            row.agent_id: row
            for row in conn.execute(select(AgentSessionRow.__table__)).all()
        }
        history = [
            (row.agent_id, row.seq, row.session_id)
            for row in conn.execute(
                select(AgentSessionHistoryRow.__table__).order_by(
                    AgentSessionHistoryRow.agent_id, AgentSessionHistoryRow.seq
                )
            ).all()
        ]
        outcomes = {
            row.agent_id: row
            for row in conn.execute(select(AgentOutcomeRow.__table__)).all()
        }
        overrides = {
            row.key: json.loads(row.value)
            for row in conn.execute(select(SettingsOverrideRow.__table__)).all()
        }
        reasoning = [
            (row.session_id, row.seq, row.answer, row.reasoning)
            for row in conn.execute(
                select(ReasoningTurnRow.__table__).order_by(
                    ReasoningTurnRow.session_id, ReasoningTurnRow.seq
                )
            ).all()
        ]

    # Agents: every field, and the two migrations the deleted tolerant loader did.
    assert set(agents) == {"builder", "legacy-modes", "reader"}
    builder = agents["builder"]
    assert (builder.name, builder.project_id, builder.backend) == (
        "Builder",
        "scufris",
        "codex",
    )
    assert (builder.model, builder.description, builder.goal, builder.task_id) == (
        "gpt-5.5",
        "builds things",
        "ship the epic",
        "20260801-100409",
    )
    assert (builder.state, builder.permission_mode) == ("done", "edit")
    # `write_enabled` -> a permission mode, both ways round...
    assert agents["legacy-modes"].permission_mode == "edit"
    assert agents["reader"].permission_mode == "manual"
    # ...and a legacy codex mode id -> the canonical backend name.
    assert agents["legacy-modes"].backend == "codex"

    # Sessions: the current pointer, the ordered history, and the spawn parent.
    assert sessions["orchestrator"].current_session_id == "orch-2"
    assert [row for row in history if row[0] == "orchestrator"] == [
        ("orchestrator", 0, "orch-1"),
        ("orchestrator", 1, "orch-2"),
        ("orchestrator", 2, "orch-3"),
    ]
    assert (
        sessions["builder"].parent_agent_id,
        sessions["builder"].parent_session_id,
    ) == ("orchestrator", "orch-1")
    # The pre-multi-session `{backend, session_id}` entry keeps that session
    # LISTED, not just current: it is a conversation the operator can still open.
    assert ("reader", 0, "reader-legacy-shape") in history
    # A pre-registry `session_id` on the agent record moves into the tables.
    assert sessions["legacy-modes"].current_session_id == "pre-registry-sess"

    # Outcomes: the whole record, including the acknowledged flag and the float ts.
    assert RunOutcome.model_validate(
        {k: v for k, v in outcomes["builder"]._mapping.items() if k != "agent_id"}
    ) == RunOutcome(
        state=AgentState.REPORTED,
        message="the epic is on lane B",
        run_id="builder:run-9",
        session_id="builder-1",
        ts=1785000000.5,
        acknowledged=False,
    )
    assert outcomes["reader"].acknowledged is True

    # Settings: every writable override, in its JSON form.
    assert overrides == {
        "agent_model": "gpt-5.6",
        "poll_seconds": 9.0,
        "disabled_tools": ["host_stats", "list_processes"],
        "host_digest_enabled": False,
    }

    # Reasoning: in file order, INCLUDING the turn where the model did not think
    # (dropping it would shift every later spoiler onto the wrong message).
    assert reasoning == [
        ("builder-1", 0, "built", "compiling"),
        ("orch-1", 0, "the first answer", "i thought about it"),
        ("orch-1", 1, "the second answer", ""),
    ]

    assert set(_completed(database)) == AGENT_STATE_SOURCES

    # A second start re-reads nothing: one gate row per source, one copy of each
    # record.
    import_agent_state(database, tmp_path)
    with database.transaction() as conn:
        assert conn.execute(select(func.count()).select_from(AgentRow)).scalar() == 3
        assert (
            conn.execute(
                select(func.count()).select_from(AgentSessionHistoryRow)
            ).scalar()
            == 6
        )
        assert (
            conn.execute(select(func.count()).select_from(ReasoningTurnRow)).scalar()
            == 3
        )
    assert set(_completed(database)) == AGENT_STATE_SOURCES


def test_the_reasoning_gate_key_cannot_collide_with_a_store(
    database: Database, tmp_path: Path
) -> None:
    """A session literally called `sessions` must not shadow the session registry.

    The sidecar files are named after a SESSION ID, so keying their gate on the
    file NAME would give this one `sessions.json` - the registry's own key. One
    of the two would then be skipped as already-imported, silently. The key is
    explicit (`reasoning/<name>`) for exactly this.
    """
    _seed(tmp_path)
    (tmp_path / "reasoning" / "sessions.json").write_text(
        json.dumps({"turns": [{"answer": "a", "reasoning": "r"}]})
    )

    import_agent_state(database, tmp_path)

    assert "sessions.json" in _completed(database)  # the registry
    assert "reasoning/sessions.json" in _completed(database)  # and the sidecar
    from scufris.db.models import AgentSessionRow, ReasoningTurnRow

    with database.transaction() as conn:
        assert conn.execute(select(func.count()).select_from(AgentSessionRow)).scalar()
        assert (
            conn.execute(
                select(ReasoningTurnRow.reasoning).where(
                    ReasoningTurnRow.session_id == "sessions"
                )
            ).scalar()
            == "r"
        )


@pytest.mark.parametrize(
    ("source", "damaged", "expected"),
    [
        ("agents.json", '[{"id": "x"}]', "not a valid agent"),
        ("agents.json", '{"builder": {}}', "not a list of agents"),
        ("sessions.json", '{"builder": {"sessions": ["s"]}}', "has no backend"),
        ("sessions.json", "[]", "not a mapping"),
        ("outcomes.json", '{"builder": {"state": "nonsense"}}', "not a valid outcome"),
        ("settings.json", '{"overrides": {"openai_api_key": "sk"}}', "not a writable"),
        (
            "settings.json",
            '{"overrides": {"poll_seconds": "soon"}}',
            "invalid value for 'poll_seconds'",
        ),
        ("settings.json", '{"profiles": 3}', "neither an `overrides` mapping"),
        ("reasoning/orch-1.json", '{"turns": [{"answer": 5}]}', "no answer/reasoning"),
        ("reasoning/orch-1.json", '{"nope": []}', "has no `turns` list"),
    ],
)
def test_a_damaged_agent_state_source_is_refused_whole(
    database: Database, tmp_path: Path, source: str, damaged: str, expected: str
) -> None:
    """One bad record fails its WHOLE source, and leaves no gate row behind.

    The JSON stores logged and skipped such a record, which is how an operator
    ended up with a database quietly missing state they could still see in their
    files. The repaired file imports on the next run because the gate stayed
    open.
    """
    _seed(tmp_path)
    (tmp_path / source).write_text(damaged)

    with pytest.raises(LegacyImportRefused, match=re.escape(expected)):
        import_agent_state(database, tmp_path)

    assert source not in _completed(database)


def test_a_damaged_source_does_not_hold_back_the_sources_after_it(
    database: Database, tmp_path: Path
) -> None:
    """A refusal isolates ONE source; the other four still import and gate.

    `sessions.json` is the FIRST of the five, so damaging it is what tells the
    per-source claim apart from "everything before the damage got in": the
    refusal still leaves the import failing (the operator has to repair the
    file), but the repair re-reads only that file rather than replaying the whole
    upgrade (review round 1, R1.5).
    """
    _seed(tmp_path)
    (tmp_path / "sessions.json").write_text("[]")

    with pytest.raises(LegacyImportRefused, match=re.escape("sessions.json")):
        import_agent_state(database, tmp_path)

    gated = set(_completed(database))
    assert "sessions.json" not in gated
    assert AGENT_STATE_SOURCES - {"sessions.json"} <= gated
