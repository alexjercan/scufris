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
    import_legacy_state,
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

    import_legacy_state(database, tmp_path)
    assert _ids(database) == ["den", "scufris"]
    assert _completed(database) == ["projects.json"]

    import_legacy_state(database, tmp_path)
    assert _ids(database) == ["den", "scufris"]

    damaged_dir, damaged_db = make_state("damaged")
    source = _write_text(damaged_dir, json.dumps(RECORDS) + "\n{trailing}")

    with pytest.raises(LegacyImportRefused) as excinfo:
        import_legacy_state(damaged_db, damaged_dir)

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

    import_legacy_state(database, tmp_path)

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
        import_legacy_state(database, tmp_path)

    message = str(excinfo.value)
    assert str(source) in message
    assert "cwd" in message

    assert _ids(database) == []
    assert _completed(database) == []

    # The operator repairs the file; the retry imports the whole store.
    _write(tmp_path, RECORDS)
    import_legacy_state(database, tmp_path)
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
        import_legacy_state(database, tmp_path)

    assert elsewhere.read_text() == "untouched"
    assert _ids(database) == []


def test_an_absent_legacy_file_is_not_an_import(
    database: Database, tmp_path: Path
) -> None:
    """A fresh install has no legacy file, and that is not a failure or an import."""
    import_legacy_state(database, tmp_path)
    assert _completed(database) == []


# --- the whole state directory -------------------------------------------------
#
# `tests/fixtures/legacy_state` is a COMPLETE pre-database state directory: one
# populated example of every shape an operator upgrading from the JSON stores
# actually has on disk, including the two legacy spellings the agent loader has to
# migrate (`write_enabled`, an `app_server` backend) and the pre-multi-session
# `{backend, session_id}` entry.
#
# There is no `host_actions.json`, and that is the fixture telling the truth: the
# host action store was memory-only, so an operator has no such file.

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "legacy_state"

# What `import_legacy_state` writes a `legacy_import` row for, given the fixtures.
ALL_SOURCES = {
    "projects.json",
    "agents.json",
    "sessions.json",
    "outcomes.json",
    "settings.json",
    "auth_sessions.json",
    "schedules.json",
    "digests.json",
    "reasoning/orch-1.json",
    "reasoning/builder-1.json",
}


def _seed(state_dir: Path) -> Path:
    """Copy the legacy fixtures into a state directory, as an upgrade would find them."""
    shutil.copytree(FIXTURES, state_dir, dirs_exist_ok=True)
    return state_dir


def test_legacy_json_state_migrates_idempotently(
    database: Database, tmp_path: Path
) -> None:
    """A whole state directory imports once, in full, and a second start is a no-op.

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
        AuthSessionRow,
        DigestRow,
        ReasoningTurnRow,
        ScheduleRow,
        SettingsOverrideRow,
    )

    _seed(tmp_path)
    import_legacy_state(database, tmp_path)

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

    # Projects, the first store to land, still ride the same one call.
    assert _ids(database) == ["scufris"]

    with database.transaction() as conn:
        auth = conn.execute(select(AuthSessionRow.__table__)).all()
        schedules = {
            row.name: row for row in conn.execute(select(ScheduleRow.__table__)).all()
        }
        digests = conn.execute(select(DigestRow.__table__).order_by(DigestRow.id)).all()

    # Auth: the live session, id and CSRF token intact - this row IS the operator's
    # login surviving the upgrade.
    assert [(row.id, row.csrf) for row in auth] == [
        ("live-session-id", "live-csrf-token")
    ]
    assert (auth[0].created_at, auth[0].last_seen) == (1785000000.0, 1785000600.0)

    # Schedules: the counters and the sentence the dashboard shows, both of them
    # the history a reset would silently discard.
    assert (schedules["watch"].runs, schedules["watch"].missed) == (41, 2)
    assert schedules["watch"].last_result == "ran (attention), delivered"
    assert schedules["watch"].next_due == 1785001200.0
    assert (schedules["daily"].runs, schedules["daily"].last_run) == (6, 1784973600.0)

    # Digests: in file order, oldest first, with the delivery outcome and the
    # per-check states the next digest compares against.
    assert [row.text for row in digests] == [
        "08:00 - all clear on 4 check(s)",
        "15:20 - disk: / is 96% full",
    ]
    assert (digests[1].delivered, digests[1].delivery_error) == (
        False,
        "telegram: timed out",
    )
    assert json.loads(digests[1].states) == {"disk": "attention", "store": "ok"}

    assert set(_completed(database)) == ALL_SOURCES

    # A second start re-reads nothing: one gate row per source, one copy of each
    # record.
    import_legacy_state(database, tmp_path)
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
        assert conn.execute(select(func.count()).select_from(ProjectRow)).scalar() == 1
        assert (
            conn.execute(select(func.count()).select_from(AuthSessionRow)).scalar() == 1
        )
        assert conn.execute(select(func.count()).select_from(ScheduleRow)).scalar() == 2
        assert conn.execute(select(func.count()).select_from(DigestRow)).scalar() == 2
    assert set(_completed(database)) == ALL_SOURCES


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

    import_legacy_state(database, tmp_path)

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
        ("auth_sessions.json", '{"sessions": {"s": {}}}', "not a valid session"),
        ("auth_sessions.json", '{"live": {}}', "has no `sessions` mapping"),
        (
            "schedules.json",
            '{"schedules": {"watch": {"runs": "many"}}}',
            "not a valid schedule",
        ),
        ("schedules.json", '{"watch": {}}', "has no `schedules` mapping"),
        ("digests.json", '{"digests": [{"at": 1.0}]}', "not a valid digest"),
        ("digests.json", '{"digests": {}}', "has no `digests` list"),
    ],
)
def test_a_damaged_source_is_refused_whole(
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
        import_legacy_state(database, tmp_path)

    assert source not in _completed(database)


@pytest.mark.parametrize("damaged", ["sessions.json", "schedules.json"])
def test_a_damaged_source_does_not_hold_back_the_other_sources(
    database: Database, tmp_path: Path, damaged: str
) -> None:
    """A refusal isolates ONE source; every other one still imports and gates.

    Both parameters are needed to tell the per-source claim apart from "everything
    before the damage got in": `sessions.json` is near the FRONT of the order and
    `schedules.json` near the back. The refusal still leaves the import failing -
    the operator has to repair the file - but the repair re-reads only that file
    rather than replaying the whole upgrade (review round 1, R1.5).
    """
    _seed(tmp_path)
    (tmp_path / damaged).write_text("[]")

    with pytest.raises(LegacyImportRefused, match=re.escape(damaged)):
        import_legacy_state(database, tmp_path)

    gated = set(_completed(database))
    assert damaged not in gated
    assert ALL_SOURCES - {damaged} <= gated


def test_repairing_a_refused_sessions_file_completes_the_import(
    database: Database, tmp_path: Path
) -> None:
    """The REPAIR the refusal asks for must import, not raise a driver error.

    `sessions.json` is the one source whose refusal changes what a LATER source
    writes: with no mappings imported, `load_agents` migrates a pre-registry
    `session_id` off the record into `agent_session` itself, and `agents.json` is
    gated on that first pass. The repaired mapping then arrives for an agent that
    already has a row, so an INSERT here is a `UNIQUE constraint failed` that no
    retry can clear - the gate row means the conflicting write is never replayed.
    The repaired file is authoritative: it is the switcher's own record, where the
    id on the agent record was only ever a pre-registry stand-in.
    """
    from scufris.db.models import AgentSessionHistoryRow, AgentSessionRow

    _seed(tmp_path)
    # `builder` in BOTH sources: a stale pre-registry id on the record, and the
    # mapping that supersedes it in the file that is damaged.
    agents = json.loads((tmp_path / "agents.json").read_text())
    for record in agents:
        if record["id"] == "builder":
            record["session_id"] = "stale-pre-registry"
    (tmp_path / "agents.json").write_text(json.dumps(agents))
    repaired = (tmp_path / "sessions.json").read_text()
    (tmp_path / "sessions.json").write_text('{"builder": ')

    with pytest.raises(LegacyImportRefused, match="sessions.json"):
        import_legacy_state(database, tmp_path)
    with database.transaction() as conn:
        migrated = conn.execute(
            select(AgentSessionRow.current_session_id).where(
                AgentSessionRow.agent_id == "builder"
            )
        ).scalar_one()
    assert migrated == "stale-pre-registry"

    # The operator repairs the file and restarts, which is what the refusal says
    # to do. Its backup is a copy of the DAMAGED file, so it must not be restored.
    (tmp_path / "sessions.json").write_text(repaired)
    import_legacy_state(database, tmp_path)

    assert "sessions.json" in _completed(database)
    with database.transaction() as conn:
        row = conn.execute(
            select(AgentSessionRow.__table__).where(
                AgentSessionRow.agent_id == "builder"
            )
        ).one()
        history = list(
            conn.execute(
                select(AgentSessionHistoryRow.session_id)
                .where(AgentSessionHistoryRow.agent_id == "builder")
                .order_by(AgentSessionHistoryRow.seq)
            ).scalars()
        )
    assert row.current_session_id == "builder-1"
    assert row.parent_agent_id == "orchestrator"
    assert row.parent_session_id == "orch-1"
    # The stale id is GONE rather than left beside the real history: it was never
    # a session the switcher owned.
    assert history == ["builder-1"]


def test_an_invalid_record_mid_file_rolls_its_whole_source_back(
    database: Database, tmp_path: Path
) -> None:
    """The records BEFORE an invalid one go too, and the source stays ungated.

    Written against `digests.json`, whose loader inserts as it validates: the first
    digest is in the transaction when the second one fails, so a loader that
    validated up front - or one that skipped the bad record, which is what
    `DigestStore._load` did - passes nothing here.
    """
    from scufris.db.models import DigestRow

    _seed(tmp_path)
    (tmp_path / "digests.json").write_text(
        json.dumps(
            {
                "digests": [
                    {"at": 1.0, "schedule": "watch", "verdict": "ok", "text": "fine"},
                    {"at": 2.0, "schedule": "watch", "verdict": "ok"},
                ]
            }
        )
    )

    with pytest.raises(LegacyImportRefused, match=re.escape("record 1")):
        import_legacy_state(database, tmp_path)

    with database.transaction() as conn:
        assert conn.execute(select(func.count()).select_from(DigestRow)).scalar() == 0
    assert "digests.json" not in _completed(database)
