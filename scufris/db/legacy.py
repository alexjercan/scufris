"""Read an operator's legacy per-store JSON files into the database, once.

The policy lives here rather than in each store's importer:

- **Backed up first.** The source is copied to ``<name>.pre-sqlite.bak`` before
  it is read, at 0600 from creation - the later store migrations bring auth
  sessions through this same function.
- **Never deleted.** Nothing here removes a legacy file. The operator does that,
  once they are satisfied, and that deletion is what makes the move one-way.
- **Damaged is refused, never treated as empty.** A file that does not parse is
  named with its line, column and the parser's own message; a record that fails
  its pydantic model fails the WHOLE import. There is no tolerant loader on this
  path: the JSON store this replaces logged and skipped such a record, and
  importing under that rule would leave a database quietly missing records the
  operator can still see in their JSON.
- **All or nothing, once.** One source is imported inside one
  ``Database.transaction()`` that also writes its ``legacy_import`` row, so a
  failure anywhere leaves no rows AND no gate: the operator repairs the file and
  the retry starts from the beginning.

The gate is a table rather than a schema version, and the import cannot ride a
schema revision: it needs the state directory and the pydantic models to do its
job, and neither belongs inside a migration.

``scufris.db.open_state_database`` is the only call site: it runs the import at
startup, after the migration and ahead of the first store read. Both orderings
matter - importing before the migration would write through models the schema
does not have yet, and importing after a store had already read would show an
operator an empty database while their projects sat in ``projects.json``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from pydantic import TypeAdapter, ValidationError
from sqlalchemy import Connection, insert, select

from .engine import FILE_MODE, Database
from .models import (
    AgentOutcomeRow,
    AgentRow,
    AgentSessionHistoryRow,
    AgentSessionRow,
    LegacyImportRow,
    ProjectRow,
    ReasoningTurnRow,
    SettingsOverrideRow,
)

logger = logging.getLogger(__name__)

BACKUP_SUFFIX = ".pre-sqlite.bak"

PROJECTS_FILENAME = "projects.json"
AGENTS_FILENAME = "agents.json"
SESSIONS_FILENAME = "sessions.json"
OUTCOMES_FILENAME = "outcomes.json"
SETTINGS_FILENAME = "settings.json"
REASONING_DIRNAME = "reasoning"

# What one store's importer does with its parsed JSON: validate it, write it on
# the OPEN connection, and return how many records it wrote. It never opens a
# transaction of its own - the one it is called inside is what makes the import
# all-or-nothing - and it raises `LegacyImportRefused` rather than dropping a
# record it cannot validate.
Loader = Callable[[Path, Connection, object], int]


class LegacyImportRefused(RuntimeError):
    """The legacy file cannot be imported, and is NOT to be treated as absent."""


def backup_path(source: Path) -> Path:
    """Where ``source`` is copied before it is read."""
    return source.with_name(f"{source.name}{BACKUP_SUFFIX}")


def import_projects(db: Database, state_dir: Path) -> bool:
    """Import ``projects.json`` from ``state_dir``. True if this run imported it."""
    return import_legacy_file(db, Path(state_dir) / PROJECTS_FILENAME, _load_projects)


def import_legacy_file(
    db: Database, source: Path, load: Loader, *, key: str | None = None
) -> bool:
    """Import one legacy JSON file into ``db``, at most once.

    Returns True if this call imported it, False if a previous run already did
    or the file does not exist (a fresh install has none). Raises
    ``LegacyImportRefused`` for a file that exists and cannot be trusted; the
    database is unchanged and the gate stays open, so a repaired file imports on
    the next run.

    ``key`` is what the ``legacy_import`` row records, defaulting to the file's
    name. The reasoning sidecar needs it explicitly: its files are named after a
    SESSION ID, and a session called ``sessions`` would produce ``sessions.json``
    and collide with the session registry's own gate row - which would silently
    skip whichever ran second. The sidecar passes ``reasoning/<name>``.

    Everything happens inside ONE transaction, including the backup and the read
    - so the check that the source has not already been imported and the writing
    of its ``legacy_import`` row cannot be separated by another process. The
    write lock is therefore held across a small file read, which is affordable
    because this runs once, at startup, on files a single host wrote by hand.
    """
    gate = key if key is not None else source.name
    with db.transaction() as conn:
        if _is_imported(conn, gate):
            return False
        if not source.is_file():
            return False
        backup = _back_up(source)
        count = load(source, conn, _parse(source))
        conn.execute(insert(LegacyImportRow).values(source=gate, imported_at=_now()))
    logger.info(
        "imported %d records from %s (the file is left in place; backup: %s)",
        count,
        source,
        backup,
    )
    return True


def _is_imported(conn: Connection, name: str) -> bool:
    row = conn.execute(
        select(LegacyImportRow.source).where(LegacyImportRow.source == name)
    ).first()
    return row is not None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _back_up(source: Path) -> Path:
    """Copy ``source`` next to itself at 0600, replacing an earlier attempt's copy.

    Created 0600 rather than chmod-ed to it afterwards, and never through a
    symlink: this function is how an operator's auth sessions will reach disk a
    second time, and a window where that copy is world-readable is the same
    exposure whether it lasts a millisecond or a day.

    A leftover backup is a previous import that failed after the copy. Nothing
    here ever writes to the source, so the file has not moved since and the
    fresh copy is the same bytes.
    """
    target = backup_path(source)
    if target.is_symlink():
        raise LegacyImportRefused(
            f"REFUSED: {target} is a symlink; refusing to write the backup"
        )
    target.unlink(missing_ok=True)
    fd = os.open(
        target,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
        FILE_MODE,
    )
    with os.fdopen(fd, "wb") as handle:
        handle.write(source.read_bytes())
    return target


def _parse(source: Path) -> object:
    """The file's JSON, or a refusal naming where it stops making sense."""
    try:
        text = source.read_text()
    except OSError as exc:
        raise LegacyImportRefused(f"REFUSED: {source} cannot be read: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LegacyImportRefused(
            f"REFUSED: {source} is damaged at line {exc.lineno} col {exc.colno}: "
            f"{exc.msg}. {backup_path(source)} is a copy of this same damaged "
            "file, not a repair - restoring it changes nothing. Repair the file "
            "from your own backup, or move it aside to import the rest of the "
            "state without it."
        ) from exc


def _load_projects(source: Path, conn: Connection, payload: object) -> int:
    """Validate every record through :class:`scufris.projects.Project` and write it.

    Inserted one at a time as they validate, so the rollback that an invalid
    record triggers is what removes the records BEFORE it. Validating the whole
    file up front would pass the same test without the transaction doing
    anything.

    ``Project`` is imported HERE rather than at module scope: since the store
    cutover, ``scufris.projects`` imports this package for its ``Database`` and
    ``ProjectRow``, and a top-level import back into it makes the two modules
    load-order dependent - importing ``scufris.projects`` first would reach this
    line before ``Project`` exists.
    """
    from ..projects import Project

    if not isinstance(payload, list):
        raise LegacyImportRefused(
            f"REFUSED: {source} is damaged: the top level is "
            f"{type(payload).__name__}, not a list of projects"
        )
    count = 0
    for index, item in enumerate(payload):
        try:
            project = Project.model_validate(item)
        except ValidationError as exc:
            raise LegacyImportRefused(
                f"REFUSED: {source} record {index} is not a valid project: {exc}"
            ) from exc
        conn.execute(insert(ProjectRow).values(**project.model_dump()))
        count += 1
    return count


# --- the agent-state sources ------------------------------------------------
#
# Five files, one policy, and an ORDER that matters: `sessions.json` is imported
# before `agents.json` because the agent loader migrates a pre-registry
# `session_id` off the record and into the session tables only when no mapping
# exists, and "an existing mapping wins" is only meaningful if the mappings are
# already there.


def import_agent_state(db: Database, state_dir: Path) -> None:
    """Import the agent, session, outcome, settings and reasoning JSON, once each.

    Each source is its own all-or-nothing import with its own gate row, and a
    refusal does not stop the sources AFTER it: every one is attempted, and the
    refusals are raised together at the end. So a damaged ``outcomes.json`` still
    fails startup - the operator must repair it, and a silently skipped source
    would be exactly the tolerant loader this package refuses to be - but the
    other four are in with their gate rows, and the retry after the repair reads
    only the file that was damaged.

    Running the later sources anyway is safe by construction: each writes its own
    tables inside its own transaction, and the one ORDER that matters (sessions
    before agents, so an existing mapping wins over a pre-registry ``session_id``
    on the record) degrades correctly - with no mappings imported, migrating the
    record's own id IS the right answer.
    """
    state_dir = Path(state_dir)
    sources: list[tuple[Path, Loader, str | None]] = [
        (state_dir / SESSIONS_FILENAME, _load_sessions, None),
        (state_dir / AGENTS_FILENAME, _load_agents, None),
        (state_dir / OUTCOMES_FILENAME, _load_outcomes, None),
        (state_dir / SETTINGS_FILENAME, _load_settings, None),
    ]
    sources += [
        (sidecar, _load_reasoning, f"{REASONING_DIRNAME}/{sidecar.name}")
        for sidecar in sorted((state_dir / REASONING_DIRNAME).glob("*.json"))
    ]
    refused: list[LegacyImportRefused] = []
    for path, loader, key in sources:
        try:
            import_legacy_file(db, path, loader, key=key)
        except LegacyImportRefused as exc:
            refused.append(exc)
    if refused:
        joined = "\n".join(str(exc) for exc in refused)
        raise LegacyImportRefused(joined) from refused[0]


def _refuse(source: Path, detail: str) -> LegacyImportRefused:
    return LegacyImportRefused(f"REFUSED: {source} {detail}")


def _require_mapping(source: Path, payload: object, of: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise _refuse(
            source, f"is damaged: the top level is {type(payload).__name__}, not {of}"
        )
    return payload


def _load_agents(source: Path, conn: Connection, payload: object) -> int:
    """Validate every agent through ``AgentRecord`` and write it.

    Two migrations run BEFORE validation, because the model no longer has the
    fields they are about and pydantic would ignore them: a legacy
    ``write_enabled`` bool becomes a permission mode (without this a
    write-enabled agent silently comes back read-only), and a legacy codex mode
    id becomes the canonical backend name. They are not optional politeness - a
    real operator's file has both, and refusing it instead would be refusing
    valid state.

    A pre-registry ``session_id`` persisted ON the record is moved into the
    session tables, where an existing mapping wins, and then dropped: the session
    tables are the only home of session ids and ``get``/``list`` re-attach them
    at read time.
    """
    from ..agent_store.records import AgentRecord
    from ..agent_store.registry import SessionRows
    from ..config import canonical_backend

    if not isinstance(payload, list):
        raise _refuse(
            source,
            f"is damaged: the top level is {type(payload).__name__}, not a list "
            "of agents",
        )
    sessions = SessionRows(conn)
    count = 0
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise _refuse(
                source, f"record {index} is {type(item).__name__}, not an object"
            )
        item = dict(item)
        if "permission_mode" not in item and "write_enabled" in item:
            item["permission_mode"] = "edit" if item.get("write_enabled") else "manual"
        item.pop("write_enabled", None)
        try:
            agent = AgentRecord.model_validate(item)
        except ValidationError as exc:
            raise _refuse(
                source, f"record {index} is not a valid agent: {exc}"
            ) from exc
        agent.backend = canonical_backend(agent.backend)
        if agent.session_id and not sessions.has(agent.id):
            sessions.add(agent.id, agent.backend, agent.session_id)
        conn.execute(
            insert(AgentRow).values(
                **agent.model_dump(mode="json", exclude={"session_id"})
            )
        )
        count += 1
    return count


def _load_sessions(source: Path, conn: Connection, payload: object) -> int:
    """Write each agent's session record and its history, in order.

    The legacy ``{backend, session_id}`` shape - written before the switcher
    existed - loads as a ONE-ELEMENT history rather than an empty one: that id is
    a conversation the operator can still open, and reading it as "current but
    never owned" would drop it off the switcher list.
    """
    entries = _require_mapping(source, payload, "a mapping of agent id -> session")
    count = 0
    for agent_id, entry in entries.items():
        if not isinstance(entry, dict):
            raise _refuse(
                source, f"entry {agent_id!r} is {type(entry).__name__}, not an object"
            )
        backend = entry.get("backend")
        if not isinstance(backend, str):
            raise _refuse(source, f"entry {agent_id!r} has no backend")
        current = entry.get("session_id")
        if not isinstance(current, (str, type(None))):
            raise _refuse(source, f"entry {agent_id!r} has a non-string session_id")
        raw = entry.get("sessions")
        if isinstance(raw, list):
            if not all(isinstance(s, str) for s in raw):
                raise _refuse(
                    source,
                    f"entry {agent_id!r} has a non-string session in its history",
                )
            history = list(raw)
        elif raw is None:
            history = [current] if current is not None else []
        else:
            raise _refuse(source, f"entry {agent_id!r} has a non-list sessions")
        conn.execute(
            insert(AgentSessionRow).values(
                agent_id=agent_id,
                backend=backend,
                current_session_id=current,
                parent_agent_id=_optional_str(
                    source, agent_id, entry, "parent_agent_id"
                ),
                parent_session_id=_optional_str(
                    source, agent_id, entry, "parent_session_id"
                ),
            )
        )
        for seq, session_id in enumerate(history):
            conn.execute(
                insert(AgentSessionHistoryRow).values(
                    agent_id=agent_id, seq=seq, session_id=session_id
                )
            )
        count += 1
    return count


def _optional_str(
    source: Path, agent_id: str, entry: dict[str, object], field: str
) -> str | None:
    value = entry.get(field)
    if value is None or isinstance(value, str):
        return value
    raise _refuse(source, f"entry {agent_id!r} has a non-string {field}")


def _load_outcomes(source: Path, conn: Connection, payload: object) -> int:
    """Validate every outcome through ``RunOutcome`` and write it."""
    from ..agent_store.outcomes import RunOutcome

    entries = _require_mapping(source, payload, "a mapping of agent id -> outcome")
    count = 0
    for agent_id, entry in entries.items():
        try:
            outcome = RunOutcome.model_validate(entry)
        except ValidationError as exc:
            raise _refuse(
                source, f"entry {agent_id!r} is not a valid outcome: {exc}"
            ) from exc
        conn.execute(
            insert(AgentOutcomeRow).values(
                agent_id=agent_id, **outcome.model_dump(mode="json")
            )
        )
        count += 1
    return count


def _load_settings(source: Path, conn: Connection, payload: object) -> int:
    """Write each persisted override, refusing any key or value that is not one.

    Strict where ``SettingsStore._load`` is tolerant, and the difference is what
    the operator can do about it. Here their `settings.json` is in front of them
    and the refusal names the key, so a repair is one edit and a restart. At LOAD
    the same strictness would be a server that will not boot because of a knob it
    no longer has, with the fix locked inside the database the failure is denying
    them.
    """
    from ..settings_store import WRITABLE_KEYS

    overrides = _overrides_from_persisted(source, payload)
    for key, value in overrides.items():
        if key not in WRITABLE_KEYS:
            raise _refuse(
                source,
                f"overrides {key!r}, which is not a writable setting "
                f"(writable keys: {sorted(WRITABLE_KEYS)})",
            )
        _validate_setting(source, key, value)
        conn.execute(
            insert(SettingsOverrideRow).values(key=key, value=json.dumps(value))
        )
    return len(overrides)


def _validate_setting(source: Path, key: str, value: object) -> None:
    """Check ``value`` against the field's own type and constraints.

    Validated field-by-field rather than by assigning onto a ``Settings``: the
    importer has a state directory, not a live settings object, and building one
    here would read the environment and fail on things that have nothing to do
    with the file being imported.
    """
    from typing import Annotated

    from ..config import Settings

    field = Settings.model_fields[key]
    annotation = (
        Annotated[tuple([field.annotation, *field.metadata])]  # type: ignore[misc]
        if field.metadata
        else field.annotation
    )
    try:
        TypeAdapter(annotation).validate_python(value)
    except ValidationError as exc:
        raise _refuse(source, f"has an invalid value for {key!r}: {exc}") from exc


def _overrides_from_persisted(source: Path, payload: object) -> dict[str, object]:
    """The override mapping from a persisted settings file.

    Accepts the current flat ``{overrides: {...}}`` shape and migrates the older
    profile-shaped ``{active, profiles: {<name>: {...}}}`` file by taking the
    active profile's overrides (falling back to ``default``). This is the ONLY
    reader of either shape now - the store reads rows - so the migration lives
    here rather than in the store it used to run in.
    """
    data = _require_mapping(source, payload, "a settings object")
    overrides = data.get("overrides")
    if isinstance(overrides, dict):
        return dict(overrides)
    profiles = data.get("profiles")
    if isinstance(profiles, dict):
        active = data.get("active")
        for name in (active, "default"):
            if isinstance(name, str) and isinstance(profiles.get(name), dict):
                return dict(profiles[name])
    raise _refuse(
        source,
        "has neither an `overrides` mapping nor a profile with one; it is not a "
        "settings file this version can read",
    )


def _load_reasoning(source: Path, conn: Connection, payload: object) -> int:
    """Write one session's captured turns as rows, in file order.

    The session id is the file's STEM, which is how the sidecar was keyed. An
    entry with an empty ``reasoning`` is kept, not skipped: the list is 1:1 with
    the assistant messages the transcript surfaces, and dropping the silent turns
    would shift every later spoiler onto the wrong message.
    """
    data = _require_mapping(source, payload, "a reasoning sidecar object")
    turns = data.get("turns")
    if not isinstance(turns, list):
        raise _refuse(source, "has no `turns` list")
    session_id = source.stem
    for seq, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise _refuse(source, f"turn {seq} is {type(turn).__name__}, not an object")
        answer = turn.get("answer")
        reasoning = turn.get("reasoning")
        if not isinstance(answer, str) or not isinstance(reasoning, str):
            raise _refuse(source, f"turn {seq} has no answer/reasoning text")
        conn.execute(
            insert(ReasoningTurnRow).values(
                session_id=session_id, seq=seq, answer=answer, reasoning=reasoning
            )
        )
    return len(turns)
