"""What each legacy source does with its parsed JSON: validate it, write it.

One function per source, all of them :data:`scufris.db.legacy.gate.Loader`: they
write on the connection they are handed, never open a transaction of their own,
and refuse a record they cannot validate rather than dropping it. The policy that
makes those refusals safe is in the package docstring.

Every store model is imported INSIDE its loader rather than at module scope: the
stores import this package for their ``Database`` and row types, and a top-level
import back into them would make the two load-order dependent.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError
from sqlalchemy import Connection, delete, insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from ..models import (
    AgentOutcomeRow,
    AgentRow,
    AgentSessionHistoryRow,
    AgentSessionRow,
    AuthSessionRow,
    DigestRow,
    ProjectRow,
    ReasoningTurnRow,
    ScheduleRow,
    SettingsOverrideRow,
)
from .gate import LegacyImportRefused


def load_projects(source: Path, conn: Connection, payload: object) -> int:
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
    from ...projects import Project

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


def _refuse(source: Path, detail: str) -> LegacyImportRefused:
    return LegacyImportRefused(f"REFUSED: {source} {detail}")


def _require_mapping(source: Path, payload: object, of: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise _refuse(
            source, f"is damaged: the top level is {type(payload).__name__}, not {of}"
        )
    return payload


def load_agents(source: Path, conn: Connection, payload: object) -> int:
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
    from ...agent_store.records import AgentRecord
    from ...agent_store.registry import SessionRows
    from ...config import canonical_backend

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


def load_sessions(source: Path, conn: Connection, payload: object) -> int:
    """Write each agent's session record and its history, in order.

    The legacy ``{backend, session_id}`` shape - written before the switcher
    existed - loads as a ONE-ELEMENT history rather than an empty one: that id is
    a conversation the operator can still open, and reading it as "current but
    never owned" would drop it off the switcher list.

    This file REPLACES whatever mapping an agent already has rather than
    inserting beside it, and that is what makes the repair after a refusal work.
    Refuse this source and ``agents.json`` is still imported and gated, and
    ``load_agents`` migrates each pre-registry ``session_id`` off the record into
    these same tables; the repaired file then arrives for an agent that already
    has a row. An insert there is a `UNIQUE constraint failed` no retry can
    clear, because the agents gate row means the conflicting write is never
    replayed. Replacing is also the RIGHT answer and not merely the surviving
    one: this file is the switcher's own record, and the id on an agent record
    was only ever the stand-in for it.
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
        values = {
            "agent_id": agent_id,
            "backend": backend,
            "current_session_id": current,
            "parent_agent_id": _optional_str(
                source, agent_id, entry, "parent_agent_id"
            ),
            "parent_session_id": _optional_str(
                source, agent_id, entry, "parent_session_id"
            ),
        }
        conn.execute(
            sqlite_insert(AgentSessionRow)
            .values(**values)
            .on_conflict_do_update(
                index_elements=[AgentSessionRow.agent_id], set_=values
            )
        )
        # The history goes with the mapping: a stale id migrated off an agent
        # record is not a conversation this file kept, so it must not be left
        # beside the history this file declares.
        conn.execute(
            delete(AgentSessionHistoryRow).where(
                AgentSessionHistoryRow.agent_id == agent_id
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


def load_outcomes(source: Path, conn: Connection, payload: object) -> int:
    """Validate every outcome through ``RunOutcome`` and write it."""
    from ...agent_store.outcomes import RunOutcome

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


def load_settings(source: Path, conn: Connection, payload: object) -> int:
    """Write each persisted override, refusing any key or value that is not one.

    Strict where ``SettingsStore._load`` is tolerant, and the difference is what
    the operator can do about it. Here their `settings.json` is in front of them
    and the refusal names the key, so a repair is one edit and a restart. At LOAD
    the same strictness would be a server that will not boot because of a knob it
    no longer has, with the fix locked inside the database the failure is denying
    them.
    """
    from ...settings_store import WRITABLE_KEYS

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

    from ...config import Settings

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


def load_reasoning(source: Path, conn: Connection, payload: object) -> int:
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


# --- the auth, schedule and digest sources ------------------------------------


def load_auth_sessions(source: Path, conn: Connection, payload: object) -> int:
    """Write every live session, validated through :class:`scufris.auth.store.Session`.

    Strict where the deleted ``SessionStore._load`` was tolerant: it kept any
    record that happened to be a dict and read missing fields as ``0.0``, which
    turns a damaged record into a session that expires at a time nobody wrote.
    A refusal here costs the operator one repair; the tolerant read cost them a
    logout they could not explain.

    Nothing here logs a session id, and the record is not pruned on the way in:
    expiry is ``SessionStore.prune``'s, which the app runs at startup with the
    settings' own windows - re-deciding it here would need those windows and would
    be a second, divergent, expiry rule.
    """
    from ...auth.store import Session

    data = _require_mapping(source, payload, "a session store object")
    entries = data.get("sessions")
    if not isinstance(entries, dict):
        raise _refuse(source, "has no `sessions` mapping")
    for session_id, entry in entries.items():
        if not isinstance(entry, dict):
            raise _refuse(
                source, f"entry {session_id!r} is {type(entry).__name__}, not an object"
            )
        try:
            session = TypeAdapter(Session).validate_python({**entry, "id": session_id})
        except ValidationError as exc:
            raise _refuse(
                source, f"entry {session_id!r} is not a valid session: {exc}"
            ) from exc
        conn.execute(
            insert(AuthSessionRow).values(
                id=session.id,
                csrf=session.csrf,
                created_at=session.created_at,
                last_seen=session.last_seen,
            )
        )
    return len(entries)


def load_schedules(source: Path, conn: Connection, payload: object) -> int:
    """Write every schedule's state, validated through ``ScheduleState``.

    Strict where ``SchedulerStore._load`` was tolerant: it dropped the WHOLE file
    on one invalid entry and logged it, which reset both schedules' run counts and
    their next due time to a fresh install's.

    The row's name is the MAPPING KEY, which is what the store looks a schedule up
    by; the copy on the record is only what the old whole-file model happened to
    carry. A name the code no longer runs is imported anyway - ``all`` only asks
    for ``watch`` and ``daily``, and refusing an operator's history because a
    schedule was renamed would be refusing valid state.
    """
    from ...scheduler import ScheduleState

    data = _require_mapping(source, payload, "a scheduler state object")
    schedules = data.get("schedules")
    if not isinstance(schedules, dict):
        raise _refuse(source, "has no `schedules` mapping")
    for name, entry in schedules.items():
        if not isinstance(entry, dict):
            raise _refuse(
                source, f"entry {name!r} is {type(entry).__name__}, not an object"
            )
        try:
            state = ScheduleState.model_validate({**entry, "name": name})
        except ValidationError as exc:
            raise _refuse(
                source, f"entry {name!r} is not a valid schedule: {exc}"
            ) from exc
        conn.execute(insert(ScheduleRow).values(**state.model_dump()))
    return len(schedules)


def load_digests(source: Path, conn: Connection, payload: object) -> int:
    """Write the recent digests in file order, validated through ``Digest``.

    File order is oldest-first, and the row ids ascend with it, which is what
    keeps ``latest`` and the store's bound (both ordered by id) reading the
    imported history the same way they read one this version wrote. The legacy
    records carry no id - the deque had no keys - so each is assigned one here.

    The store's ``MAX_DIGESTS`` bound is not re-applied: the deque that wrote this
    file was bounded by the same number, so there is nothing to trim, and the
    next :meth:`DigestStore.add` reaps anyway.
    """
    from ...digest import Digest

    data = _require_mapping(source, payload, "a digest store object")
    rows = data.get("digests")
    if not isinstance(rows, list):
        raise _refuse(source, "has no `digests` list")
    for index, row in enumerate(rows):
        try:
            digest = Digest.model_validate(row)
        except ValidationError as exc:
            raise _refuse(
                source, f"record {index} is not a valid digest: {exc}"
            ) from exc
        values = digest.model_dump(exclude={"id"})
        values["states"] = json.dumps(digest.states)
        conn.execute(insert(DigestRow).values(**values))
    return len(rows)
