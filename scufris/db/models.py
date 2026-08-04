"""The declarative schema: what the database is supposed to look like.

This module is the SOURCE OF TRUTH for the shape of the state database. The
revisions under ``migrations/versions/`` are how a database that already exists
gets here; they are not a second description of the schema, and a revision that
disagrees with this file is caught by
``test_schema_has_no_pending_autogenerate_diff`` rather than by an operator.

``projects`` mirrors :class:`scufris.projects.Project` field for field, and
``ProjectStore`` reads and writes it: ``projects.json`` is no longer
authoritative.

``agents``, ``agent_session``, ``agent_session_history``, ``agent_outcome``,
``settings_override`` and ``reasoning_turn`` are the agent-state half, and the
same is true of them: ``agents.json``, ``sessions.json``, ``outcomes.json``,
``settings.json`` and the ``reasoning/`` sidecar are no longer authoritative.

``auth_session``, ``schedule`` and ``digest`` are the rest of the app-owned
state, and close the boundary: ``auth_sessions.json``, ``schedules.json`` and
``digests.json`` are no longer authoritative.

``host_action`` and ``config_change`` are NOT here. They belong to
``scufris_hostctl``, which declares them against the same ``Base`` from its own
``models`` module; ``migrations/env.py`` imports that module so autogenerate
still sees them.

``Base`` itself is NOT declared here any more - it is ``scufris_core.Base``, so
that every package in the workspace registers its rows against one metadata
object. It is imported into this module's namespace, so ``from
scufris.db.models import Base`` still resolves for ``migrations/env.py``.

There are no FOREIGN KEYs here, and that is deliberate twice over. The engine
runs with ``foreign_keys=ON`` inside an open transaction, where ``PRAGMA
foreign_keys`` is a no-op - so Alembic's batch ALTER (the only way SQLite can
change a column) cannot turn them off for its copy-and-move, and the first
schema change to a referenced table would be stuck. And nothing here would
benefit: the stores delete an agent's session, history and outcome rows in the
same transaction as its ``agents`` row, so a cascade would only duplicate a
guarantee the transaction already gives.
"""

from __future__ import annotations

from sqlalchemy.orm import Mapped, mapped_column

from scufris_core import Base


class ProjectRow(Base):
    """One workspace record. Mirrors :class:`scufris.projects.Project`.

    Every column is NOT NULL: ``language`` and ``description`` default to the
    empty string on the pydantic record, so an absent value is ``""`` rather than
    a null, and the two spellings of "not set" are never both reachable.
    """

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(primary_key=True)
    cwd: Mapped[str]
    name: Mapped[str]
    language: Mapped[str] = mapped_column(default="")
    description: Mapped[str] = mapped_column(default="")


class AgentRow(Base):
    """One configured agent. Mirrors :class:`scufris.agent_store.AgentRecord`
    field for field EXCEPT ``session_id``, which the session tables own and which
    the store attaches at read time.

    ``state`` and ``permission_mode`` are stored as their enum VALUES (both are
    ``StrEnum``) rather than as a SQL enum: the set of lifecycle states changes
    with the run engine, and a CHECK constraint on it would turn every new state
    into a migration of a table whose rows do not need rewriting.

    The two RESERVED agents - the orchestrator and the host agent - have no row
    here. They are synthesized from settings; only their session ids and outcomes
    are persisted.
    """

    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    project_id: Mapped[str]
    backend: Mapped[str]
    model: Mapped[str] = mapped_column(default="")
    description: Mapped[str] = mapped_column(default="")
    goal: Mapped[str] = mapped_column(default="")
    task_id: Mapped[str] = mapped_column(default="")
    state: Mapped[str]
    permission_mode: Mapped[str]


class AgentSessionRow(Base):
    """Which backend an agent's sessions belong to, and which one is current.

    One row per agent, for EVERY agent - the orchestrator and the host agent
    included, which is why this is not a column on ``agents``: the reserved two
    have no ``agents`` row and their conversations still have to survive a
    restart.

    ``backend`` is what makes a stale cross-backend id unreachable: a codex
    rollout id means nothing to claude, so every accessor matches on it and a
    backend switch starts a fresh history. ``parent_agent_id`` /
    ``parent_session_id`` record who spawned this agent, so a child's
    ``request_input`` routes back to the chat that launched it; they are
    backend-independent, and a row can exist carrying only them (``backend`` "")
    before the child has ever run.
    """

    __tablename__ = "agent_session"

    agent_id: Mapped[str] = mapped_column(primary_key=True)
    backend: Mapped[str]
    current_session_id: Mapped[str | None]
    parent_agent_id: Mapped[str | None]
    parent_session_id: Mapped[str | None]


class AgentSessionHistoryRow(Base):
    """One session an agent has owned, in the order it was minted.

    Rows rather than a JSON list on ``agent_session`` (DECISION.md 4): ``add``,
    ``set_current`` and ``remove`` become row operations inside the transaction
    instead of a read-modify-write of a list, which is where the lost update was.

    ``seq`` is stored rather than derived because ORDER is what the switcher
    shows: reconstructing it from anything else - insertion rowid, an id sort -
    would either depend on a detail SQLite does not promise or reorder the
    operator's chat list.
    """

    __tablename__ = "agent_session_history"

    agent_id: Mapped[str] = mapped_column(primary_key=True)
    seq: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str]


class AgentOutcomeRow(Base):
    """An agent's most-recent durable run outcome. Mirrors
    :class:`scufris.agent_store.RunOutcome`, keyed by agent.

    The per-run EventBus closes when a run ends, so this is how the orchestrator
    observes an agent that finished while it was not watching. Kept for every
    agent including the reserved two, which is again why it is its own table.
    """

    __tablename__ = "agent_outcome"

    agent_id: Mapped[str] = mapped_column(primary_key=True)
    state: Mapped[str]
    message: Mapped[str] = mapped_column(default="")
    run_id: Mapped[str] = mapped_column(default="")
    session_id: Mapped[str | None]
    ts: Mapped[float] = mapped_column(default=0.0)
    acknowledged: Mapped[bool] = mapped_column(default=False)


class SettingsOverrideRow(Base):
    """One runtime override layered over the env-seeded ``Settings``.

    ``value`` is the JSON form of the setting, not its repr: the writable keys
    include a list (``disabled_tools``) and booleans, and JSON is the encoding
    the store already round-trips through ``Settings.model_dump(mode="json")``.
    A row per key rather than one document so two concurrent writes to different
    keys do not have to serialize on a whole-file rewrite.
    """

    __tablename__ = "settings_override"

    key: Mapped[str] = mapped_column(primary_key=True)
    value: Mapped[str]


class ReasoningTurnRow(Base):
    """One completed assistant turn's captured reasoning, in session order.

    Codex persists reasoning only as an encrypted blob, so scufris captures the
    live stream to re-show the "thinking" spoiler after a reload. Rows keyed
    ``(session_id, seq)`` rather than a per-session JSON file (DECISION.md 5):
    an append is one insert instead of a rewrite that grows with the history.

    ``answer`` is a whitespace-normalized fingerprint of the turn's final answer,
    used at merge time to guard sidecar<->transcript alignment. An entry is
    written even when the model did not think (empty ``reasoning``), which is
    what keeps the list 1:1 with the assistant messages the transcript surfaces.
    """

    __tablename__ = "reasoning_turn"

    session_id: Mapped[str] = mapped_column(primary_key=True)
    seq: Mapped[int] = mapped_column(primary_key=True)
    answer: Mapped[str]
    reasoning: Mapped[str]


class AuthSessionRow(Base):
    """One live operator session: an opaque id plus its bound CSRF token.

    Mirrors :class:`scufris.auth.store.Session`. The id is the value in the
    browser's cookie, which is why this table - and the file it sits in - is 0600:
    a readable row here IS a logged-in operator.

    ``LoginThrottle``'s failed-login window is deliberately NOT here
    (20260801-100413 DECISION.md 2): it is rate-limiting state, not the operator's,
    and persisting it would put an unauthenticated caller's input on the write path
    of the database holding these ids.
    """

    __tablename__ = "auth_session"

    id: Mapped[str] = mapped_column(primary_key=True)
    csrf: Mapped[str]
    created_at: Mapped[float]
    last_seen: Mapped[float]


class ScheduleRow(Base):
    """One host-check schedule's state. Mirrors
    :class:`scufris.scheduler.ScheduleState` field for field.

    Two rows, ever - ``watch`` and ``daily`` - keyed by the name the code uses,
    because the schedules have FIXED identities rather than being operator-defined.
    ``next_due`` of 0.0 means "not armed yet", which the scheduler distinguishes
    from a due time in the past, so it is a real value and not a null.
    """

    __tablename__ = "schedule"

    name: Mapped[str] = mapped_column(primary_key=True)
    next_due: Mapped[float] = mapped_column(default=0.0)
    last_run: Mapped[float | None]
    last_result: Mapped[str] = mapped_column(default="")
    missed: Mapped[int] = mapped_column(default=0)
    runs: Mapped[int] = mapped_column(default=0)


class DigestRow(Base):
    """One rendered digest. Mirrors :class:`scufris.digest.Digest`.

    A surrogate ``id`` rather than the timestamp: two digests can share an ``at``
    (a manual run inside the same second as a scheduled one), and the store has to
    be able to name the ONE row whose delivery outcome it is recording. It is what
    ``Digest.id`` carries back to the caller.

    ``states`` is JSON text, as ``SettingsOverrideRow.value`` already is: it is a
    per-check mapping the next digest compares against as a whole, and nothing
    queries inside it.
    """

    __tablename__ = "digest"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    at: Mapped[float]
    schedule: Mapped[str]
    verdict: Mapped[str]
    text: Mapped[str]
    delivered: Mapped[bool] = mapped_column(default=False)
    delivery_error: Mapped[str] = mapped_column(default="")
    states: Mapped[str] = mapped_column(default="{}")
