"""The two tables the host control client owns.

Declared against ``scufris_core.Base``, the workspace's one metadata object, so
the shipped Alembic environment creates and migrates them alongside the app's.
``scufris/db/migrations/env.py`` imports ``scufris_hostctl`` for exactly that
reason - the facade pulls this module in - and
``test_every_package_model_is_registered`` is what keeps the import honest.

These are PRIVATE to the package. The facade exports the stores
(``HostActionStore``, ``ConfigChangeStore``) and the records they read and
write; nothing outside ``scufris_hostctl`` names a row class.
"""

from __future__ import annotations

from sqlalchemy.orm import Mapped, mapped_column

from scufris_core import Base


class HostActionRow(Base):
    """One proposed host action and what the operator decided about it.

    A DECISION JOURNAL, not a cache of the helper's queue. The pending set
    stays the helper's - ``refresh_pending`` is additive and nothing here deletes a
    record the helper stopped listing - and what this table answers is "what did I
    approve, and why did I deny that", which the helper expires in minutes.

    ``proposal`` and ``result`` are JSON text rather than shredded columns because
    ``ProposalView`` and ``ResultFrame`` are the HELPER'S protocol types: giving
    them a schema here would make every helper protocol change a database
    migration, and nothing queries inside either one.

    ``seq`` is the queue ORDER - newest first is how a queue is read - kept as its
    own column rather than derived from ``decided_at`` (a pending action has none)
    or from the insertion rowid (which SQLite does not promise to preserve across a
    VACUUM). It is assigned by the store inside the inserting transaction, as
    ``max(seq) + 1``; SQLite can only autoincrement an INTEGER PRIMARY KEY, and the
    id here is the helper's proposal id.
    """

    __tablename__ = "host_action"

    id: Mapped[str] = mapped_column(primary_key=True)
    seq: Mapped[int] = mapped_column(unique=True)
    proposal: Mapped[str]
    decision: Mapped[str]
    decided_by: Mapped[str] = mapped_column(default="")
    decided_at: Mapped[float | None]
    reason: Mapped[str] = mapped_column(default="")
    run_id: Mapped[str | None]
    result: Mapped[str | None]
    error: Mapped[str] = mapped_column(default="")


class ConfigChangeRow(Base):
    """One NixOS configuration change: what was built, and what came of it.

    Unlike every other row here, this one is written REPEATEDLY by something that
    outlives the request that created it: the build runs for minutes to hours in
    a supervisor task, and each transition (failed, cancelled, proposed, its
    toplevel, its log tail) is a further write. That is why the builder reaches
    the store through a ``save`` callback rather than holding it.

    ``resolved`` is JSON text for the reason ``HostActionRow.proposal`` is: it is
    a nested model and nothing queries inside it. That includes
    ``building_for``'s repository match, which filters the handful of
    ``building`` rows in Python rather than earning a column that duplicates a
    field of the JSON.

    ``seq`` is the list ORDER - newest first - assigned by the store inside the
    inserting transaction as ``max(seq) + 1``, exactly as ``HostActionRow``
    documents and for the same reasons.
    """

    __tablename__ = "config_change"

    id: Mapped[str] = mapped_column(primary_key=True)
    seq: Mapped[int] = mapped_column(unique=True)
    resolved: Mapped[str]
    attr: Mapped[str]
    state: Mapped[str]
    toplevel: Mapped[str] = mapped_column(default="")
    action_id: Mapped[str] = mapped_column(default="")
    run_id: Mapped[str] = mapped_column(default="")
    log_tail: Mapped[str] = mapped_column(default="")
    error: Mapped[str] = mapped_column(default="")
    created_at: Mapped[float]
    agent: Mapped[str] = mapped_column(default="")
    requested_by: Mapped[str] = mapped_column(default="")
