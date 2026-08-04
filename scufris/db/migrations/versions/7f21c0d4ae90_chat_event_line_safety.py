"""chat event line safety

Revision ID: 7f21c0d4ae90
Revises: 4bc3435e4fdc
Create Date: 2026-08-04 16:05:12.004311

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7f21c0d4ae90"
down_revision: str | Sequence[str] | None = "4bc3435e4fdc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ACTOR_KIND_CHECK = "actor_kind IN ('operator', 'orchestrator', 'agent', 'system')"

_OLD_ACTOR_AGENT_ID_CHECK = (
    "(actor_kind = 'agent' AND actor_agent_id IS NOT NULL AND actor_agent_id <> '')"
    " OR (actor_kind <> 'agent' AND actor_agent_id IS NULL)"
)

# Every C0 control, DEL, and the three line terminators outside them that
# `str.splitlines` breaks on - U+0085, U+2028 and U+2029. `char()` rather than
# literal control characters, so none appears in this file. `char(0)` is the
# empty string in SQLite, so the range starts at 1. Do not "simplify" the
# concatenation to a literal pattern.
_LINE_UNSAFE_GLOB = (
    "('*[' || char(1) || '-' || char(31) || char(127)"
    " || char(133) || char(8232) || char(8233) || ']*')"
)

_NEW_ACTOR_AGENT_ID_CHECK = (
    "(actor_kind = 'agent' AND actor_agent_id IS NOT NULL AND actor_agent_id <> ''"
    f" AND actor_agent_id NOT GLOB {_LINE_UNSAFE_GLOB})"
    " OR (actor_kind <> 'agent' AND actor_agent_id IS NULL)"
)

_EVENT_BODY_CHECK = "body <> ''"


def _event_table(agent_id_check: str, *extra: sa.schema.SchemaItem) -> sa.Table:
    """The `event` table as it stands on one side of this revision.

    SQLite cannot add or drop a CHECK in place, so the batch operation recreates
    the table and needs the schema it is copying FROM stated rather than
    reflected: SQLite reflection does not reliably give back the CHECK
    constraints, and a recreate that loses them would drop the actor rule
    silently.
    """
    return sa.Table(
        "event",
        sa.MetaData(),
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("conversation_id", sa.String(), nullable=False),
        sa.Column("event_seq", sa.Integer(), nullable=False),
        sa.Column("actor_kind", sa.String(), nullable=False),
        sa.Column("actor_agent_id", sa.String(), nullable=True),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("body", sa.String(), nullable=False),
        sa.Column("correlation_id", sa.String(), nullable=True),
        sa.Column("causation_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.CheckConstraint(_ACTOR_KIND_CHECK, name="ck_event_actor_kind"),
        sa.CheckConstraint(agent_id_check, name="ck_event_actor_agent_id"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "conversation_id", "event_seq", name="uq_event_conversation_seq"
        ),
        *extra,
    )


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table(
        "event",
        copy_from=_event_table(_OLD_ACTOR_AGENT_ID_CHECK),
        recreate="always",
    ) as batch_op:
        batch_op.drop_constraint("ck_event_actor_agent_id", type_="check")
        batch_op.create_check_constraint(
            "ck_event_actor_agent_id", _NEW_ACTOR_AGENT_ID_CHECK
        )
        batch_op.create_check_constraint("ck_event_body", _EVENT_BODY_CHECK)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table(
        "event",
        copy_from=_event_table(
            _NEW_ACTOR_AGENT_ID_CHECK,
            sa.CheckConstraint(_EVENT_BODY_CHECK, name="ck_event_body"),
        ),
        recreate="always",
    ) as batch_op:
        batch_op.drop_constraint("ck_event_body", type_="check")
        batch_op.drop_constraint("ck_event_actor_agent_id", type_="check")
        batch_op.create_check_constraint(
            "ck_event_actor_agent_id", _OLD_ACTOR_AGENT_ID_CHECK
        )
