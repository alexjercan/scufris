"""The four tables the conversation is made of.

Declared against ``scufris_core.Base``, the workspace's one metadata object, so
the shipped Alembic environment creates and migrates them alongside the app's.
``scufris/db/migrations/env.py`` imports ``scufris_chat`` for exactly that
reason - the facade pulls this module in - and
``test_every_package_model_is_registered`` is what keeps the import honest.

These are PRIVATE to the package. The facade exports the store functions and the
records they read and write; nothing outside ``scufris_chat`` names a row class.

No FOREIGN KEYs. ``scufris/db/models.py``'s docstring records the reason - a
batch ALTER under ``foreign_keys=ON`` rebuilds the table its references point at
- and it applies here unchanged, so ``conversation_id`` and ``causation_id`` are
plain columns. What holds them together is that every write goes through
``store.py`` inside one transaction.
"""

from __future__ import annotations

from enum import StrEnum

from sqlalchemy import CheckConstraint, PrimaryKeyConstraint, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from scufris_core import Base

from .actors import ACTOR_KIND_VALUES


class DeliveryState(StrEnum):
    """Where one channel's attempt at one event has got to.

    Two states, not a boolean. A row is written ``claimed`` in the same
    transaction as the event and moved to ``confirmed`` once the channel's send
    returns, which makes the crash window READABLE: a ``claimed`` row with no
    confirmation is exactly "someone was mid-send when we died". A single
    "delivered" row would have to be written either before the send, silently
    losing the question, or after it, duplicating forever.
    """

    CLAIMED = "claimed"
    CONFIRMED = "confirmed"


#: The kind column's allowed values, rendered from the enum rather than typed
#: out, so a fifth `ActorKind` cannot land with the constraint still naming four.
_ACTOR_KIND_CHECK = "actor_kind IN ({})".format(
    ", ".join(f"'{value}'" for value in ACTOR_KIND_VALUES)
)

#: A GLOB character class covering every C0 control, DEL, and the three line
#: terminators outside them that `str.splitlines` breaks on - U+0085, U+2028 and
#: U+2029. The last three are the point: the alphabet is the one the consumer
#: splits on, so a set stopping at ASCII would leave the forgery this constraint
#: exists to refuse working one character over. Built with `char()` so no literal
#: control character has to appear in this file. `char(0)` is the empty string in
#: SQLite, so the range starts at 1; a NUL is refused by `Actor.__post_init__`
#: and, in a TEXT column, truncates rather than forging a line. Do not "simplify"
#: the concatenation to a literal pattern.
_LINE_UNSAFE_GLOB = (
    "('*[' || char(1) || '-' || char(31) || char(127)"
    " || char(133) || char(8232) || char(8233) || ']*')"
)

#: The other half of the actor rule: only an `agent` names an agent, and an
#: `agent` always does. Stated as TRUTHINESS, not as nullability, because that is
#: the predicate `Actor.__post_init__` uses - `not self.agent_id` refuses `''`
#: too, so a nullability-only constraint would let the empty string through the
#: schema and into a row `_record` cannot rebuild. The line-safety clause mirrors
#: the other half of that predicate: the id is interpolated into a LINE of
#: assembled context, so a line break in it forges an `operator:` attribution
#: under a preamble saying the operator's lines are instructions.
_ACTOR_AGENT_ID_CHECK = (
    "(actor_kind = 'agent' AND actor_agent_id IS NOT NULL AND actor_agent_id <> ''"
    f" AND actor_agent_id NOT GLOB {_LINE_UNSAFE_GLOB})"
    " OR (actor_kind <> 'agent' AND actor_agent_id IS NULL)"
)

#: An empty body renders to NO lines - `''.splitlines()` is `[]` - so the event
#: vanishes from assembled context while still consuming a window slot, and the
#: provider is seeded with a transcript missing a row `read_transcript` shows the
#: operator. Truthiness for the reason `_DELIVERY_CHANNEL_CHECK` is.
_EVENT_BODY_CHECK = "body <> ''"

#: The delivery state column's allowed values, rendered from the enum rather
#: than typed out, so a third `DeliveryState` cannot land with the constraint
#: still naming two.
_DELIVERY_STATE_CHECK = "state IN ({})".format(
    ", ".join(f"'{state.value}'" for state in DeliveryState)
)

#: The channel is a third of the primary key, so `''` is a DISTINCT delivery
#: rather than a malformed one: the row lands under a channel nothing polls and
#: the real channel still sees the event as pending. Stated as truthiness for the
#: reason `_ACTOR_AGENT_ID_CHECK` is - a repair INSERT with an uninterpolated
#: variable produces `''` far more readily than it produces a channel name.
_DELIVERY_CHANNEL_CHECK = "channel <> ''"

#: The other half of the state rule: a `confirmed` row always carries when, and
#: a `claimed` one never does. Both halves, because a row that satisfies the
#: state check alone is a delivery that says it completed at no time.
_DELIVERY_CONFIRMED_AT_CHECK = (
    f"(state = '{DeliveryState.CONFIRMED.value}' AND confirmed_at IS NOT NULL)"
    f" OR (state <> '{DeliveryState.CONFIRMED.value}' AND confirmed_at IS NULL)"
)

#: The backend is half the key, so `''` files the binding under a backend nothing
#: looks up while the real one re-seeds on every turn forever. Truthiness for the
#: reason `_DELIVERY_CHANNEL_CHECK` is.
_PROVIDER_SESSION_BACKEND_CHECK = "backend <> ''"

#: A binding to no session reads as a WARM cache and then resumes nothing - the
#: one shape worse than a miss, because a miss re-seeds and this does not.
_PROVIDER_SESSION_ID_CHECK = "provider_session_id <> ''"

#: The column the whole invalidation rests on. A zero or negative version is not
#: an older policy, it is a corrupt row: nothing mints one, so a row carrying one
#: came from a repair session or a partial write, and it would compare as stale
#: against every real version forever.
_PROVIDER_SESSION_POLICY_VERSION_CHECK = "policy_version >= 1"


class ConversationRow(Base):
    """One conversation - a durable thread, not a provider session.

    It carries almost nothing, and that is the design: the conversation is meant
    to OUTLIVE any backend under it, so the backend, the policy version and the
    provider's session id belong to the cache keyed by them, not here. See
    ``tasks/20260804-115256/DECISION.md`` section 3.
    """

    __tablename__ = "conversation"

    id: Mapped[str] = mapped_column(primary_key=True)
    created_at: Mapped[float]


class EventRow(Base):
    """One attributable utterance: an operator message, an agent report, a notice.

    One row per thing SAID, not per turn. A turn produces several, and the
    conversation is read as an ordered transcript; the finer grain is what lets
    "who said this" be answered for something inside a turn, which is what a stop
    gate has to ask. ``tasks/20260804-115256/DECISION.md`` section 1.

    ``event_seq`` is the position in that transcript, per conversation and
    starting at 1, assigned by the store inside the inserting transaction as
    ``COALESCE(MAX(event_seq), 0) + 1`` - the pattern ``HostActionRow.seq``
    already uses, and for the same reason: SQLite can only autoincrement an
    INTEGER PRIMARY KEY, and the key here is the event's own id. The unique
    constraint over ``(conversation_id, event_seq)`` is what turns a store bug
    into a failed INSERT rather than two events at position 4. Its leading column
    is ``conversation_id``, so it is also the index every transcript read uses;
    a second index on that column alone would be the same index twice.

    The actor is constrained in BOTH halves: ``actor_kind`` against the four
    kinds, and ``actor_agent_id`` against the kind - a non-empty string for
    ``agent``, NULL for the other three, which is the predicate ``Actor`` states
    as truthiness rather than the weaker nullability one.
    ``Actor`` already refuses both at construction, but that is
    code and this is a database: a migration, a repair session or a later store
    that wrote the columns directly meets the constraints instead. Half a rule
    would be worse than none here, because ``read_transcript`` rebuilds every row
    into an ``Actor``, so ONE disagreeing row makes the whole conversation
    unreadable rather than just itself.

    ``body`` is constrained non-empty because an empty one renders to no lines
    at all in assembled context: the event would vanish from the seed prompt
    while still consuming a window slot, so the provider is handed a transcript
    missing a row the operator can read back.

    ``correlation_id`` groups an exchange; ``causation_id`` names the single
    event this one answers. ``kind`` is a plain string, not an enum - nothing
    branches on it yet, and the enum lands with the first caller that does.
    """

    __tablename__ = "event"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id", "event_seq", name="uq_event_conversation_seq"
        ),
        CheckConstraint(_ACTOR_KIND_CHECK, name="ck_event_actor_kind"),
        CheckConstraint(_ACTOR_AGENT_ID_CHECK, name="ck_event_actor_agent_id"),
        CheckConstraint(_EVENT_BODY_CHECK, name="ck_event_body"),
    )

    id: Mapped[str] = mapped_column(primary_key=True)
    conversation_id: Mapped[str]
    event_seq: Mapped[int]
    actor_kind: Mapped[str]
    actor_agent_id: Mapped[str | None]
    kind: Mapped[str]
    body: Mapped[str]
    correlation_id: Mapped[str | None]
    causation_id: Mapped[str | None]
    created_at: Mapped[float]


class DeliveryRow(Base):
    """One channel's attempt at one event.

    The primary key is ``(channel, conversation_id, event_seq)`` and every part
    of it is DERIVED from the event; nothing is minted per attempt. That is the
    whole mechanism - a retry after a crash recomputes the same three values and
    collides, where a per-attempt key would produce a second row and a second
    card. The two id columns are stored as themselves rather than rendered into
    one ``idempotency_key`` string, which would need a parser to go back, turn a
    key prefix scan into a ``LIKE``, and let a conversation id containing the
    separator collide.

    Its leading column is ``channel``, so the key is also the index every
    per-channel read uses.

    Nothing declares the set of channels and no rows are fanned out when an
    event is appended: a row exists only where a channel attempted. So "what has
    this channel not been told" is a left join, which answers correctly for a
    channel that did not exist when the event was written.

    ``state`` is constrained in BOTH halves, the same reasoning ``EventRow``'s
    actor columns carry: against the two states, and ``confirmed_at`` against
    the state. The store rebuilds every row it reads, so one row whose state and
    timestamp disagree is a delivery that claims to have completed at no time.
    ``channel`` is constrained non-empty for a different reason: it is part of
    the key, so an empty one is a delivery filed under a channel nothing polls.
    """

    __tablename__ = "delivery"
    __table_args__ = (
        PrimaryKeyConstraint(
            "channel", "conversation_id", "event_seq", name="pk_delivery"
        ),
        CheckConstraint(_DELIVERY_CHANNEL_CHECK, name="ck_delivery_channel"),
        CheckConstraint(_DELIVERY_STATE_CHECK, name="ck_delivery_state"),
        CheckConstraint(_DELIVERY_CONFIRMED_AT_CHECK, name="ck_delivery_confirmed_at"),
    )

    channel: Mapped[str]
    conversation_id: Mapped[str]
    event_seq: Mapped[int]
    state: Mapped[str]
    claimed_at: Mapped[float]
    confirmed_at: Mapped[float | None]


class ProviderSessionRow(Base):
    """One live binding from a conversation to a provider's own session.

    This is a CACHE, and the row is the cached value: the conversation is the
    source of truth, and the provider session under it can be thrown away and
    rebuilt from assembled context.

    The primary key is ``(conversation_id, backend)`` while the LOOKUP key is the
    ratified triple - ``policy_version`` is a constrained column the read must
    match, not a third key column. With the triple as the key, one binding per
    policy version accumulates and a policy DOWNGRADE finds a superseded row that
    has since missed every event appended under the newer policy, and reads it as
    warm. One live binding per ``(conversation, backend)`` makes a re-seed an
    UPSERT that overwrites, so nothing accumulates and nothing comes back.

    Three CHECKs, each stated as truthiness for the reason
    ``_DELIVERY_CHANNEL_CHECK`` records - the database is what a repair session
    or a later store writing the columns directly meets, and an uninterpolated
    variable produces ``''`` far more readily than it produces a name.
    """

    __tablename__ = "provider_session"
    __table_args__ = (
        PrimaryKeyConstraint("conversation_id", "backend", name="pk_provider_session"),
        CheckConstraint(
            _PROVIDER_SESSION_BACKEND_CHECK, name="ck_provider_session_backend"
        ),
        CheckConstraint(
            _PROVIDER_SESSION_ID_CHECK, name="ck_provider_session_provider_session_id"
        ),
        CheckConstraint(
            _PROVIDER_SESSION_POLICY_VERSION_CHECK,
            name="ck_provider_session_policy_version",
        ),
    )

    conversation_id: Mapped[str]
    backend: Mapped[str]
    policy_version: Mapped[int]
    provider_session_id: Mapped[str]
    seeded_at: Mapped[float]
