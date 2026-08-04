"""The conversation Scufris owns: semantic events with typed actors.

Four tables and ten functions. A `conversation` is a durable thread that
OUTLIVES any provider session under it, and an `event` is one attributable
utterance inside it - an operator message, an agent report, a system notice -
not one turn. The grain is that fine because "who said this" decides whether an
event may authorize a stop gate, and a turn-grained row cannot answer it for
anything inside the turn.

The invariants, each held by the schema rather than by a caller's care:

- `event_seq` is per-conversation, gap-free and strictly increasing. It is
  assigned as `COALESCE(MAX(event_seq), 0) + 1` inside the caller's open
  transaction, so a rolled-back write consumes no number and two concurrent
  writers cannot claim one.
- The author is a typed `Actor` over four kinds. `Actor.parse` is the one
  boundary a string crosses, and a CHECK constraint on `actor_kind` is what a
  hand-written INSERT meets instead.
- The writer takes an OPEN `Connection` and never opens one, so a state change
  and the event describing it commit together or not at all.
- A `delivery` is keyed by `(channel, conversation_id, event_seq)`, derived from
  the event and never minted per attempt, so a retry after a crash collides
  rather than posting a second card. Idempotency is a property of the storage
  layer, which is what stops a channel added later from forgetting it.
- A `provider_session` is a CACHE, never a source of truth. One live binding per
  `(conversation, backend)`, looked up under a `policy_version` it must match,
  re-seeded from `assemble_context` on a miss - and a miss is normal, so nothing
  on that path raises. This is what makes the conversation survive `/new`, a
  compaction, a backend switch and a restart.

**This module is the whole public surface.** A sibling imports `scufris_chat`,
never `scufris_chat.store` or `scufris_chat.models`, and
`test_no_package_imports_a_sibling_private_module` enforces it. No row class is
exported: `ConversationRow`, `EventRow`, `DeliveryRow` and `ProviderSessionRow`
are private, as `packages/hostctl`'s are. `DeliveryState` is exported because it
is the enum the `delivery` CHECK constraint is rendered from, and
`tests/test_db_schema.py::test_migrated_delivery_check_lists_exactly_the_declared_states`
is what reads a migrated database back against it.

`packages/chat/src/scufris_chat/README.md` is the longer version.
"""

from __future__ import annotations

from .actors import Actor, ActorKind
from .models import DeliveryState
from .store import (
    CONTEXT_POLICY_VERSION,
    CONTEXT_WINDOW_EVENTS,
    ConversationRecord,
    EventRecord,
    SessionBinding,
    append_event,
    assemble_context,
    bind_session,
    cached_session,
    causing_event,
    claim_delivery,
    confirm_delivery,
    create_conversation,
    pending_events,
    read_transcript,
)

__all__ = [
    "CONTEXT_POLICY_VERSION",
    "CONTEXT_WINDOW_EVENTS",
    "Actor",
    "ActorKind",
    "ConversationRecord",
    "DeliveryState",
    "EventRecord",
    "SessionBinding",
    "append_event",
    "assemble_context",
    "bind_session",
    "cached_session",
    "causing_event",
    "claim_delivery",
    "confirm_delivery",
    "create_conversation",
    "pending_events",
    "read_transcript",
]
