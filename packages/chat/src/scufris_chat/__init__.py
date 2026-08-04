"""The conversation Scufris owns: semantic events with typed actors.

Two tables and four functions. A `conversation` is a durable thread that
OUTLIVES any provider session under it, and an `event` is one attributable
utterance inside it - an operator message, an agent report, a system notice -
not one turn. `tasks/20260804-115256/DECISION.md` is why the grain is that fine:
"who said this" decides whether an event may authorize a stop gate, and a
turn-grained row cannot answer it for anything inside the turn.

Three invariants, each held by the schema rather than by a caller's care:

- `event_seq` is per-conversation, gap-free and strictly increasing. It is
  assigned as `COALESCE(MAX(event_seq), 0) + 1` inside the caller's open
  transaction, so a rolled-back write consumes no number and two concurrent
  writers cannot claim one.
- The author is a typed `Actor` over four kinds. `Actor.parse` is the one
  boundary a string crosses, and a CHECK constraint on `actor_kind` is what a
  hand-written INSERT meets instead.
- The writer takes an OPEN `Connection` and never opens one, so a state change
  and the event describing it commit together or not at all.

**This module is the whole public surface.** A sibling imports `scufris_chat`,
never `scufris_chat.store` or `scufris_chat.models`, and
`test_no_package_imports_a_sibling_private_module` enforces it. No row class is
exported: `ConversationRow` and `EventRow` are private, as
`packages/hostctl`'s are.

`packages/chat/src/scufris_chat/README.md` is the longer version.
"""

from __future__ import annotations

from .actors import Actor, ActorKind
from .store import (
    ConversationRecord,
    EventRecord,
    append_event,
    causing_event,
    create_conversation,
    read_transcript,
)

__all__ = [
    "Actor",
    "ActorKind",
    "ConversationRecord",
    "EventRecord",
    "append_event",
    "causing_event",
    "create_conversation",
    "read_transcript",
]
