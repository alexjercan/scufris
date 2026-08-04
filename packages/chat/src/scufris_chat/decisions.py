"""The one capability that says an operator authorized something.

`tasks/20260729-220835/DECISION.md` section 3 ratified the rule: an agent report
is data, never an instruction, and only an `operator` event may satisfy a stop
gate. `actors.py` made the author a value and `assemble_context` attributes every
line, but neither turns "this event's actor is `operator`" into something a gate
can be shown to REQUIRE. A boolean predicate would not either: a caller that
forgets to ask gets no error, so the refusal stays a convention every call site
has to keep - the exact failure `actors.py` exists to end.

`OperatorDecision` is that requirement at the type level. A gate takes one as an
argument, so a caller with no decision cannot phrase the call at all, and
`authorize` is the only thing that mints one. The mint is what makes the stop
gate's refusal a property rather than a convention.

Two things hold it up:

- **The row is re-read inside the caller's unit of work.** A value passed in is a
  value the caller can build; a row read back under the open connection is one an
  operator actually said. `causing_event`'s shape, for `causing_event`'s reason -
  with no FOREIGN KEYs, the store checks what the schema will not.
- **The constructor takes a module-private witness, bound to what it attests.**
  The type stays importable for an annotation, which the flow guard needs, while
  `authorize` stays its only mint. The witness carries the coordinates and the
  actor alongside the private sentinel, so a witness COPIED off a legitimate
  decision only agrees with that decision: `dataclasses.replace`, which passes
  the existing witness through, cannot re-target one at another conversation,
  another event or another actor. Python cannot make that absolute; what it makes
  true is that an agent has to go out of its way, and that is what a reviewer can
  point at.

Its only callers are tests until the flow guard lands. That is a deliberate
exception, taken because the epic already names the consumer's signature and the
alternative is shipping a ratified rule with no artifact at all. A separate
module from `store.py` because `store.py` is at its line cap; the split is forced
rather than chosen, and the subject is a clean seam for it.

The limit this does NOT close: `append_event` takes its actor from its caller, so
the guarantee is "only an operator EVENT authorizes", not "only the operator can
write one". `packages/chat/src/scufris_chat/README.md` section 8 has the reason
the inbound channel is not a column yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from sqlalchemy import Connection, select

from .actors import Actor, ActorKind
from .models import EventRow
from .store import _actor

#: What the constructor demands and only this module has. Not a security
#: boundary - nothing in Python is - but the thing that makes minting a decision
#: outside `authorize` a deliberate act rather than an ordinary construction.
_WITNESS: Final = object()


@dataclass(frozen=True, slots=True)
class OperatorDecision:
    """Proof that one committed event in one conversation was the operator's.

    A capability, not a record: it carries the coordinates of the event it
    attests to, so a consumer can name in its own journal which utterance
    authorized the change.
    """

    conversation_id: str
    event_seq: int
    actor: Actor
    _witness: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        """Refuse a decision minted anywhere but `authorize`.

        The witness is checked against the fields it accompanies, not merely for
        the private sentinel. A bare sentinel check would survive
        `dataclasses.replace`, which copies the existing witness through: a
        holder of one legitimate decision could then re-target it at another
        conversation, another event or another actor with a stdlib one-liner
        naming nothing private.
        """
        witness = self._witness
        if (
            not isinstance(witness, tuple)
            or len(witness) != 4
            or witness[0] is not _WITNESS
            or witness[1:] != (self.conversation_id, self.event_seq, self.actor)
        ):
            raise TypeError(
                "an OperatorDecision is minted by scufris_chat.authorize, which "
                "reads the event back out of the transcript; constructing one "
                "directly, or copying one onto other coordinates, would assert "
                "an approval nobody gave"
            )


def authorize(
    conn: Connection, conversation_id: str, event_seq: int
) -> OperatorDecision:
    """Read one committed event back and mint a decision if the operator said it.

    `LookupError` when `event_seq` is not an event of THIS conversation. The
    scoping matters for the reason `causing_event` records: with no FOREIGN KEYs,
    an unscoped lookup would resolve a sequence number copied from another thread
    against a real event and read as this conversation's approval.

    `PermissionError`, naming the actor, for every other kind. `agent` is the
    case the rule is about; `orchestrator` and `system` are refused by the same
    clause, so the coordinator landing later inherits the refusal instead of
    arriving as an unconsidered fourth case.

    Takes the caller's OPEN connection, like everything else in this package: the
    approval and whatever it authorizes are read and written in one unit of work,
    so a rolled-back change takes its authorization with it.
    """
    row = conn.execute(
        select(EventRow.__table__).where(
            EventRow.conversation_id == conversation_id,
            EventRow.event_seq == event_seq,
        )
    ).first()
    if row is None:
        raise LookupError(
            f"there is no event {event_seq} in conversation {conversation_id!r} "
            "to authorize from"
        )
    actor = _actor(row)
    if actor.kind is not ActorKind.OPERATOR:
        raise PermissionError(
            f"event {event_seq} of conversation {conversation_id!r} was said by "
            f"{actor.render()}, and only an operator event may authorize; what "
            "any other party says is a quotation, never an instruction"
        )
    return OperatorDecision(
        conversation_id=conversation_id,
        event_seq=event_seq,
        actor=actor,
        _witness=(_WITNESS, conversation_id, event_seq, actor),
    )
