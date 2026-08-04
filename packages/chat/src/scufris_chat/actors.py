"""Who said it, as a value rather than a convention.

The alternative this replaces is an actor string compared at each call site,
which costs nothing to write and is unfakeable nowhere: the stop gate's rule is
that only an ``operator`` event may authorize, and a string comparison spread
over call sites has no single place that can be shown to hold it.

Four kinds, from ``tasks/20260729-220835/DECISION.md`` section 3.
``orchestrator`` is named separately from ``agent`` even though nothing writes
one until the coordinator lands: folding it in would change the meaning of a
ratified record to save one enum member, and the fold would have to be
re-litigated by the lane that adds the coordinator.

The wire form is a bare kind, or ``agent:<id>``. ``parse`` is the one place a
string crosses into an ``Actor``, and it is one-way on purpose: the store keeps
the kind and the id in two columns, so nothing writes the wire form and there is
no renderer to keep in step with the parse. ``EventRow``'s two CHECK constraints
hold the same rule ``Actor`` does, so an INSERT that goes around the parse meets
the database instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

AGENT_ID_SEPARATOR = ":"


class ActorKind(StrEnum):
    """The four things that can author an event."""

    OPERATOR = "operator"
    ORCHESTRATOR = "orchestrator"
    AGENT = "agent"
    SYSTEM = "system"


ACTOR_KIND_VALUES: tuple[str, ...] = tuple(kind.value for kind in ActorKind)


@dataclass(frozen=True, slots=True)
class Actor:
    """A kind and, for `agent`, which agent."""

    kind: ActorKind
    agent_id: str | None = None

    def __post_init__(self) -> None:
        """Refuse an actor whose kind and id disagree, however it was built.

        Here rather than in `parse` alone: `parse` is the one boundary a STRING
        crosses, but a caller that names the kind directly - which every writer
        in this repository does - would otherwise get no check at all. An `agent`
        with no id cannot be told from another agent, and an id on the other
        three is a claim the `actor_kind` column has nowhere to keep.
        """
        if self.kind is ActorKind.AGENT:
            if not self.agent_id:
                raise ValueError(
                    "an agent actor needs an agent id: it is what distinguishes "
                    "one agent's events from another's"
                )
        elif self.agent_id is not None:
            raise ValueError(
                f"a {self.kind} actor takes no agent id, got {self.agent_id!r}; "
                "only agent events name an agent"
            )

    @classmethod
    def parse(cls, text: str) -> Actor:
        """Read the wire form: a bare kind, or `agent:<id>`.

        The ONE place a string becomes an actor. An unknown kind raises here, so
        no caller downstream has to consider one.
        """
        kind, separator, agent_id = text.partition(AGENT_ID_SEPARATOR)
        try:
            parsed = ActorKind(kind)
        except ValueError as exc:
            raise ValueError(
                f"unknown actor kind {kind!r}; the kinds are "
                f"{', '.join(ACTOR_KIND_VALUES)}"
            ) from exc
        if separator and not agent_id:
            raise ValueError(
                f"malformed actor {text!r}: the separator promises an agent id "
                "and names none"
            )
        return cls(parsed, agent_id or None)
