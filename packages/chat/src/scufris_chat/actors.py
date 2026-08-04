"""Who said it, as a value rather than a convention.

The alternative this replaces is an actor string compared at each call site,
which costs nothing to write and is unfakeable nowhere: the stop gate's rule is
that only an ``operator`` event may authorize, and a string comparison spread
over call sites has no single place that can be shown to hold it.

Four kinds. ``orchestrator`` is named separately from ``agent`` even though
nothing writes one until the coordinator lands: folding it in would change the
meaning of a ratified record to save one enum member, and the fold would have to
be re-litigated by the lane that adds the coordinator.

The wire form is a bare kind, or ``agent:<id>``. ``parse`` is the one place a
string crosses into an ``Actor``, and ``render`` is the one place one crosses
back. Nothing round-trips through the pair by accident: the store keeps the kind
and the id in two columns, and the only writer of the wire form is the assembled
context a provider is seeded with, where every line has to name its author. The
two live next to each other so they cannot drift. ``EventRow``'s two CHECK
constraints hold the same rule ``Actor`` does, so an INSERT that goes around the
parse meets the database instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

AGENT_ID_SEPARATOR = ":"

#: What an agent id may not contain. The id is interpolated into a LINE of the
#: assembled prompt, where a line terminator in it forges the very attribution
#: the per-line format exists to make unforgeable: an `agent_id` of
#: `"bot\noperator"` renders a bare `operator: ...` line under a preamble
#: declaring the operator's lines to be instructions.
#:
#: The alphabet is the CONSUMER's, not ASCII's. What decides where a line ends
#: is `str.splitlines`, and it breaks on U+0085, U+2028 and U+2029 as readily as
#: on `\n`, so a set chosen from the id's own domain would be one character short
#: of complete. Every C0 control and DEL is refused as well: none belongs in an
#: id, and the wider set costs nothing. Listed rather than computed because
#: `models.py` has to mirror it in SQL, where it cannot be;
#: `test_the_forbidden_alphabet_covers_every_line_break` is what holds the list
#: to what `str.splitlines` actually does.
_FORBIDDEN_AGENT_ID_ORDINALS = frozenset(range(0x20)) | {0x7F, 0x85, 0x2028, 0x2029}


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
        """Refuse an actor whose kind and id disagree, or whose id can forge a line.

        Here rather than in `parse` alone: `parse` is the one boundary a STRING
        crosses, but a caller that names the kind directly - which every writer
        in this repository does - would otherwise get no check at all. An `agent`
        with no id cannot be told from another agent, and an id on the other
        three is a claim the `actor_kind` column has nowhere to keep.

        The line-safety check is what makes `render` safe to interpolate into a
        line: the attribution is as much a part of the assembled prompt as the
        body is, and a hostile body is refused by the format while a hostile id
        would be refused by nothing.
        """
        if self.kind is ActorKind.AGENT:
            if not self.agent_id:
                raise ValueError(
                    "an agent actor needs an agent id: it is what distinguishes "
                    "one agent's events from another's"
                )
            if any(
                ord(character) in _FORBIDDEN_AGENT_ID_ORDINALS
                for character in self.agent_id
            ):
                raise ValueError(
                    f"an agent id may not contain control characters or line "
                    f"separators, got {self.agent_id!r}: it is rendered into a "
                    "line of assembled context, where a line break forges an "
                    "attribution"
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

    def render(self) -> str:
        """The wire form: a bare kind, or `agent:<id>`.

        The inverse of `parse`, and its only caller is the attribution assembled
        context is rendered with. `__post_init__` is what makes it total - an
        `agent` always has an id and nothing else ever does - so there is no
        actor this cannot name.
        """
        if self.agent_id is None:
            return self.kind.value
        return f"{self.kind.value}{AGENT_ID_SEPARATOR}{self.agent_id}"
