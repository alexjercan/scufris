"""The typed arguments each verb takes, and the plan they are turned into.

A caller names a verb and typed arguments; it never supplies a command. The
argument models are what "typed" means, and ``Plan`` is what the operator
approves - an argv, not a description of one.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .taxonomy import ActionKind, ActionRefused, RiskClass


class UnitArgs(BaseModel):
    """The single argument every R1 verb takes."""

    unit: str


class GcOlderThanArgs(BaseModel):
    """Trim system generations (and the store paths they held) by age."""

    # Bounded on both ends: 0 would mean "everything including today", and a
    # value past ten years is a typo rather than an intent.
    days: int = Field(ge=1, le=3650)


class GcStoreArgs(BaseModel):
    """Delete store paths that are already dead. Touches no generation."""


class ActivateArgs(BaseModel):
    """Switch the system to an already-built configuration.

    ``toplevel`` is a store path the APP built from ``rev`` in ``repo``; the two
    provenance fields are recorded so the audit answers "which revision is this
    machine running" without trusting a description of it. They grant nothing and
    are never interpolated into a command.
    """

    toplevel: str
    repo: str = ""
    rev: str = ""


class RollbackArgs(BaseModel):
    """Return the system to a generation that already exists.

    A NUMBER, never a path: the helper resolves which store path that generation
    is, so "roll back" cannot be spelled as "activate this other thing".
    """

    generation: int = Field(ge=1)


ActionArgs = UnitArgs | GcOlderThanArgs | GcStoreArgs | ActivateArgs | RollbackArgs

_ARGS_MODEL: dict[ActionKind, type[BaseModel]] = {
    ActionKind.UNIT_START: UnitArgs,
    ActionKind.UNIT_STOP: UnitArgs,
    ActionKind.UNIT_RESTART: UnitArgs,
    ActionKind.UNIT_RELOAD: UnitArgs,
    ActionKind.GC_OLDER_THAN: GcOlderThanArgs,
    ActionKind.GC_STORE: GcStoreArgs,
    ActionKind.ACTIVATE: ActivateArgs,
    ActionKind.ROLLBACK: RollbackArgs,
}


class Step(BaseModel):
    """One command in a plan, with the wall clock it is allowed and a label.

    Steps exist because activation is not one command: the system profile is
    pointed at the built configuration, and THEN that configuration is switched
    to. Modelling that as a sequence is what lets the record say which half
    happened when the second one fails - see ``Plan.partial_detail``.
    """

    argv: list[str]
    # What this step does, in the operator's language. Rendered next to the
    # command in the preview and carried into the audit.
    label: str = ""
    timeout: float = 60.0


class Plan(BaseModel):
    """A validated action: what will run, and what the operator is agreeing to.

    Every ``Step.argv`` is built HERE, from the verb and the validated
    arguments, and carried on the plan so the preview, the audit record and the
    execution all name the same commands - the operator approves an argv, not a
    description of one.
    """

    kind: ActionKind
    risk: RiskClass
    args: dict[str, object] = Field(default_factory=dict)
    steps: list[Step]
    # A one-line statement of what this does, in the operator's language.
    summary: str
    # Set for R2: the generations this action would remove, resolved before the
    # command runs so the floor is enforced by us and not by a flag.
    generations_removed: list[int] = Field(default_factory=list)
    # What it means when a step after the first one fails, when that is not
    # simply "nothing happened". Empty for a single-step plan.
    partial_detail: str = ""
    # What a cancellation actually achieves for this class, when it is not "the
    # process group was signalled". R3 sets it because the switch runs in a
    # transient systemd unit that outlives this helper by design.
    cancel_detail: str = ""

    @property
    def argvs(self) -> list[list[str]]:
        """Every command, in order. Convenience for rendering."""
        return [step.argv for step in self.steps]


def parse_args(kind: ActionKind, raw: dict[str, object]) -> BaseModel:
    """Validate ``raw`` against the argument model for ``kind``.

    A verb that is not in the table cannot get here - ``ActionKind`` is a
    closed enum and an unknown string fails to parse at the protocol boundary.
    """
    model = _ARGS_MODEL[kind]
    try:
        return model.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - a pydantic error is a refusal
        raise ActionRefused(f"invalid arguments for {kind}: {exc}") from exc
