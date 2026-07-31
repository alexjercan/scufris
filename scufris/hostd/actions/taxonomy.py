"""The verb set and its risk classification, plus the refusal they raise.

Every other module in this package imports from here: the verbs are the closed
set nothing may extend, and the risk class is a property of the verb rather than
of a caller's opinion about it.
"""

from __future__ import annotations

from enum import StrEnum


class RiskClass(StrEnum):
    """Which class of the taxonomy an action belongs to."""

    R1 = "r1"  # service control: reversible by restoring recorded state
    R2 = "r2"  # disposable cleanup: ONE-WAY
    R3 = "r3"  # declarative config change: reversible to a recorded generation


class ActionKind(StrEnum):
    """The complete set of verbs this helper implements. Nothing else exists."""

    UNIT_START = "unit_start"
    UNIT_STOP = "unit_stop"
    UNIT_RESTART = "unit_restart"
    UNIT_RELOAD = "unit_reload"
    GC_OLDER_THAN = "gc_older_than"
    GC_STORE = "gc_store"
    ACTIVATE = "activate"
    ROLLBACK = "rollback"


RISK_OF: dict[ActionKind, RiskClass] = {
    ActionKind.UNIT_START: RiskClass.R1,
    ActionKind.UNIT_STOP: RiskClass.R1,
    ActionKind.UNIT_RESTART: RiskClass.R1,
    ActionKind.UNIT_RELOAD: RiskClass.R1,
    ActionKind.GC_OLDER_THAN: RiskClass.R2,
    ActionKind.GC_STORE: RiskClass.R2,
    ActionKind.ACTIVATE: RiskClass.R3,
    ActionKind.ROLLBACK: RiskClass.R3,
}

UNIT_KINDS: frozenset[ActionKind] = frozenset(
    {
        ActionKind.UNIT_START,
        ActionKind.UNIT_STOP,
        ActionKind.UNIT_RESTART,
        ActionKind.UNIT_RELOAD,
    }
)

R3_KINDS: frozenset[ActionKind] = frozenset(
    {
        ActionKind.ACTIVATE,
        ActionKind.ROLLBACK,
    }
)


class ActionRefused(Exception):
    """An action this helper will not build an argv for.

    Raised by validation, never by execution: by the time an action has a
    ``Plan`` it is a command the helper is willing to run once approved.
    """
