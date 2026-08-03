"""The action taxonomy: the verb set IS the risk classification.

Five risk classes exist; three of them have verbs here:

- **R1 service control** - ``unit_start``, ``unit_stop``, ``unit_restart``,
  ``unit_reload``. Reversible by restoring the recorded prior unit state.
- **R2 disposable cleanup** - ``gc_older_than``, ``gc_store``. One-way.
- **R3 declarative config change** - ``activate``, ``rollback``. Reversible by
  activating a recorded generation.

R0 needs no privilege and lives in ``scufris_host``. **R4 has no verb, and that
absence IS the enforcement** - partitioning, user and key material, the firewall,
and anything targeting scufris itself have no code path here rather than a deny
check that could have a bug.

Two properties this package is responsible for, both of which have already been
paid for once in this repo:

1. **The helper builds every argv.** A caller names a verb and typed arguments;
   it never supplies a command. There is no shell verb at any privilege under
   any approval.
2. **An argument may not become a flag.** ``shell=False`` with an explicit argv
   answers a different question - measured: a unit named
   ``-Hsomeone@host`` made systemctl open an outbound SSH connection. Every
   value is charset-validated, a leading ``-`` is refused explicitly, and
   positionals are passed after ``--``.

R3 adds a third, and it is the one the whole epic turns on: **the store path that
gets activated is not a caller's to choose.** ``activate`` takes a toplevel, but
the only code path that reaches it builds that path itself from a git revision it
resolved (``scufris/hostconfig``), the propose surfaces refuse the verb
outright, and this package still validates the path structurally before it will
name it in a command.

| Module | Owns |
|--------|------|
| `taxonomy` | the verbs, their risk classes, and `ActionRefused` |
| `models` | the typed arguments each verb takes, and `Step`/`Plan` |
| `validate` | every value that reaches an argv, and the deny-lists |
| `plans` | `build_plan`: a validated verb turned into exact commands |

This module is the package's public surface; the submodules import each other
directly rather than through it.
"""

from __future__ import annotations

from .models import (
    ActionArgs,
    ActivateArgs,
    GcOlderThanArgs,
    GcStoreArgs,
    Plan,
    RollbackArgs,
    Step,
    UnitArgs,
    parse_args,
)
from .plans import (
    GENERATION_TIMEOUT,
    PROFILE_TIMEOUT,
    PROTECTED_GENERATIONS,
    SWITCH_TIMEOUT,
    SWITCH_UNIT,
    SYSTEM_PROFILE,
    build_plan,
    generation_link,
    generations_older_than,
)
from .taxonomy import (
    R3_KINDS,
    RISK_OF,
    UNIT_KINDS,
    ActionKind,
    ActionRefused,
    RiskClass,
)
from .validate import (
    DENIED_UNIT_STEMS,
    PATH_INFO_TIMEOUT,
    normalise_unit,
    validate_toplevel,
)

__all__ = [
    "DENIED_UNIT_STEMS",
    "GENERATION_TIMEOUT",
    "PATH_INFO_TIMEOUT",
    "PROFILE_TIMEOUT",
    "PROTECTED_GENERATIONS",
    "RISK_OF",
    "R3_KINDS",
    "SWITCH_TIMEOUT",
    "SWITCH_UNIT",
    "SYSTEM_PROFILE",
    "UNIT_KINDS",
    "ActionArgs",
    "ActionKind",
    "ActionRefused",
    "ActivateArgs",
    "GcOlderThanArgs",
    "GcStoreArgs",
    "Plan",
    "RiskClass",
    "RollbackArgs",
    "Step",
    "UnitArgs",
    "build_plan",
    "generation_link",
    "generations_older_than",
    "normalise_unit",
    "parse_args",
    "validate_toplevel",
]
