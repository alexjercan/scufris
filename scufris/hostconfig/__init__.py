"""Turning a reviewed commit into a configuration the operator can approve.

This is the unprivileged half of the declarative config-change flow, and the
division of labour is the whole design:

- The configuration repository is **a project**. An agent changes it the way an
  agent changes any project - a worktree, a commit, a review. Nothing in this
  package edits, commits, branches or writes to it. It reads git and it builds.
- The build runs as the OPERATOR, never as root. Nix evaluation reads files with
  the evaluating user's privileges, so a configuration evaluated as root could
  read a host key or a sops age key into a derivation output; as ``alex`` that
  read simply fails.
- The store path that gets activated is built HERE, from a revision resolved
  HERE. A caller never supplies one - ``/api/host/actions`` refuses
  ``kind=activate`` outright - because otherwise the model would choose what gets
  activated and the closure diff would faithfully describe whatever it chose.

The build addresses the repository as ``git+file://<repo>?ref=<ref>&rev=<rev>``,
which is not a detail: nix then takes the tree from the COMMIT rather than from
the working tree, so what gets built is an identified revision, uncommitted files
are structurally excluded (and said so, rather than silently ignored), and this
package cannot dirty the repository even by accident.

| Module | Owns |
|--------|------|
| `models` | what a change is, and what a build publishes on the bus |
| `resolve` | reading git, and addressing the build at an identified revision |
| `changes` | the bounded registry in the state database, and the build that fills it |
| `render` | one change as plain text, for an agent to relay verbatim |

This module is the package's public surface; the submodules import each other
directly rather than through it.
"""

from __future__ import annotations

from .changes import (
    ChangeInFlight,
    ConfigChangeBuilder,
    ConfigChangeStore,
    UnknownChange,
)
from .models import (
    ChangeState,
    ConfigBuildBus,
    ConfigBuildDone,
    ConfigBuildError,
    ConfigBuildEvent,
    ConfigBuildOutput,
    ConfigChange,
    ConfigChangeRefused,
    ConfigSupervisor,
    Resolved,
    config_supervisor,
)
from .render import render_change
from .resolve import (
    build_argv,
    check_attr,
    default_attr,
    flake_url,
    resolve,
    toplevel_from,
)

__all__ = [
    "ChangeInFlight",
    "ChangeState",
    "ConfigBuildBus",
    "ConfigBuildDone",
    "ConfigBuildError",
    "ConfigBuildEvent",
    "ConfigBuildOutput",
    "ConfigChange",
    "ConfigChangeBuilder",
    "ConfigChangeRefused",
    "ConfigChangeStore",
    "ConfigSupervisor",
    "Resolved",
    "UnknownChange",
    "build_argv",
    "check_attr",
    "config_supervisor",
    "default_attr",
    "flake_url",
    "render_change",
    "resolve",
    "toplevel_from",
]
