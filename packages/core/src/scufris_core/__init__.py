"""The machinery every Scufris package sits on, and nothing domain-specific.

Five modules live here, and they are here for the same reason: every package
needs them and none of them owns them. `CORE_MODULES` in
`tests/test_package_boundaries.py` is the enforced allowlist; this list is its
justification.

- The transactional boundary - `Database`, `Database.transaction()`,
  `open_database` - is in `engine`. One SQLite file, one unit of work, the
  pragma hook and the nesting guard. The rules a caller has to keep are in that
  module's docstring and they have not changed by moving here.
- `Base` is in `base`. Each package declares its own rows against it; `core`
  declares none.
- `logsetup` is the one judgement call in the list: it is generic (87 lines over
  `logging`, `uuid` and `contextvars`), it is shared by modules the carve splits
  across four packages, and no package can become a workspace member while it
  imports a root module for its logging.
- `eventbus` is here because it has a second consumer in another distribution.
  `EventBus` is generic over its payload and imports nothing but the standard
  library, and `hostctl` publishes host-action and config-change events on one.
  Left at the root, `hostctl` would have to import the app to get a bus.
- `supervisor` is the generic half of the run engine over that bus - `Supervisor`,
  `RunState`, `RunPhase` - split from the agent's instantiation of it because
  `hostctl` supervises applies and config builds and the event type is a
  parameter. It is why `core` is no longer sqlalchemy-only; see the amendment in
  `tasks/20260803-213242/DECISION.md`.

**This module is the whole public surface.** A sibling package imports
`scufris_core`, never `scufris_core.engine` or `scufris_core.base`, and
`test_no_package_imports_a_sibling_private_module` enforces that - in sibling
TESTS as well as sibling source. Only `packages/core/tests` may name a submodule
here. The submodules are free to be reorganised precisely because nothing
outside names them.
"""

from __future__ import annotations

from .base import Base
from .engine import (
    DATABASE_FILENAME,
    FILE_MODE,
    SIDECAR_SUFFIXES,
    Database,
    database_path,
    open_database,
)
from .eventbus import EventBus
from .logsetup import configure_logging, new_request_id, set_request_id, truncate
from .supervisor import AgentRunStalled, RunPhase, RunState, Supervisor

__all__ = [
    "DATABASE_FILENAME",
    "FILE_MODE",
    "SIDECAR_SUFFIXES",
    "AgentRunStalled",
    "Base",
    "Database",
    "EventBus",
    "RunPhase",
    "RunState",
    "Supervisor",
    "configure_logging",
    "database_path",
    "new_request_id",
    "open_database",
    "set_request_id",
    "truncate",
]
