"""The machinery every Scufris package sits on, and nothing domain-specific.

Three things live here, and they are here for the same reason: every package
needs them and none of them owns them.

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

**This module is the whole public surface.** A sibling package imports
`scufris_core`, never `scufris_core.engine` or `scufris_core.base`, and
`test_no_package_imports_a_sibling_private_module` enforces that. The submodules
are free to be reorganised precisely because nothing outside names them.
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
from .logsetup import configure_logging, new_request_id, set_request_id, truncate

__all__ = [
    "DATABASE_FILENAME",
    "FILE_MODE",
    "SIDECAR_SUFFIXES",
    "Base",
    "Database",
    "configure_logging",
    "database_path",
    "new_request_id",
    "open_database",
    "set_request_id",
    "truncate",
]
