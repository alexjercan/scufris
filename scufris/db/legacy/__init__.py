"""Read an operator's legacy per-store JSON files into the database, once.

The policy lives here rather than in each store's importer:

- **Backed up first.** The source is copied to ``<name>.pre-sqlite.bak`` before
  it is read, at 0600 from creation - the later store migrations bring auth
  sessions through this same function.
- **Never deleted.** Nothing here removes a legacy file. The operator does that,
  once they are satisfied, and that deletion is what makes the move one-way.
- **Damaged is refused, never treated as empty.** A file that does not parse is
  named with its line, column and the parser's own message; a record that fails
  its pydantic model fails the WHOLE import. There is no tolerant loader on this
  path: the JSON store this replaces logged and skipped such a record, and
  importing under that rule would leave a database quietly missing records the
  operator can still see in their JSON.
- **All or nothing, once.** One source is imported inside one
  ``Database.transaction()`` that also writes its ``legacy_import`` row, so a
  failure anywhere leaves no rows AND no gate: the operator repairs the file and
  the retry starts from the beginning.

The gate is a table rather than a schema version, and the import cannot ride a
schema revision: it needs the state directory and the pydantic models to do its
job, and neither belongs inside a migration.

:func:`import_legacy_state` is the whole state directory in one call, and
``scufris.db.open_state_database`` is its only call site: it runs the import at
startup, after the migration and ahead of the first store read. Both orderings
matter - importing before the migration would write through models the schema
does not have yet, and importing after a store had already read would show an
operator an empty database while their projects sat in ``projects.json``.

Host actions have NO legacy source here, and their absence is deliberate rather
than an oversight: the store was memory-only, rebuilt on each boot from the root
helper's queue, so there is no file an operator could have and nothing to import.
The ``host_action`` table is the first durable home those records have had.
"""

from __future__ import annotations

from pathlib import Path

from ..engine import Database
from .gate import (
    BACKUP_SUFFIX,
    LegacyImportRefused,
    Loader,
    backup_path,
    import_legacy_file,
)
from .loaders import (
    load_agents,
    load_auth_sessions,
    load_digests,
    load_outcomes,
    load_projects,
    load_reasoning,
    load_schedules,
    load_sessions,
    load_settings,
)

__all__ = [
    "BACKUP_SUFFIX",
    "LegacyImportRefused",
    "backup_path",
    "import_legacy_file",
    "import_legacy_state",
]

PROJECTS_FILENAME = "projects.json"
AGENTS_FILENAME = "agents.json"
SESSIONS_FILENAME = "sessions.json"
OUTCOMES_FILENAME = "outcomes.json"
SETTINGS_FILENAME = "settings.json"
AUTH_SESSIONS_FILENAME = "auth_sessions.json"
SCHEDULES_FILENAME = "schedules.json"
DIGESTS_FILENAME = "digests.json"
REASONING_DIRNAME = "reasoning"


# --- the whole state directory ------------------------------------------------
#
# One policy over every source, and one ORDER that matters: `sessions.json` is
# imported before `agents.json` because the agent loader migrates a pre-registry
# `session_id` off the record and into the session tables only when no mapping
# exists, and "an existing mapping wins" is only meaningful if the mappings are
# already there.


def import_legacy_state(db: Database, state_dir: Path) -> None:
    """Import every legacy JSON source in ``state_dir``, once each.

    Each source is its own all-or-nothing import with its own gate row, and a
    refusal does not stop the sources AFTER it: every one is attempted, and the
    refusals are raised together at the end. So a damaged ``schedules.json`` still
    fails startup - the operator must repair it, and a silently skipped source
    would be exactly the tolerant loader this package refuses to be - but the
    others are in with their gate rows, and the retry after the repair reads only
    the file that was damaged.

    Running the later sources anyway is safe because each writes its own tables
    inside its own transaction, and because the one ORDER that matters survives
    being broken. That order is sessions before agents, so an existing mapping
    wins over a pre-registry ``session_id`` on the record; refuse
    ``sessions.json`` and ``load_agents`` migrates the record's own id instead,
    which is the right answer while no mappings exist. It is what happens on the
    REPAIR that makes this a property rather than a hope: the repaired file
    arrives for agents that now have rows, so ``load_sessions`` REPLACES a
    mapping and its history rather than inserting beside it (review round 1,
    R1.1). Inserting there was a ``UNIQUE constraint failed`` no retry could
    clear, because ``agents.json`` is already gated and its write is never
    replayed.

    There is no host-action source: see this module's docstring.
    """
    state_dir = Path(state_dir)
    sources: list[tuple[Path, Loader, str | None]] = [
        (state_dir / PROJECTS_FILENAME, load_projects, None),
        (state_dir / SESSIONS_FILENAME, load_sessions, None),
        (state_dir / AGENTS_FILENAME, load_agents, None),
        (state_dir / OUTCOMES_FILENAME, load_outcomes, None),
        (state_dir / SETTINGS_FILENAME, load_settings, None),
        (state_dir / AUTH_SESSIONS_FILENAME, load_auth_sessions, None),
        (state_dir / SCHEDULES_FILENAME, load_schedules, None),
        (state_dir / DIGESTS_FILENAME, load_digests, None),
    ]
    sources += [
        (sidecar, load_reasoning, f"{REASONING_DIRNAME}/{sidecar.name}")
        for sidecar in sorted((state_dir / REASONING_DIRNAME).glob("*.json"))
    ]
    refused: list[LegacyImportRefused] = []
    for path, loader, key in sources:
        try:
            import_legacy_file(db, path, loader, key=key)
        except LegacyImportRefused as exc:
            refused.append(exc)
    if refused:
        joined = "\n".join(str(exc) for exc in refused)
        raise LegacyImportRefused(joined) from refused[0]
