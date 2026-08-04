"""What Alembic runs inside, for both callers it has.

Two paths reach here and they are not symmetric:

- **The runtime path.** `scufris/db/migrate.py` puts an already-open
  ``Connection`` on ``config.attributes`` before invoking ``command.upgrade``.
  That connection comes from the app's own engine, so the migration inherits the
  production pragmas - WAL, ``synchronous=FULL``, a busy timeout and foreign
  keys - instead of running the schema change on SQLite's defaults. This is the
  path an operator's startup takes, and it is proven by
  ``test_migration_connection_uses_production_pragmas``.
- **The development path.** ``alembic revision --autogenerate`` from the repo
  root, against the URL in the root ``alembic.ini``. A maintainer's tool for
  writing a revision; never the runtime.

Offline mode (``alembic upgrade --sql``) is refused rather than half-supported:
this schema is applied to a local SQLite file by the process that owns it, so
there is no reviewer to hand a SQL script to, and a silently-wrong emitted
script is worse than an error.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import Connection, engine_from_config, pool

import scufris_chat  # noqa: F401 - registers this package's tables
import scufris_hostctl  # noqa: F401 - registers this package's tables
from scufris.db.migrate import MIGRATION_CONTEXT_OPTS
from scufris.db.models import Base

config = context.config

# Every workspace member declaring a table has to be IMPORTED before this line,
# or its tables are absent from the metadata and autogenerate silently proposes
# dropping them. `test_every_package_model_is_registered` fails when a package
# ships a `models` module that nothing here imports.
target_metadata = Base.metadata


def _configure_and_run(connection: Connection) -> None:
    # Spelled out by key rather than splatted: ``configure`` is typed as a wall
    # of named keyword arguments, and a ``**dict`` does not check against it.
    # The VALUES still come from the one constant, which is what the drift test
    # compares under.
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=MIGRATION_CONTEXT_OPTS["render_as_batch"],
        compare_type=MIGRATION_CONTEXT_OPTS["compare_type"],
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations() -> None:
    if context.is_offline_mode():
        raise RuntimeError(
            "offline migrations are not supported: the state database is a local "
            "SQLite file migrated in place by the process that opens it"
        )

    connection = config.attributes.get("connection")
    if connection is not None:
        _configure_and_run(connection)
        return

    # The development path. NullPool because this engine exists for the length of
    # one CLI command; the runtime never gets here.
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    try:
        with connectable.connect() as conn:
            _configure_and_run(conn)
    finally:
        connectable.dispose()


run_migrations()
