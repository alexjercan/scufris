#!/usr/bin/env python
"""The unit of work, on its own, with no application around it.

    python examples/core_unit_of_work.py

`scufris_core` is the whole persistence machinery and none of the schema, and
the point of carving it out is that this claim can be DEMONSTRATED rather than
asserted: this script opens a real database, declares its own table against the
shared `Base`, writes inside one transaction, throws inside a second, and counts
what survived. It imports `scufris` nowhere, opens no socket and needs no host.

    1. open           - a real SQLite file under a temporary directory
    2. commit         - two rows written in ONE `Database.transaction()`
    3. roll back      - a third row, then an exception; the transaction unwinds
    4. count          - two rows, which is the claim

The toy `note` table is this script's, not `core`'s. That is the shape every
package takes: `core` owns `Base` and the transaction, each package owns its own
rows. `packages/core/src/scufris_core/engine.py` has the rules a real caller has
to keep - above all that a transaction never spans an `await`.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Run from a checkout without installing it. Only the member's `src` is needed:
# this script imports `scufris_core`, never `scufris`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "core" / "src"))

from sqlalchemy import func, insert, select  # noqa: E402
from sqlalchemy.orm import Mapped, mapped_column  # noqa: E402

from scufris_core import Base, open_database  # noqa: E402


class NoteRow(Base):
    """A table this example owns, registered against the shared metadata."""

    __tablename__ = "note"

    id: Mapped[int] = mapped_column(primary_key=True)
    body: Mapped[str]


class Boom(RuntimeError):
    """Raised inside a transaction to prove the rollback is real."""


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp)
        database = open_database(state_dir)
        try:
            print(f"1. opened {database.path.name} under a temporary directory")

            # No Alembic here: this example owns one table and creates it
            # directly. The APP's schema is migrated, never created from the
            # models - see scufris/db/migrate.py. `metadata` holds exactly
            # `note`, because nothing here imports a package that declares rows.
            Base.metadata.create_all(database.engine)

            with database.transaction() as connection:
                connection.execute(
                    insert(NoteRow), [{"body": "first"}, {"body": "second"}]
                )
            print("2. committed two rows in one transaction")

            try:
                with database.transaction() as connection:
                    connection.execute(insert(NoteRow), {"body": "third"})
                    raise Boom("something went wrong after the write")
            except Boom as exc:
                print(f"3. rolled back after: {exc}")

            with database.transaction() as connection:
                surviving = connection.execute(
                    select(func.count()).select_from(NoteRow)
                ).scalar_one()
            print(f"4. surviving rows: {surviving}")

            if surviving != 2:
                print(f"FAILED: expected 2 surviving rows, got {surviving}")
                return 1
        finally:
            database.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
