"""The declarative schema: what the database is supposed to look like.

This module is the SOURCE OF TRUTH for the shape of the state database. The
revisions under ``migrations/versions/`` are how a database that already exists
gets here; they are not a second description of the schema, and a revision that
disagrees with this file is caught by
``test_schema_has_no_pending_autogenerate_diff`` rather than by an operator.

``projects`` mirrors :class:`scufris.projects.Project` field for field, and
``ProjectStore`` reads and writes it: ``projects.json`` is no longer
authoritative.

``legacy_import`` is the bookkeeping the one-way JSON import needs; see
``legacy.py``.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """The metadata Alembic compares a database against."""


class ProjectRow(Base):
    """One workspace record. Mirrors :class:`scufris.projects.Project`.

    Every column is NOT NULL: ``language`` and ``description`` default to the
    empty string on the pydantic record, so an absent value is ``""`` rather than
    a null, and the two spellings of "not set" are never both reachable.
    """

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(primary_key=True)
    cwd: Mapped[str]
    name: Mapped[str]
    language: Mapped[str] = mapped_column(default="")
    description: Mapped[str] = mapped_column(default="")


class LegacyImportRow(Base):
    """One legacy JSON file that has been imported, in full, exactly once.

    The row IS the gate: it is written in the same transaction as the records it
    stands for, so it exists only if that whole import committed. A source with
    a row here is never read again - which is what makes a second startup a
    no-op rather than a duplicate import.

    Keyed by the file's NAME, not its path: the state directory can move, and
    what has been imported is a fact about this database's contents.
    """

    __tablename__ = "legacy_import"

    source: Mapped[str] = mapped_column(primary_key=True)
    # ISO-8601 UTC. A string rather than a DateTime because nothing sorts or
    # filters on it - it is there for an operator reading the table by hand
    # after a migration they want to account for.
    imported_at: Mapped[str]
