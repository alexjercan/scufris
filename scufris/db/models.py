"""The declarative schema: what the database is supposed to look like.

This module is the SOURCE OF TRUTH for the shape of the state database. The
revisions under ``migrations/versions/`` are how a database that already exists
gets here; they are not a second description of the schema, and a revision that
disagrees with this file is caught by
``test_schema_has_no_pending_autogenerate_diff`` rather than by an operator.

One table today. ``projects`` mirrors :class:`scufris.projects.Project` field for
field, and nothing in the app reads or writes it yet - ``ProjectStore`` is still
on ``projects.json`` until the store cutover task. Creating the table before the
store moves is what lets this land green on its own.
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
