"""The one declarative base every package registers its rows against.

Alone in its own module, and deliberately: the classes that USE it are domain
tables and belong to the packages that own them, while the metadata object
Alembic compares a database against has to be a single shared thing or the
migration history splits in two.

A package declares its own rows against this `Base`; `scufris.db.migrations`
imports whichever modules declare them so that one autogenerate run sees the
whole schema.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """The metadata Alembic compares a database against."""
