"""The Alembic environment, shipped as package data rather than at the repo root.

The wheel is built with ``only-include = ["scufris"]`` (`pyproject.toml`), so a
root ``alembic/`` directory would never reach an operator and ``upgrade head``
would have nothing to run at their startup. This package is a package for one
reason: ``importlib.resources.files`` has to be able to find ``env.py`` and
``versions/`` inside an installed distribution, which is how
``scufris.db.migrate`` points Alembic at them.

Nothing imports from here. The runner is `scufris/db/migrate.py`; the schema
itself is `scufris/db/models.py`.
"""
