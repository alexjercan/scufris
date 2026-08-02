# Move the configuration-change registry onto the database

- PRIORITY: 70
- TAGS: refactor, v0.2.0, storage, reliability
- KIND: TASK
- ACTIVITY: -
- GATES: -
- RESOLUTION: -
- PARENT: 20260729-102145

## Story

As a Scufris operator, I want the configuration-change registry on the same
transactional boundary as every other app-owned store, so that a restart during
a NixOS build does not answer "there was never any such change".

## Notes

- Found by review round 1 (R1.2) of 20260801-100413, by the boundary test's
  discovery walk rather than by its hand-written store list.
- `ConfigChangeStore` (`scufris/hostconfig/changes.py`) is still an in-memory
  bounded `OrderedDict` with a `_reap`, which is exactly the shape
  `HostActionStore` had before 20260801-100413 migrated it. That task is the
  worked example: a row model, one Alembic revision, one `db.transaction()` per
  method, and `asyncio.to_thread` at every `async def` call site.
- It is app-owned state reached from `app.state.config_changes`, so
  `test_post_host_state_uses_declared_persistence_boundary` in
  `tests/test_db_state_boundary.py` currently EXCLUDES it by name. Removing that
  exclusion is this task's proof - the test then covers every store with no
  exceptions left.
- There is no legacy JSON source: the store was memory-only, so this is the
  same "no legacy file" case host actions were, and
  `scufris/db/legacy/__init__.py`'s docstring should say so for both.
- Out of scope for 20260801-100413, whose Steps name auth, host actions, the
  schedule and the digest history only. Migrating a fifth store was not planned
  there and would have materially exceeded that plan.
