# Understanding: the chat conversation and event tables

## What changes

`packages/chat` comes into existence, holding `conversation` and `event`. It is
the first NEW package of the rewrite - `core`, `host`, `hostd` and `hostctl`
were moves of tested code, this is not.

Nothing is deleted. `chat`'s table names collide with none of the ten in
`scufris/db/models.py`, which is what makes it the only package buildable
alongside the old stack.

## Surfaces

- `packages/chat/src/scufris_chat/` - new. `__init__.py` is the whole public
  surface, per the rule `core`'s docstring states and
  `test_no_package_imports_a_sibling_private_module` enforces.
- `tests/test_package_boundaries.py:220` - `DECLARED_GRAPH` gains
  `"scufris_chat": frozenset({"scufris_core"})`.
- `tests/test_examples.py` - `OFFLINE` and `EXAMPLES_BY_MEMBER` gain
  `chat_conversation.py`.
- `scufris/db/migrations/env.py:27` - gains `import scufris_chat` next to the
  existing `import scufris_hostctl  # noqa: F401 - registers this package's
  tables`, or autogenerate will not see the new rows.
- One Alembic revision on the single linear chain, which `20260803-214750`
  squashed to one baseline.

## Data and interfaces

Two tables against `scufris_core.Base`. No FOREIGN KEYs - `db/models.py`'s
docstring records why, and the reason (batch ALTER under `foreign_keys=ON`
inside an open transaction) applies here unchanged.

The unit of work is `Database.transaction()` yielding a `sqlalchemy.Connection`.
SQLAlchemy Core, not ORM. There is no `sessionmaker` in this repository and this
task does not introduce one.

`event_seq` is assigned INSIDE the writing transaction. Not a timestamp (two
events can share a microsecond), not caller-supplied (the caller is the thing
being audited), not a UUID (order is the point).

The actor is a typed value with three cases: `operator`, `agent:<id>`, `system`.
This is the mechanism the stop gate rests on, not documentation - `20260804-115321`
refuses an `agent:<id>` actor, and it can only do that if the type makes the
distinction unfakeable.

## Sketches

```
write_event(conn, conversation_id, actor, ...) inside ONE transaction:

  BEGIN
    seq = SELECT COALESCE(MAX(event_seq), 0) + 1
            FROM event WHERE conversation_id = ?
    INSERT INTO event (conversation_id, event_seq, actor, ...)
    <caller's state change goes here, same transaction>
  COMMIT

  Rollback anywhere -> no row, no consumed seq, no half state.
```

## Shape

The two open design questions are settled here because the table cannot be
designed without them:

- **Per-turn granularity.** One event per turn is cheap to render and cannot
  attribute a tool call. One event per meaningful thing said is what the mockup
  draws - tool calls and agent reports are separately attributable there - and
  costs a query per render. The mockup is the target, so the finer grain is the
  likely answer, but the alternative gets written down as rejected rather than
  forgotten.
- **Retention.** v0.2.0 deletes no events. The table grows without bound. This
  is recorded as a CHOICE so that a large table later reads as a known
  consequence rather than an oversight.

## Consequences and open questions

- **The example is not optional.** `_import_roots()` globs `packages/*/src/*`,
  so `scufris_chat` becomes a workspace member the instant the directory exists,
  and `EXAMPLES_BY_MEMBER` requires it to name an offline example that imports
  it. `examples/chat_conversation.py` ships in THIS task, minimal.
  `20260804-115322` grows it into the lane demo; it does not create it.
- **Do not wire chat into the app here.** `DECLARED_GRAPH` is checked for
  EQUALITY, not containment. Adding `scufris_chat` to the root `scufris` entry
  before the root actually imports it fails the check. That edge lands in Lane 6.
- **Open:** whether `conversation` needs a `backend` column or whether that
  belongs entirely to the session cache in `20260804-115320`. Leaning: the cache
  owns it, because the conversation is meant to outlive any backend. Resolve
  while writing the table, not before.
