# Record the chat conversation and event tables with typed actors

- PRIORITY: 100
- TAGS: feature, v0.2.0, lane1, chat
- KIND: TASK
- ACTIVITY: WORKING
- GATES: PLAN
- RESOLUTION: -
- PARENT: 20260801-154211

## Story

As the operator, I want every meaningful thing said in a conversation recorded
as a semantic event with a typed author, so that Scufris owns the conversation
rather than borrowing the provider's, and so that "who said this" is a fact the
database enforces instead of a convention the renderer follows.

This is the first task of Lane 1 and the first table in `packages/chat`. It is
built ALONGSIDE the old stack: `conversation`, `event`, `delivery` and
`activity` collide with no existing `__tablename__`, so nothing is deleted here.

The two questions the Steps used to defer - per-turn granularity and retention -
are SETTLED in `tasks/20260804-115256/DECISION.md`, along with the actor-kind
set, the absent `backend` column, and who opens the transaction. Build what that
record says; do not re-litigate it.

## Steps

- [ ] Create the member skeleton. `packages/chat/pyproject.toml` declares
      `scufris-chat`, depends on `scufris-core` and `sqlalchemy>=2.0` and
      nothing else, sources `scufris-core = { workspace = true }`, and builds
      `src/scufris_chat` - copy the shape of `packages/hostctl/pyproject.toml`.
      Add `packages/chat/src/scufris_chat/__init__.py` and run `uv lock`.
      `[tool.uv.workspace] members = ["packages/*"]` already claims it and
      `flake.nix` reaches it through `workspace.deps.all`, so neither file is
      edited. EXPECT RED HERE: `_import_roots()` globs `packages/*/src/*`, so
      `test_every_member_has_an_example` and the EQUALITY check on
      `DECLARED_GRAPH` both fail the moment this directory exists. The last two
      Steps close them; do not leave the tree in this state at a checkpoint.
- [ ] Write the failing tests first, in `packages/chat/tests/test_chat_events.py`
      (a package-owned test directory, collected by
      `testpaths = ["tests", "packages/*/tests"]`). All four DoD tests, red for
      the missing behaviour and not merely for a missing import: run each and
      read the failure before writing any of the module.
- [ ] Declare the rows in `packages/chat/src/scufris_chat/models.py`, PRIVATE to
      the package as `packages/hostctl/src/scufris_hostctl/models.py` is.
      `ConversationRow` (`conversation`) and `EventRow` (`event`) against
      `scufris_core.Base`. `event` carries `conversation_id`, `event_seq`,
      `actor_kind`, `actor_agent_id`, `kind`, `body`, `correlation_id`,
      `causation_id` and `created_at`, with `UniqueConstraint(conversation_id,
      event_seq)` and a CHECK constraint pinning `actor_kind` to the four kinds.
      No FOREIGN KEYs - `scufris/db/models.py`'s docstring records why, and the
      reason (batch ALTER under `foreign_keys=ON`) applies here unchanged.
- [ ] Type the actor in `packages/chat/src/scufris_chat/actors.py`: an
      `ActorKind` enum over `operator`, `orchestrator`, `agent`, `system` and a
      frozen `Actor` parsed at ONE boundary. `agent` requires an id and the
      other three refuse one; an unknown kind raises at the parse. Four kinds,
      not three, per DECISION.md section 2 - `tasks/20260729-220835/DECISION.md`
      section 3 is the ratified list and names `orchestrator` separately.
- [ ] Write the store in `packages/chat/src/scufris_chat/store.py` as functions
      over an OPEN `sqlalchemy.Connection`, never over `Database`:
      `create_conversation`, `append_event`, `read_transcript`, `causing_event`,
      returning frozen `ConversationRecord` / `EventRecord`. The writer cannot
      open a transaction, which is what makes "the state change and its event
      commit together" structural rather than a rule callers are asked to keep.
      Assign `event_seq` inside the caller's connection as
      `COALESCE(MAX(event_seq), 0) + 1` scoped to the conversation, the pattern
      `packages/hostctl/src/scufris_hostctl/actions.py:234` already uses;
      `BEGIN IMMEDIATE` is what makes it safe under two writers.
- [ ] Export the whole public surface from
      `packages/chat/src/scufris_chat/__init__.py` - `Actor`, `ActorKind`, the
      two records, the four functions - and no row class. A sibling imports
      `scufris_chat` and never a submodule;
      `test_no_package_imports_a_sibling_private_module` enforces it in sibling
      tests too.
- [ ] Register the tables with Alembic. Add `import scufris_chat  # noqa: F401`
      next to the existing `scufris_hostctl` import in
      `scufris/db/migrations/env.py:27`, then write ONE revision on top of
      `4119562b5fd9` through the maintainer loop in `scufris/README.md` section
      "Writing a revision" (`rm -f .alembic-scratch.db*`, `alembic upgrade
      head`, `alembic revision --autogenerate`, `ruff check --fix` and `ruff
      format` on `scufris/db/migrations/versions/`). Review the generated file
      rather than keeping it unread. Add `conversation` and `event` to the
      hand-written set in `test_declared_tables_are_the_only_ones`
      (`tests/test_db_migrations.py:581`) - its docstring currently asserts
      their ABSENCE, so update the prose with the list - and add
      `test_migration_creates_the_chat_tables` next to it.
- [ ] Close the two member gates the first Step opened. `DECLARED_GRAPH`
      (`tests/test_package_boundaries.py:220`) gains
      `"scufris_chat": frozenset({"scufris_core"})` and NOTHING is added to the
      `"scufris"` entry - the root does not import chat until Lane 6, and the
      check is EQUALITY. `tests/test_examples.py` gains
      `chat_conversation.py` on `OFFLINE` and
      `"scufris_chat": "chat_conversation.py"` in `EXAMPLES_BY_MEMBER`.
- [ ] Add `examples/chat_conversation.py` in MINIMAL form, on the shape of
      `examples/core_unit_of_work.py`: a real SQLite file under a temporary
      directory, `sys.path` extended to `packages/chat/src` and
      `packages/core/src` only, one transaction that mints a conversation and
      appends an operator message and an agent report, and a read-back that
      prints the transcript with its actors. It imports `scufris` nowhere.
      `20260804-115322` grows this same file into the lane demo; it does not
      create it.
- [ ] Document the package: `packages/chat/src/scufris_chat/README.md` (what the
      two tables are for, the seq invariant, the actor type, the
      connection-passing rule) and one row for it in the `packages/*` line of
      the AGENTS.md sources-of-truth table.
- [ ] Verify: `python -m pytest`, `ruff check .`, `ruff format --check .`,
      `mypy .`, and `uv lock --check`. Run `tatr check` with the records.

## Definition of Done

- `event_seq` is per-conversation, gap-free and strictly increasing under
  concurrent writers
  (test: `test_event_seq_is_monotonic_under_concurrent_writers`).
- The seq is assigned inside the same transaction that inserts the event, so a
  rolled-back write consumes no number
  (test: `test_rolled_back_event_consumes_no_seq`).
- An actor is a typed value; an unknown actor string cannot be persisted, and a
  known one round-trips (test: `test_actor_must_be_a_known_kind`).
- A caused event resolves to the event that caused it
  (test: `test_causation_resolves_to_the_causing_event`).
- The shipped migration creates `conversation` and `event` on a fresh database
  (test: `test_migration_creates_the_chat_tables`).
- The chat example runs offline, standing alone in a fresh interpreter
  (cmd: `python examples/chat_conversation.py`).

## Notes

- Source: `tasks/20260729-220835/DECISION.md` sections 1, 3 and 4. Section 4 is
  explicit that the state change and its event commit in ONE transaction.
  `tasks/20260804-115256/DECISION.md` settles what that record left open.
- `packages/chat` depends on `core` only. It must not import `flow`, `agents` or
  any sibling's `models` or `repo` module; `tests/test_package_boundaries.py`
  enforces this.
- The concurrency test needs REAL threads and a real file: each writer opens its
  own `Database.transaction()`, and `busy_timeout=5000` is what turns the
  contention into a wait. `transaction()` refuses a thread running an event
  loop, so this test is synchronous.
- `kind` on `event` is a plain string, per DECISION.md section 1. The enum lands
  with the first caller that branches on it (`20260804-115320`,
  `20260804-115321`).
- Lane 1 of `tasks/20260801-154211/TASK.md`.
