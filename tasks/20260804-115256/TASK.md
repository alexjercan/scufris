# Record the chat conversation and event tables with typed actors

- PRIORITY: 100
- TAGS: feature, v0.2.0, lane1, chat
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
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

- [x] Create the member skeleton. `packages/chat/pyproject.toml` declares
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
- [x] Write the failing tests first, in `packages/chat/tests/test_chat_events.py`
      (a package-owned test directory, collected by
      `testpaths = ["tests", "packages/*/tests"]`). All four DoD tests, red for
      the missing behaviour and not merely for a missing import: run each and
      read the failure before writing any of the module.
- [x] Declare the rows in `packages/chat/src/scufris_chat/models.py`, PRIVATE to
      the package as `packages/hostctl/src/scufris_hostctl/models.py` is.
      `ConversationRow` (`conversation`) and `EventRow` (`event`) against
      `scufris_core.Base`. `event` carries `conversation_id`, `event_seq`,
      `actor_kind`, `actor_agent_id`, `kind`, `body`, `correlation_id`,
      `causation_id` and `created_at`, with `UniqueConstraint(conversation_id,
      event_seq)` and a CHECK constraint pinning `actor_kind` to the four kinds.
      No FOREIGN KEYs - `scufris/db/models.py`'s docstring records why, and the
      reason (batch ALTER under `foreign_keys=ON`) applies here unchanged.
- [x] Type the actor in `packages/chat/src/scufris_chat/actors.py`: an
      `ActorKind` enum over `operator`, `orchestrator`, `agent`, `system` and a
      frozen `Actor` parsed at ONE boundary. `agent` requires an id and the
      other three refuse one; an unknown kind raises at the parse. Four kinds,
      not three, per DECISION.md section 2 - `tasks/20260729-220835/DECISION.md`
      section 3 is the ratified list and names `orchestrator` separately.
- [x] Write the store in `packages/chat/src/scufris_chat/store.py` as functions
      over an OPEN `sqlalchemy.Connection`, never over `Database`:
      `create_conversation`, `append_event`, `read_transcript`, `causing_event`,
      returning frozen `ConversationRecord` / `EventRecord`. The writer cannot
      open a transaction, which is what makes "the state change and its event
      commit together" structural rather than a rule callers are asked to keep.
      Assign `event_seq` inside the caller's connection as
      `COALESCE(MAX(event_seq), 0) + 1` scoped to the conversation, the pattern
      `packages/hostctl/src/scufris_hostctl/actions.py:234` already uses;
      `BEGIN IMMEDIATE` is what makes it safe under two writers.
- [x] Export the whole public surface from
      `packages/chat/src/scufris_chat/__init__.py` - `Actor`, `ActorKind`, the
      two records, the four functions - and no row class. A sibling imports
      `scufris_chat` and never a submodule;
      `test_no_package_imports_a_sibling_private_module` enforces it in sibling
      tests too.
- [x] Register the tables with Alembic. Add `import scufris_chat  # noqa: F401`
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
- [x] Close the two member gates the first Step opened. `DECLARED_GRAPH`
      (`tests/test_package_boundaries.py:220`) gains
      `"scufris_chat": frozenset({"scufris_core"})` and NOTHING is added to the
      `"scufris"` entry - the root does not import chat until Lane 6, and the
      check is EQUALITY. `tests/test_examples.py` gains
      `chat_conversation.py` on `OFFLINE` and
      `"scufris_chat": "chat_conversation.py"` in `EXAMPLES_BY_MEMBER`.
- [x] Add `examples/chat_conversation.py` in MINIMAL form, on the shape of
      `examples/core_unit_of_work.py`: a real SQLite file under a temporary
      directory, `sys.path` extended to `packages/chat/src` and
      `packages/core/src` only, one transaction that mints a conversation and
      appends an operator message and an agent report, and a read-back that
      prints the transcript with its actors. It imports `scufris` nowhere.
      `20260804-115322` grows this same file into the lane demo; it does not
      create it.
- [x] Document the package: `packages/chat/src/scufris_chat/README.md` (what the
      two tables are for, the seq invariant, the actor type, the
      connection-passing rule) and one row for it in the `packages/*` line of
      the AGENTS.md sources-of-truth table.
- [x] Verify: `python -m pytest`, `ruff check .`, `ruff format --check .`,
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

## Close-out

**What.** `packages/chat` exists as the sixth workspace member, with `conversation`
and `event` declared against `scufris_core.Base`, an `Actor` value over four
kinds, a store of four functions over an open `Connection`, revision
`18c9104709b8` on top of `4119562b5fd9`, and `examples/chat_conversation.py`.
Nothing was deleted: the old stack's tables do not collide with these two.

**Why it is shaped this way.** `tasks/20260804-115256/DECISION.md` settled the
five forks before implementation, and none of them was re-opened here. The one
shape worth restating is that the store is FUNCTIONS taking a connection rather
than a class holding a `Database`: it is what makes "the state change and its
event commit together" a thing the type signature enforces rather than a rule a
caller is asked to keep. A class holding a `Database` would be able to open a
second unit of work, which `transaction()` refuses anyway - so the class would
only add a way to get it wrong.

**Alternatives at the code level.** `causing_event` on a `causation_id` that
resolves to nothing RAISES `LookupError` rather than returning `None`. Returning
`None` was the smaller change and was rejected: with no FOREIGN KEYs such an id
is reachable, and "this event started something" and "its cause is missing" mean
opposite things to a reader. A row `id` was added as the primary key even though
the Step's column list does not name one - `causation_id` has to point at
something, and `(conversation_id, event_seq)` as a composite key would put the
transcript position inside every reference to an event.

**Two Steps were wrong and were corrected.**

- Step 8 says "NOTHING is added to the `scufris` entry" of `DECLARED_GRAPH`.
  That contradicts Step 7, which adds `import scufris_chat` to
  `scufris/db/migrations/env.py` - and `env.py` is in the root distribution, so
  the import IS a `scufris -> scufris_chat` edge. `DECLARED_GRAPH` is checked
  for EQUALITY, so omitting it fails:
  `scufris -> scufris_chat: not allowed by the declared graph (imported by
  db.migrations.env)`. The edge is declared, with a comment recording that it is
  registration-only and that Lane 6 adds the real callers. Nothing else in the
  root reaches chat.
- No Step mentions `[tool.ruff.lint.isort] known-first-party`, which lists the
  five members by name. Without `scufris_chat` on it, ruff sorted the new import
  into the THIRD-PARTY block in all three files that use it - including above
  `from alembic import context` in `env.py`. Added.

**Difficulties.** The first `nix develop` after creating `packages/chat` failed
with `internal error: accessed dependencies from pyproject.nix project, not
uv.lock`; the new files were unstaged, and the flake reads the git tree.
`git add` fixed it. Worth knowing for the next member carve, and it is already
half-recorded in AGENTS.md's "New workspace member" note.

**A flake, not a regression.** One full-suite run failed
`tests/test_app.py::test_agent_run_reaches_done_and_persists_session` on
`assert st["turns"] == 1` getting 0 - a poll for a background agent run. It did
not reproduce: two further full-suite runs and three targeted runs were green,
`tests/test_app.py` is green on the base, and this diff adds no agent-run
behaviour. Recorded rather than fixed; it is not this task's.

**Evidence.** `nix flake check` passes all six checks (ruff, ruff-format, mypy,
pytest, records, filesize). All six DoD proofs run green individually, including
the concurrency test's 8 threads x 5 events on a real file. `uv lock --check`
clean. The generated revision was read before keeping it: two `create_table`
calls, the CHECK, the unique constraint, correct nullability, correct
`down_revision`.

**Next time.** The TDD beat was worth its cost in exactly one place. Writing the
modules with `Actor`'s id-rule and the CHECK constraint deliberately absent
produced two real behavioural reds - `DID NOT RAISE ValueError`, then
`DID NOT RAISE IntegrityError` - which proved both gates are load-bearing. The
seq tests, by contrast, were green on the first cut, because the design (the
caller owns the transaction) makes the defect they guard unreachable from this
API. That is a fine outcome, but it means those two tests prove the SCHEMA and
the engine, not this store's care, and a future refactor that lets the store
open its own transaction is what they would actually catch.

## Round 2

Ten findings, all ten addressed; none disputed.

**The theme was half-invariants.** Three of the four substantive findings are the
same shape: a rule stated in one place and enforced in one and a half.

- The actor rule was CHECK-constrained on the KIND and not on the id, so
  `actor_kind='operator'` with an `actor_agent_id` was schema-legal - and because
  `read_transcript` rebuilds every row into an `Actor`, one such row made the
  WHOLE conversation raise, not just itself. `ck_event_actor_agent_id` now holds
  the other half, in the model and in revision `18c9104709b8`.
- The kind list could not drift between the enum and the model (the model renders
  it from the enum) but could between the enum and the shipped REVISION, because
  `compare_metadata` does not diff CHECK constraints - so a fifth kind would have
  been green across the entire suite and raised only on a migrated operator
  database. `test_migrated_actor_check_lists_exactly_the_declared_kinds` reads
  the text off `sqlite_master` and closes it; adding a fifth kind now goes red
  there, checked.
- `causing_event` refused a dangling `causation_id` while `append_event` accepted
  any `conversation_id` and minted a transcript belonging to no conversation.
  The argument for one is the argument for the other: there are no FOREIGN KEYs,
  so the store makes the check the schema will not. `causing_event`'s lookup is
  now scoped to the event's own conversation too - a cause in another thread
  resolves to a real event, which makes it worse unscoped than a dangling id, not
  better.

**The build declared a membership it did not have.** `env.py` imports
`scufris_chat` from the ROOT distribution, and `scufris-chat` was in neither
`[project.dependencies]` nor `[tool.uv.sources]`. In-repo it passed because
`uv sync` installs every workspace member; an installed root wheel would have
raised `ModuleNotFoundError` at `alembic upgrade head`, the one path the import
exists for. Declared and relocked. This is the other half of the plan gap Step 8
already had, and no Step named it either.

**`Actor.render` was deleted.** It had no production caller - the store keeps the
kind and the id as two columns, so the wire form is never written - and no Step
named a render direction. The parse stays one-way, which is now stated in the
module docstring so the next reader does not restore the symmetry by reflex. The
example formats the actor for display itself. `parse` also got stricter:
`operator:` promised an id and named none, and used to parse as a bare kind.

**Evidence.** Every new guard was sabotaged before it was trusted: a fifth
`ActorKind` fails the two kind tests; dropping `ck_event_actor_agent_id` from the
revision fails `test_migration_creates_the_chat_tables`; dropping it from the
model fails `test_actor_must_be_a_known_kind`. The four store findings were red
for behaviour before the fix, not for an import. Full suite green (1117 passed,
1 skipped), `ruff check`, `ruff format --check`, `mypy` (242 files),
`uv lock --check`, the filesize gate, `tatr check`, and all six DoD proofs
including `python examples/chat_conversation.py`.

**Next time.** The finding to learn from is the CHECK-drift one, because the
suite was designed to catch drift and did not: `test_schema_has_no_pending_
autogenerate_diff` reads as "the models and the migration agree", and it means
"they agree about everything autogenerate compares". Anything hand-written into a
revision - CHECK text, a trigger, a partial index - is outside that guarantee and
needs its own assertion against a migrated database.

## Round 3

One finding, R2.1, addressed and widened; not disputed.

**A rule stated twice in two different logics.** The round-2 fix wrote the actor
rule into the schema as NULLABILITY (`actor_agent_id IS NOT NULL`) while
`Actor.__post_init__` states it as TRUTHINESS (`not self.agent_id`). Those are
the same predicate everywhere except the empty string, so `('agent', '')` passed
the CHECK and then made `read_transcript` raise for the whole conversation - the
exact failure R1.2 was raised to prevent, narrowed to one value.

**Fixed wider than asked.** The finding's proposed predicate,
`(actor_kind = 'agent') = (actor_agent_id IS NOT NULL AND actor_agent_id <> '')`,
closes `('agent', '')` and leaves `('operator', '')`: false = false is true, so
the row is accepted, and `_record` then builds `Actor(ActorKind.OPERATOR, '')`,
which `__post_init__` refuses on `'' is not None`. Same blast radius, one column
value over. The shipped constraint is the full predicate instead:
`(actor_kind = 'agent' AND actor_agent_id IS NOT NULL AND actor_agent_id <> '')
OR (actor_kind <> 'agent' AND actor_agent_id IS NULL)`. An equality between two
booleans reads compactly and cannot express "non-empty on one side, NULL on the
other"; the disjunction can, so compactness lost.

**Evidence.** Both tests were run red first and failed on
`DID NOT RAISE IntegrityError`, not on an import:
`test_migration_creates_the_chat_tables` (shipped revision, real migrated file)
and `test_actor_must_be_a_known_kind` (model metadata). Each now asserts both
empty-string rows rejected and `('agent', 'builder')` accepted, so the
constraint cannot pass by rejecting everything. Full suite green under
`nix develop` (exit 0, no failures), `ruff check .`, `ruff format --check .`
(242 files), `mypy .` (242 files), `uv lock --check`, `tatr check`, and
`python examples/chat_conversation.py` (exit 0).

**Next time.** The review named this shape three times across two rounds, so the
lesson belongs in the code, not only in the record: `models.py:34`,
`EventRow`'s docstring and `README.md:79` now say the predicate is truthiness
and why. When an invariant lives in both a dataclass and a CHECK, write the
constraint from the Python predicate's exact semantics rather than from what the
column's nullability suggests - `not x` and `x IS NOT NULL` differ on every
falsy value the column can hold.
