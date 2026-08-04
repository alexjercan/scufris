# Decision: Carve the code into a uv workspace of per-service packages

- DATE: 20260803-230416
- STATUS: ACCEPTED
- TASK: 20260803-213242
- TAGS: architecture,packaging,storage,maintainability

## Context

The epic record accepted five decisions and deferred writing them down to
20260803-214746, the child that first makes them real. That child has now
bootstrapped the workspace and `packages/core`, so the decisions are load-bearing
for the four carves that follow and for `tasks/20260729-102157`, which builds the
new conversation, agent and flow packages inside this shape. They are recorded
here, before anything depends on them, because a carve that is half done is the
worst time to relitigate the cut.

Two facts about the codebase constrain everything below. It uses SQLAlchemy
CORE - there is no `sessionmaker` and no ORM `Session` anywhere in it, so the
unit of work is `Database.transaction()` yielding a `sqlalchemy.Connection`, and
any wording in a child task about "the session" means that. And it declares zero
`ForeignKey`s, for the two reasons `scufris/db/models.py` documents: SQLite's
`PRAGMA foreign_keys` is a no-op inside an open transaction, so Alembic's batch
ALTER - the only way SQLite can change a column - would be stuck on the first
change to a referenced table; and the stores already delete related rows in one
transaction, so a cascade would duplicate a guarantee the transaction gives.

## Decision

**1. Ten units, cut by ownership of a record.** `packages/core`,
`packages/hostd`, `packages/host`, `packages/hostctl`, `packages/agents`,
`packages/chat`, `packages/flow`, `packages/telegram`, the `scufris/`
composition root, and `web/` unchanged.

The cut is not new. `chat`, `agents` and the `hostd`/tatr pair are the four
records of `tasks/20260729-220835/DECISION.md` section 2 given directories; the
host trio splits by PRIVILEGE, a boundary the operating system already enforces -
`host` reads and needs none, `hostd` runs as root in a separate process,
`hostctl` is the unprivileged client between them.

This is a monolith of modules, not microservices: one process, one database, one
deployment, each package instantiated in-process by the root. Nothing runs in a
container and nothing talks HTTP to a sibling. What is bought is the ENFORCED
boundary, not the distribution.

**2. A package imports a sibling's public API and nothing else.** What a
sibling's `__init__` exports is fair game; `<sibling>.models`, `<sibling>.repo`
and every other submodule are not. Row classes and repositories stay private to
the package that owns the tables, so changing `agents`' schema cannot break
`flow`.

That is the whole rule, and it is one test:
`test_no_package_imports_a_sibling_private_module` in
`tests/test_package_boundaries.py` AST-walks every member's imports and fails on
any dotted sibling name. It is read from the SOURCE tree rather than by
importing, so it can report on a member whose wiring is not finished - which is
the state every carve task is in while it runs.

**3. No Protocol ports in `core`.** Every seam here has one implementation and
one caller. Direct sibling imports of the public facade cost nothing and buy the
same boundary, so `core` defines no `Protocol` port and no package is written
against one. If a second implementation ever appears, introducing a Protocol at
that one seam is a one-file change made with evidence - which is how the
`AgentBackend` protocol earned its place.

**4. Tables are owned by packages; only the machinery is central.** `core` owns
the engine, `Database`, `Database.transaction()` and `Base`, knows about no
domain table and imports no sibling. Each package declares its own tables, row
classes and repository functions against the shared `Base`. A package NEVER
opens a transaction: the open `Connection` is passed in.

So one operator turn is one transaction opened by the root and threaded through
`chat.append_event(connection, ...)` and `agents.record_run(connection, ...)`,
committed once - which is what `tasks/20260729-220835/DECISION.md` section 4
requires of a state change and its event. Cross-package references are plain
string columns with integrity enforced inside that transaction; distributing
table ownership changes nothing about the no-foreign-keys position above.

**5. One `Base`, one Alembic history.** `Base` is `scufris_core.Base`. Every
package defines its rows against it, so importing a package's models registers
its tables onto the one shared metadata. `env.py` stays at the root and imports
every package's models before reading `Base.metadata`. One `versions/`
directory, one linear chain, one `alembic upgrade head`.

The one new failure mode is a package whose models `env.py` forgets to import:
it vanishes from the metadata silently and autogenerate emits a `drop_table` for
it. `test_every_package_model_is_registered` exists for that and for nothing
else.

**6. `logsetup` belongs to `core`.** `scufris/logsetup.py` moved to
`scufris_core.logsetup`, re-exported from the facade as `configure_logging`,
`new_request_id`, `set_request_id` and `truncate`.

It is 87 lines importing only `logging`, `uuid` and `contextvars` - nothing from
`scufris` - and eleven modules import it that the carve splits across at least
four future packages. `scufris/hostd/main.py` was the concrete blocker: `hostd`
cannot become a member while it imports a root module.

This is what the `core` ALLOWLIST is for. `test_core_is_domain_free` checks an
explicit list of module names rather than a property, precisely so that adding
something to `core` costs a line in the test and a written justification like
this one.

**7. `core` is smaller than the epic's first table said.** It shipped as the
engine, `Database`, `Base` and `logsetup`, with `sqlalchemy` as its only
dependency. Checked against the tree, most of the rest does not exist or does
not belong: `scufris/enums.py` is ten domain symbols that each travel with their
package later; there is no `ids` module (`python-ulid` is declared and imported
by zero files); there is no `time` module and no candidate helper; there is no
`scufris/errors.py`, errors being local and domain-specific. `alembic` stays at
the root with `scufris/db/migrate.py`, whose `script_location` resolves inside
`scufris.db.migrations`.

`EventBus`, the generic half of `Supervisor` and `RunPhase` belong in `core` by
the rule, but move in 20260803-214749, where `hostctl` is the second consumer
that makes them evidence rather than a guess.

## Alternatives considered

**Cutting by layer** (`models`, `services`, `api`) - rejected. Layers put every
feature's pieces in three packages, so no package is ever complete and the
boundary tests say nothing about ownership.

**A documented convention instead of a test** for the import rule - rejected.
"Packages do not reach into each other" is exactly the claim that decays one
plausible import at a time; the split is only worth doing alongside the check.

**Hexagonal ports up front** - rejected. One caller is not an abstraction, and a
port per seam adds a file, an indirection and a fake to every boundary in
exchange for a substitutability nothing asks for.

**One central module holding every table and every repository** - rejected. It
reads as centralization but inverts the dependency: the package everything
depends on would have to know about everything, no schema could change without
touching it, and every boundary above it would be cosmetic.

**Per-package version tables via Alembic branch labels** - rejected. That buys
multiple heads, `upgrade heads` instead of `head`, a merge revision whenever two
packages migrate in one release, and a concept every future reader must learn -
for no benefit on a single SQLite file.

**Each entry point configures its own logging**, leaving `logsetup` at the root -
rejected. It buys `core` no smallness, since the module is already generic, and
costs a second log format that drifts from the first plus a duplicated
request-id contextvar. One format across the app is the property `logsetup`
exists to hold.

**A property check instead of an allowlist** for `test_core_is_domain_free` -
rejected. "Declares no table" is satisfied trivially by everything `core` is
already planned to gain, so it would wave through exactly the junk-drawer decay
it is named for.

## Consequences

- Adding a module to `core` is a deliberate act: a line in `CORE_MODULES` and a
  justification in a task record. That is the intended friction.
- Every open branch conflicts on `uv.lock`, and a maintainer following
  `Database` to its definition takes one more hop.
- The dev venv is a nix derivation built from the lock, so a new member does not
  resolve until `uv lock` has run AND the shell has been re-entered. This is an
  ordering constraint on every carve task, now recorded in `AGENTS.md`.
- The boundary claims are falsifiable rather than aspirational, and the four
  later carves are mechanical repeats of a path already proven.
- Nothing structural is foreclosed: a member can be dissolved back into the root
  by deleting its `pyproject.toml` and moving the tree.
- Open and owned by this epic, before `packages/telegram` is carved: whether
  host approvals are conversation events. Under the accepted graph `hostctl`
  cannot reach `chat`, so a host approval cannot become a conversation event
  without the composition root re-implementing the join that
  `tasks/20260729-220835/DECISION.md` requires. That answer decides which
  package owns the approval card and is NOT settled here.

## Amendment: `core` is no longer sqlalchemy-only

- DATE: 20260804-030000
- TASK: 20260803-214749

Decision 1 gave `core` one dependency, `sqlalchemy`, and its `pyproject.toml`
said in as many words that nothing there needed pydantic. Carving `hostctl`
falsifies that. `hostctl` supervises its applies and its config builds, so the
generic half of `Supervisor` had to move to `core`, and it carries `RunState` -
a `BaseModel`.

`RunState` stays a `BaseModel` rather than becoming a dataclass or a TypedDict:
`scufris/api/agent_runs.py` returns it straight to the HTTP surface, so its
serialization is a wire contract, and pydantic is what enforces the `RunPhase`
field's membership. Downgrading it to a dataclass would move that validation
into the router and change the response model, which is a behaviour change
smuggled into a move.

The list is still short and still argued per entry. `core` now depends on
`sqlalchemy` and `pydantic`, and on nothing else.
