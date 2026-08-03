# EPIC: Carve the code into a uv workspace of per-service packages

- PRIORITY: 106
- TAGS: goal,epic,v0.2.0,architecture,packaging,maintainability
- KIND: EPIC
- ACTIVITY: -
- GATES: -
- RESOLUTION: -

## Epic

Carve the code into a `uv` workspace of per-service packages under one
composition root, so that "this component is complete" becomes a property the
build enforces rather than a claim in a README, and so that the shape of the
target architecture is visible in the directory tree before the new store is
written into it.

The model is a monolith of modules, not microservices: one process, one
database, one deployment. Each package is a service-shaped unit that `scufris`
instantiates in-process. Nothing runs in a container and nothing talks over HTTP
to a sibling. The value bought is the ENFORCED BOUNDARY, not the distribution.

This epic carves the code that SURVIVES and deletes the code that is
unambiguously dead. It does not build the new conversation, agent or flow
packages - those are `tasks/20260729-102157` - and it does not delete the code
they replace, which happens only once the replacement is live.

### The ten units

| Directory | Distribution | Import | Owns |
|---|---|---|---|
| `packages/core` | `scufris-core` | `scufris_core` | the SQLAlchemy engine, `Database`, `Database.transaction()` and `Base`. That is all of it |
| `packages/hostd` | `scufris-hostd` | `scufris_hostd` | the root helper: socket protocol, verbs, audit. Complete; frozen |
| `packages/host` | `scufris-host` | `scufris_host` | read-only host inspection; feeds Stats |
| `packages/hostctl` | `scufris-hostctl` | `scufris_hostctl` | the unprivileged client that DRIVES `hostd`: actions, previews, approvals, NixOS changes, the audit bridge |
| `packages/agents` | `scufris-agents` | `scufris_agents` | presets, instances, runs, backends, provider sessions |
| `packages/chat` | `scufris-chat` | `scufris_chat` | the semantic conversation: events, actors, delivery |
| `packages/flow` | `scufris-flow` | `scufris_flow` | the tatr reader, the flow guard, assignments, projects |
| `packages/telegram` | `scufris-telegram` | `scufris_telegram` | one delivery channel over `chat` |
| `scufris/` | `scufris` | `scufris` | composition root: the FastAPI app, auth, the nav registry, static, the CLI |
| `web/` | - | - | the frontend build, unchanged |

`chat`, `agents` and the `hostd`/tatr pair are the four records of
`tasks/20260729-220835/DECISION.md` section 2 given directories. The cut is that
decision's, not a new one.

The host trio splits by privilege: `host` reads and needs none, `hostd` is root
in a separate process and applies things, `hostctl` is the unprivileged client
between them. `hostctl` is named for its job - it is the thing that controls
`hostd` - rather than for host operations in general, which is the whole trio.

### The rule that makes it a boundary

A package may import a sibling's PUBLIC API - what its `__init__` exports. It
may never import a sibling's `models` or `repo` module.

That is the whole rule. It is greppable, it is one test, and it is what stops
the workspace from being nine folders that all import each other. Row classes
and repositories stay private to the package that owns the tables, so changing
`agents`' schema cannot break `flow`.

Ports and Protocols in `core` are deliberately NOT adopted. Every seam here has
exactly one implementation and one caller, and `AGENTS.md` is right that one
caller is not an abstraction. Direct sibling imports of the public facade cost
nothing and buy the same boundary. If a second implementation ever appears,
introducing a Protocol at that seam is a one-file change made with evidence.

### Dependency direction

Notation: `A -> B` means **A depends on B**. One direction, everywhere.

```text
host     -> nothing                     (psutil and pydantic; it opens no database)
core     -> nothing                     (sqlalchemy; every other package depends on it)
hostd    -> core, host                  (host.run: the read-only command seam;
                                         core: main.py's configure_logging)
hostctl  -> core, host, hostd           (and over the socket, a real process boundary)
agents   -> core
chat     -> core
flow     -> core, chat, agents
telegram -> core, chat, hostctl, host   (see below - this edge is real today)
scufris  -> all of them
```

No cycles. `flow`, `hostctl`, `telegram` and `scufris` each import more than one
sibling; for the first three that is coordination, and for `scufris` it is
composition.

**`telegram -> hostctl, host` is a real edge, not a mistake to design away.**
Five modules import `host_actions` or `host_approvals` today
(`telegram/wiring.py:34,35,40`, `render.py:36`, `contracts.py:17`,
`approvals.py:25`, `bot.py:25`) and three import `metrics.HostStats`
(`wiring.py:42`, `render.py:38`, `contracts.py:19`). Telegram renders host
approval cards; that is a product behavior, not an accident.

**Open, and owned by this epic: are host approvals conversation events?**
`tasks/20260729-220835/DECISION.md` requires that an approval decided from
either channel writes ONE decision event, and replaces
`TelegramApprovals._announced` with a durable `(channel, idempotency_key)`
delivery table. Under the graph above `hostctl` cannot reach `chat`, so a host
approval cannot become a conversation event without the composition root
re-implementing the join. Three ways out, to be decided in this epic's
`DECISION.md` before `packages/telegram` is carved:

1. `hostctl -> chat`, and host approvals are ordinary conversation events.
2. An event port owned by the root, which both `hostctl` and `chat` see.
3. Host approvals are a FIFTH record outside the four, with their own delivery.

Do not carve `packages/telegram` until this is answered; the answer decides
which package owns the approval card.

**The host trio's edges are the code's, not a design choice.** Six `hostd`
modules already import `scufris.host.run`, `.models`, `.storage` and `.units`
(`engine.py:33`, `preview.py:26-29`, `nixos.py:41-43`, `executor.py:25`,
`actions/validate.py:18`, `actions/plans.py:12-13`), and `hostconfig` imports
`host.run` too. `host.run` is read-only command plumbing - `Runner`,
`run_command`, `CommandResult`, `nix_cli` - so the root helper reusing it crosses
no privilege line. The alternatives were duplicating those types on the two sides
of a wire protocol, or hoisting a module that shells out to `nix` and `systemctl`
into `core`, which is the junk-drawer decay `test_core_is_domain_free` exists to
catch. This is why `host` is carved BEFORE `hostd` (2026-08-03, from the
understanding pass on the four children).

### The database

Ownership is distributed; the MACHINERY is central.

- `core` owns the engine, `Database`, `Base`, and the unit of work. It knows
  about no domain table and imports no sibling.
- Each package owns its own tables, row classes and repository functions.
- A package NEVER opens a transaction. The connection is passed in.

So one operator turn is one transaction opened by the root, threaded through
`chat.append_event(connection, ...)` and `agents.record_run(connection, ...)`,
and committed once - which is what `tasks/20260729-220835/DECISION.md` section 4
requires of a state change and its event.

**The unit of work is `Database.transaction()`, which yields a
`sqlalchemy.Connection`.** This codebase uses SQLAlchemy CORE, not the ORM:
there is no `sessionmaker` and no ORM `Session` anywhere in it. Every package
API that touches storage therefore takes a `Connection`, and any wording in a
child task about "the session" means this and nothing else.

**`core` is smaller than a shared package usually is, and that is the point.**
It is the engine, `Database`, `Base` and nothing else. There is no `ids` module
(`python-ulid` is declared and imported by zero files), no `time` module, and no
generic error type (errors are local and domain-specific). `scufris/enums.py`
does NOT move: all ten of its symbols - `ORCHESTRATOR_ID`, `HOST_AGENT_ID`,
`Audience`, `audience_for`, `AuthMode`, `AuthPolicy`, `Backend`,
`PermissionMode`, `AgentState`, `RunPhase` - belong to `agents`, auth or the
composition root. Each travels with its package when that package is built.
Hoisting them into `core` up front is exactly the junk-drawer decay the
Sequencing section warns about.

The alternative - one module holding every table and every repository - is
rejected. It reads as centralization but inverts the dependency: the package
every module depends on would have to know about every module, no schema could
change without touching it, and the boundaries above would be cosmetic.

**Cross-package references need no foreign keys, because nothing here uses
them.** There are zero `ForeignKey` declarations in the codebase and
`db/models.py` documents why: SQLite's `PRAGMA foreign_keys` is a no-op inside
an open transaction, so Alembic's batch ALTER - the only way SQLite can change a
column - would be stuck on the first change to a referenced table; and the
stores already delete related rows in one transaction, so a cascade would
duplicate a guarantee the transaction gives. References are plain string columns
with integrity enforced in the transaction. Distributing table ownership changes
none of that.

**Alembic: one `Base`, many files.** `Base` moves to `core`. Each package
defines its rows against that same `Base`, so importing a package's `models`
registers its tables onto the shared metadata. `env.py` stays at the root and
imports every package's models before reading `Base.metadata`. One `versions/`
directory, one linear chain, one `alembic upgrade head`.

The one new failure mode: a package `env.py` forgets to import silently vanishes
from the metadata, so its tables are never created and autogenerate would emit a
`drop_table` for them. `test_every_package_model_is_registered` catches it.

Its justification is narrower than "it protects operator data" - during v0.2.0
there IS no operator data to lose, because 20260803-214750 squashes to one
baseline and refuses any pre-v0.2.0 database. The risk it actually carries is a
package whose tables silently never exist, which is a broken feature rather than
a destroyed database, and `test_declared_tables_are_the_only_ones`
(`tests/test_db_migrations.py:478`) catches only the opposite direction. Keep
the test; do not oversell it.

Per-package version tables via Alembic branch labels are rejected: multiple
heads, `upgrade heads` instead of `head`, merge revisions whenever two packages
migrate in one release, and a concept every future reader must learn - for no
benefit on a single SQLite file.

### How this is proven

Per `tasks/20260801-154211/TASK.md`, the primary proof of a package is a
runnable example plus unit tests written test-first. A package is not done
because it imports; it is done because `examples/<package>_*.py` runs green
against a temporary database and fakes, with no host, no provider and no
network.

## Done Means

1. The workspace resolves and every package imports independently
   (cmd: `uv sync && uv run python -c "import scufris_core, scufris_host, scufris_hostctl, scufris_hostd, scufris"`).
2. No package imports a sibling's `models` or `repo` module
   (test: `test_no_package_imports_a_sibling_private_module`).
3. The declared dependency graph is acyclic and matches the real imports
   (test: `test_package_import_graph_matches_the_declared_graph`).
4. `core` holds only its allowlisted modules, so growing it requires editing the
   allowlist and justifying the entry (test: `test_core_is_domain_free`).
5. Every package's models are reachable from the migration metadata, so no
   package's tables can be silently absent
   (test: `test_every_package_model_is_registered`).
6. Every carved package has a runnable offline example, and the gate's opt-in
   list is proven to cover every workspace member - a package that never adds
   itself cannot leave the gate green
   (test: `test_every_package_has_a_gated_example`).
7. No test leaves the canonical gate when it moves into a package
   (cmd: `rg -q 'packages/\*/tests' pyproject.toml`).
8. No unambiguously dead code survives: no `/api/agent/*` router, no JSON import
   path (cmd: `! rg -q 'legacy_agent|db/legacy|db\.legacy|legacy_import' --glob '!tasks/**' --glob '!CHANGELOG.md' .`).
9. The packaged build is unchanged in behavior, and the root helper is still
   reachable at the path the NixOS module execs. Since 20260803-214747 the
   console script ships from its own distribution, so the output that carries it
   is `.#scufris-hostd`, not `.#scufris`
   (cmd: `nix flake check && nix build .#scufris .#scufris-web && nix build .#scufris-hostd && test -x result/bin/scufris-hostd`).
10. The tree answers "who owns this concern" without reading code
    (manual: the maintainer names the owning package for a given concern from
    the directory listing alone).

## Child Tasks

- [ ] 20260803-214746 (p105) bootstrap the uv workspace and the `core` package
- [ ] 20260803-214748 (p104) move read-only inspection into `packages/host`
- [ ] 20260803-214747 (p103) move the root helper into `packages/hostd`
- [ ] 20260803-214749 (p102) move the host control client into `packages/hostctl`
- [ ] 20260803-214750 (p101) delete the legacy agent router and the JSON import
      path, and squash the migration history to one baseline

## Decisions

- Accepted in this record: the ten-unit cut, the public-API-only import rule,
  the rejection of Protocol ports, distributed tables over central CRUD, and one
  Alembic history. To be written to `DECISION.md` by 20260803-214746, the task
  that first makes them real.

## Manual Acceptance

- (pending) 20260803-214746: `core` is small enough that its contents are
  obvious, and does not read as a junk drawer.

## Sequencing

- Runs FIRST in v0.2.0, before any new package is built. Every child here is
  either a pure move of complete, tested code or a deletion of code with no
  replacement, so the app keeps working throughout and a failure is
  unambiguously the carve rather than new logic.
- **`hostctl` (20260803-214749) is the one child that is NOT a pure move.**
  `EventBus` and the generic half of `Supervisor` have to be hoisted into `core`
  before it can run, `Settings` has to be narrowed out of `hostconfig/service.py`,
  and `host_watch.py` stays at the root. Its NOTES.md carries the detail. Plan it
  last of the three host carves and expect real edits, not `git mv`.
- The realistic decay path is `core` becoming a junk drawer - every "shared"
  package in every repository eventually does. `test_core_is_domain_free` is
  the guard, and it must be an explicit ALLOWLIST of the module names permitted
  under `scufris_core`, not a property check. A property check ("declares no
  `__tablename__`, imports no sibling") is satisfied trivially by `EventBus`,
  the generic `Supervisor`, `RunPhase` and `logsetup` - the exact growth already
  planned for this package - so it cannot catch the decay it is named for. An
  allowlist forces an edit and a justification per entry, which is the point.

## Notes

- Spike: tasks/20260729-220835/SPIKE.md
- Decision: tasks/20260729-220835/DECISION.md - section 2 is the record
  ownership this package cut mirrors; section 4 is why the session is threaded
  rather than opened per package.
- `uv2nix.lib.workspace.loadWorkspace` is already the loader in `flake.nix:61`,
  so uv2nix already models this repository as a workspace. **`flake.nix:74`
  `members` is NOT the member declaration** - it sits inside
  `mkEditablePyprojectOverlay` under the comment "Optional: Only enable editable
  for these packages", so it is an EDITABILITY filter. Setting it to
  `["scufris"]` would make `packages/core` non-editable in the dev shell, which
  is the opposite of what is wanted. Membership comes from
  `[tool.uv.workspace]` in the root `pyproject.toml`, which does not exist yet.
  Either leave `members` commented (all members editable) or list every member
  and keep it in sync as each package lands.
- `mkApplication` (`flake.nix:113,117`) builds its output from the STRUCTURE of
  the package it is given, so moving the `scufris-hostd` console script to
  another distribution removes `bin/scufris-hostd` from `packages.scufris` -
  which is what `nix/scufris-hostd.nix:45-50` defaults to and `:147` execs. This
  breaks at BUILD time. 20260803-214747 owns the fix.
- `api/errors.py:16` imports `hostd.protocol.ErrorCode`: the HTTP error mapper
  depends on the root helper's wire protocol. Legal under the import rule, but
  decide in 20260803-214747 whether the app should map its own codes instead.
- Not in scope: splitting the frontend build, splitting the database, running
  any package as a separate process, introducing Protocol seams, and building
  or deleting the agent/conversation/flow stack.
