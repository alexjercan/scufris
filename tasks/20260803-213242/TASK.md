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
| `packages/core` | `scufris-core` | `scufris_core` | ids, time, enums, settings, the SQLAlchemy engine/session/`Base`, the unit of work, error types |
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

```text
core     <- everything
host     <- nothing            (psutil and pydantic; it opens no database)
hostd    <- host               (host.run: the read-only command seam)
hostctl  <- core, host, hostd  (and over the socket, a real process boundary)
agents   <- core
chat     <- core
flow     <- core, chat, agents
telegram <- core, chat
scufris  <- all of them
```

No cycles. `flow` and `scufris` are the only packages that import more than one
sibling, and both do it for the same reason: coordination is their job.

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

- `core` owns the engine, the session factory, `Base`, and the unit of work. It
  knows about no domain table and imports no sibling.
- Each package owns its own tables, row classes and repository functions.
- A package NEVER opens a transaction. The session is passed in.

So one operator turn is one transaction opened by the root, threaded through
`chat.append_event(session, ...)` and `agents.record_run(session, ...)`, and
committed once - which is what `tasks/20260729-220835/DECISION.md` section 4
requires of a state change and its event.

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
from the metadata, and autogenerate will emit a `drop_table` for it. That is
what `test_every_package_model_is_registered` exists to catch, and it is the
only reason that test is worth its weight.

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
2. No package imports a sibling's `models` or `repo` module, and the dependency
   graph above has no cycles
   (test: `test_no_package_imports_a_sibling_private_module`).
3. `core` depends on no sibling and declares no domain table
   (test: `test_core_is_domain_free`).
4. Every package's models are reachable from the migration metadata, so no
   package's tables can be silently absent
   (test: `test_every_package_model_is_registered`).
5. Each carved package has a runnable offline example that proves it works on
   its own (cmd: `python -m pytest tests/test_examples.py`).
6. No unambiguously dead code survives: no `/api/agent/*` router, no JSON import
   path (cmd: `! rg -q 'legacy_agent|db/legacy|db\.legacy' --glob '!tasks/**' .`).
7. The packaged build is unchanged in behavior
   (cmd: `nix flake check && nix build .#scufris .#scufris-web`).
8. The tree answers "who owns this concern" without reading code
   (manual: the maintainer names the owning package for a given concern from the
   directory listing alone).

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
  the guard, and it is worth keeping strict enough to be annoying.

## Notes

- Spike: tasks/20260729-220835/SPIKE.md
- Decision: tasks/20260729-220835/DECISION.md - section 2 is the record
  ownership this package cut mirrors; section 4 is why the session is threaded
  rather than opened per package.
- `uv2nix.lib.workspace.loadWorkspace` is already the loader in `flake.nix:61`
  and the `members` knob is already present, commented, at `flake.nix:74`. The
  multi-member path is supported, not new.
- `scufris/telegram/` imports `metrics.HostStats` (`contracts.py`, `render.py`,
  `wiring.py`), which the declared `telegram <- core, chat` graph does not allow.
  Telegram is root code until 20260729-102157 carves it, so nothing breaks here -
  recorded so it is not rediscovered then.
- Not in scope: splitting the frontend build, splitting the database, running
  any package as a separate process, introducing Protocol seams, and building
  or deleting the agent/conversation/flow stack.
