# Bootstrap the uv workspace and the core package

- PRIORITY: 105
- TAGS: refactor, v0.2.0, architecture, packaging
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
- PARENT: 20260803-213242

## Story

As the Scufris maintainer, I want the `uv` workspace to exist with one real
member - `core` - so that the packaging machinery is proven against the smallest
possible package before four more move into it.

`core` holds the transactional boundary (`engine.py`, `Database`,
`Database.transaction()`), the shared `Base`, and - per the logsetup decision
below - `logsetup.py`. It imports no sibling and declares no domain table. It is
deliberately tiny: see the Notes for the four things this task was originally
told to move that turned out not to exist or not to belong.

## Steps

- [x] Write `tests/test_package_boundaries.py` FIRST, with a helper
      `_domain_free(package_root, allowed)` and three tests:
      - `test_core_is_domain_free`: an explicit ALLOWLIST of module names
        permitted under `scufris_core` - `{"", "engine", "base", "logsetup"}` -
        plus "no module declares `__tablename__`" and "the only `DeclarativeBase`
        subclass defined here is `Base`". An allowlist, not a property check: a
        property check is satisfied trivially by everything `core` is later
        planned to gain (`EventBus`, the generic `Supervisor`, `RunPhase`), so it
        cannot catch the decay it is named for.
      - `test_the_domain_free_check_rejects_the_pre_move_tree`: the SAME helper
        asserted to FAIL against `scufris/db`, where `Base` sits beside THIRTEEN
        `__tablename__` classes in `models.py`. This is the red-first proof that
        survives the move, rather than a one-shot manual demonstration.
      - `test_no_package_imports_a_sibling_private_module`: AST-walk every
        `packages/*/src/*` module's `Import`/`ImportFrom` nodes; a package may
        name a sibling's distribution root, never `<sibling>.<submodule>`.
      Red on base: `scufris_core` does not exist (verified,
      `ModuleNotFoundError`).
- [x] Root `pyproject.toml`: add `[tool.uv.workspace] members = ["packages/*"]`
      and `[tool.uv.sources] scufris-core = { workspace = true }`; add
      `scufris-core` to `dependencies`. Leave `sqlalchemy` in the root deps -
      `scufris/db/models.py` still declares thirteen tables against it.
      (Scratch-verified: `uv lock` resolves the member as
      `source = { editable = "packages/core" }`.)
- [x] Widen `testpaths` at `pyproject.toml:43` to
      `["tests", "packages/*/tests"]`. The flake check runs a bare
      `python -m pytest`, so the test modules 20260803-214747 and -214748 move
      into packages would stop running while `nix flake check` stayed green.
      (Scratch-verified: a glob matching nothing is tolerated, rc=0.)
- [x] Create `packages/core/pyproject.toml`: name `scufris-core`,
      `requires-python = ">=3.13"` matching the root, `dependencies =
      ["sqlalchemy>=2.0"]` and nothing else, hatchling build backend with
      `[tool.hatch.build.targets.wheel] packages = ["src/scufris_core"]`.
- [x] `git mv scufris/db/engine.py packages/core/src/scufris_core/engine.py`.
      Its 327 lines are already domain-free; the module docstring's closing
      paragraph ("The schema itself is NOT here") stays true and now names a
      cross-package boundary.
- [x] Move the `Base` class (`scufris/db/models.py:42`) to
      `packages/core/src/scufris_core/base.py`. Leave the thirteen row classes
      where they are: they are domain tables and belong to the packages that
      will own them. (Thirteen, not twelve - `LegacyImportRow` at `models.py:348`
      survives until 20260803-214750.) `models.py` gains
      `from scufris_core import Base`; because the rows still subclass it,
      `from scufris.db.models import Base` keeps resolving, so
      `scufris/db/migrations/env.py:28` and `tests/test_db_migrations.py:45`
      need NO edit.
- [x] `git mv scufris/logsetup.py
      packages/core/src/scufris_core/logsetup.py` (see the logsetup decision in
      the Notes). Re-point its ELEVEN importers under `scufris/` and
      `tests/test_logsetup.py` at `scufris_core`; update the `logsetup` mention
      in `scufris/README.md`. The census, pasted rather than counted by hand -
      run on the base tree, `rg -l 'from \.+logsetup import|from
      scufris\.logsetup import' scufris/`, which is eleven and not the twelve an
      earlier pass wrote (four of them are function-local imports inside the MCP
      servers' `main()`, which is why an eyeball count missed the shape):
      `agent/appserver.py`, `agent_mcp_server.py`, `api/request_log.py`,
      `backends/claude.py`, `backends/codex.py`, `backends/opencode.py`,
      `cli.py`, `den_mcp_server.py`, `host_mcp_server.py`, `hostd/main.py`,
      `mcp_server.py`.
- [x] Write `packages/core/src/scufris_core/__init__.py` as the ONLY surface a
      sibling may import: `DATABASE_FILENAME`, `FILE_MODE`, `SIDECAR_SUFFIXES`,
      `Base`, `Database`, `database_path`, `open_database`, plus
      `configure_logging`, `new_request_id`, `set_request_id`, `truncate`, with
      an explicit `__all__`. `SIDECAR_SUFFIXES` (`engine.py:66`) is on the list
      because `tests/test_db_state_boundary.py:38` already imports it - the
      facade sketched in NOTES.md was one name short of what exists.
- [x] Re-point the direct importers of the moved engine. The census is FOUR
      source files and SIX test files, not the "18 modules plus 25+ test files"
      an earlier pass wrote down - that count was of importers of `scufris.db`,
      which is unaffected because `scufris/db/__init__.py` stays as the
      composition facade:
      - `scufris/db/__init__.py:19`, `scufris/db/migrate.py:48`,
        `scufris/db/legacy/__init__.py:44`, `scufris/db/legacy/gate.py:19`
        (`db/legacy/` is not deleted until 20260803-214750, which runs LAST).
      - `tests/test_db_state_boundary.py:38`, `tests/test_agent_run_router.py:42`,
        `tests/test_legacy_agent_router.py:41`, `tests/test_domain_routers.py:47`,
        `tests/test_orchestrator_routers.py:48`, and any `scufris.db.engine`
        reference left by `rg -n 'scufris\.db\.engine|from \.\.?engine import'`.
      `scufris/db/__init__.py` keeps re-exporting `Database`, `database_path`,
      `open_database`, `DATABASE_FILENAME` alongside `open_state_database` /
      `state_database`, so the ~40 modules importing `from scufris.db import ...`
      are untouched. This is not a shim: those four names compose with
      `upgrade_to_head` and `import_legacy_state`, both root-owned.
- [x] Leave `scufris/db/migrate.py` at the root. It is Alembic-coupled and
      resolves `script_location` inside `scufris.db.migrations`, so moving it
      would give `core` an alembic dependency it does not otherwise need.
- [x] `scripts/check_file_size.py`: add `"packages"` to `COVERED_ROOTS`.
      Without it every file moved in this task and the four after it is silently
      exempt from the 600-line cap. In the same edit, widen `cap_for` from
      `relative.startswith("tests/")` to also accept `"/tests/" in relative`, so
      a test module under `packages/*/tests/` gets the 900-line TEST_CAP instead
      of the 600-line SOURCE_CAP - 20260803-214748 moves eight there, and
      `tests/test_app.py` is already on the ratchet at over 600.
- [x] Leave `flake.nix:74` `members` commented. It is an EDITABILITY filter on
      `mkEditablePyprojectOverlay`, not the member declaration - setting it to
      `["scufris"]` would make `packages/core` non-editable in the dev shell.
      Membership comes from `[tool.uv.workspace]`. Run `uv lock` and commit the
      regenerated `uv.lock`.
- [x] Re-enter `nix develop` (or run `nix flake check`) after `uv lock`. The dev
      venv is a nix derivation built from the lock, so `import scufris_core`
      cannot resolve until it is rebuilt. This is an ordering constraint on the
      work, not an optional step: run the suite only after the rebuild.
- [x] Add `examples/core_unit_of_work.py`: open a temp SQLite database through
      `open_database`, write two rows in one `Database.transaction()`, roll a
      second transaction back, print the surviving count. No host, no provider,
      no network, no import of `scufris`.
- [x] Add `tests/test_examples.py` running an explicit OPT-IN list of offline
      examples - `OFFLINE = ("core_unit_of_work.py",)` - each as a subprocess,
      failing on non-zero exit. Opt-in is the honest version: `auth_session.py`
      boots a real uvicorn and `host_inspect.py` / `nixos_change.py` need a real
      NixOS box, and no marker distinguishing them exists yet. This is the
      harness the later packages plug into.
- [x] Write `tasks/20260803-213242/DECISION.md` - the EPIC folder, per this
      task's Definition of Done - recording the ten-unit cut, the
      public-API-only import rule, the rejection of Protocol ports, distributed
      tables over central CRUD, one Alembic history, and the logsetup decision.
- [x] Update `scufris/README.md` section 9 and the `AGENTS.md` sources table:
      the module map now has two roots.

## Definition of Done

- `core` imports on its own and depends on no sibling
  (cmd: `python -c "import scufris_core, scufris_core.engine"`, run in a dev
  shell rebuilt after `uv lock`).
- `core` declares no domain table, and the check that says so is falsifiable
  (test: `test_core_is_domain_free`,
  `test_the_domain_free_check_rejects_the_pre_move_tree`).
- The boundary machinery exists and runs over the packages that exist; its
  red-first proof is 20260803-214747's
  (test: `test_no_package_imports_a_sibling_private_module`).
- The 600-line cap still covers every moved file
  (cmd: `python scripts/check_file_size.py && rg -q '"packages"' scripts/check_file_size.py`).
- Tests that move into a package still run in the canonical gate
  (cmd: `rg -q 'packages/\*/tests' pyproject.toml`).
- The whole suite still passes with the moved engine
  (cmd: `python -m pytest`).
- The example runs green offline and is gated
  (cmd: `python -m pytest tests/test_examples.py -k core`).
- The build is unchanged in behavior
  (cmd: `nix flake check && nix build .#scufris`).
- The PUBLISHED artifact set installs clean. Added after review round 1: the
  proof above exercises the uv2nix path only, and the root wheel now declares
  `Requires-Dist: scufris-core`, which no registry can satisfy
  (cmd: `uv build --all-packages && uv venv /tmp/smoke &&
  uv pip install --python /tmp/smoke/bin/python dist/*-py3-none-any.whl &&
  /tmp/smoke/bin/scufris --version`).
- Every workspace member ships the root's version, so that artifact set is one
  release (test: `test_every_workspace_member_shares_the_root_version`,
  `test_a_member_left_behind_is_rejected`).
- The decisions behind the carve are recorded before four packages depend on
  them (cmd: `test -f tasks/20260803-213242/DECISION.md`).

## Notes

- Parent: 20260803-213242. Read its Epic section before starting; it carries the
  dependency graph, the boundary rule and the Alembic mechanics.
- `uv2nix.lib.workspace.loadWorkspace` is already the loader (`flake.nix:61`),
  and it discovers members from `[tool.uv.workspace]`. `flake.nix:74` is a
  different knob and stays commented.
- Keep `core` small enough that its contents are obvious. If something is
  arguably domain-specific, it does not go here.

### The logsetup decision (this task's, per its Step)

`scufris/logsetup.py` MOVES to `scufris_core`, and the allowlist names it
explicitly. Evidence: 87 lines, imports only `logging`, `uuid` and
`contextvars` - nothing from `scufris` - and it is imported by twelve modules
that the carve splits across at least four future packages (`hostd/main.py`,
`cli.py`, four MCP servers, three backends, `api/request_log.py`,
`agent/appserver.py`). `scufris/hostd/main.py:17` is the concrete blocker:
20260803-214747 cannot make `hostd` a member while it imports a root module.

The alternative - `hostd/main.py` configures its own logging - was rejected: it
buys `core` no smallness (the module is already generic) and costs a second log
format that drifts from the first, plus a duplicated request-id contextvar. One
format across the app is the property `logsetup` exists to hold.

This is exactly the workflow the allowlist is for: adding to `core` is a
deliberate act with a written justification, not a property check that silently
waves things through.

### Scope corrected 2026-08-03 after the understanding pass

Four things this task originally said to move are gone from it, all verified
against the tree:

- `scufris/enums.py` STAYS at the root. All ten symbols are domain -
  `ORCHESTRATOR_ID`, `HOST_AGENT_ID`, `Audience`, `audience_for`, `AuthMode`,
  `AuthPolicy`, `Backend`, `PermissionMode`, `AgentState`, `RunPhase` - and
  each travels with its package later.
- `ids` - `python-ulid` is declared in `pyproject.toml` and imported by zero
  files. Creating the module would be speculative.
- `time` and generic error types - neither exists. There is no
  `scufris/errors.py`; errors are local and domain-specific.
- "the session factory" - a mis-description. This codebase uses SQLAlchemy
  CORE: no `sessionmaker`, no ORM `Session`. The unit of work is
  `Database.transaction()`, which yields a `Connection`.

`core`'s declared dependencies drop from five to one. `alembic`, `pydantic`,
`pydantic-settings` and `python-ulid` are all unused by what actually moves.

Resolving the contradiction 20260803-214749's NOTES routed here: `eventbus.py`
and the generic half of `supervisor.py` do NOT move in this task, and neither
does `RunPhase`. They belong to `core` on the rule - generic async plumbing, no
domain knowledge - but nothing needs them there until `hostctl` does, and
`RunPhase` is the supervisor's own phase enum, so it travels with its owner.
Move all three in 20260803-214749, where a second consumer is the evidence. Two
callers is an abstraction; one is a guess. So "all of `enums.py` stays at the
root" is right for THIS task, and is not contradicted by `RunPhase` leaving
later with the module that defines its meaning.

### Proofs run on the base branch

- `python -c "import scufris_core"` -> `ModuleNotFoundError`. Red.
- `rg -q '"packages"' scripts/check_file_size.py` -> rc 1. Red.
- `rg -q 'packages/\*/tests' pyproject.toml` -> rc 1. Red.
- `python -m pytest tests/test_examples.py -k core` -> file absent. Red.
- `python -m pytest` -> green today; it is the regression guard for the move,
  not a red-first proof.

### Scratch-verified mechanics

- A `[tool.uv.workspace]` member with `[tool.uv.sources] x = { workspace = true }`
  locks as `source = { editable = "packages/core" }` (uv 0.11.28).
- `testpaths = ["tests", "packages/*/tests"]` with no `packages/` directory
  present collects and exits 0.

## Close-out

### What and why

The repository is a `uv` workspace with one member, `packages/core` ->
`scufris_core`, holding `engine.py` (moved whole, 327 lines), `base.py` (the
`Base` class alone, out of `db/models.py`) and `logsetup.py`. Its `__init__`
is the entire public surface - eleven names with an explicit `__all__` - and
`sqlalchemy` is its only dependency.

Two claims that were README prose are now tests in
`tests/test_package_boundaries.py`, both reading the SOURCE tree with `ast`
rather than importing it, so a later carve can add a member before its wiring is
finished and still be checked. `test_core_is_domain_free` is an allowlist plus
two domain arms; `test_the_domain_free_check_rejects_the_pre_move_tree` points
the same helper at `scufris/db` and asserts both domain arms fire, so the green
one cannot be green by checking nothing.

`examples/core_unit_of_work.py` is the runnable proof: it declares its own toy
table against the shared `Base`, commits two rows in one transaction, rolls a
second back, and counts two survivors, importing `scufris` nowhere.
`tests/test_examples.py` runs it as a subprocess off an explicit `OFFLINE`
opt-in list - the harness the four later carves plug into.

The five epic decisions plus the logsetup one and the "what `core` is NOT" list
are `tasks/20260803-213242/DECISION.md`, written before four packages depend on
them.

### Alternatives

- **A property check instead of an allowlist for `core`.** Rejected in the plan
  and confirmed while writing it: "declares no table" passes for `EventBus`,
  `Supervisor` and `RunPhase`, which is exactly the junk drawer the test is
  named against. Falsified by hand - a `junk.py` declaring one row makes all
  three arms fire; the module was deleted, not committed.
- **Moving `tests/test_logsetup.py` into `packages/core/tests/`,** which would
  have given the widened `testpaths` a live subject instead of a glob matching
  nothing. Rejected as scope the plan did not ask for: 20260803-214748 moves
  eight test modules there and is the honest first subject.
- **`hostd/main.py` configures its own logging** instead of moving `logsetup`.
  Recorded and rejected in DECISION.md section 6.

### Difficulties and diagnosis

- **`nix flake check`'s pytest failed with `ModuleNotFoundError: scufris_core`
  while the dev shell was green.** The dev venv installs every member EDITABLE
  against `$REPO_ROOT`, and `mkCheckWith` never set it: the checks had been
  finding `scufris` only because pytest puts the rootdir on `sys.path`, and
  nothing puts `packages/core/src` there. Fixed by exporting `REPO_ROOT=$PWD`
  in the shared check preamble, which points the editable finder at the sandbox's
  writable copy. This was a latent hole in the gate, not a new one: any package
  layout would have hit it.
- **The falsifier test broke the moment `Base` left `scufris/db`.** Seeding the
  declarative-class walk from `DeclarativeBase` alone found nothing once
  `models.py` merely imported `Base`. `DECLARATIVE_ROOTS` now names `Base` too,
  which is correct rather than a workaround: `Base` is the workspace's single
  shared base by contract, and every tree but `core` subclasses it without
  defining it.
- **Ruff sorted `scufris_core` into the third-party block.** Added
  `[tool.ruff.lint.isort] known-first-party = ["scufris", "scufris_core"]` so
  members group with the app.

### Evidence

- `python -c "import scufris_core, scufris_core.engine"` - ok, in a dev shell
  rebuilt after `uv lock`.
- `python -m pytest` - 1112 passed, 1 skipped.
- `tests/test_package_boundaries.py`, `tests/test_examples.py` - 4 passed.
- `python scripts/check_file_size.py` - clean, with `packages` covered.
- `ruff check .`, `ruff format --check .`, `mypy .` - clean over 234 files.
- `nix flake check` - all 7 checks pass. `nix build .#scufris` - built, and
  `./result/bin/scufris --help` runs.
- `uv.lock` resolves `scufris-core` as `source = { editable = "packages/core" }`.

### Review round 1

Twelve findings, all addressed in 038b332; one pushback, recorded on R1.1's
Response line. The BLOCKER is the one that mattered: creating a second
distribution silently made the published root wheel uninstallable, because
`uv build` builds only the root while the root now declares
`Requires-Dist: scufris-core`. The release job's own smoke test could not have
caught it - it installed one wheel chosen by `ls`. Fixed by building and
installing the whole set, and made durable by teaching
`release_tools.check_agreement` about member versions, so the set is guaranteed
to be one release rather than assumed to be.

The reviewer's read of the gap was exact: the DoD proved the uv2nix path and
nothing proved `uv build`, which is the artifact the release actually
publishes. That proof is now in the DoD.

Post-fix evidence:

- `uv build --all-packages` + one-resolution install of both wheels ->
  `scufris 0.1.0`. The old path (root-only build, single wheel) still fails
  with "requirements are unsatisfiable", so the fix is falsifiable.
- `python -m pytest` - 1114 passed, 1 skipped. The first run hit
  `tests/test_app.py::test_agent_run_reaches_done_and_persists_session`
  (`assert None == 'mock-session'`); it passes in isolation and the rerun was
  green. Same order-dependent failure the reviewer saw on a neighbouring test,
  tracked by 20260803-043935 and 20260803-100411.
- `ruff format --check .`, `ruff check`, `mypy .` - clean over 234 files.
- `nix flake check` - all checks pass. `nix build .#scufris` - built.
- `python -m scripts.release_tools check` - version sources agree on 0.1.0,
  members included.

### Reflection

The ordering constraint is the thing worth carrying forward: `uv lock` then
re-enter `nix develop` BEFORE running anything, because the dev venv is a
derivation built from the lock. It cost one confusing red suite here and is now
a line in `AGENTS.md`. The second is that `flake.nix` had to change for the
carve to be provable at the gate at all - worth checking early in each of the
four remaining moves rather than at the end.

The third came out of review: splitting one distribution into two changes the
RELEASE, not just the source tree, and nothing in this task's original proof
set looked at a built artifact. The four remaining carves each add a member and
each inherits that hazard; `check_agreement`'s member arm and the
`--all-packages` smoke step now cover them without another thought. The
recurring failure is narrower than it looks - both wrong counts in this record
were typed by hand. Paste the `rg -l` output.
