# Move the host control client into packages/hostctl

- PRIORITY: 102
- TAGS: refactor, v0.2.0, architecture, host
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
- PARENT: 20260803-213242

## Story

As the Scufris maintainer, I want the unprivileged host control client moved
into `packages/hostctl`, so that the completed host-agency pillar is parked
behind one boundary and can be left alone while the rewrite happens next to it.

`hostctl` is the client that DRIVES `hostd`: it builds an action, gets a
preview, holds it for operator approval, dispatches the approved action over the
socket, watches the result, and bridges approval requests out to a channel. Plus
the NixOS configuration change flow with generation rollback.

**This is the one child that is NOT a pure move.** `EventBus` and the generic
half of `Supervisor` have to be hoisted into `core` first, `Settings` has to be
narrowed out of `hostconfig/service.py`, two row classes have to leave
`scufris/db/models.py`, and one test file has to be split. Budget for real
edits, not `git mv`.

## Steps

Ordered. Each of 1-4 is committable on its own and leaves the suite green;
5-12 land the package.

### 1. Hoist `eventbus` into `core`

- [x] `git mv scufris/eventbus.py
      packages/core/src/scufris_core/eventbus.py`. 130 lines, imports nothing
      from `scufris`, so the move is mechanical.
- [x] Export `EventBus` from `packages/core/src/scufris_core/__init__.py`
      `__all__` and extend that module's docstring with why it is here: the
      second consumer (`hostctl`) is in another distribution, so the bus cannot
      stay at the root without `hostctl` importing the app.
- [x] Add `"eventbus"` to `CORE_MODULES` in
      `tests/test_package_boundaries.py:34`, with the justification in the
      comment above it (that block is the allowlist's own documentation).
- [x] Re-point the eight importers - confirmed by grep, the TASK's earlier line
      numbers had drifted: `scufris/supervisor.py:39`,
      `scufris/orchestrator/turn.py:22`, `scufris/orchestrator/runs.py:32`,
      `scufris/api/sse.py:16`, `scufris/api/agent_runs.py:33`,
      `scufris/hostclient.py:47`, `scufris/hostconfig/models.py:15`,
      `scufris/hostconfig/service.py:24` (this last one is NOT in the parent's
      list). Plus five test modules: `tests/test_supervisor.py:17`,
      `tests/test_eventbus.py:8`, `tests/test_orchestrator_routers.py:49`,
      `tests/test_chat_router.py:32`, `tests/domain_router_fakes.py:20`,
      `tests/test_legacy_agent_router.py:41`.
- [x] `git mv tests/test_eventbus.py packages/core/tests/test_eventbus.py`. It
      tests only the bus and imports nothing else; `packages/*/tests` is already
      in `testpaths`, so it stays in the canonical gate.

### 2. Split `supervisor` into `core` and the root

- [x] Move `supervisor.py:1-411` (module docstring, `AgentRunStalled`,
      `RunState`, `_Run`, `Supervisor`, the `EventT`/`MakeStream`/`Reservation`/
      `OnComplete` aliases) to
      `packages/core/src/scufris_core/supervisor.py`. Drop `from .agent import
      StreamError, StreamEvent` (`supervisor.py:37`) from the moved half - it is
      used only by the agent region.
- [x] Leave `supervisor.py:413-450` at the root - `AgentSupervisor`,
      `_agent_error_event`, `_agent_error_detail`, `agent_supervisor()` - now
      importing `Supervisor` from `scufris_core`. Confirmed by reading the file:
      the region is exactly the `# --- the agent's supervisor ---` block and
      nothing above it touches `StreamEvent`.
- [x] Move `RunPhase` (`scufris/enums.py:131-138`) INTO
      `scufris_core/supervisor.py` rather than giving it a module of its own.
      It is the supervisor's phase enum, it has no other owner, and one module
      is one allowlist entry instead of two. Record in DECISION.md.
- [x] Delete `RunPhase` from `scufris/enums.py` and re-point its two live
      readers - `scufris/orchestrator/runs.py:31` and `tests/test_enums.py:12`
      - at `scufris_core`. No compat re-export from `enums.py`: a shim with two
      call sites is not worth the concept. The other nine symbols stay.
- [x] Export `RunPhase`, `RunState`, `Supervisor`, `AgentRunStalled` from
      `scufris_core/__init__.py`, and add `"supervisor"` to `CORE_MODULES` with
      its justification.
- [x] Re-point the six `Supervisor` importers:
      `scufris/orchestrator/turn.py:23`, `orchestrator/runs.py:36`,
      `hostclient.py:48`, `app.py:99`, `host_approvals.py:56`,
      `hostconfig/models.py:16`, `api/agent_runs.py:45`,
      `api/legacy_agent.py:47`. `AgentSupervisor` and `agent_supervisor` still
      come from `scufris.supervisor`; only `Supervisor`, `RunState` and
      `RunPhase` move.
- [x] Split `tests/test_supervisor.py` the same way the source splits: the
      generic lifecycle tests move to `packages/core/tests/test_supervisor.py`,
      anything asserting on `StreamError`/`agent_supervisor` stays at the root.

### 3. `core` gains pydantic

- [x] Add `"pydantic>=2.0.0"` to `packages/core/pyproject.toml` dependencies -
      `RunState(BaseModel)` needs it - and correct that file's "nothing here
      needs pydantic" comment, which this step falsifies.
- [x] Record in `tasks/20260803-213242/DECISION.md`: `core` is no longer
      sqlalchemy-only, and why `RunState` stays a `BaseModel` (it is serialized
      straight to the API by `api/agent_runs.py`).
- [x] `uv lock` and `uv sync`.

### 4. Narrow `Settings` out of `hostconfig/service.py`

- [x] `ConfigChangeService.__init__` takes `config_repo: Path` and
      `config_attr: str` instead of `settings: Settings`. Confirmed by reading
      the file: those are the only two fields it uses, at
      `service.py:99`, `:100` and `:107`.
- [x] Update the one production caller, `scufris/app.py:461`, to pass
      `settings.host_config_repo` and `settings.host_config_attr`, and the one
      test caller at `tests/test_nixos_config_change.py:360-361`.
- [x] Drop `from ..config import Settings` (`service.py:23`). After this the
      `hostconfig/` tree imports nothing from the root but `db` and `eventbus`.

### 5. Create the distribution

- [x] `packages/hostctl/pyproject.toml`: name `scufris-hostctl`, version
      `0.1.0`, `requires-python >=3.13`, wheel packages `src/scufris_hostctl`,
      dependencies `scufris-core`, `scufris-host`, `scufris-hostd`, `pydantic`,
      `sqlalchemy`. It needs `scufris-host` and `scufris-hostd` as REAL
      dependencies, not protocol types: `hostconfig/changes.py:21-22` uses
      `scufris_host`'s `Outcome`/`Runner`/`run_command` and `scufris_hostd`'s
      `Executor`/`run_action`, and `hostconfig/resolve.py:17` uses
      `Runner`/`nix_cli`/`run_command`. All are facade exports, so the import
      rule holds. `sqlalchemy` because it owns two tables.
- [x] Root `pyproject.toml`: add `"scufris-hostctl"` to `dependencies`,
      `scufris-hostctl = { workspace = true }` to `[tool.uv.sources]`, and
      `"scufris_hostctl"` to `[tool.ruff.lint.isort] known-first-party`. No
      exact pin - that is `hostd`'s wire-protocol problem, not this one.
- [x] `flake.nix` needs NO change: membership comes from
      `[tool.uv.workspace] members = ["packages/*"]` and `mkEditablePyproject`'s
      `members` filter is still commented out. Confirmed by reading
      `flake.nix:70-75`. `uv lock` is what carries the new member.

### 6. Move the modules

- [x] `scufris/host_actions.py` -> `packages/hostctl/src/scufris_hostctl/actions.py`
- [x] `scufris/host_approvals.py` -> `.../scufris_hostctl/approvals.py`
- [x] `scufris/hostclient.py` -> `.../scufris_hostctl/client.py`
- [x] `scufris/hostconfig/` -> `.../scufris_hostctl/hostconfig/` (all six
      modules, name unchanged)
- [x] The three flat modules drop their `host`/`host_` prefix: inside
      `scufris_hostctl` it is redundant. `hostconfig/` keeps its name because it
      is already a package and renaming it buys nothing. Record in DECISION.md;
      this is the only naming change in the task.
- [x] Add `packages/hostctl/src/scufris_hostctl/README.md`, matching
      `scufris_host`'s and `scufris_hostd`'s: what the package is, the
      `propose -> preview -> approve -> apply -> audit -> roll back` contract
      from the CLIENT side, and the R3 change flow.

### 7. Move the two tables

- [x] Move `HostActionRow` (`scufris/db/models.py:265-298`) and
      `ConfigChangeRow` (`:301-336`) - docstrings included - to
      `packages/hostctl/src/scufris_hostctl/models.py`, declared against
      `scufris_core.Base`.
- [x] Update `scufris/db/models.py`'s module docstring, which currently names
      `host_action` and `config_change` as app-owned (`:18-22`).
- [x] `scufris/db/migrations/env.py` imports `scufris_hostctl.models` alongside
      `scufris.db.models` before reading `Base.metadata`, or the two tables
      vanish from autogenerate. This is the failure mode step 9 tests.
- [x] `tests/test_db_state_boundary.py:38,310-312` imports `HostActionRow` from
      `scufris.db.models`. Re-point it at
      `Base.metadata.tables["host_action"]` rather than at
      `scufris_hostctl.models` - a root test reaching into a package's private
      models module is exactly what the boundary rule is about, even though
      `tests/` is outside `_import_roots()`.
- [x] No migration revision: the tables' definitions are unchanged, only their
      declaring module. `test_schema_has_no_pending_autogenerate_diff` proves
      it, and it goes RED if `env.py` forgot the import.

### 8. The facade

- [x] `packages/hostctl/src/scufris_hostctl/__init__.py` exports everything the
      root imports today. From the grep of every importer, at minimum:
      `HostActionStore`, `HostActionRecord`, `render_action`,
      `confirmation_for`, `AlreadyDecided`, `UnknownAction`,
      `HostApprovalService`, `ConfirmationRequired`, `decision_message`,
      `HostdClient`, `HostdError`, `HostdUnavailable`, `HostSupervisor`,
      `ConfigChange`, `ConfigChangeBuilder`, `ConfigChangeStore`,
      `ConfigChangeService`, `ConfigChangeRefused`, `ConfigSupervisor`,
      `ChangeState`, `ChangeInFlight`, `UnknownChange`, `Resolved`,
      `render_change`, `HostConfigDeps`'s inputs.
- [x] `ConfigChange`, `ConfigSupervisor` and the bus aliases live in
      `hostconfig/models.py`; the import rule forbids the root reaching into a
      sibling's `models`, so the facade is how they travel. Same for
      `HostSupervisor` out of `client.py`.
- [x] NOT exported: `HostActionRow`, `ConfigChangeRow`. The tables are private.

### 9. Prove the metadata

- [x] Write `test_every_package_model_is_registered` in
      `tests/test_db_migrations.py` (not `test_package_boundaries.py`, whose
      whole method is `ast` and no imports - this one has to import).
      For every `packages/*/src/*/models.py` on disk, parse out its
      `__tablename__` values and assert each is in `Base.metadata.tables` after
      importing exactly what `migrations/env.py` imports. That is the real
      guard: a package whose `env.py` import is missing has tables that are
      never created.
- [x] Its docstring states the narrow claim, per the epic: this catches a
      package whose tables silently never exist, which is a broken feature. It
      is not data-loss protection - v0.2.0 has no operator data to lose.
- [x] `test_declared_tables_are_the_only_ones`
      (`tests/test_db_migrations.py:478`) must stay green with the same
      fourteen names. It checks the opposite direction and does not replace
      this.

### 10. Split the config-change suite

- [x] `tests/test_nixos_config_change.py` is 818 lines and splits at the
      `_app` helper (`:355`). The SERVICE half - `:189-354` plus
      `test_the_change_registry_stays_bounded` (`:797`) - moves to
      `packages/hostctl/tests/test_nixos_config_change.py`. The APP half -
      everything from `:355` that boots `create_app` - stays at
      `tests/test_nixos_config_change.py`.
- [x] The moved half must be SELF-CONTAINED: `packages/*/tests` does not see
      `tests/conftest.py`, so the `_Helper`, `_login`, `_settings`, `ORIGIN`
      imports at `:28` cannot travel, and neither can the `database` fixture
      `:797` uses. The `config_repo` fixture (`:136`) is already local and
      travels as-is. Follow the precedent the comment at `:55-58` records: this
      suite already owns its own fixture shapes rather than reaching across a
      distribution boundary.
- [x] Without the split, `pytest packages/hostctl/tests` runs an empty
      directory and DoD 3 is vacuous.
- [x] Move `tests/test_host_action_decisions.py` too if, on inspection, it
      boots no app - decide during the work, and defer with a reason in the
      commit if it does.

### 11. The example

- [x] `examples/hostctl_approval_flow.py`: build an action against a FAKE hostd
      (the `FakeExecutor` seam `packages/hostd/tests/test_nixos_activation.py`
      and `examples/hostd_socket_roundtrip.py` already use), preview it,
      approve it, dispatch it, print the audit trail. Offline, no root, no real
      socket, temp database.
- [x] Add `"hostctl_approval_flow.py"` to `OFFLINE` in
      `tests/test_examples.py:33`. The parametrized id then makes
      `pytest tests/test_examples.py -k hostctl` select it.

### 12. Re-point every importer and the docs

- [x] The complete list, from grep - larger than the parent's estimate.
      Production: `scufris/app.py:46,66,70,74,80`,
      `scufris/api/host.py:43,50,59`, `scufris/api/hostconfig.py:23`,
      `scufris/api/errors.py:17`, `scufris/api/agent_runs.py:35`,
      `scufris/host_watch.py:29,30`, `scufris/host_approval_bridge.py:20,21`,
      `scufris/mcp_host_tools/actions.py:81,168` (function-local imports),
      `scufris/telegram/wiring.py:36,37,42`, `telegram/render.py:38`,
      `telegram/contracts.py:19`, `telegram/approvals.py:25`,
      `telegram/bot.py:25`.
- [x] Tests: `tests/domain_router_fakes.py:17,21,27`,
      `tests/test_domain_routers.py:51,54,63,64`,
      `tests/test_route_contract.py:38`,
      `tests/test_host_action_decisions.py:31`,
      `tests/test_telegram_approvals.py:649`,
      `tests/test_host_mcp_server.py:367`,
      `tests/test_nixos_config_change.py:34,236,353`.
- [x] Examples: `examples/host_action.py:37,38`,
      `examples/host_agent.py:55,297`, `examples/nixos_change.py:41,42`.
- [x] `scufris/host_watch.py` STAYS at the root. It imports eleven root modules
      (`agent_diagnostics`, `agent_store`, `checks`, `config`, `digest`,
      `health`, `host`, `host_approvals`, `hostclient`, `hostd.audit`,
      `scheduler`), most of which v0.2.0 deletes; moving it would make
      `hostctl` import agents and projects.
- [x] `scufris/host_approval_bridge.py` STAYS at the root. It couples approvals
      to the conversation, which does not exist yet. The epic's open question -
      are host approvals conversation events - decides where it finally lands.
- [x] Docs, matching what 6d998c8 touched for `hostd`: `AGENTS.md:18` (the
      workspace-members row) and a `packages/hostctl/.../README.md` row at
      `:19-20`; `README.md:31-32`; `scufris/README.md` - the module map rows at
      `:481-484` move to the packages table at `:466-467`, and the diagram
      labels at `:33,35,69` and the prose at `:179-190` get the new import
      root; `CHANGELOG.md`.

## Definition of Done

Every `cmd:` below was run on `master` at plan time and is RED for the intended
missing change, except where noted.

- The package imports on its own
  (cmd: `uv run python -c "import scufris_hostctl"`) - base: exit 1,
  `ModuleNotFoundError`.
- It owns its tables and they are still reachable from the migration metadata
  (test: `test_every_package_model_is_registered`) - base: exit 5, the test
  does not exist.
- Its own suite passes unmoved in behavior, is not empty, and still runs in the
  canonical gate
  (cmd: `uv run python -m pytest packages/hostctl/tests && uv run python -m pytest --collect-only -q | rg -q packages/hostctl`).
  Base: exit 4 (no such directory), and `rg` exit 1.
- `core` grew only by an allowlisted, justified entry
  (test: `test_core_is_domain_free`). NOT red on base - it is a GUARD that is
  green now and must stay green. It goes red the moment `eventbus` or
  `supervisor` lands in `core` without the matching `CORE_MODULES` edit, which
  is the failure it exists to catch.
- No package reaches into a sibling's internals
  (test: `test_no_package_imports_a_sibling_private_module`). Same guard shape;
  it earns its keep here because `hostctl` is the first member with a private
  `models` the root used to import directly.
- The approval flow is provable without root and without a real socket
  (cmd: `uv run python -m pytest tests/test_examples.py -k hostctl`) - base:
  exit 5, no test selected.
- The packaged build is unchanged and the app still boots on NixOS
  (cmd: `nix flake check && nix build .#scufris .#scufris-web .#scufris-hostd && nix build .#scufris-vm-test`).
  The parent's `.#checks.x86_64-linux.scufris-vm-test` does not exist:
  `flake.nix:210-217` puts the VM tests in `packages`, DELIBERATELY out of
  `checks`, so the light gate stays fast. `scufris-vm-test` is the right one -
  it boots the real app, which imports `hostctl` through `app.py` - and
  `scufris-hostd-vm-test` covers the helper, which this task does not touch.

## Notes

- Parent: 20260803-213242.
- Named for its job: it is the client that controls `hostd`. `host` reads and
  needs no privilege; `hostd` is root in another process; `hostctl` is the
  unprivileged client between them.
- This pillar is COMPLETE for the target architecture. Move it, do not improve
  it. Its page gets unlinked later; the code stays.
- Baseline at plan time: full suite green on `master` after `uv sync --frozen`
  (`uv run python -m pytest`, exit 0).
- The dev shell's `VIRTUAL_ENV` does not match `.venv` and `uv` warns and
  ignores it. `uv run` targets `.venv`, which is the one with the workspace
  members installed; a bare `python -m pytest` from the nix dev env cannot
  import `scufris_core` at all.
- Assumption: `test_host_action_decisions.py` is app-level and stays. Step 10
  says to check rather than assume.
- Assumption: the three flat modules get their `host` prefix dropped (step 6).
  If review prefers a literal `git mv`, that is a rename-only change and costs
  nothing to reverse.

## Close-out

**What.** `packages/hostctl` ships `scufris-hostctl`: `actions` (the decision
journal, `confirmation_for`, `render_action`), `approvals` (the one decision
seam), `client` (the socket), `hostconfig/` (the R3 change flow) and `models`
(the two tables it owns), behind a facade the root reaches only through. Four
preparatory commits made it possible: `eventbus` and the generic run engine
hoisted into `scufris_core`, `core` gaining `pydantic` for `RunState`, and
`ConfigChangeService` narrowed from `Settings` to two values.

**Why this shape.** The four load-bearing choices are in `DECISION.md`:
`RunPhase` inside `scufris_core.supervisor` rather than its own module, the
generic/agent cut of `Supervisor` at the banner the file already carried,
globally unique test basenames (and no `conftest.py` in a package suite), and
the dropped `host`/`host_` prefixes. `host_watch.py` and
`host_approval_bridge.py` stayed at the root as planned - both couple approvals
to things the client must not know about.

**Alternatives.** `--import-mode=importlib` instead of renaming the split test
halves (a suite-wide rewrite bought for a filename); a `RunPhase` re-export from
`scufris/enums.py` (two call sites); moving the watch and bridge modules too.
All recorded and rejected in `DECISION.md`.

**Difficulties.**

- `packages/hostctl/tests/conftest.py` looked like the obvious place for the
  package suite's `database` fixture and broke fifteen root modules: pytest
  imports every `conftest.py` under the bare name `conftest`, so it won the name
  for the whole run and the root's `from conftest import ...` lines resolved
  into it. Diagnosed from the collection error naming the WRONG file's path for
  a root import. The fixture now lives in the test module; no package has a
  `conftest.py`. Recorded as an amendment to DECISION.md point 3.
- The plan's "no real socket" for the example is not achievable and should not
  be: `HostdClient` is the socket. `examples/hostctl_approval_flow.py` runs the
  real helper in-process on a temporary socket over `FakeRunner`/`FakeExecutor`
  instead - no root, no network, no NixOS machine. DECISION.md point 5.

**Evidence.**

- `uv run python -m pytest`: 1124 passed, 1 skipped.
- `uv run mypy .`: no issues in 244 files. `ruff check` / `ruff format --check`:
  clean.
- `uv run python -c "import scufris_hostctl"`: exit 0.
- `uv run python -m pytest packages/hostctl/tests`: 12 passed; and
  `pytest --collect-only -q | rg packages/hostctl` matches, so it stays in the
  canonical gate.
- `tests/test_db_migrations.py::test_every_package_model_is_registered` and
  `test_declared_tables_are_the_only_ones`: both pass, same fourteen names.
- `tests/test_package_boundaries.py`: 3 passed, including
  `test_core_is_domain_free` and
  `test_no_package_imports_a_sibling_private_module`.
- `uv run python -m pytest tests/test_examples.py -k hostctl`: 1 passed.
- `nix flake check` exit 0; `nix build .#scufris .#scufris-web .#scufris-hostd`
  and `nix build .#scufris-vm-test` both exit 0.

**Reflection.** The assumption in the plan's Notes held:
`test_host_action_decisions.py` boots `create_app` eight times, so it is an app
test and stayed. The two assumptions the plan flagged as cheap to reverse - the
dropped prefixes, and the test-file split - were both worth taking; the split in
particular is what makes `pytest packages/hostctl/tests` a real gate rather than
an empty directory. The one thing worth carrying forward is the `conftest.py`
trap: it is invisible until a second test root exists, and every future package
suite inherits it.

### Review round 1

Seven findings, all fixed, none disputed. The one that changed the design is
R1.1: the `SCHEMA_ASSEMBLY` exemption in
`test_no_package_imports_a_sibling_private_module` rested on a false premise -
Alembic needs the import SIDE EFFECT, not the row classes, and the facade
already produces it - so `env.py` imports `scufris_hostctl` and the boundary
rule now ships with zero holes. DECISION.md point 6 records it. The rest were
prose: a README claim the plan's overturned "no real socket" clause left behind
(R1.2), four stale pre-move module names and one stale `:class:` target (R1.3,
R1.4), three "thirteen row classes" counts that are eleven now (R1.5), two
no-op `models` imports (R1.6), and task IDs in the moved `models.py` docstrings
(R1.7).

R1.7 was fixed only where it was raised. The same task-ID-in-docstring pattern
is repo-wide and entirely pre-existing; sweeping it belongs to its own task
rather than to a carve branch.

Re-verified after the fixes: `pytest` 1124 passed / 1 skipped, `mypy` clean on
244 files, `ruff check` and `ruff format --check` clean, and all seven
Definition of Done proofs re-run green including `nix flake check`,
`nix build .#scufris .#scufris-web .#scufris-hostd` and
`nix build .#scufris-vm-test`.

### Review round 2

One finding, R2.1, fixed. All seven round-1 findings were confirmed and ticked.

`test_every_package_model_is_registered` was vacuous in the run that gates the
branch: it imported `env.py`'s module list into the CURRENT interpreter, where
`scufris_hostctl` is already loaded by the app, the example test and the
package suite, so the tables were registered whatever `env.py` said. It now
builds the metadata in a `python -c` child (`_tables_env_registers`), which
sees only what `env.py` names. DECISION.md point 7 records it.

The fix was proven by breaking what it guards, not by argument: with
`scufris/db/migrations/env.py:27` deleted, the full suite is 1 failed / 1123
passed, where before the fix it was 1124 passed and exit 0. Restored, the suite
is 1124 passed / 1 skipped again.

Re-verified after the fix: `mypy` clean on 244 files, `ruff check` and
`ruff format --check` clean, and all seven Definition of Done proofs green
including `nix flake check`, `nix build .#scufris .#scufris-web .#scufris-hostd`
and `nix build .#scufris-vm-test`.
