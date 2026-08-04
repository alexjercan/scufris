# Move the root helper into packages/hostd

- PRIORITY: 103
- TAGS: refactor, v0.2.0, architecture, host
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
- PARENT: 20260803-213242

## Story

As the Scufris maintainer, I want the root helper moved into `packages/hostd`,
because it is complete and its boundary - a unix socket - is externally
observable, so a behavior change here is caught rather than absorbed.

It runs SECOND, after `packages/host` (20260803-214748, landed as 09cf946). That
reorder resolved NOTES.md open question 1: `scufris/hostd/` now imports
`scufris_host` (`engine.py:33`, `preview.py:26`, `nixos.py:41`, `executor.py:25`,
`actions/plans.py:12`, `actions/validate.py:18`) and `scufris_core`
(`main.py:17`, `configure_logging`) and nothing else from the root. The tree is
import-clean today; the blocker is gone.

No behavior changes. This is a move.

## Steps

- [x] Write `DECISION.md` first: it is the record for the four choices below
      (the drift guard's real shape, `api/errors.py`, the facade rule, the
      `create_subprocess` sweep's root). Later steps implement it.
- [x] Create `packages/hostd/pyproject.toml`: name `scufris-hostd`, version
      `0.1.0`, `requires-python = ">=3.13"`, dependencies
      `scufris-core`, `scufris-host`, `pydantic>=2.0.0`. Hatchling wheel over
      `src/scufris_hostd`, mirroring `packages/host/pyproject.toml`.
      `scufris-core` is on the list because `main.py` imports
      `configure_logging`; the epic's graph says `hostd -> host` only, so amend
      it to `hostd -> core, host` in the parent record.
- [x] `git mv scufris/hostd packages/hostd/src/scufris_hostd`, including its
      `README.md`. Rewrite its own relative imports only where the move breaks
      them (`from ..hostd.X` does not appear inside the tree; `from .X` does and
      survives).
- [x] Widen `scufris_hostd/__init__.py` by ONE name: `encode`, from `protocol`.
      NOTES.md claims the facade already covers every app-side name; it does
      not - `hostclient.py:44` imports `encode` and `__all__` omits it. Nothing
      else is missing (checked against every root import site).
- [x] Move the three pure-hostd test modules with `git mv` to
      `packages/hostd/tests/`: `test_hostd_audit.py`, `test_host_actions.py`,
      `test_nixos_activation.py`. They take only `tmp_path` from pytest, so they
      need no root `conftest.py`; `test_nixos_activation.py` imports
      `test_host_actions` by module name, which keeps working because both land
      in one directory with no `__init__.py`. Their `scufris_host` imports
      (`FakeRunner`, `CommandResult`, `ok_result`, `NIX_FEATURES`) stay as they
      are - `hostd` depends on `host`.
- [x] Move the console script: drop `scufris-hostd` from the root
      `[project.scripts]`, add `[project.scripts] scufris-hostd =
      "scufris_hostd.main:main"` to `packages/hostd/pyproject.toml`. Rewrite the
      root comment that says "shipped from the same wheel so the two halves
      cannot drift" - that sentence stops being true here.
- [x] Add to the root `pyproject.toml`: `"scufris-hostd==0.1.0"` in
      `dependencies` (EXACT - see the drift guard below), `scufris-hostd =
      { workspace = true }` in `[tool.uv.sources]`, and `scufris_hostd` in
      `[tool.ruff.lint.isort] known-first-party`. Then `uv lock`.
- [x] Re-point every root SOURCE importer through the facade `scufris_hostd`.
      `test_no_package_imports_a_sibling_private_module` makes this mandatory,
      not stylistic: it walks `packages/*/src/*` plus `scufris/` and fails any
      `scufris_hostd.<submodule>` import. The sites, re-grepped:
      `hostclient.py:36-45` (three submodules, one merged import),
      `app.py:86,87`, `api/host.py:59,60`, `api/errors.py:16`,
      `api/auth.py:51`, `host_watch.py:30`, `host_actions.py:31,32`,
      `host_approvals.py:48-50`, `checks.py:41`, `hostconfig/changes.py:25`,
      `hostconfig/service.py:23`. Re-grep before editing; the line numbers moved
      once already in 09cf946.
- [x] Re-point the root tests and examples that import `hostd`. Every name they
      use is on the facade, so all of these become `from scufris_hostd import`:
      `tests/conftest.py:34`, `test_domain_routers.py:73-76`,
      `test_host_mcp_server.py:368-370`, `test_telegram_approvals.py:39,650-652`,
      `test_host_digest.py:47`, `test_host_action_api.py:31`,
      `test_host_action_decisions.py:32`, `test_nixos_config_change.py:52`,
      `examples/host_action.py:39`, `examples/host_agent.py:56`,
      `examples/host_digest.py:38`, `examples/nixos_change.py:48`,
      `examples/telegram_approval.py:50,225`.
- [x] Fix the three root tests that name the moving tree BY PATH, not by import:
      `test_db_state_boundary.py:273` (`scufris/hostd` -> the package source
      dir; add an `is_dir()` assert, because `glob` on a missing directory makes
      that check pass vacuously), `test_auth_machine.py:322` (the
      `create_subprocess` exemption), and `test_check_file_size.py:46` (a
      `cap_for` string argument naming a file that will not exist).
- [x] Root the `create_subprocess` sweep in `test_auth_machine.py` at
      `packages/*/src/*` as well as `scufris/`, and re-point the
      `hostd/executor.py` exemption at its new path. Leaving the sweep on
      `scufris/` alone drops `checked` from 6 to 5 - exactly its
      `assert checked >= 5` floor - and silently stops policing the helper's
      spawn. Every later carve task erodes it further.
- [x] Add `test_the_app_pins_hostd_to_one_exact_version` to
      `tests/test_release.py`: read both `pyproject.toml` files with `tomllib`,
      assert the root's `scufris-hostd` requirement is `==<the version in
      packages/hostd/pyproject.toml>`. This REPLACES the DoD's
      `test_hostd_and_app_report_the_same_protocol_version`; see DECISION.md for
      why that name had no honest subject and why the check has to be
      file-based.
- [x] Add `packages.scufris-hostd` to `flake.nix` - a second `mkApplication`
      over the same `runtimeVenv` with `package = pythonSet.scufris-hostd` -
      and re-point `nix/scufris-hostd.nix:47,48` (`package` default and
      `defaultText`) at it. `mkApplication` builds its output from the STRUCTURE
      of the package it is given, so `${pkgs.scufris}/bin/scufris-hostd`
      disappears the moment the console script moves; `:147` execs that path.
      This breaks at BUILD time, so the VM check is the proof.
- [x] Amend the parent epic (`tasks/20260803-213242/TASK.md`): Done Means item 9
      still reads `nix build .#scufris .#scufris-web && test -x
      result/bin/scufris-hostd`, which this task makes false. It becomes
      `nix build .#scufris-hostd && test -x result/bin/scufris-hostd`. Also
      amend the dependency graph line `hostd -> host` to `hostd -> core, host`.
- [x] Add `examples/hostd_socket_roundtrip.py`: a `HostdServer` over a unix
      socket in a temp directory, backed by `FakeExecutor` and `FakeFiles`,
      driven by a raw socket client through `propose -> preview -> approve ->
      apply -> audit`. No host, no network, no root. `tests/conftest.py:242-320`
      already stands one up in a thread; lift that shape. Add it to `OFFLINE` in
      `tests/test_examples.py:36`.
- [x] Update the path references that name the moving tree. Re-grep rather than
      trusting these line numbers: `README.md:32,361`, `AGENTS.md:20,84,128`
      (`:84` and `:128` are live instructions that go stale),
      `scufris/README.md:90,190,318,480`, `web/src/host-types.ts:4`,
      `docs/RELEASING.md` (one line: the `scufris-hostd` pin is bumped with the
      member versions, and the new test is what fails when it is not).
- [x] Add the CHANGELOG entry, in the shape 09cf946 used.

## Definition of Done

- The helper imports from its own distribution
  (cmd: `uv run python -c "import scufris_hostd"` - red on base: verified,
  exits 1).
- The app pins the helper to one exact version, and the pin is checked against
  the member it names
  (test: `test_the_app_pins_hostd_to_one_exact_version`).
- The hostd suite passes unmoved in behavior AND stays in the canonical gate; red
  on base, since the collect grep exits 1
  (cmd: `python -m pytest packages/hostd/tests && python -m pytest --collect-only | rg -q packages/hostd`).
- No root module reaches around the facade into the helper's internals
  (test: `test_no_package_imports_a_sibling_private_module`).
- The console script survives the move to a second distribution
  (cmd: `nix build .#scufris-hostd && test -x result/bin/scufris-hostd` - red on
  base: there is no such flake output).
- The package proves itself offline
  (cmd: `python -m pytest tests/test_examples.py -k hostd` - red on base: no
  such example).
- REGRESSION GUARDS, green on base and required to stay green: the privileged
  helper still builds and activates under NixOS
  (cmd: `nix build .#scufris-hostd-vm-test`), and the whole
  gate passes (cmd: `nix flake check`).

## Notes

- Parent: 20260803-213242. Decisions: `DECISION.md` in this folder.
- `hostd` is COMPLETE for the target architecture. Do not improve it while
  moving it; a behavior change here would be indistinguishable from a carve
  failure.
- It is the only package that legitimately runs as a separate process. Its
  boundary is a unix socket, not an import rule.
- **Scratch-verified, and it changes the plan.** With
  `[tool.uv.sources] scufris-hostd = { workspace = true }`, uv DROPS the version
  specifier: a deliberately mismatched `scufris-hostd==0.2.0` against a member at
  `0.1.0` resolved and synced clean, and `uv.lock` recorded only
  `{ name = "scufris-hostd", editable = "packages/hostd" }`. The built wheel
  DOES carry `Requires-Dist: scufris-hostd==0.2.0`. So the exact pin binds only
  downstream wheel installs, and no gate in this repo - `uv lock`, `uv sync`,
  `nix build` (which resolves from the lock) - would notice it going stale. The
  pin is worth having and it needs a test of its own; that is why the DoD's
  drift guard is a file-based check rather than an `importlib.metadata`
  comparison.
- `importlib.metadata.version("scufris")` raises `PackageNotFoundError` in the
  local `.venv`, so a metadata-based drift test would be environment-dependent
  on top of testing the wrong thing.
- Steps left deliberately alone: `tests/test_route_contract.py:241` (`"hostd"`
  is a route tag, not a path) and every `scufris-hostd` string in
  `nix/scufris-service.nix`, `nix/tests/*.nix`, `config.py` and `README.md` that
  names the UNIT or the socket. Those are operator-facing names and do not move.
- `packages/hostd/src/scufris_hostd/actions/models.py` is the first real subject
  for `test_no_package_imports_a_sibling_private_module`'s spirit. Nothing
  outside `hostd` imports it today; keep it that way.

## Close-out

**What and why.** `scufris/hostd/` is now `packages/hostd/src/scufris_hostd`,
the `scufris-hostd` distribution, with its console script and its three test
modules. Its boundary is a unix socket, so the carve is checked where that
boundary actually is: `examples/hostd_socket_roundtrip.py` drives
propose -> preview -> approve -> apply -> audit through a raw `AF_UNIX` socket
against a `FakeRunner`/`FakeExecutor` host, and `scufris-hostd-vm-test` runs the
real root unit on a real socket with a real activation. Nothing an operator sees
changed: same unit, same socket, same `PROTOCOL_VERSION = 1`, same audit lines.

The guarantee that DID change is stated in DECISION.md: the two halves of the
socket protocol no longer ship from one wheel. The replacement is an exact
`scufris-hostd==0.1.0` pin plus `test_the_app_pins_hostd_to_one_exact_version`,
which is file-based because uv drops the specifier for a workspace source - so
`uv lock`, `uv sync` and `nix build` all stay green against a stale pin, while
the published wheel carries it.

**Alternatives.** DECISION.md holds them: `importlib.metadata` equality, a
`hello` handshake, the app declaring its own error codes, leaving the spawn
sweep rooted at `scufris/`, and duplicating `configure_logging` to keep the
`hostd -> host` diagram. Two more were decided during implementation and added
as 5a and 5b - see below.

**Difficulties and diagnosis.**

1. `tests/test_nixos_config_change.py` imports `test_host_actions`, which the
   plan said only `test_nixos_activation.py` did. Collection failed the moment
   the module moved. It now declares its own three fixture constants
   (DECISION.md 5a) rather than reaching across a distribution boundary.
2. `tests/test_domain_routers.py` was 892 lines against a 900-line cap;
   collapsing four submodule imports into one facade import made it 901 and
   `check_file_size` failed. The fakes moved verbatim to
   `tests/domain_router_fakes.py`, leaving 617 lines (DECISION.md 5b).
3. The DoD first named the VM guard as
   `nix build .#checks.x86_64-linux.scufris-hostd-vm-test`. There is no such
   attribute: the VM tests are deliberately in `packages`, not `checks`, so the
   light gate stays fast. Corrected in the DoD to
   `nix build .#scufris-hostd-vm-test` (review R1.4).
4. `python -m pytest` from the worktree cannot see `scufris_core`: the dev
   shell's venv installs members editable against the MAIN checkout. Ran as
   `uv run --no-sync python -m pytest` here; `nix flake check` is the real gate
   and it passes.

**Evidence.**

- `uv run python -c "import scufris_hostd"` - ok.
- `python -m pytest` - 1119 passed, 1 skipped. `packages/hostd/tests` collects
  under the canonical run (`--collect-only | rg -q packages/hostd`).
- `pytest tests/test_release.py -k pins` - the drift guard passes.
- `nix flake check` - exit 0, "all checks passed!".
- `nix build .#scufris-hostd && test -x result/bin/scufris-hostd` - ok, and
  `packages.scufris` no longer carries it (`ls result/bin` is `scufris` alone).
- `nix build .#scufris-hostd-vm-test` - passes: the root unit builds and
  activates from the new flake output.
- `nix build .#scufris .#scufris-web` - ok. `web: npm run ci` - green.
- `examples/hostd_socket_roundtrip.py` runs offline and is in `OFFLINE`.

**Reflection.** Two of the four implementation surprises were test-to-test
coupling that no import graph over `scufris/` would have shown, because both
crossed test modules rather than source modules. The next carve child should
grep the moving tree's test modules for imports BY MODULE NAME before planning,
not after. The file-size ratchet firing on an import rewrite is the same lesson
from the other side: a file near the cap is a file that will block the next
mechanical change, and there are three carve children left that will rewrite
imports in exactly this file.
