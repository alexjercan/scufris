# Move read-only host inspection into packages/host

- PRIORITY: 104
- TAGS: refactor, v0.2.0, architecture, host
- KIND: TASK
- ACTIVITY: WORKING
- GATES: PLAN
- RESOLUTION: -
- PARENT: 20260803-213242

## Story

As the Scufris maintainer, I want read-only host inspection moved into
`packages/host`, so that the one product surface surviving the rewrite
untouched - Stats - sits behind a package that needs no privilege and imports no
sibling but `core`.

No behavior changes. This is a move.

The move's real work is not `git mv`. `scufris/` reaches into the host tree's
INTERNALS in thirteen places - `hostd/engine.py:33`, `hostd/preview.py:26-29`,
`hostd/nixos.py:41-43`, `hostd/executor.py:25`, `hostd/actions/plans.py:12-13`,
`hostd/actions/validate.py:18`, `hostconfig/resolve.py:17`,
`hostconfig/changes.py:23`, `app.py:57`, `api/host.py:33` - and once the tree is
a sibling distribution every one of them violates the epic's one rule.
`test_no_package_imports_a_sibling_private_module` is written and green today
precisely because `core` gave it nothing to police; its own docstring says it
"earns a red run once a second package is carved out beside `core`". This task
is that run. Routing those thirteen through the facade is the task.

## Steps

- [ ] Create `packages/host/` with its own `pyproject.toml` declaring ONLY
      `psutil` and `pydantic`. NOT `scufris-core`: confirmed by inspection, the
      whole moving tree (`scufris/host/*.py`, `metrics.py`, `processes.py`)
      imports nothing but stdlib, `psutil` and `pydantic` - no `scufris`
      module, not even `logsetup`, and no database. The epic declares
      `host -> nothing`; this package is the proof the rule is real. Mirror
      `packages/core/pyproject.toml`: same `requires-python`, hatchling,
      `packages = ["src/scufris_host"]`, `version = "0.1.0"` (the root's - it is
      what `release_tools.check_agreement` asserts and
      `tests/test_release.py:94` checks).
- [ ] `git mv scufris/host packages/host/src/scufris_host`, with
      `scufris/host/README.md` inside it. Move `scufris/metrics.py` and
      `scufris/processes.py` to `packages/host/src/scufris_host/` beside it -
      both qualify by the same inspection, and `HostStats` is what Stats serves.
      Move `scufris/host/__init__.py`'s `HostInspector`/`HostOverview` unchanged.
- [ ] Widen `scufris_host/__init__.py` to cover every name `scufris/` needs, so
      no root module has to reach past it. The existing `__all__` already
      exports all but four; ADD `nix_cli` (from `run`, wanted by six `hostd` and
      `hostconfig` modules), `MIN_HOST_OVERVIEW_TTL` and `HostOverviewCache`
      (from `overview`, wanted by `app.py:57` and `api/host.py:33`), and
      re-export `metrics`' and `processes`' public names. Verified: the three
      name sets are pairwise disjoint - `host.__all__` vs metrics' fifteen vs
      processes' six - so the flat facade needs no aliasing.
- [ ] Rewrite every `scufris/` import of the moved tree to the facade. Deep
      imports that MUST become facade imports (the thirteen above). Already
      shallow, so a pure rename: `checks.py:46,207`, `host_watch.py:26`,
      `app.py:56,81,86`, `api/host.py:32,53,54`, `mcp_host_tools/inspection.py`
      (fifteen function-local imports of `..host`), `telegram/wiring.py:42`,
      `telegram/render.py:38`, `telegram/contracts.py:19`. `scufris/digest.py`
      does not import the tree at all; leave it alone. The telegram three make
      the `telegram -> host` edge the epic's graph declares explicit.
- [ ] Wire the workspace: add `scufris-host` to the root `dependencies` and
      `[tool.uv.sources]`, add `scufris_host` to
      `[tool.ruff.lint.isort] known-first-party`, DROP `psutil` from the root
      `dependencies` (after the move zero root modules import it; `types-psutil`
      stays in the dev group for the package's own type checking), and
      regenerate `uv.lock`. `[tool.uv.workspace] members = ["packages/*"]` and
      `testpaths = ["tests", "packages/*/tests"]` already cover the new member -
      no change needed, and that is the point of how they were written.
- [ ] Move the five test modules that test the moved tree to
      `packages/host/tests/`: `test_host_inspection.py`, `test_host_nix_store.py`,
      `test_host_thermal.py`, `test_metrics.py`, `test_processes.py`. Verified
      they use no fixture from `tests/conftest.py`, so the new directory needs no
      `conftest.py`. `test_host_nix_store.py` does
      `from test_host_inspection import ok`; that import survives only because
      both land in the same directory. Everything else named `test_host_*` tests
      root modules (`host_actions`, `hostd`, the MCP server) and STAYS.
- [ ] Re-point the root tests that import the moved names: ~20 modules under
      `tests/`, including `conftest.py:41` (whose `make_fixture_stats()` builds a
      `HostStats`) and `test_host_mcp_server.py`'s function-local
      `from scufris.host import ...`. Root tests may import
      `scufris_host` deeply - `_import_roots()` globs `packages/*/src/*` plus
      `scufris/`, not `tests/` - but use the facade anyway.
- [ ] Add `examples/host_report_fixture.py`, an OFFLINE example rendering every
      report from canned fixtures, and add it to `tests/test_examples.py`'s
      `OFFLINE` tuple (today `("core_unit_of_work.py",)`). The seam already
      exists and is documented as existing for this: `FakeRunner` +
      `ok_result` live in `run.py` "so the example script ... can drive an
      inspection without touching the real host", and `HostInspector.__init__`
      takes `config_repo`, `system` and `cpu_sysfs` as `Path`s, so the sysfs- and
      filesystem-backed reports point at `tmp_path`. Cover the seventeen
      `render_*` functions in `render.py`. Scope OUT `HostStats`: `metrics.py`
      calls `psutil` at module level and shells to `nvidia-smi` via `subprocess`
      directly, so it has no injectable seam and adding one is the behavior
      change this task forbids. Leave `examples/host_inspect.py` and
      `examples/nixos_change.py` unlisted - they need a real NixOS box - but do
      re-point their imports.
- [ ] Write `test_stats_endpoint_matches_inspector_output` in `tests/`. It does
      not exist; the nearest coverage is `tests/test_app.py` and
      `tests/test_route_contract.py`. Cheap: assert the `/api/stats` body equals
      the fake `Collector`'s `collect()` payload. It is the one assertion that
      makes "Stats still serves the same payload" falsifiable across the move.
      It belongs in `tests/`, not the package: it exercises the composition
      root's route, not the package.
- [ ] Update the prose and non-Python references naming the moving tree:
      `README.md:31,135`, `AGENTS.md:19,127,128` (the line numbers in the brief
      were stale), `nix/tests/scufris-vm.nix:66`, `web/src/stats-types.ts:103`,
      and add a CHANGELOG entry under `[Unreleased]` as the `core` carve did.
      `flake.nix` needs NO change: `mkCheckWith` already exports `REPO_ROOT`
      for editable members.

## Definition of Done

- The package imports on its own and depends on NO sibling, `core` included
  (cmd: `uv run python -c "import scufris_host"`;
  test: `test_no_package_imports_a_sibling_private_module`).
- Its own suite passes unmoved in behavior AND still runs in the canonical gate
  (cmd: `python -m pytest packages/host/tests && python -m pytest --collect-only | rg -q packages/host`).
- The offline example renders every report from fixtures with no host access
  (cmd: `python -m pytest tests/test_examples.py -k host_report_fixture`).
- Stats still serves the same payload
  (test: `test_stats_endpoint_matches_inspector_output`).
- The whole gate is green
  (cmd: `python -m pytest && ruff check . && mypy scufris packages`).

## Notes

- Parent: 20260803-213242.
- This package needs NO privilege. If something here needs root, it belongs in
  `hostctl` or `hostd`.
- Stats is the one page that survives the UI demolition intact. Keep it working
  at every commit.
- Proofs run RED on the base (`e7cb027`), in `nix develop`: `import
  scufris_host` -> 1; `pytest packages/host/tests` -> 4 (no such path);
  `--collect-only | rg packages/host` -> 1; `pytest tests/test_examples.py -k
  host_report_fixture` -> 5; `-k test_stats_endpoint_matches_inspector_output`
  -> 5. `tests/test_package_boundaries.py` and the five moving test modules are
  green on the base, which is the no-behavior-change baseline.
- Run verification through `nix develop --command`. A bare `uv run` targets
  `.venv`, which has no pytest; the dev shell is the environment the flake
  checks use.
- The facade-widening decision is recorded in `DECISION.md`.
